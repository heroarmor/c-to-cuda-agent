/* lanczos.cu -- CUDA conversion of benchmark/complex/manybody/lanczos.c
 * Gather-based H-apply (one thread per basis state, no atomics).
 * Dot products via GPU tree reduction + host sum of block partials.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                 \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

/* ------------------------------------------------------------------ */
/*  Kernels                                                           */
/* ------------------------------------------------------------------ */

/* out = H * in  (gather formulation: each output element computed
   independently by reading from the flipped state, no atomics) */
__global__ void applyH_kernel(int Nspin, const double *in, double *out) {
    long dim = 1L << Nspin;
    long s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= dim) return;

    double v = in[s];
    double diag = 0.0;
    double off = 0.0;
    for (int i = 0; i < Nspin; ++i) {
        int j = (i + 1) % Nspin;
        int bi = (int)((s >> i) & 1), bj = (int)((s >> j) & 1);
        if (bi == bj)
            diag += 0.25;
        else {
            diag -= 0.25;
            off += 0.5 * in[s ^ ((1L << i) | (1L << j))];
        }
    }
    out[s] = diag * v + off;
}

/* tree-reduce dot product: each block produces one partial sum */
__global__ void dot_partial_kernel(long n, const double *a, const double *b,
                                    double *partials) {
    __shared__ double cache[256];
    int tid = threadIdx.x;
    long i = blockIdx.x * blockDim.x + tid;

    double sum = 0.0;
    long stride = gridDim.x * blockDim.x;
    while (i < n) {
        sum += a[i] * b[i];
        i += stride;
    }
    cache[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) cache[tid] += cache[tid + s];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = cache[0];
}

__global__ void w_axpy_kernel(long n, double *w, double c, const double *v) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] -= c * v[i];
}

__global__ void scale_kernel(long n, double *out, const double *in, double s) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] / s;
}

/* ------------------------------------------------------------------ */
/*  Host helper: dot product w/ GPU partial reduction                 */
/* ------------------------------------------------------------------ */
static double dot(long n, const double *d_a, const double *d_b,
                  double *d_partials, int nblocks, double *h_partials) {
    int threads = 256;
    dot_partial_kernel<<<nblocks, threads>>>(n, d_a, d_b, d_partials);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_partials, d_partials, nblocks * sizeof(double),
                           cudaMemcpyDeviceToHost));
    double s = 0.0;
    for (int i = 0; i < nblocks; ++i) s += h_partials[i];
    return s;
}

/* ------------------------------------------------------------------ */
/*  Tridiagonal eigenvalue helpers (host, identical to baseline)      */
/* ------------------------------------------------------------------ */
static int sturm_count(int m, const double *a, const double *b, double x) {
    int count = 0;
    double d = a[0] - x;
    if (d < 0) ++count;
    for (int k = 1; k < m; ++k) {
        if (d == 0.0) d = 1e-300;
        d = (a[k] - x) - b[k] * b[k] / d;
        if (d < 0) ++count;
    }
    return count;
}

static double smallest_eig(int m, const double *a, const double *b) {
    double lo = -1e6, hi = 1e6;
    for (int it = 0; it < 200; ++it) {
        double mid = 0.5 * (lo + hi);
        if (sturm_count(m, a, b, mid) >= 1) hi = mid;
        else lo = mid;
    }
    return 0.5 * (lo + hi);
}

/* ------------------------------------------------------------------ */
/*  Lanczos ground-state energy (GPU accelerated)                     */
/* ------------------------------------------------------------------ */
static double ground_energy(int Nspin, int maxiter) {
    long dim = 1L << Nspin;
    int m = maxiter < dim ? maxiter : (int)dim;
    int threads = 256;
    int nblocks = (int)((dim + threads - 1) / threads);

    /* allocate device memory */
    double *d_V, *d_w, *d_partials;
    size_t V_bytes = (size_t)m * dim * sizeof(double);
    size_t w_bytes = (size_t)dim * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_V, V_bytes));
    CUDA_CHECK(cudaMalloc(&d_w, w_bytes));
    CUDA_CHECK(cudaMalloc(&d_partials, nblocks * sizeof(double)));

    double *a = (double *)malloc((size_t)m * sizeof(double));
    double *b = (double *)malloc((size_t)m * sizeof(double));
    double *h_partials = (double *)malloc((size_t)nblocks * sizeof(double));
    double *h_tmp = (double *)malloc(w_bytes);

    /* initialise random V[0] on host, copy to device */
    unsigned long long rs = 0x1234567 ^ (unsigned long long)Nspin;
    double nrm = 0.0;
    for (long i = 0; i < dim; ++i) {
        rs = rs * 6364136223846793005ULL + 1442695040888963407ULL;
        double r = (double)(rs >> 11) / 9007199254740992.0 - 0.5;
        h_tmp[i] = r;
        nrm += r * r;
    }
    nrm = sqrt(nrm);
    for (long i = 0; i < dim; ++i) h_tmp[i] /= nrm;
    CUDA_CHECK(cudaMemcpy(d_V, h_tmp, w_bytes, cudaMemcpyHostToDevice));

    b[0] = 0.0;

    /* w = H * V[0] */
    applyH_kernel<<<nblocks, threads>>>(Nspin, d_V, d_w);
    CUDA_CHECK(cudaDeviceSynchronize());

    a[0] = dot(dim, d_V, d_w, d_partials, nblocks, h_partials);
    /* w -= a[0] * V[0] */
    w_axpy_kernel<<<nblocks, threads>>>(dim, d_w, a[0], d_V);
    CUDA_CHECK(cudaDeviceSynchronize());

    int j;
    for (j = 1; j < m; ++j) {
        double beta = sqrt(dot(dim, d_w, d_w, d_partials, nblocks, h_partials));
        if (beta < 1e-12) break;
        b[j] = beta;

        /* V[j] = w / beta */
        scale_kernel<<<nblocks, threads>>>(dim, d_V + (size_t)j * dim, d_w, beta);
        CUDA_CHECK(cudaDeviceSynchronize());

        /* w = H * V[j] */
        applyH_kernel<<<nblocks, threads>>>(Nspin, d_V + (size_t)j * dim, d_w);
        CUDA_CHECK(cudaDeviceSynchronize());

        /* w -= beta * V[j-1] */
        w_axpy_kernel<<<nblocks, threads>>>(dim, d_w, beta,
                                            d_V + (size_t)(j - 1) * dim);
        CUDA_CHECK(cudaDeviceSynchronize());

        a[j] = dot(dim, d_V + (size_t)j * dim, d_w, d_partials, nblocks, h_partials);

        /* w -= a[j] * V[j] */
        w_axpy_kernel<<<nblocks, threads>>>(dim, d_w, a[j],
                                            d_V + (size_t)j * dim);
        CUDA_CHECK(cudaDeviceSynchronize());

        /* full reorthogonalization (two passes) */
        for (int twice = 0; twice < 2; ++twice)
            for (int k = 0; k <= j; ++k) {
                double c = dot(dim, d_V + (size_t)k * dim, d_w,
                               d_partials, nblocks, h_partials);
                w_axpy_kernel<<<nblocks, threads>>>(dim, d_w, c,
                                                    d_V + (size_t)k * dim);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
    }
    int mused = j;
    double e0 = smallest_eig(mused, a, b);

    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_partials));
    free(a); free(b); free(h_partials); free(h_tmp);
    return e0;
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 16;
    int m = (argc > 2) ? atoi(argv[2]) : 120;

    clock_t t0 = clock();
    double e0 = ground_energy(N, m);
    clock_t t1 = clock();

    double ref4 = ground_energy(4, 50);

    printf("lanczos N=%d (dim=%ld)  E0=%.8f  E0/N=%.6f  time=%.3f s\n",
           N, 1L << N, e0, e0 / N, (double)(t1 - t0) / CLOCKS_PER_SEC);
    printf("        check: E0(N=4 ring)=%.8f  (exact -2.0, err=%.1e)\n",
           ref4, fabs(ref4 + 2.0));
    return 0;
}
