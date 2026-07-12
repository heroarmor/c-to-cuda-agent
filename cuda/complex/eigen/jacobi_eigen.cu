/* jacobi_eigen.cu -- CUDA conversion of benchmark/complex/eigen/jacobi_eigen.c
 *
 * Cyclic Jacobi eigenvalue solver. Two kernel kinds collaborate per sweep:
 *   1) jacobi_sweep_kernel -- one block runs the (intrinsically sequential)
 *      pair loop; threads parallelise each rotation's length-n row/col update.
 *   2) offnorm_kernel       -- multi-block reduction of the off-diagonal norm
 *      used for the convergence test (replaces a per-sweep full-matrix copy).
 *
 * The cyclic ordering is sequential, so `sweeps` (a reported field) depends on
 * exact arithmetic; rotations use round-to-nearest, no-FMA intrinsics so the
 * GPU matrix stays bit-identical to the CPU baseline and the sweep count matches.
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

/* round-to-nearest, no-contraction rotate of a pair (prevents FMA divergence) */
__device__ static inline double rot_a(double c, double s, double a, double b) {
    return __dsub_rn(__dmul_rn(c, a), __dmul_rn(s, b));   /* c*a - s*b */
}
__device__ static inline double rot_b(double c, double s, double a, double b) {
    return __dadd_rn(__dmul_rn(s, a), __dmul_rn(c, b));   /* s*a + c*b */
}

/* one cyclic Jacobi sweep on a single block */
__global__ void jacobi_sweep_kernel(int n, double *A) {
    __shared__ double cs[2];
    int tid = threadIdx.x;
    int nt  = blockDim.x;

    for (int p = 0; p < n; ++p) {
        for (int q = p + 1; q < n; ++q) {
            if (tid == 0) {
                double apq = A[p * n + q];
                if (fabs(apq) > 1e-300) {
                    double app = A[p * n + p];
                    double aqq = A[q * n + q];
                    double theta = (aqq - app) / (2.0 * apq);
                    double t = (theta >= 0.0 ? 1.0 : -1.0)
                               / (fabs(theta) + sqrt(theta * theta + 1.0));
                    cs[0] = 1.0 / sqrt(t * t + 1.0);
                    cs[1] = t * cs[0];
                } else {
                    cs[0] = 1.0;
                    cs[1] = 0.0;
                }
            }
            __syncthreads();
            double c = cs[0], s = cs[1];

            for (int i = tid; i < n; i += nt) {            /* rotate columns p,q */
                double aip = A[i * n + p];
                double aiq = A[i * n + q];
                A[i * n + p] = rot_a(c, s, aip, aiq);
                A[i * n + q] = rot_b(c, s, aip, aiq);
            }
            __syncthreads();
            for (int i = tid; i < n; i += nt) {            /* rotate rows p,q */
                double api = A[p * n + i];
                double aqi = A[q * n + i];
                A[p * n + i] = rot_a(c, s, api, aqi);
                A[q * n + i] = rot_b(c, s, api, aqi);
            }
            __syncthreads();
        }
    }
}

/* second kernel kind: sum of squares of the strict upper triangle (off-norm^2) */
__global__ void offnorm_kernel(int n, const double *A, double *partial) {
    extern __shared__ double sh[];
    int tid = threadIdx.x;
    long total = (long)n * n;
    double s = 0.0;
    for (long idx = (long)blockIdx.x * blockDim.x + tid; idx < total;
         idx += (long)blockDim.x * gridDim.x) {
        int p = (int)(idx / n), q = (int)(idx % n);
        if (q > p) { double a = A[idx]; s += a * a; }
    }
    sh[tid] = s;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k) sh[tid] += sh[tid + k];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sh[0];
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 128;
    double tol = 1e-12;
    int max_sweeps = 100;

    size_t bytes = (size_t)n * n * sizeof(double);
    double *h_A = (double *)calloc(n * n, sizeof(double));
    for (int i = 0; i < n; ++i) {
        h_A[i * n + i] = 2.0;
        if (i > 0)     h_A[i * n + (i - 1)] = -1.0;
        if (i < n - 1) h_A[i * n + (i + 1)] = -1.0;
    }
    double trace0 = 0.0;
    for (int i = 0; i < n; ++i) trace0 += h_A[i * n + i];

    double *d_A;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int rblocks = 256;
    double *d_partial, *h_partial = (double *)malloc(rblocks * sizeof(double));
    CUDA_CHECK(cudaMalloc(&d_partial, rblocks * sizeof(double)));

    int sweeps = 0;
    clock_t t0 = clock();
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        jacobi_sweep_kernel<<<1, threads>>>(n, d_A);
        CUDA_CHECK(cudaGetLastError());

        offnorm_kernel<<<rblocks, threads, threads * sizeof(double)>>>(n, d_A, d_partial);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_partial, d_partial, rblocks * sizeof(double), cudaMemcpyDeviceToHost));
        double off = 0.0;
        for (int i = 0; i < rblocks; ++i) off += h_partial[i];
        if (sqrt(off) < tol) { sweeps = sweep + 1; break; }
    }
    clock_t t1 = clock();
    if (sweeps == 0) sweeps = max_sweeps;

    CUDA_CHECK(cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost));
    double tr = 0.0, emin = h_A[0], emax = h_A[0];
    for (int i = 0; i < n; ++i) {
        double e = h_A[i * n + i];
        tr += e;
        if (e < emin) emin = e;
        if (e > emax) emax = e;
    }
    const double PI = 3.14159265358979323846;
    double emin_exact = 2.0 - 2.0 * cos(PI / (n + 1));
    double emax_exact = 2.0 - 2.0 * cos(n * PI / (n + 1));

    printf("jacobi_eigen n=%d  sweeps=%d  sum(eig)=%.6f (trace=%.6f)\n",
           n, sweeps, tr, trace0);
    printf("            lambda_min=%.6f (exact %.6f)  lambda_max=%.6f (exact %.6f)  time=%.3f s\n",
           emin, emin_exact, emax, emax_exact, (double)(t1 - t0) / CLOCKS_PER_SEC);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_partial));
    free(h_partial);
    free(h_A);
    return 0;
}
