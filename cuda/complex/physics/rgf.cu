/* rgf.cu -- CUDA conversion of benchmark/complex/physics/rgf.c
 * Host-side sequential control of forward/backward block sweeps,
 * with BxB matrix operations accelerated on GPU. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

/* ---------------------------------------------------------------
 * Kernel: C = A * B   (n x n matrix multiply)
 * 2D grid of 2D thread blocks; each thread computes one element.
 * --------------------------------------------------------------- */
__global__ void mat_mul_kernel(const double *A, const double *B,
                                double *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k)
            sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

/* ---------------------------------------------------------------
 * Kernel: C = A - B  (element-wise, n*n elements)
 * --------------------------------------------------------------- */
__global__ void mat_sub_kernel(const double *A, const double *B,
                                double *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nn = n * n;
    if (idx < nn) C[idx] = A[idx] - B[idx];
}

/* ---------------------------------------------------------------
 * Kernel: Aout = inverse(Ain)  (n x n, Gauss-Jordan)
 * Single block with n threads (one per row).
 * Uses shared memory: 2 * n * n doubles.
 * --------------------------------------------------------------- */
__global__ void mat_inv_kernel(const double *Ain, double *Aout, int n) {
    extern __shared__ double smem[];
    double *a   = smem;                /* workspace */
    double *inv = smem + (size_t)n * n; /* inverse  */

    int row = threadIdx.x;
    if (row >= n) return;

    /* load Ain and initialise inverse to identity */
    for (int j = 0; j < n; ++j) {
        a[row * n + j]   = Ain[row * n + j];
        inv[row * n + j] = (row == j) ? 1.0 : 0.0;
    }
    __syncthreads();

    for (int col = 0; col < n; ++col) {

        /* ---------- partial pivoting ---------- */
        __shared__ double pv[1024];
        __shared__ int    pr[1024];

        pv[row] = (row >= col) ? fabs(a[row * n + col]) : -1.0;
        pr[row] = row;
        __syncthreads();

        for (int s = 1; s < n; s <<= 1) {
            if ((row & ((s << 1) - 1)) == 0 && row + s < n) {
                if (pv[row + s] > pv[row]) {
                    pv[row] = pv[row + s];
                    pr[row] = pr[row + s];
                }
            }
            __syncthreads();
        }
        int piv = pr[0];

        /* ---------- row swap ---------- */
        if (piv != col) {
            if (row == 0) {
                for (int j = 0; j < n; ++j) {
                    double t        = a[col * n + j];
                    a[col * n + j]  = a[piv * n + j];
                    a[piv * n + j]  = t;
                    t               = inv[col * n + j];
                    inv[col * n + j]= inv[piv * n + j];
                    inv[piv * n + j]= t;
                }
            }
            __syncthreads();
        }

        /* ---------- normalise pivot row ---------- */
        double piv_val = a[col * n + col];
        if (row == col) {
            for (int j = 0; j < n; ++j) {
                a[col * n + j]   /= piv_val;
                inv[col * n + j] /= piv_val;
            }
        }
        __syncthreads();

        /* ---------- eliminate from other rows ---------- */
        if (row != col) {
            double factor = a[row * n + col];
            for (int j = 0; j < n; ++j) {
                a[row * n + j]   -= factor * a[col * n + j];
                inv[row * n + j] -= factor * inv[col * n + j];
            }
        }
        __syncthreads();
    }

    /* write result */
    for (int j = 0; j < n; ++j)
        Aout[row * n + j] = inv[row * n + j];
}

/* ---------------------------------------------------------------
 * Kernel: C = A + B  (element-wise, n*n elements)
 * --------------------------------------------------------------- */
__global__ void mat_add_kernel(const double *A, const double *B,
                                double *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nn = n * n;
    if (idx < nn) C[idx] = A[idx] + B[idx];
}

/* ---------------------------------------------------------------
 * Host main
 * --------------------------------------------------------------- */
int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 256;
    int B = (argc > 2) ? atoi(argv[2]) : 32;

    if (B > 1024) {
        fprintf(stderr, "block size %d too large (max 1024)\n", B);
        return 1;
    }
    size_t bb = (size_t)B * B;

    /* build D and C on the host */
    double *D = (double *)calloc(bb, sizeof *D);
    double *C = (double *)calloc(bb, sizeof *C);
    if (!D || !C) { fprintf(stderr, "host alloc failed\n"); return 1; }
    for (int i = 0; i < B; ++i) {
        D[i * B + i] = 4.0;
        if (i > 0)     D[i * B + (i - 1)] = -1.0;
        if (i < B - 1) D[i * B + (i + 1)] = -1.0;
        C[i * B + i] = -1.0;
    }

    /* allocate device memory */
    double *d_D, *d_C, *d_g, *d_G, *d_t1, *d_t2, *d_t3, *d_M;
    size_t device_bytes = bb * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_D, device_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, device_bytes));
    CUDA_CHECK(cudaMalloc(&d_g, (size_t)N * bb * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_G, (size_t)N * bb * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_t1, device_bytes));
    CUDA_CHECK(cudaMalloc(&d_t2, device_bytes));
    CUDA_CHECK(cudaMalloc(&d_t3, device_bytes));
    CUDA_CHECK(cudaMalloc(&d_M, device_bytes));

    CUDA_CHECK(cudaMemcpy(d_D, D, device_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, C, device_bytes, cudaMemcpyHostToDevice));

    /* kernel launch config */
    dim3 mul_block(16, 16);
    dim3 mul_grid((B + 15) / 16, (B + 15) / 16);
    int vec_threads = 256;
    int vec_blocks = ((int)bb + vec_threads - 1) / vec_threads;
    int inv_threads = B;
    size_t inv_shmem = 2 * bb * sizeof(double);

    clock_t tic = clock();

    #define d_gp(i) (d_g + (size_t)(i) * bb)
    #define d_Gp(i) (d_G + (size_t)(i) * bb)

    /* ---------- forward sweep ---------- */
    mat_inv_kernel<<<1, inv_threads, inv_shmem>>>(d_D, d_gp(0), B);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 1; i < N; ++i) {
        mat_mul_kernel<<<mul_grid, mul_block>>>(d_C, d_gp(i - 1), d_t1, B);
        CUDA_CHECK(cudaDeviceSynchronize());
        mat_mul_kernel<<<mul_grid, mul_block>>>(d_t1, d_C, d_t2, B);
        CUDA_CHECK(cudaDeviceSynchronize());
        mat_sub_kernel<<<vec_blocks, vec_threads>>>(d_D, d_t2, d_M, B);
        CUDA_CHECK(cudaDeviceSynchronize());
        mat_inv_kernel<<<1, inv_threads, inv_shmem>>>(d_M, d_gp(i), B);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /* ---------- backward sweep ---------- */
    CUDA_CHECK(cudaMemcpy(d_Gp(N - 1), d_gp(N - 1),
                          bb * sizeof(double), cudaMemcpyDeviceToDevice));

    for (int i = N - 2; i >= 0; --i) {
        mat_mul_kernel<<<mul_grid, mul_block>>>(d_gp(i), d_C, d_t1, B);
        CUDA_CHECK(cudaDeviceSynchronize());
        mat_mul_kernel<<<mul_grid, mul_block>>>(d_t1, d_Gp(i + 1), d_t2, B);
        CUDA_CHECK(cudaDeviceSynchronize());
        mat_mul_kernel<<<mul_grid, mul_block>>>(d_t2, d_C, d_t3, B);
        CUDA_CHECK(cudaDeviceSynchronize());
        mat_mul_kernel<<<mul_grid, mul_block>>>(d_t3, d_gp(i), d_t1, B);
        CUDA_CHECK(cudaDeviceSynchronize());
        mat_add_kernel<<<vec_blocks, vec_threads>>>(d_gp(i), d_t1, d_Gp(i), B);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /* copy G back to host */
    double *G = (double *)malloc((size_t)N * bb * sizeof *G);
    if (!G) { fprintf(stderr, "host alloc failed\n"); return 1; }
    CUDA_CHECK(cudaMemcpy(G, d_G, (size_t)N * bb * sizeof(double),
                          cudaMemcpyDeviceToHost));

    /* trace = sum of diagonals of all diagonal blocks */
    double trace = 0.0;
    for (int i = 0; i < N; ++i) {
        double *Gi = G + (size_t)i * bb;
        for (int d = 0; d < B; ++d)
            trace += Gi[d * B + d];
    }

    clock_t toc = clock();

    printf("rgf blocks=%d blocksize=%d (n=%d)  trace(inv)=%.6f  time=%.3f s\n",
           N, B, N * B, trace, (double)(toc - tic) / CLOCKS_PER_SEC);

    #undef d_gp
    #undef d_Gp

    free(D); free(C); free(G);
    CUDA_CHECK(cudaFree(d_D));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_G));
    CUDA_CHECK(cudaFree(d_t1));
    CUDA_CHECK(cudaFree(d_t2));
    CUDA_CHECK(cudaFree(d_t3));
    CUDA_CHECK(cudaFree(d_M));
    return 0;
}
