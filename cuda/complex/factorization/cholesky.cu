/* cholesky.cu -- CUDA conversion of benchmark/complex/factorization/cholesky.c
 * Per-column Cholesky on GPU with host-side sequential loop.
 * A diagonal reduction kernel + a parallel off-diagonal update kernel per column. */
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

#define BLOCK_SIZE 256

/* Reduction over A[j][0..j-1]^2 → d = A[j][j] - sum */
__global__ void chol_diag_kernel(int n, int j, const double *A, double *d_out) {
    __shared__ double cache[BLOCK_SIZE];
    int tid = threadIdx.x;
    double sum = 0.0;
    for (int k = tid; k < j; k += blockDim.x) {
        double v = A[j * n + k];
        sum += v * v;
    }
    cache[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) cache[tid] += cache[tid + s];
        __syncthreads();
    }
    if (tid == 0) *d_out = A[j * n + j] - cache[0];
}

/* Off-diagonal update: A[i][j] = (A[i][j] - sum_k A[i][k]*A[j][k]) / A[j][j] */
__global__ void chol_offdiag_kernel(int n, int j, double *A) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + (j + 1);
    if (i < n) {
        double sum = A[i * n + j];
        for (int k = 0; k < j; k++)
            sum -= A[i * n + k] * A[j * n + k];
        A[i * n + j] = sum / A[j * n + j];
    }
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 512;

    double *B  = (double *)malloc((size_t)n * n * sizeof *B);
    double *A  = (double *)malloc((size_t)n * n * sizeof *A);
    double *A0 = (double *)malloc((size_t)n * n * sizeof *A0);
    double *b  = (double *)malloc((size_t)n * sizeof *b);
    if (!B || !A || !A0 || !b) { fprintf(stderr, "alloc failed\n"); return 1; }

    unsigned long s = 13572468ul;
    for (size_t i = 0; i < (size_t)n * n; ++i) {
        s = s * 1103515245ul + 12345ul;
        B[i] = ((s >> 16) & 0x7fff) / 32768.0 - 0.5;
    }
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) sum += B[i * n + k] * B[j * n + k];
            A0[i * n + j] = sum + (i == j ? n : 0.0);
        }
    for (size_t i = 0; i < (size_t)n * n; ++i) A[i] = A0[i];
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) sum += A0[i * n + j];
        b[i] = sum;
    }

    double *d_A;
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_A, A, (size_t)n * n * sizeof(double),
                          cudaMemcpyHostToDevice));

    double *d_diag;
    CUDA_CHECK(cudaMalloc(&d_diag, sizeof(double)));

    int info = 0;
    clock_t t0 = clock();
    for (int j = 0; j < n && info == 0; ++j) {
        chol_diag_kernel<<<1, BLOCK_SIZE>>>(n, j, d_A, d_diag);
        CUDA_CHECK(cudaDeviceSynchronize());

        double d;
        CUDA_CHECK(cudaMemcpy(&d, d_diag, sizeof(double),
                              cudaMemcpyDeviceToHost));
        if (d <= 0.0) { info = -1; break; }
        d = sqrt(d);
        CUDA_CHECK(cudaMemcpy(d_A + j * n + j, &d, sizeof(double),
                              cudaMemcpyHostToDevice));

        int rows = n - (j + 1);
        if (rows > 0) {
            int blocks = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
            chol_offdiag_kernel<<<blocks, BLOCK_SIZE>>>(n, j, d_A);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    clock_t t1 = clock();

    CUDA_CHECK(cudaMemcpy(A, d_A, (size_t)n * n * sizeof(double),
                          cudaMemcpyDeviceToHost));

    if (info == 0) {
        for (int i = 0; i < n; ++i) {
            double s = b[i];
            for (int j = 0; j < i; ++j) s -= A[i * n + j] * b[j];
            b[i] = s / A[i * n + i];
        }
        for (int i = n - 1; i >= 0; --i) {
            double s = b[i];
            for (int j = i + 1; j < n; ++j) s -= A[j * n + i] * b[j];
            b[i] = s / A[i * n + i];
        }
    }

    double err = 0.0;
    for (int i = 0; i < n; ++i) { double e = fabs(b[i] - 1.0); if (e > err) err = e; }

    printf("cholesky n=%d  spd=%s  max|x-1|=%.2e  time=%.3f s\n",
           n, info == 0 ? "yes" : "NO", err,
           (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(B); free(A); free(A0); free(b);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_diag));
    return 0;
}
