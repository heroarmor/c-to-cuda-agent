/* qr.cu -- CUDA conversion of benchmark/complex/factorization/qr.c
 * Hybrid CPU/GPU Householder QR.
 * Panel operations (norm, reflector computation) on the CPU;
 * trailing submatrix update (rank-1 Householder application) on the GPU. */
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

/* Copy column k, rows [start, n), of A into contiguous buffer col. */
__global__ void copy_col_kernel(int n, int k, const double *A, double *col) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + k;
    if (i < n)
        col[i - k] = A[i * n + k];
}

/* Apply Householder reflector to trailing submatrix A[k:n, k:n].
 * For each column j >= k:
 *   dot = sum_{i >= k} v[i] * A[i][j]
 *   A[i][j] -= tau * dot * v[i]    for all i >= k */
__global__ void qr_trail_kernel(int n, int k, double tau,
                                const double *v, double *A) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + k;
    if (j >= n) return;
    double dot = 0.0;
    for (int i = k; i < n; ++i)
        dot += v[i] * A[i * n + j];
    double f = tau * dot;
    for (int i = k; i < n; ++i)
        A[i * n + j] -= f * v[i];
}

static void back_subst(int n, const double *A, double *b) {
    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < n; ++j) s -= A[i * n + j] * b[j];
        b[i] = s / A[i * n + i];
    }
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 512;

    double *A  = (double *)malloc((size_t)n * n * sizeof *A);
    double *A0 = (double *)malloc((size_t)n * n * sizeof *A0);
    double *b  = (double *)malloc((size_t)n * sizeof *b);
    if (!A || !A0 || !b) { fprintf(stderr, "alloc failed\n"); return 1; }

    unsigned long s = 24681357ul;
    for (size_t i = 0; i < (size_t)n * n; ++i) {
        s = s * 1103515245ul + 12345ul;
        A0[i] = ((s >> 16) & 0x7fff) / 32768.0 - 0.5;
    }
    for (int i = 0; i < n; ++i) A0[i * n + i] += n;
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

    double *d_v;
    CUDA_CHECK(cudaMalloc(&d_v, (size_t)n * sizeof(double)));

    double *d_col;
    CUDA_CHECK(cudaMalloc(&d_col, (size_t)n * sizeof(double)));
    double *h_col = (double *)malloc((size_t)n * sizeof *h_col);
    if (!h_col) { fprintf(stderr, "alloc failed\n"); return 1; }

    const int BLOCK = 256;
    clock_t t0 = clock();

    for (int k = 0; k < n; ++k) {
        /* 1. Copy column k to host. */
        int rows = n - k;
        {
            int blocks = (rows + BLOCK - 1) / BLOCK;
            copy_col_kernel<<<blocks, BLOCK>>>(n, k, d_A, d_col);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_col, d_col, rows * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }

        /* 2. Compute Householder reflector on the CPU. */
        double normx = 0.0;
        for (int i = 0; i < rows; ++i) normx += h_col[i] * h_col[i];
        normx = sqrt(normx);
        if (normx == 0.0) continue;

        double alpha = (h_col[0] > 0.0) ? -normx : normx;
        double *v = h_col;   /* reuse column buffer as v */
        v[0] -= alpha;

        double vnorm2 = 0.0;
        for (int i = 0; i < rows; ++i) vnorm2 += v[i] * v[i];
        if (vnorm2 < 1e-300) continue;

        double tau = 2.0 / vnorm2;

        /* 3. Upload v to device. */
        CUDA_CHECK(cudaMemcpy(d_v + k, v, rows * sizeof(double),
                              cudaMemcpyHostToDevice));

        /* 4. Apply reflector to trailing submatrix on GPU. */
        {
            int cols = n - k;
            int blocks = (cols + BLOCK - 1) / BLOCK;
            qr_trail_kernel<<<blocks, BLOCK>>>(n, k, tau, d_v, d_A);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        /* 5. Apply reflector to b on host. */
        {
            double dot = 0.0;
            for (int i = k; i < n; ++i) dot += v[i - k] * b[i];
            double f = tau * dot;
            for (int i = k; i < n; ++i) b[i] -= f * v[i - k];
        }
    }

    CUDA_CHECK(cudaMemcpy(A, d_A, (size_t)n * n * sizeof(double),
                          cudaMemcpyDeviceToHost));

    back_subst(n, A, b);

    clock_t t1 = clock();

    double err = 0.0;
    for (int i = 0; i < n; ++i) {
        double e = fabs(b[i] - 1.0);
        if (e > err) err = e;
    }

    printf("qr n=%d  max|x-1|=%.2e  time=%.3f s\n",
           n, err, (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(A); free(A0); free(b); free(h_col);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_col));
    return 0;
}
