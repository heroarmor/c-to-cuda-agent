/* lu.cu -- CUDA conversion of benchmark/complex/factorization/lu.c
 * Hybrid CPU/GPU LU factorization with partial pivoting.
 * Panel operations (pivot search, multiplier computation) on the CPU;
 * row swaps and trailing submatrix update on the GPU. */
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
__global__ void copy_col_kernel(int n, int k, int start,
                                const double *A, double *col) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (i < n)
        col[i - start] = A[i * n + k];
}

/* Write contiguous buffer col into column k, rows [start, n), of A. */
__global__ void write_col_kernel(int n, int k, int start,
                                 const double *col, double *A) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (i < n)
        A[i * n + k] = col[i - start];
}

/* Swap two entire rows of A. */
__global__ void swap_rows_kernel(int n, double *A, int r1, int r2) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        double t = A[r1 * n + j];
        A[r1 * n + j] = A[r2 * n + j];
        A[r2 * n + j] = t;
    }
}

/* Trailing submatrix update: A[i][j] -= A[i][k] * A[k][j]  for i,j > k. */
__global__ void trailing_update_kernel(int n, int k, double *A) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + (k + 1);
    int j = blockIdx.x * blockDim.x + threadIdx.x + (k + 1);
    if (i < n && j < n)
        A[i * n + j] -= A[i * n + k] * A[k * n + j];
}

/* Host forward/back solve -- same code as the baseline. */
static void lu_solve(int n, const double *A, const int *piv, double *b) {
    for (int k = 0; k < n; ++k)
        if (piv[k] != k) {
            double t = b[k]; b[k] = b[piv[k]]; b[piv[k]] = t;
        }
    for (int i = 0; i < n; ++i) {
        double s = b[i];
        for (int j = 0; j < i; ++j) s -= A[i * n + j] * b[j];
        b[i] = s;
    }
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
    int *piv   = (int *)malloc((size_t)n * sizeof *piv);
    if (!A || !A0 || !b || !piv) { fprintf(stderr, "alloc failed\n"); return 1; }

    unsigned long s = 987654321ul;
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

    double *d_panel;
    CUDA_CHECK(cudaMalloc(&d_panel, (size_t)n * sizeof(double)));
    double *h_panel = (double *)malloc((size_t)n * sizeof *h_panel);
    if (!h_panel) { fprintf(stderr, "alloc failed\n"); return 1; }

    const int BLOCK = 256;
    clock_t t0 = clock();

    for (int k = 0; k < n; ++k) {
        /* 1. Copy column k (rows k..n-1) to host. */
        {
            int rows = n - k;
            int blocks = (rows + BLOCK - 1) / BLOCK;
            copy_col_kernel<<<blocks, BLOCK>>>(n, k, k, d_A, d_panel);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_panel, d_panel, rows * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }

        /* 2. Find pivot row (same search as baseline). */
        int p = k;
        double best = fabs(h_panel[0]);
        for (int i = 1; i < n - k; ++i) {
            double v = fabs(h_panel[i]);
            if (v > best) { best = v; p = k + i; }
        }
        piv[k] = p;

        /* 3. Swap entire rows k and p on GPU. */
        if (p != k) {
            int blocks = (n + BLOCK - 1) / BLOCK;
            swap_rows_kernel<<<blocks, BLOCK>>>(n, d_A, k, p);
            CUDA_CHECK(cudaGetLastError());
        }

        /* 4. Compute multipliers on the host.
         *    After the swap: pivot = h_panel[p-k]  (new A[k][k]).
         *    For i > k the multiplier is A[i][k] / pivot, where A[i][k] after
         *    the swap is either h_panel[0] (if i==p) or h_panel[i-k]. */
        double pivot_val = h_panel[p - k];
        for (int i = 1; i < n - k; ++i) {
            int row = k + i;
            h_panel[i] = (row == p ? h_panel[0] : h_panel[i]) / pivot_val;
        }

        /* 5. Write multipliers to column k, rows k+1..n-1. */
        if (k + 1 < n) {
            int rows = n - (k + 1);
            int blocks = (rows + BLOCK - 1) / BLOCK;
            CUDA_CHECK(cudaMemcpy(d_panel, h_panel + 1, rows * sizeof(double),
                                  cudaMemcpyHostToDevice));
            write_col_kernel<<<blocks, BLOCK>>>(n, k, k + 1, d_panel, d_A);
            CUDA_CHECK(cudaGetLastError());
        }

        /* 6. Trailing submatrix update on GPU (the O(n^2) GEMM-like work). */
        if (k + 1 < n) {
            int m = n - k - 1;
            dim3 block(16, 16);
            dim3 grid((m + 15) / 16, (m + 15) / 16);
            trailing_update_kernel<<<grid, block>>>(n, k, d_A);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    /* Copy factored matrix back to host. */
    CUDA_CHECK(cudaMemcpy(A, d_A, (size_t)n * n * sizeof(double),
                          cudaMemcpyDeviceToHost));

    /* Solve A x = b on host (same code as baseline). */
    lu_solve(n, A, piv, b);

    clock_t t1 = clock();

    double err = 0.0;
    for (int i = 0; i < n; ++i) {
        double e = fabs(b[i] - 1.0);
        if (e > err) err = e;
    }

    printf("lu n=%d  max|x-1|=%.2e  time=%.3f s\n",
           n, err, (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(A); free(A0); free(b); free(piv); free(h_panel);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_panel));
    return 0;
}
