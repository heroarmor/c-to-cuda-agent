/* spmv.cu -- CUDA conversion of benchmark/moderate/sparse-linalg/spmv.c
 * One thread per row; each thread iterates over its non-zeros in CSR format.
 */
#include <stdio.h>
#include <stdlib.h>
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

__global__ void spmv_kernel(int n, const int *row_ptr, const int *col_idx,
                            const double *val, const double *x, double *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double s = 0.0;
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
            s += val[k] * x[col_idx[k]];
        y[i] = s;
    }
}

/* Build the 5-point Laplacian in CSR (same as baseline). */
static int build_laplacian(int m, int **row_ptr, int **col_idx, double **val) {
    int n = m * m;
    int *rp = (int *)malloc((size_t)(n + 1) * sizeof *rp);
    int *ci = (int *)malloc((size_t)5 * n * sizeof *ci);
    double *v = (double *)malloc((size_t)5 * n * sizeof *v);
    int nnz = 0;
    rp[0] = 0;
    for (int iy = 0; iy < m; ++iy) {
        for (int ix = 0; ix < m; ++ix) {
            int row = iy * m + ix;
            if (iy > 0)     { ci[nnz] = row - m; v[nnz] = -1.0; nnz++; }
            if (ix > 0)     { ci[nnz] = row - 1; v[nnz] = -1.0; nnz++; }
            ci[nnz] = row; v[nnz] = 4.0; nnz++;
            if (ix < m - 1) { ci[nnz] = row + 1; v[nnz] = -1.0; nnz++; }
            if (iy < m - 1) { ci[nnz] = row + m; v[nnz] = -1.0; nnz++; }
            rp[row + 1] = nnz;
        }
    }
    *row_ptr = rp; *col_idx = ci; *val = v;
    return nnz;
}

int main(int argc, char **argv) {
    int m = (argc > 1) ? atoi(argv[1]) : 1024;
    int n = m * m;

    int *h_row_ptr, *h_col_idx;
    double *h_val;
    int nnz = build_laplacian(m, &h_row_ptr, &h_col_idx, &h_val);

    double *h_x = (double *)malloc((size_t)n * sizeof *h_x);
    double *h_y = (double *)malloc((size_t)n * sizeof *h_y);
    for (int i = 0; i < n; ++i) h_x[i] = 1.0;

    int *d_row_ptr, *d_col_idx;
    double *d_val, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (size_t)(n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, (size_t)nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_val, (size_t)nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)n * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr, (size_t)(n + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, h_col_idx, (size_t)nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val, h_val, (size_t)nnz * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, (size_t)n * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, (size_t)n * sizeof(double)));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    int reps = 50;
    clock_t t0 = clock();
    for (int r = 0; r < reps; ++r)
        spmv_kernel<<<blocks, threads>>>(n, d_row_ptr, d_col_idx, d_val, d_x, d_y);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_t t1 = clock();

    CUDA_CHECK(cudaMemcpy(h_y, d_y, (size_t)n * sizeof(double),
                          cudaMemcpyDeviceToHost));

    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += h_y[i];

    printf("spmv n=%d nnz=%d reps=%d  checksum=%.1f  time=%.3f s\n",
           n, nnz, reps, sum, (double)(t1 - t0) / CLOCKS_PER_SEC);

    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    free(h_row_ptr); free(h_col_idx); free(h_val); free(h_x); free(h_y);
    return 0;
}
