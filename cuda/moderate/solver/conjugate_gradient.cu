/* conjugate_gradient.cu -- CUDA conversion of
 * benchmark/moderate/solver/conjugate_gradient.c
 *
 * Host orchestrates the CG iteration with device-resident vectors.
 * Per iteration: SpMV kernel, two dot-product (reduction) kernels,
 * three AXPY kernels.
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

#define BLOCK_SIZE 256

/* CSR SpMV: y = A * x */
__global__ void spmv_kernel(int n, const int *rp, const int *ci,
                            const double *v, const double *x, double *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double s = 0.0;
    for (int k = rp[i]; k < rp[i + 1]; ++k)
        s += v[k] * x[ci[k]];
    y[i] = s;
}

/* Dot product partial reduction: each block sums a[i]*b[i] into partial[bid] */
__global__ void dot_kernel(int n, const double *a, const double *b,
                           double *partial) {
    extern __shared__ double smem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    double s = 0.0;
    for (int i = idx; i < n; i += gridDim.x * blockDim.x)
        s += a[i] * b[i];
    smem[tid] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = smem[0];
}

/* AXPY: y += a * x */
__global__ void axpy_kernel(int n, double a, const double *x, double *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

/* AXPBY: y = a * x + b * y */
__global__ void axpby_kernel(int n, double a, const double *x, double b,
                             double *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + b * y[i];
}

static double compute_dot(int n, const double *d_a, const double *d_b,
                          double *d_partial, double *h_partial, int nblocks) {
    dot_kernel<<<nblocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(
        n, d_a, d_b, d_partial);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial,
                          (size_t)nblocks * sizeof(double),
                          cudaMemcpyDeviceToHost));
    double s = 0.0;
    for (int i = 0; i < nblocks; ++i) s += h_partial[i];
    return s;
}

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
    int m = (argc > 1) ? atoi(argv[1]) : 512;
    int maxit = (argc > 2) ? atoi(argv[2]) : 10000;
    double tol = 1e-8;
    int n = m * m;

    /* --- build matrix on CPU --- */
    int *h_rp, *h_ci;
    double *h_v;
    build_laplacian(m, &h_rp, &h_ci, &h_v);

    /* --- allocate device memory --- */
    int *d_rp, *d_ci;
    double *d_v, *d_x, *d_b, *d_r, *d_p, *d_Ap;
    CUDA_CHECK(cudaMalloc(&d_rp, (size_t)(n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ci, (size_t)5 * n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_v,  (size_t)5 * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x,  (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b,  (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r,  (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p,  (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Ap, (size_t)n * sizeof(double)));

    /* --- copy matrix to device --- */
    CUDA_CHECK(cudaMemcpy(d_rp, h_rp, (size_t)(n + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ci, h_ci, (size_t)5 * n * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v, (size_t)5 * n * sizeof(double),
                          cudaMemcpyHostToDevice));

    /* --- initialise vectors on host then copy to device --- */
    double *h_b = (double *)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; ++i) h_b[i] = 1.0;
    CUDA_CHECK(cudaMemset(d_x, 0, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, (size_t)n * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r, h_b, (size_t)n * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p, h_b, (size_t)n * sizeof(double),
                          cudaMemcpyHostToDevice));

    /* --- dot-product scratch buffers --- */
    int dot_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (dot_blocks > 256) dot_blocks = 256;
    double *d_partial, *h_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, (size_t)dot_blocks * sizeof(double)));
    h_partial = (double *)malloc((size_t)dot_blocks * sizeof(double));
    if (!h_partial) { fprintf(stderr, "alloc failed\n"); return 1; }

    /* --- launch configuration for element-wise kernels --- */
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* --- initial residual norm --- */
    double rs_old = compute_dot(n, d_r, d_r, d_partial, h_partial, dot_blocks);
    double bnorm = sqrt(rs_old);

    /* --- CG iteration --- */
    clock_t t0 = clock();
    int k = 0;
    for (; k < maxit; ++k) {
        spmv_kernel<<<blocks, BLOCK_SIZE>>>(n, d_rp, d_ci, d_v, d_p, d_Ap);
        CUDA_CHECK(cudaDeviceSynchronize());

        double pAp = compute_dot(n, d_p, d_Ap, d_partial, h_partial,
                                 dot_blocks);
        double alpha = rs_old / pAp;

        axpy_kernel<<<blocks, BLOCK_SIZE>>>(n, alpha, d_p, d_x);
        axpy_kernel<<<blocks, BLOCK_SIZE>>>(n, -alpha, d_Ap, d_r);
        CUDA_CHECK(cudaDeviceSynchronize());

        double rs_new = compute_dot(n, d_r, d_r, d_partial, h_partial,
                                    dot_blocks);

        if (sqrt(rs_new) <= tol * bnorm) { ++k; break; }

        double beta = rs_new / rs_old;
        axpby_kernel<<<blocks, BLOCK_SIZE>>>(n, 1.0, d_r, beta, d_p);
        CUDA_CHECK(cudaDeviceSynchronize());

        rs_old = rs_new;
    }
    clock_t t1 = clock();

    /* --- verification: compute b - A x on CPU --- */
    spmv_kernel<<<blocks, BLOCK_SIZE>>>(n, d_rp, d_ci, d_v, d_x, d_Ap);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_b, d_Ap, (size_t)n * sizeof(double),
                          cudaMemcpyDeviceToHost));
    double res = 0.0;
    for (int i = 0; i < n; ++i) {
        double b_val = 1.0;
        double d = b_val - h_b[i];
        res += d * d;
    }

    printf("cg n=%d  iters=%d  relres=%.2e  time=%.3f s\n",
           n, k, sqrt(res) / bnorm, (double)(t1 - t0) / CLOCKS_PER_SEC);

    /* --- cleanup --- */
    CUDA_CHECK(cudaFree(d_rp));
    CUDA_CHECK(cudaFree(d_ci));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_Ap));
    CUDA_CHECK(cudaFree(d_partial));
    free(h_rp); free(h_ci); free(h_v); free(h_b); free(h_partial);
    return 0;
}
