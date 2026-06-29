/* spgemm.cu -- CUDA conversion of benchmark/complex/sparse-linalg/spgemm.c
 * SpGEMM C = A * A (CSR) where A is the 2D 5-point Laplacian.
 * One thread per row; per-row SPA fits in thread-local storage.
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

static int build_laplacian(int m, int **row_ptr, int **col_idx, double **val) {
    int n = m * m;
    int *rp = (int *)malloc((size_t)(n + 1) * sizeof *rp);
    int *ci = (int *)malloc((size_t)5 * n * sizeof *ci);
    double *v = (double *)malloc((size_t)5 * n * sizeof *v);
    int nnz = 0;
    rp[0] = 0;
    for (int iy = 0; iy < m; ++iy)
        for (int ix = 0; ix < m; ++ix) {
            int row = iy * m + ix;
            if (iy > 0)     { ci[nnz] = row - m; v[nnz] = -1.0; nnz++; }
            if (ix > 0)     { ci[nnz] = row - 1; v[nnz] = -1.0; nnz++; }
            ci[nnz] = row; v[nnz] = 4.0; nnz++;
            if (ix < m - 1) { ci[nnz] = row + 1; v[nnz] = -1.0; nnz++; }
            if (iy < m - 1) { ci[nnz] = row + m; v[nnz] = -1.0; nnz++; }
            rp[row + 1] = nnz;
        }
    *row_ptr = rp; *col_idx = ci; *val = v;
    return nnz;
}

__global__ void count_kernel(int n, const int *Arp, const int *Aci, int *cnt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int loc[32];
    int nc = 0;
    for (int k = Arp[i]; k < Arp[i + 1]; ++k) {
        int a = Aci[k];
        for (int kk = Arp[a]; kk < Arp[a + 1]; ++kk) {
            int col = Aci[kk];
            int dup = 0;
            for (int t = 0; t < nc; ++t) { if (loc[t] == col) { dup = 1; break; } }
            if (!dup) loc[nc++] = col;
        }
    }
    cnt[i] = nc;
}

__global__ void numeric_kernel(int n, const int *Arp, const int *Aci, const double *Av,
                                const int *Crp, int *Cci, double *Cv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int    loc_col[32];
    double loc_val[32];
    int nc = 0;
    for (int k = Arp[i]; k < Arp[i + 1]; ++k) {
        int a = Aci[k];
        double av = Av[k];
        for (int kk = Arp[a]; kk < Arp[a + 1]; ++kk) {
            int col = Aci[kk];
            int dup = 0;
            for (int t = 0; t < nc; ++t) {
                if (loc_col[t] == col) {
                    loc_val[t] += av * Av[kk];
                    dup = 1;
                    break;
                }
            }
            if (!dup) {
                loc_col[nc] = col;
                loc_val[nc] = av * Av[kk];
                nc++;
            }
        }
    }
    int pos = Crp[i];
    for (int t = 0; t < nc; ++t) {
        Cci[pos + t] = loc_col[t];
        Cv[pos + t]  = loc_val[t];
    }
}

int main(int argc, char **argv) {
    int m = (argc > 1) ? atoi(argv[1]) : 512;
    int n = m * m;

    int *h_Arp, *h_Aci;
    double *h_Av;
    int Anz = build_laplacian(m, &h_Arp, &h_Aci, &h_Av);

    int *d_Arp, *d_Aci;
    double *d_Av;
    CUDA_CHECK(cudaMalloc(&d_Arp, (size_t)(n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_Aci, (size_t)Anz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_Av, (size_t)Anz * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_Arp, h_Arp, (size_t)(n + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Aci, h_Aci, (size_t)Anz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Av, h_Av, (size_t)Anz * sizeof(double),
                          cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    int *d_cnt;
    CUDA_CHECK(cudaMalloc(&d_cnt, (size_t)n * sizeof(int)));

    clock_t t0 = clock();
    count_kernel<<<blocks, threads>>>(n, d_Arp, d_Aci, d_cnt);
    CUDA_CHECK(cudaDeviceSynchronize());

    int *h_cnt = (int *)malloc((size_t)n * sizeof(int));
    int *h_Crp = (int *)malloc((size_t)(n + 1) * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_cnt, d_cnt, (size_t)n * sizeof(int),
                          cudaMemcpyDeviceToHost));

    long total = 0;
    h_Crp[0] = 0;
    for (int i = 0; i < n; ++i) {
        total += h_cnt[i];
        h_Crp[i + 1] = (int)total;
    }

    int *d_Crp, *d_Cci;
    double *d_Cv;
    CUDA_CHECK(cudaMalloc(&d_Crp, (size_t)(n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_Cci, (size_t)total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_Cv, (size_t)total * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_Crp, h_Crp, (size_t)(n + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));

    numeric_kernel<<<blocks, threads>>>(n, d_Arp, d_Aci, d_Av, d_Crp, d_Cci, d_Cv);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_t t1 = clock();

    double *h_Cv = (double *)malloc((size_t)total * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_Cv, d_Cv, (size_t)total * sizeof(double),
                          cudaMemcpyDeviceToHost));

    double sum = 0.0;
    for (long k = 0; k < total; ++k) sum += h_Cv[k];

    printf("spgemm n=%d  nnz(A)=%d  nnz(A^2)=%ld  checksum=%.1f  time=%.3f s\n",
           n, Anz, total, sum, (double)(t1 - t0) / CLOCKS_PER_SEC);

    CUDA_CHECK(cudaFree(d_Arp)); CUDA_CHECK(cudaFree(d_Aci)); CUDA_CHECK(cudaFree(d_Av));
    CUDA_CHECK(cudaFree(d_cnt)); CUDA_CHECK(cudaFree(d_Crp));
    CUDA_CHECK(cudaFree(d_Cci)); CUDA_CHECK(cudaFree(d_Cv));
    free(h_Arp); free(h_Aci); free(h_Av);
    free(h_cnt); free(h_Crp); free(h_Cv);
    return 0;
}
