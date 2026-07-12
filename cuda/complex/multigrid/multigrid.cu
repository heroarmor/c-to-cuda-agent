/* multigrid.cu -- CUDA conversion of benchmark/complex/multigrid/multigrid.c
 * Geometric multigrid V-cycle for the 2D Poisson equation.
 * Host-driven recursion; each grid operation is a device kernel.
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

// -- kernels ----------------------------------------------------------------

__global__ void smooth_rb_kernel(int n, double *u, const double *f,
                                  double h2, int color) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= n - 1 || j >= n - 1) return;
    if (((i + j) & 1) != color) return;
    u[i * n + j] = 0.25 * (h2 * f[i * n + j] +
                           u[(i - 1) * n + j] + u[(i + 1) * n + j] +
                           u[i * n + j - 1] + u[i * n + j + 1]);
}

__global__ void residual_kernel(int n, const double *u, const double *f,
                                double h2, double *r) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= n - 1 || j >= n - 1) return;
    double Au = (4.0 * u[i * n + j] - u[(i - 1) * n + j] - u[(i + 1) * n + j] -
                 u[i * n + j - 1] - u[i * n + j + 1]) / h2;
    r[i * n + j] = f[i * n + j] - Au;
}

__global__ void restrict_kernel(int nf, const double *rf, int nc, double *rc) {
    int J = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int I = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (I >= nc - 1 || J >= nc - 1) return;
    int i = 2 * I, j = 2 * J;
    rc[I * nc + J] =
        0.2500 *  rf[i * nf + j] +
        0.1250 * (rf[(i - 1) * nf + j] + rf[(i + 1) * nf + j] +
                  rf[i * nf + j - 1] + rf[i * nf + j + 1]) +
        0.0625 * (rf[(i - 1) * nf + j - 1] + rf[(i - 1) * nf + j + 1] +
                  rf[(i + 1) * nf + j - 1] + rf[(i + 1) * nf + j + 1]);
}

__global__ void prolong_coincident_kernel(int nc, const double *ec,
                                           int nf, double *uf) {
    int J = blockIdx.x * blockDim.x + threadIdx.x;
    int I = blockIdx.y * blockDim.y + threadIdx.y;
    if (I >= nc || J >= nc) return;
    uf[(2 * I) * nf + (2 * J)] += ec[I * nc + J];
}

__global__ void prolong_horizontal_kernel(int nc, const double *ec,
                                           int nf, double *uf) {
    int J = blockIdx.x * blockDim.x + threadIdx.x;
    int I = blockIdx.y * blockDim.y + threadIdx.y;
    if (I >= nc || J >= nc - 1) return;
    uf[(2 * I) * nf + (2 * J + 1)] += 0.5 * (ec[I * nc + J] + ec[I * nc + J + 1]);
}

__global__ void prolong_vertical_kernel(int nc, const double *ec,
                                         int nf, double *uf) {
    int J = blockIdx.x * blockDim.x + threadIdx.x;
    int I = blockIdx.y * blockDim.y + threadIdx.y;
    if (I >= nc - 1 || J >= nc) return;
    uf[(2 * I + 1) * nf + (2 * J)] += 0.5 * (ec[I * nc + J] + ec[(I + 1) * nc + J]);
}

__global__ void prolong_center_kernel(int nc, const double *ec,
                                       int nf, double *uf) {
    int J = blockIdx.x * blockDim.x + threadIdx.x;
    int I = blockIdx.y * blockDim.y + threadIdx.y;
    if (I >= nc - 1 || J >= nc - 1) return;
    uf[(2 * I + 1) * nf + (2 * J + 1)] +=
        0.25 * (ec[I * nc + J] + ec[I * nc + J + 1] +
                ec[(I + 1) * nc + J] + ec[(I + 1) * nc + J + 1]);
}

// -- host helpers -----------------------------------------------------------

static dim3 grid_2d(int nx, int ny, int bx, int by) {
    return dim3((nx + bx - 1) / bx, (ny + by - 1) / by);
}

static void launch_smooth_rb(int n, double *d_u, const double *d_f,
                              double h2, int iters) {
    dim3 block(16, 16);
    dim3 grid = grid_2d(n - 2, n - 2, block.x, block.y);
    for (int it = 0; it < iters; ++it) {
        smooth_rb_kernel<<<grid, block>>>(n, d_u, d_f, h2, 0);
        smooth_rb_kernel<<<grid, block>>>(n, d_u, d_f, h2, 1);
    }
}

// -- recursive V-cycle (host-driven) ---------------------------------------

static void vcycle(int n, double *d_u, const double *d_f, double h) {
    double h2 = h * h;
    if (n <= 3) {
        dim3 block(16, 16);
        dim3 grid = grid_2d(n - 2, n - 2, block.x, block.y);
        smooth_rb_kernel<<<grid, block>>>(n, d_u, d_f, h2, 0);
        smooth_rb_kernel<<<grid, block>>>(n, d_u, d_f, h2, 1);
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }
    int nc = (n - 1) / 2 + 1;
    double *d_r, *d_rc, *d_ec;
    size_t sz_n = (size_t)n * n * sizeof(double);
    size_t sz_c = (size_t)nc * nc * sizeof(double);

    launch_smooth_rb(n, d_u, d_f, h2, 2);                       /* pre-smooth */

    CUDA_CHECK(cudaMalloc(&d_r, sz_n));
    {
        dim3 block(16, 16);
        dim3 grid = grid_2d(n - 2, n - 2, block.x, block.y);
        residual_kernel<<<grid, block>>>(n, d_u, d_f, h2, d_r);
    }
    CUDA_CHECK(cudaMalloc(&d_rc, sz_c));
    CUDA_CHECK(cudaMalloc(&d_ec, sz_c));
    CUDA_CHECK(cudaMemset(d_ec, 0, sz_c));
    {
        dim3 block(16, 16);
        dim3 grid = grid_2d(nc - 2, nc - 2, block.x, block.y);
        restrict_kernel<<<grid, block>>>(n, d_r, nc, d_rc);
    }
    CUDA_CHECK(cudaFree(d_r));

    vcycle(nc, d_ec, d_rc, 2.0 * h);                            /* recurse */

    {
        dim3 block(16, 16);
        prolong_coincident_kernel<<<grid_2d(nc,     nc,     block.x, block.y),
                                     block>>>(nc, d_ec, n, d_u);
        prolong_horizontal_kernel<<<grid_2d(nc,     nc - 1, block.x, block.y),
                                     block>>>(nc, d_ec, n, d_u);
        prolong_vertical_kernel<<<grid_2d(nc - 1, nc,     block.x, block.y),
                                     block>>>(nc, d_ec, n, d_u);
        prolong_center_kernel<<<grid_2d(nc - 1, nc - 1, block.x, block.y),
                                  block>>>(nc, d_ec, n, d_u);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_rc));
    CUDA_CHECK(cudaFree(d_ec));

    launch_smooth_rb(n, d_u, d_f, h2, 2);                       /* post-smooth */
}

// -- main -------------------------------------------------------------------

int main(int argc, char **argv) {
    int L      = (argc > 1) ? atoi(argv[1]) : 9;
    int cycles = (argc > 2) ? atoi(argv[2]) : 12;
    int n = (1 << L) + 1;
    double h = 1.0 / (n - 1);
    double h2 = h * h;
    size_t bytes = (size_t)n * n * sizeof(double);

    double *u = (double *)calloc((size_t)n * n, sizeof(double));
    double *f = (double *)calloc((size_t)n * n, sizeof(double));
    double *r = (double *)malloc(bytes);
    if (!u || !f || !r) { fprintf(stderr, "host alloc failed\n"); return 1; }
    for (int i = 1; i < n - 1; ++i)
        for (int j = 1; j < n - 1; ++j)
            f[i * n + j] = 1.0;

    double *d_u, *d_f, *d_r;
    CUDA_CHECK(cudaMalloc(&d_u, bytes));
    CUDA_CHECK(cudaMalloc(&d_f, bytes));
    CUDA_CHECK(cudaMalloc(&d_r, bytes));
    CUDA_CHECK(cudaMemcpy(d_u, u, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f, f, bytes, cudaMemcpyHostToDevice));

    /* initial residual norm -------------------------------------------------- */
    {
        dim3 block(16, 16);
        dim3 grid = grid_2d(n - 2, n - 2, block.x, block.y);
        residual_kernel<<<grid, block>>>(n, d_u, d_f, h2, d_r);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(r, d_r, bytes, cudaMemcpyDeviceToHost));

    double r0 = 0.0;
    for (int i = 1; i < n - 1; ++i)
        for (int j = 1; j < n - 1; ++j)
            r0 += r[i * n + j] * r[i * n + j];
    r0 = sqrt(r0);

    /* solve ------------------------------------------------------------------ */
    clock_t t0 = clock();
    double rk = r0;
    for (int c = 0; c < cycles; ++c) {
        vcycle(n, d_u, d_f, h);
        {
            dim3 block(16, 16);
            dim3 grid = grid_2d(n - 2, n - 2, block.x, block.y);
            residual_kernel<<<grid, block>>>(n, d_u, d_f, h2, d_r);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(r, d_r, bytes, cudaMemcpyDeviceToHost));
        rk = 0.0;
        for (int i = 1; i < n - 1; ++i)
            for (int j = 1; j < n - 1; ++j)
                rk += r[i * n + j] * r[i * n + j];
        rk = sqrt(rk);
    }
    clock_t t1 = clock();

    printf("multigrid L=%d (n=%d^2)  cycles=%d  res0=%.3e  res=%.3e  reduction=%.2e  time=%.3f s\n",
           L, n, cycles, r0, rk, rk / r0, (double)(t1 - t0) / CLOCKS_PER_SEC);

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_f));
    CUDA_CHECK(cudaFree(d_r));
    free(u); free(f); free(r);
    return 0;
}
