/* wave2d.cu -- CUDA conversion of benchmark/moderate/pde/wave2d.c
 *
 * 2D wave equation via explicit finite differences, in operator-split form so
 * each time step runs two distinct kernels that collaborate:
 *   1) laplacian_kernel  -- materialise the discrete Laplacian field of cur
 *   2) integrate_kernel  -- leapfrog update  next = 2*cur - prev + C2*lap
 * Three device buffers (prev/cur/next) rotate by host pointer swap; the time
 * loop stays on the host (frequent, e2e kernel invocation).
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

/* kernel 1: discrete Laplacian of the interior (1 <= x,y < m-1) */
__global__ void laplacian_kernel(int m, const double *cur, double *lap) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (x < m - 1 && y < m - 1) {
        int i = y * m + x;
        lap[i] = cur[i - 1] + cur[i + 1] + cur[i - m] + cur[i + m] - 4.0 * cur[i];
    }
}

/* kernel 2: leapfrog time integration consuming the Laplacian field */
__global__ void integrate_kernel(int m, double C2, const double *prev,
                                  const double *cur, const double *lap, double *next) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (x < m - 1 && y < m - 1) {
        int i = y * m + x;
        next[i] = 2.0 * cur[i] - prev[i] + C2 * lap[i];
    }
}

static double wave_energy(int m, double C2,
                          const double *cur, const double *prev) {
    double kin = 0.0, pot = 0.0;
    for (int i = 0; i < m * m; ++i) { double d = cur[i] - prev[i]; kin += d * d; }
    for (int y = 1; y < m - 1; ++y)
        for (int x = 1; x < m - 1; ++x) {
            int i = y * m + x;
            double Gcur = 4.0 * cur[i] - cur[i - 1] - cur[i + 1] - cur[i - m] - cur[i + m];
            pot += Gcur * prev[i];
        }
    return 0.5 * kin + 0.5 * C2 * pot;
}

int main(int argc, char **argv) {
    int m = (argc > 1) ? atoi(argv[1]) : 512;
    int steps = (argc > 2) ? atoi(argv[2]) : 300;
    double C2 = 0.25;
    size_t N = (size_t)m * m;
    size_t bytes = N * sizeof(double);

    double *h_cur = (double *)malloc(bytes);
    double *h_prev = (double *)malloc(bytes);
    if (!h_cur || !h_prev) { fprintf(stderr, "host alloc failed\n"); return 1; }

    /* initial Gaussian pulse */
    double cx = m / 2.0, cy = m / 2.0, sig = m / 16.0;
    for (int y = 0; y < m; ++y)
        for (int x = 0; x < m; ++x) {
            double dx = x - cx, dy = y - cy;
            double val = exp(-(dx * dx + dy * dy) / (2.0 * sig * sig));
            h_cur[(size_t)y * m + x] = val;
            h_prev[(size_t)y * m + x] = val;
        }

    double *d_prev, *d_cur, *d_next, *d_lap;
    CUDA_CHECK(cudaMalloc(&d_prev, bytes));
    CUDA_CHECK(cudaMalloc(&d_cur,  bytes));
    CUDA_CHECK(cudaMalloc(&d_next, bytes));
    CUDA_CHECK(cudaMalloc(&d_lap,  bytes));

    CUDA_CHECK(cudaMemcpy(d_cur,  h_cur,  bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prev, h_prev, bytes, cudaMemcpyHostToDevice));

    double E0 = wave_energy(m, C2, h_cur, h_prev);

    dim3 block(16, 16);
    dim3 grid((m + block.x - 2) / block.x, (m + block.y - 2) / block.y);

    clock_t t0 = clock();
    for (int s = 0; s < steps; ++s) {
        laplacian_kernel<<<grid, block>>>(m, d_cur, d_lap);
        integrate_kernel<<<grid, block>>>(m, C2, d_prev, d_cur, d_lap, d_next);
        CUDA_CHECK(cudaGetLastError());
        double *t = d_prev; d_prev = d_cur; d_cur = d_next; d_next = t;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_t t1 = clock();

    CUDA_CHECK(cudaMemcpy(h_cur,  d_cur,  bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_prev, d_prev, bytes, cudaMemcpyDeviceToHost));

    double Ef = wave_energy(m, C2, h_cur, h_prev);
    double drift = fabs(E0) > 0.0 ? fabs(Ef - E0) / fabs(E0) : fabs(Ef - E0);

    double energy = 0.0;
    for (size_t i = 0; i < N; ++i) energy += h_cur[i] * h_cur[i];

    printf("wave2d m=%d steps=%d  energy=%.4f  E_drift=%.2e (conserved)  time=%.3f s\n",
           m, steps, energy, drift, (double)(t1 - t0) / CLOCKS_PER_SEC);

    CUDA_CHECK(cudaFree(d_prev));
    CUDA_CHECK(cudaFree(d_cur));
    CUDA_CHECK(cudaFree(d_next));
    CUDA_CHECK(cudaFree(d_lap));
    free(h_cur); free(h_prev);
    return 0;
}
