/* heat2d.cu -- CUDA conversion of benchmark/easy/pde/heat2d.c
 * 2D thread grid, one thread per interior cell; grids resident on device.
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

__global__ void heat2d_kernel(int m, double r, const double *u, double *un) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (x < m - 1 && y < m - 1) {
        int i = y * m + x;
        un[i] = u[i] + r * (u[i - 1] + u[i + 1] + u[i - m] + u[i + m] - 4.0 * u[i]);
    }
}

int main(int argc, char **argv) {
    int m = (argc > 1) ? atoi(argv[1]) : 512;
    int steps = (argc > 2) ? atoi(argv[2]) : 200;
    double r = 0.2;
    size_t N = (size_t)m * m;
    size_t bytes = N * sizeof(double);

    double *h_u = (double *)malloc(bytes);
    double *h_un = (double *)malloc(bytes);
    if (!h_u || !h_un) { fprintf(stderr, "alloc failed\n"); return 1; }

    for (size_t i = 0; i < N; ++i) h_u[i] = 0.0;
    for (int y = m / 4; y < 3 * m / 4; ++y)
        for (int x = m / 4; x < 3 * m / 4; ++x)
            h_u[(size_t)y * m + x] = 1.0;

    double *d_u, *d_un;
    CUDA_CHECK(cudaMalloc(&d_u, bytes));
    CUDA_CHECK(cudaMalloc(&d_un, bytes));

    CUDA_CHECK(cudaMemcpy(d_u, h_u, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_un, h_un, bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((m + block.x - 2) / block.x, (m + block.y - 2) / block.y);

    clock_t t0 = clock();
    for (int s = 0; s < steps; ++s) {
        heat2d_kernel<<<grid, block>>>(m, r, d_u, d_un);
        double *tmp = d_u; d_u = d_un; d_un = tmp;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_t t1 = clock();

    CUDA_CHECK(cudaMemcpy(h_u, d_u, bytes, cudaMemcpyDeviceToHost));

    double total = 0.0;
    for (size_t i = 0; i < N; ++i) total += h_u[i];

    printf("heat2d m=%d steps=%d  total_heat=%.4f  time=%.3f s\n",
           m, steps, total, (double)(t1 - t0) / CLOCKS_PER_SEC);

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_un));
    free(h_u); free(h_un);
    return 0;
}
