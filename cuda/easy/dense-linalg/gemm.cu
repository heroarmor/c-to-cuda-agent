/* gemm.cu -- CUDA conversion of benchmark/moderate/dense-linalg/gemm.c
 * 2D thread grid with shared-memory tiling for data reuse.
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

#define TILE_SIZE 16

__global__ void gemm_kernel(int n, const float *A, const float *B, float *C) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;
    int num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int t_col = t * TILE_SIZE + threadIdx.x;
        int t_row = t * TILE_SIZE + threadIdx.y;

        if (row < n && t_col < n)
            As[threadIdx.y][threadIdx.x] = A[row * n + t_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (t_row < n && col < n)
            Bs[threadIdx.y][threadIdx.x] = B[t_row * n + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = acc;
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 512;
    size_t bytes = (size_t)n * n * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    if (!h_A || !h_B || !h_C) { fprintf(stderr, "alloc failed\n"); return 1; }

    for (size_t i = 0; i < (size_t)n * n; ++i) {
        h_A[i] = 1.0f / (float)((i % 100) + 1);
        h_B[i] = (float)(i % 7);
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    clock_t t0 = clock();
    gemm_kernel<<<grid, block>>>(n, d_A, d_B, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_t t1 = clock();

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    double trace = 0.0;
    for (int i = 0; i < n; ++i) trace += h_C[i * n + i];

    double secs = (double)(t1 - t0) / CLOCKS_PER_SEC;
    double gflop = 2.0 * n * n * n / 1e9;
    printf("gemm n=%d  trace=%.4f  time=%.3f s  (%.2f GFLOP/s)\n",
           n, trace, secs, secs > 0 ? gflop / secs : 0.0);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);
    return 0;
}
