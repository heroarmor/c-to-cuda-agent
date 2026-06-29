/* tensor_contraction.cu -- CUDA conversion of benchmark/moderate/tensor/tensor_contraction.c
 * 2D thread grid with shared-memory tiling (tensor contraction as GEMM).
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

__global__ void tensor_contraction_kernel(
    const double *A, const double *B, double *C,
    int IJ, int KL, int M)
{
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    double sum = 0.0;
    int num_tiles = (M + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int t_col = t * TILE_SIZE + threadIdx.x;
        int t_row = t * TILE_SIZE + threadIdx.y;

        if (row < IJ && t_col < M)
            As[threadIdx.y][threadIdx.x] = A[(size_t)row * M + t_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;

        if (t_row < M && col < KL)
            Bs[threadIdx.y][threadIdx.x] = B[(size_t)t_row * KL + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < IJ && col < KL)
        C[(size_t)row * KL + col] = sum;
}

int main(int argc, char **argv) {
    int d = (argc > 1) ? atoi(argv[1]) : 32;
    int I = d, J = d, K = d, L = d, M = d;
    int IJ = I * J, KL = K * L;

    size_t szA = (size_t)I * J * M;
    size_t szB = (size_t)M * K * L;
    size_t szC = (size_t)I * J * K * L;

    double *h_A = (double *)malloc(szA * sizeof(double));
    double *h_B = (double *)malloc(szB * sizeof(double));
    double *h_C = (double *)malloc(szC * sizeof(double));
    if (!h_A || !h_B || !h_C) { fprintf(stderr, "alloc failed\n"); return 1; }

    for (size_t i = 0; i < szA; ++i) h_A[i] = 1.0 / (double)((i % 97) + 1);
    for (size_t i = 0; i < szB; ++i) h_B[i] = (double)(i % 13) - 6.0;

    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, szA * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, szB * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, szC * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, szA * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, szB * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, szC * sizeof(double)));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((KL + TILE_SIZE - 1) / TILE_SIZE, (IJ + TILE_SIZE - 1) / TILE_SIZE);

    clock_t t0 = clock();
    tensor_contraction_kernel<<<grid, block>>>(d_A, d_B, d_C, IJ, KL, M);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_t t1 = clock();

    CUDA_CHECK(cudaMemcpy(h_C, d_C, szC * sizeof(double), cudaMemcpyDeviceToHost));

    double sum = 0.0;
    for (size_t i = 0; i < szC; ++i) sum += h_C[i];

    double secs = (double)(t1 - t0) / CLOCKS_PER_SEC;
    double gflop = 2.0 * (double)szC * M / 1e9;
    printf("tensor_contraction d=%d  C[%d^4]  checksum=%.4f  time=%.3f s  (%.2f GFLOP/s)\n",
           d, d, sum, secs, secs > 0 ? gflop / secs : 0.0);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);
    return 0;
}
