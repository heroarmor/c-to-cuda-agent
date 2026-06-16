/* saxpy.cu -- CUDA conversion of benchmark/easy/dense-linalg/saxpy.c
 * One thread per element; trivially data-parallel.
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

__global__ void saxpy_kernel(int n, float a, const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : (1 << 24);
    float a = 2.0f;
    size_t bytes = (size_t)n * sizeof(float);

    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);
    if (!h_x || !h_y) { fprintf(stderr, "alloc failed\n"); return 1; }

    for (int i = 0; i < n; ++i) { h_x[i] = 1.0f; h_y[i] = (float)i; }

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    clock_t t0 = clock();
    saxpy_kernel<<<blocks, threads>>>(n, a, d_x, d_y);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_t t1 = clock();

    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += h_y[i];

    printf("saxpy n=%d  checksum=%.0f  time=%.3f s\n",
           n, sum, (double)(t1 - t0) / CLOCKS_PER_SEC);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    free(h_x); free(h_y);
    return 0;
}
