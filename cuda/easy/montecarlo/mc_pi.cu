/* mc_pi.cu -- CUDA conversion of benchmark/easy/montecarlo/mc_pi.c
 * One thread per chunk with a private LCG state; final reduction on host.
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

/* per-stream 64-bit LCG returning a double in [0, 1) */
__device__ static inline double lcg(unsigned long long *s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(*s >> 11) / 9007199254740992.0;
}

__global__ void mc_pi_kernel(long long per, int chunks, long long *inside_counts) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= chunks) return;

    unsigned long long state =
        0x9E3779B97F4A7C15ULL ^ ((unsigned long long)(c + 1) * 0xD1B54A32D192ED03ULL);
    long long inside = 0;
    for (long long i = 0; i < per; ++i) {
        double x = lcg(&state), y = lcg(&state);
        if (x * x + y * y <= 1.0) ++inside;
    }
    inside_counts[c] = inside;
}

int main(int argc, char **argv) {
    long long samples = (argc > 1) ? atoll(argv[1]) : 100000000LL;
    int chunks        = (argc > 2) ? atoi(argv[2]) : 1024;
    long long per = samples / chunks;

    long long *d_counts;
    CUDA_CHECK(cudaMalloc(&d_counts, chunks * sizeof(long long)));

    int threads = 256;
    int blocks = (chunks + threads - 1) / threads;

    clock_t t0 = clock();
    mc_pi_kernel<<<blocks, threads>>>(per, chunks, d_counts);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_t t1 = clock();

    long long *h_counts = (long long *)malloc(chunks * sizeof(long long));
    if (!h_counts) { fprintf(stderr, "host alloc failed\n"); return 1; }
    CUDA_CHECK(cudaMemcpy(h_counts, d_counts, chunks * sizeof(long long),
                          cudaMemcpyDeviceToHost));

    long long total = 0;
    for (int c = 0; c < chunks; ++c) total += h_counts[c];

    long long n = per * chunks;
    double pi = 4.0 * (double)total / (double)n;
    printf("mc_pi samples=%lld chunks=%d  pi=%.6f  err=%.2e  time=%.3f s\n",
           n, chunks, pi, pi - 3.14159265358979323846,
           (double)(t1 - t0) / CLOCKS_PER_SEC);

    CUDA_CHECK(cudaFree(d_counts));
    free(h_counts);
    return 0;
}
