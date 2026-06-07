/* gemm.c -- dense single-precision matrix multiply  C = A * B   (N x N)
 *
 * Pattern: dense linear algebra (BLAS-3). The canonical GPU kernel: high
 * arithmetic intensity, regular access, compute-bound.
 * GPU conversion: 2D thread grid, one thread per C[i][j]; shared-memory tiling
 * for data reuse. Naive triple loop here is the reference to be transformed.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void gemm(int n, const float *A, const float *B, float *C) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < n; ++k)
                acc += A[i * n + k] * B[k * n + j];
            C[i * n + j] = acc;
        }
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 512;

    float *A = malloc((size_t)n * n * sizeof *A);
    float *B = malloc((size_t)n * n * sizeof *B);
    float *C = malloc((size_t)n * n * sizeof *C);
    if (!A || !B || !C) { fprintf(stderr, "alloc failed\n"); return 1; }

    for (size_t i = 0; i < (size_t)n * n; ++i) {
        A[i] = 1.0f / (float)((i % 100) + 1);
        B[i] = (float)(i % 7);
    }

    clock_t t0 = clock();
    gemm(n, A, B, C);
    clock_t t1 = clock();

    double trace = 0.0;
    for (int i = 0; i < n; ++i) trace += C[i * n + i];

    double secs = (double)(t1 - t0) / CLOCKS_PER_SEC;
    double gflop = 2.0 * n * n * n / 1e9;
    printf("gemm n=%d  trace=%.4f  time=%.3f s  (%.2f GFLOP/s)\n",
           n, trace, secs, secs > 0 ? gflop / secs : 0.0);

    free(A); free(B); free(C);
    return 0;
}
