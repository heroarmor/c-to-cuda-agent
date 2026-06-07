/* saxpy.c -- BLAS-1 vector update:  y = a*x + y
 *
 * Pattern: dense linear algebra (BLAS-1), element-wise / embarrassingly parallel.
 * GPU conversion: one thread per element; trivially data-parallel and memory-bound.
 * A good "hello world" target for the C->CUDA agent.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void saxpy(int n, float a, const float *x, float *y) {
    for (int i = 0; i < n; ++i)
        y[i] = a * x[i] + y[i];
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : (1 << 24);   /* ~16M elements */
    float a = 2.0f;

    float *x = malloc((size_t)n * sizeof *x);
    float *y = malloc((size_t)n * sizeof *y);
    if (!x || !y) { fprintf(stderr, "alloc failed\n"); return 1; }

    for (int i = 0; i < n; ++i) { x[i] = 1.0f; y[i] = (float)i; }

    clock_t t0 = clock();
    saxpy(n, a, x, y);
    clock_t t1 = clock();

    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += y[i];

    printf("saxpy n=%d  checksum=%.0f  time=%.3f s\n",
           n, sum, (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(x); free(y);
    return 0;
}
