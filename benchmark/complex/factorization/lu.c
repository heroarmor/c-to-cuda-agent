/* lu.c -- dense LU factorization with partial pivoting, then solve A x = b.
 *
 * Pattern: dense linear algebra, the direct-solver workhorse. Tier 3: the panel
 * factorization carries a data dependency down the columns, so the
 * high-performance GPU design (cf. MAGMA) is *hybrid* -- keep the small panel
 * factorization on the CPU and push the large trailing-submatrix update (a rank-1
 * / GEMM update) to the GPU. Converting it well means partitioning host vs device
 * work, not just optimizing one kernel.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static void lu_factor(int n, double *A, int *piv) {
    for (int k = 0; k < n; ++k) {
        int p = k;
        double best = fabs(A[k * n + k]);
        for (int i = k + 1; i < n; ++i) {
            double v = fabs(A[i * n + k]);
            if (v > best) { best = v; p = i; }
        }
        piv[k] = p;
        if (p != k)
            for (int j = 0; j < n; ++j) {
                double t = A[k * n + j]; A[k * n + j] = A[p * n + j]; A[p * n + j] = t;
            }
        double d = A[k * n + k];
        for (int i = k + 1; i < n; ++i) {
            double f = A[i * n + k] / d;
            A[i * n + k] = f;                                   /* store L multiplier */
            for (int j = k + 1; j < n; ++j)                    /* trailing update (the GEMM) */
                A[i * n + j] -= f * A[k * n + j];
        }
    }
}

static void lu_solve(int n, const double *A, const int *piv, double *b) {
    for (int k = 0; k < n; ++k)
        if (piv[k] != k) { double t = b[k]; b[k] = b[piv[k]]; b[piv[k]] = t; }
    for (int i = 0; i < n; ++i) {                              /* forward solve, L (unit diag) */
        double s = b[i];
        for (int j = 0; j < i; ++j) s -= A[i * n + j] * b[j];
        b[i] = s;
    }
    for (int i = n - 1; i >= 0; --i) {                         /* back solve, U */
        double s = b[i];
        for (int j = i + 1; j < n; ++j) s -= A[i * n + j] * b[j];
        b[i] = s / A[i * n + i];
    }
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 512;

    double *A  = malloc((size_t)n * n * sizeof *A);
    double *A0 = malloc((size_t)n * n * sizeof *A0);
    double *b  = malloc((size_t)n * sizeof *b);
    int *piv   = malloc((size_t)n * sizeof *piv);

    unsigned long s = 987654321ul;
    for (size_t i = 0; i < (size_t)n * n; ++i) {
        s = s * 1103515245ul + 12345ul;
        A0[i] = ((s >> 16) & 0x7fff) / 32768.0 - 0.5;
    }
    for (int i = 0; i < n; ++i) A0[i * n + i] += n;            /* diagonally dominant */
    for (size_t i = 0; i < (size_t)n * n; ++i) A[i] = A0[i];

    for (int i = 0; i < n; ++i) {                             /* b = A * ones (x_true = 1) */
        double sum = 0.0;
        for (int j = 0; j < n; ++j) sum += A0[i * n + j];
        b[i] = sum;
    }

    clock_t t0 = clock();
    lu_factor(n, A, piv);
    lu_solve(n, A, piv, b);
    clock_t t1 = clock();

    double err = 0.0;
    for (int i = 0; i < n; ++i) { double e = fabs(b[i] - 1.0); if (e > err) err = e; }

    printf("lu n=%d  max|x-1|=%.2e  time=%.3f s\n",
           n, err, (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(A); free(A0); free(b); free(piv);
    return 0;
}
