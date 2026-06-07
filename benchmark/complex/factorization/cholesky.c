/* cholesky.c -- dense Cholesky factorization A = L L^T of an SPD matrix,
 *               then solve A x = b.
 *
 * Pattern: dense linear algebra. Tier 3, same hybrid CPU-GPU story as LU (a
 * column/panel data dependency plus a trailing SYRK/GEMM update); batched
 * tiny-Cholesky for millions of small matrices is a hot GPU topic (MAGMA).
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* In-place lower Cholesky factor (reads/writes lower triangle + diagonal). */
static int chol(int n, double *A) {
    for (int j = 0; j < n; ++j) {
        double d = A[j * n + j];
        for (int k = 0; k < j; ++k) d -= A[j * n + k] * A[j * n + k];
        if (d <= 0.0) return -1;                               /* not positive definite */
        d = sqrt(d);
        A[j * n + j] = d;
        for (int i = j + 1; i < n; ++i) {
            double sum = A[i * n + j];
            for (int k = 0; k < j; ++k) sum -= A[i * n + k] * A[j * n + k];
            A[i * n + j] = sum / d;
        }
    }
    return 0;
}

static void chol_solve(int n, const double *L, double *b) {
    for (int i = 0; i < n; ++i) {                              /* forward: L y = b */
        double s = b[i];
        for (int j = 0; j < i; ++j) s -= L[i * n + j] * b[j];
        b[i] = s / L[i * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {                         /* back: L^T x = y */
        double s = b[i];
        for (int j = i + 1; j < n; ++j) s -= L[j * n + i] * b[j];
        b[i] = s / L[i * n + i];
    }
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 512;

    double *B  = malloc((size_t)n * n * sizeof *B);
    double *A  = malloc((size_t)n * n * sizeof *A);
    double *A0 = malloc((size_t)n * n * sizeof *A0);
    double *b  = malloc((size_t)n * sizeof *b);

    unsigned long s = 13572468ul;
    for (size_t i = 0; i < (size_t)n * n; ++i) {
        s = s * 1103515245ul + 12345ul;
        B[i] = ((s >> 16) & 0x7fff) / 32768.0 - 0.5;
    }
    /* A = B B^T + n I  -> symmetric positive definite, well conditioned */
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) sum += B[i * n + k] * B[j * n + k];
            A0[i * n + j] = sum + (i == j ? n : 0.0);
        }
    for (size_t i = 0; i < (size_t)n * n; ++i) A[i] = A0[i];

    for (int i = 0; i < n; ++i) {                             /* b = A * ones */
        double sum = 0.0;
        for (int j = 0; j < n; ++j) sum += A0[i * n + j];
        b[i] = sum;
    }

    clock_t t0 = clock();
    int info = chol(n, A);
    if (info == 0) chol_solve(n, A, b);
    clock_t t1 = clock();

    double err = 0.0;
    for (int i = 0; i < n; ++i) { double e = fabs(b[i] - 1.0); if (e > err) err = e; }

    printf("cholesky n=%d  spd=%s  max|x-1|=%.2e  time=%.3f s\n",
           n, info == 0 ? "yes" : "NO", err, (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(B); free(A); free(A0); free(b);
    return 0;
}
