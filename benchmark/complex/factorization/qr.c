/* qr.c -- Householder QR factorization, then solve the square system A x = b
 *         by applying Q^T to b and back-substituting with R.
 *
 * Pattern: dense linear algebra (orthogonal factorization), the stable backbone
 * of least-squares and many eigen/SVD methods. Tier 3: each Householder
 * reflector depends on the previous one (column-by-column dependency) while the
 * trailing update is a rank-1 / block GEMM -- the same host-panel / device-update
 * partitioning challenge as LU/Cholesky.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* Householder QR on n x n A (R left in upper triangle); Q^T applied to b.
 * Then back-substitution R x = (Q^T b); solution returned in b. */
static void qr_solve(int n, double *A, double *b) {
    double *v = malloc((size_t)n * sizeof *v);
    for (int k = 0; k < n; ++k) {
        double normx = 0.0;
        for (int i = k; i < n; ++i) normx += A[i * n + k] * A[i * n + k];
        normx = sqrt(normx);
        if (normx == 0.0) continue;
        double alpha = (A[k * n + k] > 0.0) ? -normx : normx;   /* sign for stability */
        for (int i = k; i < n; ++i) v[i] = A[i * n + k];
        v[k] -= alpha;
        double vnorm2 = 0.0;
        for (int i = k; i < n; ++i) vnorm2 += v[i] * v[i];
        if (vnorm2 < 1e-300) continue;

        for (int j = k; j < n; ++j) {                          /* reflect columns of A */
            double dot = 0.0;
            for (int i = k; i < n; ++i) dot += v[i] * A[i * n + j];
            double f = 2.0 * dot / vnorm2;
            for (int i = k; i < n; ++i) A[i * n + j] -= f * v[i];
        }
        double dot = 0.0;                                      /* reflect b */
        for (int i = k; i < n; ++i) dot += v[i] * b[i];
        double f = 2.0 * dot / vnorm2;
        for (int i = k; i < n; ++i) b[i] -= f * v[i];
    }
    free(v);

    for (int i = n - 1; i >= 0; --i) {                         /* R x = Q^T b */
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

    unsigned long s = 24681357ul;
    for (size_t i = 0; i < (size_t)n * n; ++i) {
        s = s * 1103515245ul + 12345ul;
        A0[i] = ((s >> 16) & 0x7fff) / 32768.0 - 0.5;
    }
    for (int i = 0; i < n; ++i) A0[i * n + i] += n;            /* well conditioned */
    for (size_t i = 0; i < (size_t)n * n; ++i) A[i] = A0[i];

    for (int i = 0; i < n; ++i) {                             /* b = A * ones */
        double sum = 0.0;
        for (int j = 0; j < n; ++j) sum += A0[i * n + j];
        b[i] = sum;
    }

    clock_t t0 = clock();
    qr_solve(n, A, b);
    clock_t t1 = clock();

    double err = 0.0;
    for (int i = 0; i < n; ++i) { double e = fabs(b[i] - 1.0); if (e > err) err = e; }

    printf("qr n=%d  max|x-1|=%.2e  time=%.3f s\n",
           n, err, (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(A); free(A0); free(b);
    return 0;
}
