/* jacobi_eigen.c -- all eigenvalues of a symmetric matrix via the cyclic
 *                   Jacobi rotation method.
 *
 * Test matrix: the 1D Laplacian tridiag(-1, 2, -1), whose eigenvalues are known
 * analytically, lambda_k = 2 - 2 cos(k*pi/(n+1)), giving an exact correctness
 * check.
 *
 * Pattern: dense symmetric eigensolver -- a stand-in for the eigen/SVD bucket,
 * the hardest dense problem. Tier 3: iterative and convergence-dependent, with
 * each rotation touching a pair of rows/columns. The GPU-friendly variants are
 * block / one-sided Jacobi (the latter is the workhorse of GPU SVD).
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* Cyclic Jacobi: A symmetric (n x n), destroyed; eigenvalues left on diagonal. */
static int jacobi(int n, double *A, int max_sweeps, double tol) {
    int sweep = 0;
    for (; sweep < max_sweeps; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q) off += A[p * n + q] * A[p * n + q];
        if (sqrt(off) < tol) break;

        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q) {
                double apq = A[p * n + q];
                if (fabs(apq) < 1e-300) continue;
                double app = A[p * n + p], aqq = A[q * n + q];
                double theta = (aqq - app) / (2.0 * apq);
                double t = (theta >= 0.0 ? 1.0 : -1.0) /
                           (fabs(theta) + sqrt(theta * theta + 1.0));
                double c = 1.0 / sqrt(t * t + 1.0), s = t * c;

                for (int i = 0; i < n; ++i) {                  /* rotate columns p, q */
                    double aip = A[i * n + p], aiq = A[i * n + q];
                    A[i * n + p] = c * aip - s * aiq;
                    A[i * n + q] = s * aip + c * aiq;
                }
                for (int i = 0; i < n; ++i) {                  /* rotate rows p, q */
                    double api = A[p * n + i], aqi = A[q * n + i];
                    A[p * n + i] = c * api - s * aqi;
                    A[q * n + i] = s * api + c * aqi;
                }
            }
    }
    return sweep;
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 128;

    double *A = calloc((size_t)n * n, sizeof *A);
    for (int i = 0; i < n; ++i) {
        A[i * n + i] = 2.0;
        if (i > 0)     A[i * n + (i - 1)] = -1.0;
        if (i < n - 1) A[i * n + (i + 1)] = -1.0;
    }
    double trace0 = 0.0;
    for (int i = 0; i < n; ++i) trace0 += A[i * n + i];

    clock_t t0 = clock();
    int sweeps = jacobi(n, A, 100, 1e-12);
    clock_t t1 = clock();

    double tr = 0.0, emin = A[0], emax = A[0];
    for (int i = 0; i < n; ++i) {
        double e = A[i * n + i];
        tr += e;
        if (e < emin) emin = e;
        if (e > emax) emax = e;
    }
    const double PI = 3.14159265358979323846;
    double emin_exact = 2.0 - 2.0 * cos(PI / (n + 1));
    double emax_exact = 2.0 - 2.0 * cos(n * PI / (n + 1));

    printf("jacobi_eigen n=%d  sweeps=%d  sum(eig)=%.6f (trace=%.6f)\n",
           n, sweeps, tr, trace0);
    printf("            lambda_min=%.6f (exact %.6f)  lambda_max=%.6f (exact %.6f)  time=%.3f s\n",
           emin, emin_exact, emax, emax_exact, (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(A);
    return 0;
}
