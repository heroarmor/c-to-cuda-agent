/* conjugate_gradient.c -- Conjugate Gradient solver for  A x = b
 *
 * A is the 2D 5-point Laplacian on an m x m grid (symmetric positive definite).
 *
 * Pattern: THE host+device showcase. Each iteration the *host* orchestrates
 * several *device* kernels -- one SpMV, two dot products (reductions) and a few
 * AXPYs -- plus a convergence test. A good C->CUDA conversion must therefore
 * handle host-side control flow, repeated kernel launches, reductions, and
 * vectors that stay resident on the device across iterations -- i.e. far more
 * than optimizing a single kernel.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static int build_laplacian(int m, int **row_ptr, int **col_idx, double **val) {
    int n = m * m;
    int *rp = malloc((size_t)(n + 1) * sizeof *rp);
    int *ci = malloc((size_t)5 * n * sizeof *ci);
    double *v = malloc((size_t)5 * n * sizeof *v);
    int nnz = 0;
    rp[0] = 0;
    for (int iy = 0; iy < m; ++iy) {
        for (int ix = 0; ix < m; ++ix) {
            int row = iy * m + ix;
            if (iy > 0)     { ci[nnz] = row - m; v[nnz] = -1.0; nnz++; }
            if (ix > 0)     { ci[nnz] = row - 1; v[nnz] = -1.0; nnz++; }
            ci[nnz] = row; v[nnz] = 4.0; nnz++;
            if (ix < m - 1) { ci[nnz] = row + 1; v[nnz] = -1.0; nnz++; }
            if (iy < m - 1) { ci[nnz] = row + m; v[nnz] = -1.0; nnz++; }
            rp[row + 1] = nnz;
        }
    }
    *row_ptr = rp; *col_idx = ci; *val = v;
    return nnz;
}

static void spmv(int n, const int *rp, const int *ci, const double *v,
                 const double *x, double *y) {
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int k = rp[i]; k < rp[i + 1]; ++k) s += v[k] * x[ci[k]];
        y[i] = s;
    }
}

static double dot(int n, const double *a, const double *b) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

int main(int argc, char **argv) {
    int m = (argc > 1) ? atoi(argv[1]) : 512;     /* grid side; n = m*m */
    int maxit = (argc > 2) ? atoi(argv[2]) : 10000;
    double tol = 1e-8;
    int n = m * m;

    int *rp, *ci;
    double *v;
    build_laplacian(m, &rp, &ci, &v);

    double *x  = calloc((size_t)n, sizeof *x);   /* initial guess 0 */
    double *b  = malloc((size_t)n * sizeof *b);
    double *r  = malloc((size_t)n * sizeof *r);
    double *p  = malloc((size_t)n * sizeof *p);
    double *Ap = malloc((size_t)n * sizeof *Ap);
    for (int i = 0; i < n; ++i) b[i] = 1.0;

    /* r = b - A x  (x = 0  =>  r = b),  p = r */
    for (int i = 0; i < n; ++i) { r[i] = b[i]; p[i] = r[i]; }
    double rs_old = dot(n, r, r);
    double bnorm = sqrt(rs_old);

    clock_t t0 = clock();
    int k = 0;
    for (; k < maxit; ++k) {
        spmv(n, rp, ci, v, p, Ap);
        double alpha = rs_old / dot(n, p, Ap);
        for (int i = 0; i < n; ++i) x[i] += alpha * p[i];
        for (int i = 0; i < n; ++i) r[i] -= alpha * Ap[i];
        double rs_new = dot(n, r, r);
        if (sqrt(rs_new) <= tol * bnorm) { ++k; break; }
        double beta = rs_new / rs_old;
        for (int i = 0; i < n; ++i) p[i] = r[i] + beta * p[i];
        rs_old = rs_new;
    }
    clock_t t1 = clock();

    /* residual check: || b - A x || */
    spmv(n, rp, ci, v, x, Ap);
    double res = 0.0;
    for (int i = 0; i < n; ++i) { double d = b[i] - Ap[i]; res += d * d; }

    printf("cg n=%d  iters=%d  relres=%.2e  time=%.3f s\n",
           n, k, sqrt(res) / bnorm, (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(rp); free(ci); free(v);
    free(x); free(b); free(r); free(p); free(Ap);
    return 0;
}
