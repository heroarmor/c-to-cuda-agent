/* multigrid.c -- geometric multigrid V-cycle for the 2D Poisson equation
 *
 *     -Laplacian u = f   on the unit square,   u = 0 on the boundary
 *
 * Grids have (2^L + 1) points per side. Components: red-black Gauss-Seidel
 * smoothing, full-weighting restriction, bilinear prolongation, recursive
 * V-cycle, exact 1-point coarse solve.
 *
 * Pattern: multigrid (the geometric cousin of algebraic multigrid / AMG used as
 * a GPU preconditioner, cf. AmgX). Tier 3: a hierarchy of grids with smoothing,
 * inter-grid transfers and a recursive control structure -- the host drives the
 * V-cycle while each smoother/transfer is a device kernel over a different grid
 * size. Optimal O(N) solver, so it converges in a handful of cycles.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static void smooth_rb(int n, double *u, const double *f, double h2, int iters) {
    for (int it = 0; it < iters; ++it)
        for (int color = 0; color < 2; ++color)
            for (int i = 1; i < n - 1; ++i)
                for (int j = 1; j < n - 1; ++j)
                    if (((i + j) & 1) == color)
                        u[i * n + j] = 0.25 * (h2 * f[i * n + j] +
                                               u[(i - 1) * n + j] + u[(i + 1) * n + j] +
                                               u[i * n + j - 1] + u[i * n + j + 1]);
}

static double residual(int n, const double *u, const double *f, double h2, double *r) {
    double nrm = 0.0;
    for (int i = 0; i < n * n; ++i) r[i] = 0.0;
    for (int i = 1; i < n - 1; ++i)
        for (int j = 1; j < n - 1; ++j) {
            double Au = (4.0 * u[i * n + j] - u[(i - 1) * n + j] - u[(i + 1) * n + j] -
                         u[i * n + j - 1] - u[i * n + j + 1]) / h2;
            double rr = f[i * n + j] - Au;
            r[i * n + j] = rr;
            nrm += rr * rr;
        }
    return sqrt(nrm);
}

static void restrict_fw(int nf, const double *rf, int nc, double *rc) {
    for (int i = 0; i < nc * nc; ++i) rc[i] = 0.0;
    for (int I = 1; I < nc - 1; ++I)
        for (int J = 1; J < nc - 1; ++J) {
            int i = 2 * I, j = 2 * J;
            rc[I * nc + J] =
                0.2500 *  rf[i * nf + j] +
                0.1250 * (rf[(i - 1) * nf + j] + rf[(i + 1) * nf + j] +
                          rf[i * nf + j - 1] + rf[i * nf + j + 1]) +
                0.0625 * (rf[(i - 1) * nf + j - 1] + rf[(i - 1) * nf + j + 1] +
                          rf[(i + 1) * nf + j - 1] + rf[(i + 1) * nf + j + 1]);
        }
}

static void prolong_add(int nc, const double *ec, int nf, double *uf) {
    for (int I = 0; I < nc; ++I)                                /* coincident points */
        for (int J = 0; J < nc; ++J)
            uf[(2 * I) * nf + (2 * J)] += ec[I * nc + J];
    for (int I = 0; I < nc; ++I)                                /* horizontal edges  */
        for (int J = 0; J < nc - 1; ++J)
            uf[(2 * I) * nf + (2 * J + 1)] += 0.5 * (ec[I * nc + J] + ec[I * nc + J + 1]);
    for (int I = 0; I < nc - 1; ++I)                            /* vertical edges    */
        for (int J = 0; J < nc; ++J)
            uf[(2 * I + 1) * nf + (2 * J)] += 0.5 * (ec[I * nc + J] + ec[(I + 1) * nc + J]);
    for (int I = 0; I < nc - 1; ++I)                            /* cell centers      */
        for (int J = 0; J < nc - 1; ++J)
            uf[(2 * I + 1) * nf + (2 * J + 1)] +=
                0.25 * (ec[I * nc + J] + ec[I * nc + J + 1] +
                        ec[(I + 1) * nc + J] + ec[(I + 1) * nc + J + 1]);
}

static void vcycle(int n, double *u, const double *f, double h) {
    double h2 = h * h;
    if (n <= 3) {                                              /* one interior point: exact */
        smooth_rb(n, u, f, h2, 1);
        return;
    }
    smooth_rb(n, u, f, h2, 2);                                 /* pre-smooth */

    double *r = malloc((size_t)n * n * sizeof *r);
    residual(n, u, f, h2, r);

    int nc = (n - 1) / 2 + 1;
    double *rc = malloc((size_t)nc * nc * sizeof *rc);
    double *ec = calloc((size_t)nc * nc, sizeof *ec);
    restrict_fw(n, r, nc, rc);

    vcycle(nc, ec, rc, 2.0 * h);                               /* recurse on coarse grid */

    prolong_add(nc, ec, n, u);                                 /* correct fine solution */
    smooth_rb(n, u, f, h2, 2);                                 /* post-smooth */

    free(r); free(rc); free(ec);
}

int main(int argc, char **argv) {
    int L      = (argc > 1) ? atoi(argv[1]) : 9;     /* grid = (2^L + 1)^2 */
    int cycles = (argc > 2) ? atoi(argv[2]) : 12;
    int n = (1 << L) + 1;
    double h = 1.0 / (n - 1);

    double *u = calloc((size_t)n * n, sizeof *u);
    double *f = calloc((size_t)n * n, sizeof *f);
    double *r = malloc((size_t)n * n * sizeof *r);
    for (int i = 1; i < n - 1; ++i)
        for (int j = 1; j < n - 1; ++j) f[i * n + j] = 1.0;

    double r0 = residual(n, u, f, h * h, r);

    clock_t t0 = clock();
    double rk = r0;
    for (int c = 0; c < cycles; ++c) {
        vcycle(n, u, f, h);
        rk = residual(n, u, f, h * h, r);
    }
    clock_t t1 = clock();

    printf("multigrid L=%d (n=%d^2)  cycles=%d  res0=%.3e  res=%.3e  reduction=%.2e  time=%.3f s\n",
           L, n, cycles, r0, rk, rk / r0, (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(u); free(f); free(r);
    return 0;
}
