/* spgemm.c -- sparse general matrix-matrix product  C = A * A  (CSR).
 *
 * A is the 2D 5-point Laplacian; A*A is the (13-point) stencil A^2.
 *
 * Pattern: sparse linear algebra (SpGEMM). Tier 3 and notoriously hard on GPUs:
 * the output sparsity pattern is unknown a priori, so it needs a symbolic
 * (counting) pass plus irregular per-row accumulation, with severe load
 * imbalance. State-of-the-art GPU methods use iterative row merging / hashing.
 * Here a dense sparse-accumulator (SPA) does the per-row merge.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int build_laplacian(int m, int **row_ptr, int **col_idx, double **val) {
    int n = m * m;
    int *rp = malloc((size_t)(n + 1) * sizeof *rp);
    int *ci = malloc((size_t)5 * n * sizeof *ci);
    double *v = malloc((size_t)5 * n * sizeof *v);
    int nnz = 0;
    rp[0] = 0;
    for (int iy = 0; iy < m; ++iy)
        for (int ix = 0; ix < m; ++ix) {
            int row = iy * m + ix;
            if (iy > 0)     { ci[nnz] = row - m; v[nnz] = -1.0; nnz++; }
            if (ix > 0)     { ci[nnz] = row - 1; v[nnz] = -1.0; nnz++; }
            ci[nnz] = row; v[nnz] = 4.0; nnz++;
            if (ix < m - 1) { ci[nnz] = row + 1; v[nnz] = -1.0; nnz++; }
            if (iy < m - 1) { ci[nnz] = row + m; v[nnz] = -1.0; nnz++; }
            rp[row + 1] = nnz;
        }
    *row_ptr = rp; *col_idx = ci; *val = v;
    return nnz;
}

int main(int argc, char **argv) {
    int m = (argc > 1) ? atoi(argv[1]) : 512;   /* grid side; n = m*m */
    int n = m * m;

    int *Arp, *Aci;
    double *Av;
    build_laplacian(m, &Arp, &Aci, &Av);

    double *acc  = malloc((size_t)n * sizeof *acc);   /* sparse accumulator values */
    int    *idx  = malloc((size_t)n * sizeof *idx);   /* list of touched columns   */
    char   *used = calloc((size_t)n, 1);

    int *Crp = malloc((size_t)(n + 1) * sizeof *Crp);
    Crp[0] = 0;

    clock_t t0 = clock();

    /* symbolic pass: count nonzeros per row of C */
    long total = 0;
    for (int i = 0; i < n; ++i) {
        int cnt = 0;
        for (int k = Arp[i]; k < Arp[i + 1]; ++k) {
            int a = Aci[k];
            for (int kk = Arp[a]; kk < Arp[a + 1]; ++kk) {
                int col = Aci[kk];
                if (!used[col]) { used[col] = 1; idx[cnt++] = col; }
            }
        }
        for (int t = 0; t < cnt; ++t) used[idx[t]] = 0;
        total += cnt;
        Crp[i + 1] = (int)total;
    }

    int    *Cci = malloc((size_t)total * sizeof *Cci);
    double *Cv  = malloc((size_t)total * sizeof *Cv);

    /* numeric pass: accumulate C[i, :] = sum_a A[i,a] * A[a, :] */
    for (int i = 0; i < n; ++i) {
        int cnt = 0;
        for (int k = Arp[i]; k < Arp[i + 1]; ++k) {
            int a = Aci[k];
            double av = Av[k];
            for (int kk = Arp[a]; kk < Arp[a + 1]; ++kk) {
                int col = Aci[kk];
                if (!used[col]) { used[col] = 1; idx[cnt++] = col; acc[col] = 0.0; }
                acc[col] += av * Av[kk];
            }
        }
        int pos = Crp[i];
        for (int t = 0; t < cnt; ++t) {
            Cci[pos + t] = idx[t];
            Cv[pos + t]  = acc[idx[t]];
            used[idx[t]] = 0;
        }
    }
    clock_t t1 = clock();

    double sum = 0.0;
    for (long k = 0; k < total; ++k) sum += Cv[k];

    printf("spgemm n=%d  nnz(A)=%d  nnz(A^2)=%ld  checksum=%.1f  time=%.3f s\n",
           n, Arp[n], total, sum, (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(Arp); free(Aci); free(Av);
    free(acc); free(idx); free(used);
    free(Crp); free(Cci); free(Cv);
    return 0;
}
