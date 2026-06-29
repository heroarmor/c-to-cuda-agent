/* spmv.c -- sparse matrix-vector product  y = A*x  in CSR format
 *
 * Pattern: sparse linear algebra (Berkeley dwarf). Irregular memory access and
 * low arithmetic intensity -- the workhorse inside every iterative solver.
 * GPU conversion: one thread (or warp) per row; performance is dominated by the
 * sparse format and load balancing across rows.
 *
 * Test matrix: 2D 5-point Laplacian on an m x m grid (n = m*m unknowns).
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Build the 5-point Laplacian in CSR. Returns nnz; allocates row_ptr/col_idx/val. */
static int build_laplacian(int m, int **row_ptr, int **col_idx, double **val) {
    int n = m * m;
    int *rp = malloc((size_t)(n + 1) * sizeof *rp);
    int *ci = malloc((size_t)5 * n * sizeof *ci);   /* up to 5 nnz per row */
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

static void spmv(int n, const int *row_ptr, const int *col_idx,
                 const double *val, const double *x, double *y) {
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
            s += val[k] * x[col_idx[k]];
        y[i] = s;
    }
}

int main(int argc, char **argv) {
    int m = (argc > 1) ? atoi(argv[1]) : 1024;   /* grid side; n = m*m */
    int n = m * m;

    int *row_ptr, *col_idx;
    double *val;
    int nnz = build_laplacian(m, &row_ptr, &col_idx, &val);

    double *x = malloc((size_t)n * sizeof *x);
    double *y = malloc((size_t)n * sizeof *y);
    for (int i = 0; i < n; ++i) x[i] = 1.0;

    int reps = 50;
    clock_t t0 = clock();
    for (int r = 0; r < reps; ++r) spmv(n, row_ptr, col_idx, val, x, y);
    clock_t t1 = clock();

    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += y[i];

    printf("spmv n=%d nnz=%d reps=%d  checksum=%.1f  time=%.3f s\n",
           n, nnz, reps, sum, (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(row_ptr); free(col_idx); free(val); free(x); free(y);
    return 0;
}
