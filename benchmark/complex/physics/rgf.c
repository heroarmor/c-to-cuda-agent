/* rgf.c -- Recursive Green's Function: diagonal blocks of the inverse of a
 *          block-tridiagonal matrix  (a.k.a. selected inversion).
 *
 * Computes the diagonal blocks of G = A^{-1} for a block-tridiagonal A WITHOUT
 * forming the full inverse. This is the "compute the diagonal" task ubiquitous
 * in physics: G = (zI - H)^{-1} is a Green's function whose diagonal gives the
 * local density of states / charge density (electronic structure, and NEGF
 * quantum transport). See SelInv / PEXSI and the recursive Green's function
 * (RGF) method.
 *
 * Test matrix: the 2D 5-point Laplacian on a (B x N) grid, which is
 * block-tridiagonal with diagonal blocks tridiag(-1, 4, -1) and coupling
 * blocks -I -- symmetric positive definite, so the inverse is well defined.
 *
 * Pattern: selected inversion / NEGF. The forward+backward block sweep is a
 * sequential recurrence (host-side control), but every step is dense block
 * linear algebra (B x B GEMMs and inverses) -> batched dense kernels on the GPU.
 * Independent energy points / right-hand sides are embarrassingly parallel, the
 * usual source of GPU throughput here.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static int B;   /* block size, set in main; used by the block helpers below */

static void mat_mul(const double *A, const double *Bm, double *C) {   /* C = A * Bm */
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < B; ++j) {
            double s = 0.0;
            for (int k = 0; k < B; ++k) s += A[i * B + k] * Bm[k * B + j];
            C[i * B + j] = s;
        }
}

static void mat_sub(const double *A, const double *Bm, double *C) {   /* C = A - Bm */
    for (int i = 0; i < B * B; ++i) C[i] = A[i] - Bm[i];
}

/* Out-of-place inverse of a B x B matrix via Gauss-Jordan with partial pivoting. */
static void mat_inv(const double *Ain, double *Aout) {
    int n = B;
    double *a = malloc((size_t)n * n * sizeof *a);
    memcpy(a, Ain, (size_t)n * n * sizeof *a);
    for (int i = 0; i < n * n; ++i) Aout[i] = 0.0;
    for (int i = 0; i < n; ++i) Aout[i * n + i] = 1.0;

    for (int col = 0; col < n; ++col) {
        int piv = col;
        double best = fabs(a[col * n + col]);
        for (int r = col + 1; r < n; ++r) {
            double vv = fabs(a[r * n + col]);
            if (vv > best) { best = vv; piv = r; }
        }
        if (piv != col)
            for (int k = 0; k < n; ++k) {
                double t = a[col * n + k]; a[col * n + k] = a[piv * n + k]; a[piv * n + k] = t;
                t = Aout[col * n + k]; Aout[col * n + k] = Aout[piv * n + k]; Aout[piv * n + k] = t;
            }
        double d = a[col * n + col];
        for (int k = 0; k < n; ++k) { a[col * n + k] /= d; Aout[col * n + k] /= d; }
        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            double f = a[r * n + col];
            for (int k = 0; k < n; ++k) {
                a[r * n + k]    -= f * a[col * n + k];
                Aout[r * n + k] -= f * Aout[col * n + k];
            }
        }
    }
    free(a);
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 256;   /* number of diagonal blocks */
    B     = (argc > 2) ? atoi(argv[2]) : 32;    /* block size */
    size_t bb = (size_t)B * B;

    /* diagonal block D = tridiag(-1, 4, -1);  coupling block C = -I */
    double *D = calloc(bb, sizeof *D);
    double *C = calloc(bb, sizeof *C);
    for (int i = 0; i < B; ++i) {
        D[i * B + i] = 4.0;
        if (i > 0)     D[i * B + (i - 1)] = -1.0;
        if (i < B - 1) D[i * B + (i + 1)] = -1.0;
        C[i * B + i] = -1.0;
    }

    double *g = malloc((size_t)N * bb * sizeof *g);   /* left-connected blocks */
    double *G = malloc((size_t)N * bb * sizeof *G);    /* diagonal blocks of inverse */
    double *t1 = malloc(bb * sizeof *t1);
    double *t2 = malloc(bb * sizeof *t2);
    double *t3 = malloc(bb * sizeof *t3);
    double *M  = malloc(bb * sizeof *M);
    #define gp(i) (&g[(size_t)(i) * bb])
    #define Gp(i) (&G[(size_t)(i) * bb])

    clock_t tic = clock();

    /* forward sweep: g_0 = D^{-1};  g_i = (D - C g_{i-1} C)^{-1} */
    mat_inv(D, gp(0));
    for (int i = 1; i < N; ++i) {
        mat_mul(C, gp(i - 1), t1);   /* C g_{i-1}        */
        mat_mul(t1, C, t2);          /* C g_{i-1} C      */
        mat_sub(D, t2, M);           /* D - C g_{i-1} C  */
        mat_inv(M, gp(i));
    }

    /* backward sweep: G_{N-1} = g_{N-1};
       G_i = g_i + g_i C G_{i+1} C g_i   (using symmetric coupling C) */
    memcpy(Gp(N - 1), gp(N - 1), bb * sizeof(double));
    for (int i = N - 2; i >= 0; --i) {
        mat_mul(gp(i), C, t1);       /* g_i C                 */
        mat_mul(t1, Gp(i + 1), t2);  /* g_i C G_{i+1}         */
        mat_mul(t2, C, t3);          /* g_i C G_{i+1} C       */
        mat_mul(t3, gp(i), t1);      /* g_i C G_{i+1} C g_i   */
        for (size_t k = 0; k < bb; ++k) Gp(i)[k] = gp(i)[k] + t1[k];
    }

    /* trace(A^{-1}) = sum of diagonals of all diagonal blocks */
    double trace = 0.0;
    for (int i = 0; i < N; ++i) {
        double *Gi = Gp(i);
        for (int d = 0; d < B; ++d) trace += Gi[d * B + d];
    }
    clock_t toc = clock();

    printf("rgf blocks=%d blocksize=%d (n=%d)  trace(inv)=%.6f  time=%.3f s\n",
           N, B, N * B, trace, (double)(toc - tic) / CLOCKS_PER_SEC);

    #undef gp
    #undef Gp
    free(D); free(C); free(g); free(G);
    free(t1); free(t2); free(t3); free(M);
    return 0;
}
