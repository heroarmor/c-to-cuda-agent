/* lanczos.c -- ground-state energy of the 1D antiferromagnetic Heisenberg
 *              spin-1/2 chain (periodic) by matrix-free Lanczos.
 *
 *     H = sum_i  S_i . S_{i+1}   (indices mod N)
 *       = sum_i  [ Sz_i Sz_{i+1} + 1/2 (S+_i S-_{i+1} + S-_i S+_{i+1}) ]
 *
 * The Hamiltonian acts on a 2^N-dimensional Hilbert space and is never stored:
 * its action on a vector is computed on the fly (diagonal Sz Sz terms plus
 * spin-flip off-diagonal terms on antiparallel bonds). Lanczos with full
 * reorthogonalization builds a small tridiagonal matrix whose smallest
 * eigenvalue (found by Sturm-sequence bisection) is the ground-state energy.
 *
 * Pattern: exact diagonalization for quantum many-body systems -- the workhorse
 * inside DMRG/Davidson (~85% of the time). Complex tier: matrix-free sparse
 * operator over an exponentially large state space with irregular (bit-flip)
 * memory access. GPU conversion: one thread per basis state for the H-apply
 * (a scatter into flipped states) plus dense vector reductions for Lanczos.
 *
 * Verification: a built-in N=4 ring is solved every run; its ground-state energy
 * is exactly -2.0, an analytic check of the operator and the eigensolver.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* out = H_Heisenberg(periodic) * in, for an Nspin chain (real, symmetric) */
static void applyH(int Nspin, const double *in, double *out) {
    long dim = 1L << Nspin;
    for (long s = 0; s < dim; ++s) out[s] = 0.0;
    for (long s = 0; s < dim; ++s) {
        double v = in[s];
        double diag = 0.0;
        for (int i = 0; i < Nspin; ++i) {
            int j = (i + 1) % Nspin;
            int bi = (int)((s >> i) & 1), bj = (int)((s >> j) & 1);
            if (bi == bj) diag += 0.25;
            else {
                diag -= 0.25;
                long s2 = s ^ ((1L << i) | (1L << j));   /* flip antiparallel pair */
                out[s2] += 0.5 * v;
            }
        }
        out[s] += diag * v;
    }
}

static double dot(long n, const double *a, const double *b) {
    double s = 0.0;
    for (long i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

/* number of eigenvalues of the symmetric tridiagonal (a[], b[]) below x */
static int sturm_count(int m, const double *a, const double *b, double x) {
    int count = 0;
    double d = a[0] - x;
    if (d < 0) ++count;
    for (int k = 1; k < m; ++k) {
        if (d == 0.0) d = 1e-300;
        d = (a[k] - x) - b[k] * b[k] / d;
        if (d < 0) ++count;
    }
    return count;
}

/* smallest eigenvalue of the symmetric tridiagonal via bisection */
static double smallest_eig(int m, const double *a, const double *b) {
    double lo = -1e6, hi = 1e6;
    for (int it = 0; it < 200; ++it) {
        double mid = 0.5 * (lo + hi);
        if (sturm_count(m, a, b, mid) >= 1) hi = mid;
        else lo = mid;
    }
    return 0.5 * (lo + hi);
}

/* ground-state energy of the Nspin Heisenberg ring via Lanczos */
static double ground_energy(int Nspin, int maxiter) {
    long dim = 1L << Nspin;
    int m = maxiter < dim ? maxiter : (int)dim;

    double *V = calloc((size_t)m * dim, sizeof *V);    /* Lanczos basis (for reorth) */
    double *w = malloc((size_t)dim * sizeof *w);
    double *a = malloc((size_t)m * sizeof *a);
    double *b = malloc((size_t)m * sizeof *b);

    unsigned long long rs = 0x1234567 ^ (unsigned long long)Nspin;
    double nrm = 0.0;
    for (long i = 0; i < dim; ++i) {
        rs = rs * 6364136223846793005ULL + 1442695040888963407ULL;
        double r = (double)(rs >> 11) / 9007199254740992.0 - 0.5;
        V[i] = r; nrm += r * r;
    }
    nrm = sqrt(nrm);
    for (long i = 0; i < dim; ++i) V[i] /= nrm;

    int j = 0;
    b[0] = 0.0;
    applyH(Nspin, &V[0], w);
    a[0] = dot(dim, &V[0], w);
    for (long i = 0; i < dim; ++i) w[i] -= a[0] * V[i];

    for (j = 1; j < m; ++j) {
        double beta = sqrt(dot(dim, w, w));
        if (beta < 1e-12) break;
        b[j] = beta;
        for (long i = 0; i < dim; ++i) V[(size_t)j * dim + i] = w[i] / beta;

        applyH(Nspin, &V[(size_t)j * dim], w);
        for (long i = 0; i < dim; ++i) w[i] -= beta * V[(size_t)(j - 1) * dim + i];
        a[j] = dot(dim, &V[(size_t)j * dim], w);
        for (long i = 0; i < dim; ++i) w[i] -= a[j] * V[(size_t)j * dim + i];

        for (int twice = 0; twice < 2; ++twice)        /* full reorthogonalization */
            for (int k = 0; k <= j; ++k) {
                double c = dot(dim, &V[(size_t)k * dim], w);
                for (long i = 0; i < dim; ++i) w[i] -= c * V[(size_t)k * dim + i];
            }
    }
    int mused = j;
    double e0 = smallest_eig(mused, a, b);

    free(V); free(w); free(a); free(b);
    return e0;
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 16;       /* number of spins */
    int m = (argc > 2) ? atoi(argv[2]) : 120;      /* Lanczos iterations */

    clock_t t0 = clock();
    double e0 = ground_energy(N, m);
    clock_t t1 = clock();

    double ref4 = ground_energy(4, 50);            /* analytic check: must be -2.0 */

    printf("lanczos N=%d (dim=%ld)  E0=%.8f  E0/N=%.6f  time=%.3f s\n",
           N, 1L << N, e0, e0 / N, (double)(t1 - t0) / CLOCKS_PER_SEC);
    printf("        check: E0(N=4 ring)=%.8f  (exact -2.0, err=%.1e)\n",
           ref4, fabs(ref4 + 2.0));
    return 0;
}
