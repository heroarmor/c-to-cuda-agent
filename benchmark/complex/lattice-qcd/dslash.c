/* dslash.c -- Wilson-Dirac operator ("Dslash") on a 4D lattice with an SU(3)
 *             gauge field, plus a conjugate-gradient solve of the normal system.
 *
 * This is the central kernel of lattice quantum chromodynamics. A quark field is
 * a spinor with 4 spin x 3 color = 12 complex components per lattice site. The
 * Wilson operator is a nearest-neighbor hopping term in 4 dimensions:
 *
 *   (D psi)(x) = psi(x)
 *     - kappa * sum_mu [ (1 - gamma_mu) U_mu(x)   psi(x+mu)
 *                      + (1 + gamma_mu) U_mu^dag(x-mu) psi(x-mu) ]
 *
 * Each hop multiplies by a 3x3 complex SU(3) link matrix (color) and a spin
 * projector built from the Dirac gamma matrices.
 *
 * Pattern: lattice QCD -- a complex-arithmetic 4D stencil with small dense
 * (3x3 and 4x4) blocks per link; the flagship HPC-physics kernel (cf. QUDA).
 * Complex tier: heavy complex SU(3) algebra, 8-way 4D neighbor access, and a
 * Krylov solve built on top (D is not Hermitian, so CG runs on D^dag D).
 * GPU conversion: one thread per site, gauge links in (texture/constant) memory,
 * spin projection to halve the work, the CG loop orchestrated on the host.
 *
 * Verification: the Wilson operator is gamma5-Hermitian, D^dag = gamma5 D gamma5,
 * i.e. M = gamma5 D is Hermitian. We check <phi, M psi> = conj(<psi, M phi>) for
 * random spinors. The CG residual on D^dag D x = b is reported as a second check.
 */
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <time.h>

typedef double complex cx;

static int L;                 /* lattice size per dimension */
static long SITES;            /* L^4 */
static long NF;               /* SITES * 12 (complex components in a spinor field) */

/* gamma matrices (Euclidean, Hermitian) and gamma5 = diag(1,1,-1,-1) */
static cx gamma_[4][4][4];
static cx gamma5[4][4];

static void init_gammas(void) {
    for (int m = 0; m < 4; ++m)
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) gamma_[m][i][j] = 0.0;
    cx i_ = _Complex_I;
    /* chiral basis: g_k = [[0,-i sigma_k],[i sigma_k,0]], g4 = [[0,1],[1,0]] */
    /* gamma1 */
    gamma_[0][0][3] = -i_; gamma_[0][1][2] = -i_; gamma_[0][2][1] =  i_; gamma_[0][3][0] =  i_;
    /* gamma2 */
    gamma_[1][0][3] = -1; gamma_[1][1][2] =  1; gamma_[1][2][1] =  1; gamma_[1][3][0] = -1;
    /* gamma3 */
    gamma_[2][0][2] = -i_; gamma_[2][1][3] =  i_; gamma_[2][2][0] =  i_; gamma_[2][3][1] = -i_;
    /* gamma4 */
    gamma_[3][0][2] =  1; gamma_[3][1][3] =  1; gamma_[3][2][0] =  1; gamma_[3][3][1] =  1;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) gamma5[i][j] = 0.0;
    gamma5[0][0] = 1; gamma5[1][1] = 1; gamma5[2][2] = -1; gamma5[3][3] = -1;
}

static inline long site_index(int x, int y, int z, int t) {
    return ((((long)t) * L + z) * L + y) * L + x;
}

/* neighbor site in +/-direction mu (0..3 -> x,y,z,t), periodic */
static long neighbor(long s, int mu, int dir) {
    int c[4];
    c[0] = (int)(s % L); c[1] = (int)((s / L) % L);
    c[2] = (int)((s / ((long)L * L)) % L); c[3] = (int)((s / ((long)L * L * L)) % L);
    c[mu] = (c[mu] + dir + L) % L;
    return site_index(c[0], c[1], c[2], c[3]);
}

/* spinor accessor: field[(site*4 + spin)*3 + color] */
static inline cx *SP(cx *f, long site) { return &f[site * 12]; }

/* ---- SU(3) link storage: U[(site*4 + mu)*9 + (a*3+b)] ---- */
static cx *U;

static double urand(unsigned long long *s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(*s >> 11) / 9007199254740992.0 - 0.5;
}

/* build a random SU(3) matrix into m[9] (row-major 3x3) */
static void random_su3(cx *m, unsigned long long *rs) {
    for (int i = 0; i < 9; ++i) m[i] = urand(rs) + urand(rs) * _Complex_I;
    /* Gram-Schmidt on the three columns -> unitary */
    for (int col = 0; col < 3; ++col) {
        for (int prev = 0; prev < col; ++prev) {
            cx proj = 0;
            for (int r = 0; r < 3; ++r) proj += conj(m[r * 3 + prev]) * m[r * 3 + col];
            for (int r = 0; r < 3; ++r) m[r * 3 + col] -= proj * m[r * 3 + prev];
        }
        double nrm = 0;
        for (int r = 0; r < 3; ++r) nrm += creal(m[r * 3 + col]) * creal(m[r * 3 + col]) +
                                            cimag(m[r * 3 + col]) * cimag(m[r * 3 + col]);
        nrm = sqrt(nrm);
        for (int r = 0; r < 3; ++r) m[r * 3 + col] /= nrm;
    }
    /* fix determinant to 1: multiply column 0 by conj(det) (|det|=1 for unitary) */
    cx det = m[0] * (m[4] * m[8] - m[5] * m[7])
           - m[1] * (m[3] * m[8] - m[5] * m[6])
           + m[2] * (m[3] * m[7] - m[4] * m[6]);
    cx phase = conj(det);
    for (int r = 0; r < 3; ++r) m[r * 3 + 0] *= phase;
}

/* color matvec: out[spin][a] = sum_b U[a][b] in[spin][b]; dagger if conj-transpose */
static void su3_mul(const cx *Um, const cx *in, cx *out, int dagger) {
    for (int s = 0; s < 4; ++s)
        for (int a = 0; a < 3; ++a) {
            cx acc = 0;
            for (int b = 0; b < 3; ++b)
                acc += (dagger ? conj(Um[b * 3 + a]) : Um[a * 3 + b]) * in[s * 3 + b];
            out[s * 3 + a] = acc;
        }
}

/* spin projection: out = (I - sign*gamma_mu) in ; sign=+1 -> (1-g), -1 -> (1+g) */
static void spin_proj(int mu, int sign, const cx *in, cx *out) {
    for (int s = 0; s < 4; ++s)
        for (int c = 0; c < 3; ++c) {
            cx acc = in[s * 3 + c];
            for (int sp = 0; sp < 4; ++sp)
                acc -= (cx)sign * gamma_[mu][s][sp] * in[sp * 3 + c];
            out[s * 3 + c] = acc;
        }
}

static double kappa = 0.12;

/* out = D in  (Wilson-Dirac operator) */
static void apply_D(cx *out, const cx *in) {
    cx t1[12], t2[12];
    for (long site = 0; site < SITES; ++site) {
        cx *o = SP(out, site);
        const cx *c = &in[site * 12];
        for (int k = 0; k < 12; ++k) o[k] = c[k];               /* diagonal "1" term */
        for (int mu = 0; mu < 4; ++mu) {
            long fwd = neighbor(site, mu, +1), bwd = neighbor(site, mu, -1);
            const cx *Uf = &U[(site * 4 + mu) * 9];
            const cx *Ub = &U[(bwd * 4 + mu) * 9];
            /* forward: (1 - gamma_mu) U_mu(x) in(x+mu) */
            su3_mul(Uf, &in[fwd * 12], t1, 0);
            spin_proj(mu, +1, t1, t2);
            for (int k = 0; k < 12; ++k) o[k] -= kappa * t2[k];
            /* backward: (1 + gamma_mu) U_mu^dag(x-mu) in(x-mu) */
            su3_mul(Ub, &in[bwd * 12], t1, 1);
            spin_proj(mu, -1, t1, t2);
            for (int k = 0; k < 12; ++k) o[k] -= kappa * t2[k];
        }
    }
}

static void apply_g5(cx *out, const cx *in) {
    for (long site = 0; site < SITES; ++site) {
        const cx *c = &in[site * 12];
        cx *o = &out[site * 12];
        for (int s = 0; s < 4; ++s)
            for (int col = 0; col < 3; ++col)
                o[s * 3 + col] = gamma5[s][s] * c[s * 3 + col];
    }
}

/* out = D^dag in = gamma5 D gamma5 in */
static void apply_Ddag(cx *out, const cx *in, cx *scratch) {
    apply_g5(scratch, in);
    apply_D(out, scratch);
    apply_g5(out, out);
}

static cx vdot(const cx *a, const cx *b) {           /* <a,b> = sum conj(a) b */
    cx s = 0;
    for (long i = 0; i < NF; ++i) s += conj(a[i]) * b[i];
    return s;
}

int main(int argc, char **argv) {
    L = (argc > 1) ? atoi(argv[1]) : 6;
    SITES = (long)L * L * L * L;
    NF = SITES * 12;
    init_gammas();

    U = malloc((size_t)SITES * 4 * 9 * sizeof *U);
    unsigned long long rs = 0xC0FFEEULL;
    for (long s = 0; s < SITES; ++s)
        for (int mu = 0; mu < 4; ++mu) random_su3(&U[(s * 4 + mu) * 9], &rs);

    cx *psi = malloc((size_t)NF * sizeof *psi);
    cx *phi = malloc((size_t)NF * sizeof *phi);
    cx *Dpsi = malloc((size_t)NF * sizeof *Dpsi);
    cx *Dphi = malloc((size_t)NF * sizeof *Dphi);
    cx *tmp  = malloc((size_t)NF * sizeof *tmp);
    for (long i = 0; i < NF; ++i) {
        psi[i] = urand(&rs) + urand(&rs) * _Complex_I;
        phi[i] = urand(&rs) + urand(&rs) * _Complex_I;
    }

    clock_t t0 = clock();
    apply_D(Dpsi, psi);
    apply_D(Dphi, phi);
    clock_t t1 = clock();

    /* gamma5-Hermiticity: a = <phi, g5 D psi>,  b = <psi, g5 D phi>, expect a = conj(b) */
    apply_g5(tmp, Dpsi);  cx a = vdot(phi, tmp);
    apply_g5(tmp, Dphi);  cx b = vdot(psi, tmp);
    double herm_err = cabs(a - conj(b));

    /* CG on D^dag D x = rhs (Hermitian positive definite) */
    cx *x = calloc((size_t)NF, sizeof *x);
    cx *r = malloc((size_t)NF * sizeof *r);
    cx *p = malloc((size_t)NF * sizeof *p);
    cx *Ap = malloc((size_t)NF * sizeof *Ap);
    cx *sc = malloc((size_t)NF * sizeof *sc);
    for (long i = 0; i < NF; ++i) { r[i] = psi[i]; p[i] = r[i]; }   /* rhs = psi, x0 = 0 */
    double rs0 = creal(vdot(r, r)), rsold = rs0;
    int it = 0, maxit = 500;
    for (; it < maxit; ++it) {
        apply_D(tmp, p);                /* Ap = D^dag (D p) */
        apply_Ddag(Ap, tmp, sc);
        double alpha = rsold / creal(vdot(p, Ap));
        for (long i = 0; i < NF; ++i) { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; }
        double rsnew = creal(vdot(r, r));
        if (sqrt(rsnew / rs0) < 1e-8) { ++it; break; }
        double beta = rsnew / rsold;
        for (long i = 0; i < NF; ++i) p[i] = r[i] + beta * p[i];
        rsold = rsnew;
    }
    double relres = sqrt(creal(vdot(r, r)) / rs0);

    printf("dslash L=%d (sites=%ld)  gamma5-herm_err=%.2e  CG iters=%d relres=%.2e  D-apply=%.3f s\n",
           L, SITES, herm_err, it, relres, (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(U); free(psi); free(phi); free(Dpsi); free(Dphi); free(tmp);
    free(x); free(r); free(p); free(Ap); free(sc);
    return 0;
}
