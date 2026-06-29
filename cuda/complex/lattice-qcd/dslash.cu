/* dslash.cu -- CUDA conversion of benchmark/complex/lattice-qcd/dslash.c
 * Wilson-Dirac operator ("Dslash") on a 4D lattice with an SU(3) gauge field,
 * plus a conjugate-gradient solve of the normal system.
 *
 * One thread per lattice site; gauge links in global memory;
 * CG loop orchestrated on host with GPU-accelerated dot products.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                 \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

typedef cuDoubleComplex cx;

static inline __host__ __device__ cx cx_cv(double r, double i) { return make_cuDoubleComplex(r, i); }
static inline __host__ __device__ cx cx_r(double r) { return make_cuDoubleComplex(r, 0); }

/* gamma matrices (chiral basis) in constant memory */
__constant__ cx d_gamma_[4][4][4];
__constant__ cx d_gamma5[4][4];

/* host copies of gamma matrices (for host-side reduction) */
static cx h_gamma5[4][4];

static int L;
static long SITES;
static long NF;

__device__ long site_index_d(int x, int y, int z, int t, int Ld) {
    return ((((long)t) * Ld + z) * Ld + y) * Ld + x;
}

__device__ long neighbor_d(long s, int mu, int dir, int Ld) {
    int c[4];
    c[0] = (int)(s % Ld); c[1] = (int)((s / Ld) % Ld);
    c[2] = (int)((s / ((long)Ld * Ld)) % Ld); c[3] = (int)((s / ((long)Ld * Ld * Ld)) % Ld);
    c[mu] = (c[mu] + dir + Ld) % Ld;
    return site_index_d(c[0], c[1], c[2], c[3], Ld);
}

__device__ void su3_mul_d(const cx *Um, const cx *in, cx *out, int dagger) {
    for (int s = 0; s < 4; ++s)
        for (int a = 0; a < 3; ++a) {
            cx acc = cx_r(0);
            for (int b = 0; b < 3; ++b)
                acc = cuCadd(acc, cuCmul(dagger ? cuConj(Um[b * 3 + a]) : Um[a * 3 + b],
                                          in[s * 3 + b]));
            out[s * 3 + a] = acc;
        }
}

__device__ void spin_proj_d(int mu, int sign, const cx *in, cx *out) {
    for (int s = 0; s < 4; ++s)
        for (int c = 0; c < 3; ++c) {
            cx acc = in[s * 3 + c];
            for (int sp = 0; sp < 4; ++sp)
                acc = cuCsub(acc, cuCmul(cuCmul(cx_r(sign), d_gamma_[mu][s][sp]), in[sp * 3 + c]));
            out[s * 3 + c] = acc;
        }
}

__global__ void apply_D_kernel(long SITES_d, int Ld, double kappa,
                                const cx *U, const cx *in, cx *out) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= SITES_d) return;

    cx t1[12], t2[12];
    const cx *c = &in[site * 12];
    cx *o = &out[site * 12];
    for (int k = 0; k < 12; ++k) o[k] = c[k];
    cx kappa_cx = cx_r(kappa);

    for (int mu = 0; mu < 4; ++mu) {
        long fwd = neighbor_d(site, mu, 1, Ld);
        long bwd = neighbor_d(site, mu, -1, Ld);
        const cx *Uf = &U[(site * 4 + mu) * 9];
        const cx *Ub = &U[(bwd * 4 + mu) * 9];

        su3_mul_d(Uf, &in[fwd * 12], t1, 0);
        spin_proj_d(mu, 1, t1, t2);
        for (int k = 0; k < 12; ++k)
            o[k] = cuCsub(o[k], cuCmul(kappa_cx, t2[k]));

        su3_mul_d(Ub, &in[bwd * 12], t1, 1);
        spin_proj_d(mu, -1, t1, t2);
        for (int k = 0; k < 12; ++k)
            o[k] = cuCsub(o[k], cuCmul(kappa_cx, t2[k]));
    }
}

__global__ void apply_g5_kernel(long SITES_d, const cx *in, cx *out) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= SITES_d) return;
    for (int s = 0; s < 4; ++s)
        for (int col = 0; col < 3; ++col)
            out[site * 12 + s * 3 + col] =
                cuCmul(d_gamma5[s][s], in[site * 12 + s * 3 + col]);
}

__global__ void axpy_kernel(long n, cx alpha, const cx *x, cx *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = cuCadd(y[i], cuCmul(alpha, x[i]));
}

__global__ void axpby_kernel(long n, cx alpha, const cx *x, cx beta, cx *p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = cuCadd(cuCmul(alpha, x[i]), cuCmul(beta, p[i]));
}

__global__ void vdot_kernel(long n, const cx *a, const cx *b, cx *partial) {
    extern __shared__ cx sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    cx sum = cx_r(0);
    while (i < n) {
        sum = cuCadd(sum, cuCmul(cuConj(a[i]), b[i]));
        i += blockDim.x * gridDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = cuCadd(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

static cx vdot_gpu(const cx *d_a, const cx *d_b, long n,
                   cx *d_partial, int blocks, int threads) {
    vdot_kernel<<<blocks, threads, threads * sizeof(cx)>>>(n, d_a, d_b, d_partial);
    cx *h_partial = (cx *)malloc((size_t)blocks * sizeof(cx));
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, (size_t)blocks * sizeof(cx),
                          cudaMemcpyDeviceToHost));
    cx result = cx_r(0);
    for (int i = 0; i < blocks; ++i) result = cuCadd(result, h_partial[i]);
    free(h_partial);
    return result;
}

/* ---- host helpers ---- */
static double urand(unsigned long long *s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(*s >> 11) / 9007199254740992.0 - 0.5;
}

static void random_su3(cx *m, unsigned long long *rs) {
    for (int i = 0; i < 9; ++i)
        m[i] = cx_cv(urand(rs), urand(rs));
    for (int col = 0; col < 3; ++col) {
        for (int prev = 0; prev < col; ++prev) {
            cx proj = cx_r(0);
            for (int r = 0; r < 3; ++r)
                proj = cuCadd(proj, cuCmul(cuConj(m[r * 3 + prev]), m[r * 3 + col]));
            for (int r = 0; r < 3; ++r)
                m[r * 3 + col] = cuCsub(m[r * 3 + col], cuCmul(proj, m[r * 3 + prev]));
        }
        double nrm = 0;
        for (int r = 0; r < 3; ++r)
            nrm += cuCreal(m[r * 3 + col]) * cuCreal(m[r * 3 + col]) +
                   cuCimag(m[r * 3 + col]) * cuCimag(m[r * 3 + col]);
        nrm = sqrt(nrm);
        for (int r = 0; r < 3; ++r)
            m[r * 3 + col] = cuCdiv(m[r * 3 + col], cx_r(nrm));
    }
    cx t1 = cuCmul(m[0], cuCsub(cuCmul(m[4], m[8]), cuCmul(m[5], m[7])));
    cx t2 = cuCmul(m[1], cuCsub(cuCmul(m[3], m[8]), cuCmul(m[5], m[6])));
    cx t3 = cuCmul(m[2], cuCsub(cuCmul(m[3], m[7]), cuCmul(m[4], m[6])));
    cx det = cuCadd(cuCsub(t1, t2), t3);
    cx phase = cuConj(det);
    for (int r = 0; r < 3; ++r)
        m[r * 3 + 0] = cuCmul(m[r * 3 + 0], phase);
}

static void init_gammas(void) {
    cx h_gamma[4][4][4];
    for (int m = 0; m < 4; ++m)
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) h_gamma[m][i][j] = cx_r(0);
    cx i_ = cx_cv(0, 1);
    cx ni_ = cx_cv(0, -1);
    h_gamma[0][0][3] = ni_; h_gamma[0][1][2] = ni_; h_gamma[0][2][1] = i_; h_gamma[0][3][0] = i_;
    h_gamma[1][0][3] = cx_r(-1); h_gamma[1][1][2] = cx_r(1);
    h_gamma[1][2][1] = cx_r(1); h_gamma[1][3][0] = cx_r(-1);
    h_gamma[2][0][2] = ni_; h_gamma[2][1][3] = i_; h_gamma[2][2][0] = i_; h_gamma[2][3][1] = ni_;
    h_gamma[3][0][2] = cx_r(1); h_gamma[3][1][3] = cx_r(1);
    h_gamma[3][2][0] = cx_r(1); h_gamma[3][3][1] = cx_r(1);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) h_gamma5[i][j] = cx_r(0);
    h_gamma5[0][0] = cx_r(1); h_gamma5[1][1] = cx_r(1);
    h_gamma5[2][2] = cx_r(-1); h_gamma5[3][3] = cx_r(-1);

    CUDA_CHECK(cudaMemcpyToSymbol(d_gamma_, h_gamma, sizeof(h_gamma)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_gamma5, h_gamma5, sizeof(h_gamma5)));
}

int main(int argc, char **argv) {
    L = (argc > 1) ? atoi(argv[1]) : 6;
    SITES = (long)L * L * L * L;
    NF = SITES * 12;

    init_gammas();

    size_t U_bytes = (size_t)SITES * 4 * 9 * sizeof(cx);
    cx *h_U = (cx *)malloc(U_bytes);
    unsigned long long rs = 0xC0FFEEULL;
    for (long s = 0; s < SITES; ++s)
        for (int mu = 0; mu < 4; ++mu) random_su3(&h_U[(s * 4 + mu) * 9], &rs);

    size_t spinor_bytes = (size_t)NF * sizeof(cx);

    cx *h_psi = (cx *)malloc(spinor_bytes);
    cx *h_phi = (cx *)malloc(spinor_bytes);
    cx *h_Dpsi = (cx *)malloc(spinor_bytes);
    cx *h_Dphi = (cx *)malloc(spinor_bytes);
    cx *h_tmp  = (cx *)malloc(spinor_bytes);
    for (long i = 0; i < NF; ++i) {
        h_psi[i] = cx_cv(urand(&rs), urand(&rs));
        h_phi[i] = cx_cv(urand(&rs), urand(&rs));
    }

    cx *d_U, *d_psi, *d_phi, *d_Dpsi, *d_Dphi;
    cx *d_tmp, *d_sc, *d_x, *d_r, *d_p, *d_Ap, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_U, U_bytes));
    CUDA_CHECK(cudaMalloc(&d_psi, spinor_bytes));
    CUDA_CHECK(cudaMalloc(&d_phi, spinor_bytes));
    CUDA_CHECK(cudaMalloc(&d_Dpsi, spinor_bytes));
    CUDA_CHECK(cudaMalloc(&d_Dphi, spinor_bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp, spinor_bytes));
    CUDA_CHECK(cudaMalloc(&d_sc, spinor_bytes));
    CUDA_CHECK(cudaMalloc(&d_x, spinor_bytes));
    CUDA_CHECK(cudaMalloc(&d_r, spinor_bytes));
    CUDA_CHECK(cudaMalloc(&d_p, spinor_bytes));
    CUDA_CHECK(cudaMalloc(&d_Ap, spinor_bytes));

    int vthreads = 256;
    int vblocks = (NF + vthreads - 1) / vthreads;
    int partial_blocks = 64;
    if (partial_blocks > vblocks) partial_blocks = vblocks;
    CUDA_CHECK(cudaMalloc(&d_partial, (size_t)partial_blocks * sizeof(cx)));

    int sthreads = 256;
    int sblocks = (SITES + sthreads - 1) / sthreads;

    CUDA_CHECK(cudaMemcpy(d_U, h_U, U_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_psi, h_psi, spinor_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_phi, h_phi, spinor_bytes, cudaMemcpyHostToDevice));

    clock_t t0 = clock();
    apply_D_kernel<<<sblocks, sthreads>>>(SITES, L, 0.12, d_U, d_psi, d_Dpsi);
    CUDA_CHECK(cudaDeviceSynchronize());
    apply_D_kernel<<<sblocks, sthreads>>>(SITES, L, 0.12, d_U, d_phi, d_Dphi);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_t t1 = clock();

    CUDA_CHECK(cudaMemcpy(h_Dpsi, d_Dpsi, spinor_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Dphi, d_Dphi, spinor_bytes, cudaMemcpyDeviceToHost));

    for (long site = 0; site < SITES; ++site) {
        for (int s = 0; s < 4; ++s)
            for (int col = 0; col < 3; ++col) {
                h_tmp[site * 12 + s * 3 + col] =
                    cuCmul(h_gamma5[s][s], h_Dpsi[site * 12 + s * 3 + col]);
            }
    }
    cx a = cx_r(0);
    for (long i = 0; i < NF; ++i) a = cuCadd(a, cuCmul(cuConj(h_phi[i]), h_tmp[i]));

    for (long site = 0; site < SITES; ++site) {
        for (int s = 0; s < 4; ++s)
            for (int col = 0; col < 3; ++col) {
                h_tmp[site * 12 + s * 3 + col] =
                    cuCmul(h_gamma5[s][s], h_Dphi[site * 12 + s * 3 + col]);
            }
    }
    cx b = cx_r(0);
    for (long i = 0; i < NF; ++i) b = cuCadd(b, cuCmul(cuConj(h_psi[i]), h_tmp[i]));

    double herm_err = cuCabs(cuCsub(a, cuConj(b)));

    CUDA_CHECK(cudaMemset(d_x, 0, spinor_bytes));
    CUDA_CHECK(cudaMemcpy(d_r, d_psi, spinor_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_p, d_r, spinor_bytes, cudaMemcpyDeviceToDevice));

    cx rsold_cx = vdot_gpu(d_r, d_r, NF, d_partial, partial_blocks, vthreads);
    double rs0 = cuCreal(rsold_cx), rsold = rs0;
    int it = 0, maxit = 500;

    for (; it < maxit; ++it) {
        apply_D_kernel<<<sblocks, sthreads>>>(SITES, L, 0.12, d_U, d_p, d_tmp);
        apply_g5_kernel<<<sblocks, sthreads>>>(SITES, d_tmp, d_sc);
        apply_D_kernel<<<sblocks, sthreads>>>(SITES, L, 0.12, d_U, d_sc, d_Ap);
        apply_g5_kernel<<<sblocks, sthreads>>>(SITES, d_Ap, d_Ap);

        cx denom_cx = vdot_gpu(d_p, d_Ap, NF, d_partial, partial_blocks, vthreads);
        double denom = cuCreal(denom_cx);
        double alpha = rsold / denom;

        axpy_kernel<<<vblocks, vthreads>>>(NF, cx_r(alpha), d_p, d_x);
        axpy_kernel<<<vblocks, vthreads>>>(NF, cx_r(-alpha), d_Ap, d_r);

        cx rsnew_cx = vdot_gpu(d_r, d_r, NF, d_partial, partial_blocks, vthreads);
        double rsnew = cuCreal(rsnew_cx);

        if (sqrt(rsnew / rs0) < 1e-8) { ++it; break; }

        double beta = rsnew / rsold;
        axpby_kernel<<<vblocks, vthreads>>>(NF, cx_r(1), d_r, cx_r(beta), d_p);

        rsold = rsnew;
    }

    cx r2 = vdot_gpu(d_r, d_r, NF, d_partial, partial_blocks, vthreads);
    double relres = sqrt(cuCreal(r2) / rs0);

    printf("dslash L=%d (sites=%ld)  gamma5-herm_err=%.2e  CG iters=%d relres=%.2e  D-apply=%.3f s\n",
           L, SITES, herm_err, it, relres, (double)(t1 - t0) / CLOCKS_PER_SEC);

    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_psi));
    CUDA_CHECK(cudaFree(d_phi));
    CUDA_CHECK(cudaFree(d_Dpsi));
    CUDA_CHECK(cudaFree(d_Dphi));
    CUDA_CHECK(cudaFree(d_tmp));
    CUDA_CHECK(cudaFree(d_sc));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_Ap));
    CUDA_CHECK(cudaFree(d_partial));
    free(h_U); free(h_psi); free(h_phi); free(h_Dpsi); free(h_Dphi); free(h_tmp);
    return 0;
}
