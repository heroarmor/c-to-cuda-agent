/* tensor_contraction.c -- rank-3 x rank-3 tensor contraction:
 *
 *     C[i,j,k,l] = sum_m A[i,j,m] * B[m,k,l]
 *
 * Pattern: tensor algebra -- the generalization of GEMM to multi-dimensional
 * arrays, central to quantum chemistry, tensor networks and deep learning.
 * Tier 3: although it can be reshaped into one big GEMM ((I*J) x M times
 * M x (K*L)), the high-performance route fuses the index permutation/transpose
 * with the matmul ("contraction without transposition") instead of materializing
 * transposed copies -- a nontrivial layout/mapping decision for an auto-converter.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    int d = (argc > 1) ? atoi(argv[1]) : 32;    /* all extents = d */
    int I = d, J = d, K = d, L = d, M = d;

    size_t szA = (size_t)I * J * M;
    size_t szB = (size_t)M * K * L;
    size_t szC = (size_t)I * J * K * L;
    double *A = malloc(szA * sizeof *A);
    double *B = malloc(szB * sizeof *B);
    double *C = calloc(szC, sizeof *C);
    if (!A || !B || !C) { fprintf(stderr, "alloc failed\n"); return 1; }

    for (size_t i = 0; i < szA; ++i) A[i] = 1.0 / (double)((i % 97) + 1);
    for (size_t i = 0; i < szB; ++i) B[i] = (double)(i % 13) - 6.0;

    clock_t t0 = clock();
    /* loop order chosen so the inner (k,l) sweep is unit-stride in B and C */
    for (int i = 0; i < I; ++i)
        for (int j = 0; j < J; ++j)
            for (int m = 0; m < M; ++m) {
                double a = A[((size_t)i * J + j) * M + m];
                const double *Bm = &B[(size_t)m * K * L];
                double *Cij = &C[(((size_t)i * J + j) * K) * L];
                for (int kl = 0; kl < K * L; ++kl)
                    Cij[kl] += a * Bm[kl];
            }
    clock_t t1 = clock();

    double sum = 0.0;
    for (size_t i = 0; i < szC; ++i) sum += C[i];

    double secs = (double)(t1 - t0) / CLOCKS_PER_SEC;
    double gflop = 2.0 * (double)szC * M / 1e9;
    printf("tensor_contraction d=%d  C[%d^4]  checksum=%.4f  time=%.3f s  (%.2f GFLOP/s)\n",
           d, d, sum, secs, secs > 0 ? gflop / secs : 0.0);

    free(A); free(B); free(C);
    return 0;
}
