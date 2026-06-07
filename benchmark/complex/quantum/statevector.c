/* statevector.c -- quantum circuit simulation by dense state-vector evolution.
 *
 * Represents the full n-qubit state as a 2^n vector of complex amplitudes and
 * applies a circuit gate by gate. Each single-qubit gate updates amplitude pairs
 * that differ in one bit; CNOT swaps amplitudes conditioned on a control bit.
 * The circuit here is several layers of Ry rotations + a CNOT entangling ladder.
 *
 * Pattern: quantum computing / state-vector simulation. The state doubles in
 * size per qubit (2^n * 16 bytes), so it is memory-bound and bandwidth-hungry;
 * the SIMT model maps naturally (one thread per amplitude pair). Complex tier:
 * strided gather/scatter access whose stride depends on the target qubit, and a
 * long sequence of kernels sharing one huge resident state vector.
 * GPU conversion: one thread per amplitude pair, gate matrix in registers,
 * state vector resident across all gate kernels.
 *
 * Verification: every gate is unitary, so the 2-norm of the state is invariant,
 * ||psi||^2 = 1, and must stay 1 to roundoff regardless of the circuit.
 */
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <time.h>

typedef double complex cx;

/* apply a 2x2 gate [[u00,u01],[u10,u11]] to qubit q of the state vector */
static void apply1(cx *psi, long dim, int q, cx u00, cx u01, cx u10, cx u11) {
    long bit = 1L << q;
    for (long k = 0; k < dim; ++k)
        if (!(k & bit)) {
            cx a = psi[k], b = psi[k | bit];
            psi[k]       = u00 * a + u01 * b;
            psi[k | bit] = u10 * a + u11 * b;
        }
}

/* controlled-NOT: flip target where control bit is set */
static void cnot(cx *psi, long dim, int ctrl, int tgt) {
    long cb = 1L << ctrl, tb = 1L << tgt;
    for (long k = 0; k < dim; ++k)
        if ((k & cb) && !(k & tb)) {
            cx t = psi[k]; psi[k] = psi[k | tb]; psi[k | tb] = t;
        }
}

int main(int argc, char **argv) {
    int n      = (argc > 1) ? atoi(argv[1]) : 20;   /* qubits */
    int layers = (argc > 2) ? atoi(argv[2]) : 6;
    long dim = 1L << n;

    cx *psi = malloc((size_t)dim * sizeof *psi);
    for (long k = 0; k < dim; ++k) psi[k] = 0.0;
    psi[0] = 1.0;                                    /* start in |00...0> */

    clock_t t0 = clock();
    for (int L = 0; L < layers; ++L) {
        for (int q = 0; q < n; ++q) {                /* layer of Ry(theta) rotations */
            double th = 0.1 * (q + 1) + 0.3 * L;
            double c = cos(th / 2), s = sin(th / 2);
            apply1(psi, dim, q, c, -s, s, c);
        }
        for (int q = 0; q + 1 < n; ++q)              /* entangling CNOT ladder */
            cnot(psi, dim, q, q + 1);
    }
    clock_t t1 = clock();

    double norm2 = 0.0, p0 = 0.0;
    cx checksum = 0.0;
    for (long k = 0; k < dim; ++k) {
        double pr = creal(psi[k]) * creal(psi[k]) + cimag(psi[k]) * cimag(psi[k]);
        norm2 += pr;
        checksum += psi[k] * (double)((k % 7) + 1);
    }
    p0 = creal(psi[0]) * creal(psi[0]) + cimag(psi[0]) * cimag(psi[0]);

    printf("statevector n=%d layers=%d (dim=%ld)  ||psi||^2=%.12f (unit)  P(|0>)=%.6f  re(chk)=%.6f  time=%.3f s\n",
           n, layers, dim, norm2, p0, creal(checksum), (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(psi);
    return 0;
}
