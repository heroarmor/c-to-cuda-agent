/* nbody.c -- direct (all-pairs) gravitational N-body simulation
 *
 * O(N^2) force evaluation with softening, velocity-Verlet style integration.
 *
 * Pattern: N-body (Berkeley dwarf). The all-pairs force loop is a classic
 * compute-bound GPU kernel; softening removes the singularity so there is no
 * divergence in the inner loop.
 * GPU conversion: one thread per body accumulates the force from all others,
 * staging blocks of bodies through shared memory (the textbook tiled N-body).
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 4096;
    int steps = (argc > 2) ? atoi(argv[2]) : 10;
    double dt = 0.001, eps2 = 1e-6;

    double *x  = malloc((size_t)n * sizeof *x);
    double *y  = malloc((size_t)n * sizeof *y);
    double *z  = malloc((size_t)n * sizeof *z);
    double *vx = calloc((size_t)n, sizeof *vx);
    double *vy = calloc((size_t)n, sizeof *vy);
    double *vz = calloc((size_t)n, sizeof *vz);
    double *ax = malloc((size_t)n * sizeof *ax);
    double *ay = malloc((size_t)n * sizeof *ay);
    double *az = malloc((size_t)n * sizeof *az);
    double *m  = malloc((size_t)n * sizeof *m);

    /* deterministic pseudo-random initial positions in [-0.5, 0.5]^3 */
    unsigned long s = 1234567ul;
    for (int i = 0; i < n; ++i) {
        s = s * 1103515245ul + 12345ul; x[i] = ((s >> 16) & 0x7fff) / 32768.0 - 0.5;
        s = s * 1103515245ul + 12345ul; y[i] = ((s >> 16) & 0x7fff) / 32768.0 - 0.5;
        s = s * 1103515245ul + 12345ul; z[i] = ((s >> 16) & 0x7fff) / 32768.0 - 0.5;
        m[i] = 1.0 / n;
    }

    clock_t t0 = clock();
    for (int step = 0; step < steps; ++step) {
        for (int i = 0; i < n; ++i) {
            double fx = 0.0, fy = 0.0, fz = 0.0;
            for (int j = 0; j < n; ++j) {
                double dx = x[j] - x[i], dy = y[j] - y[i], dz = z[j] - z[i];
                double r2 = dx * dx + dy * dy + dz * dz + eps2;
                double inv = 1.0 / sqrt(r2);
                double f = m[j] * inv * inv * inv;
                fx += f * dx; fy += f * dy; fz += f * dz;
            }
            ax[i] = fx; ay[i] = fy; az[i] = fz;
        }
        for (int i = 0; i < n; ++i) {
            vx[i] += dt * ax[i]; vy[i] += dt * ay[i]; vz[i] += dt * az[i];
            x[i]  += dt * vx[i]; y[i]  += dt * vy[i]; z[i]  += dt * vz[i];
        }
    }
    clock_t t1 = clock();

    double cx = 0.0;
    double px = 0.0, py = 0.0, pz = 0.0;
    for (int i = 0; i < n; ++i) {
        cx += x[i];
        px += m[i] * vx[i]; py += m[i] * vy[i]; pz += m[i] * vz[i];
    }
    /* invariant: total momentum is conserved by the symmetric pairwise forces.
       Velocities start at zero, so |momentum| must stay ~0 (roundoff only). */
    double pmag = sqrt(px * px + py * py + pz * pz);
    double secs = (double)(t1 - t0) / CLOCKS_PER_SEC;
    printf("nbody n=%d steps=%d  sum_x=%.6f  |momentum|=%.2e (conserved ~0)  time=%.3f s  (%.2f Gpair/s)\n",
           n, steps, cx, pmag, secs,
           secs > 0 ? ((double)n * n * steps) / secs / 1e9 : 0.0);

    free(x); free(y); free(z);
    free(vx); free(vy); free(vz);
    free(ax); free(ay); free(az); free(m);
    return 0;
}
