/* wave2d.c -- 2D wave equation by explicit finite differences
 *
 *     d2u/dt2 = c^2 (d2u/dx2 + d2u/dy2),   fixed (Dirichlet) boundaries
 *
 * Leapfrog time stepping using three time levels (prev / cur / next).
 *
 * Pattern: structured-grid / stencil PDE with a *second-order* time recurrence:
 * each step needs the two previous levels. GPU conversion: 2D thread grid per
 * step; the time loop stays on the host, the three buffers stay resident on the
 * device and are rotated by pointer swap (no copies).
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static void step(int m, double C2, const double *prev, const double *cur, double *next) {
    for (int y = 1; y < m - 1; ++y)
        for (int x = 1; x < m - 1; ++x) {
            int i = y * m + x;
            double lap = cur[i - 1] + cur[i + 1] + cur[i - m] + cur[i + m] - 4.0 * cur[i];
            next[i] = 2.0 * cur[i] - prev[i] + C2 * lap;
        }
}

/* Discrete leapfrog energy  E = 1/2||cur-prev||^2 + (C2/2)<G cur, prev>, where
   G is the (4u - neighbors) graph Laplacian. The scheme conserves this exactly
   in exact arithmetic, so its drift over a run measures only roundoff. */
static double wave_energy(int m, double C2, const double *cur, const double *prev) {
    double kin = 0.0, pot = 0.0;
    for (int i = 0; i < m * m; ++i) { double d = cur[i] - prev[i]; kin += d * d; }
    for (int y = 1; y < m - 1; ++y)
        for (int x = 1; x < m - 1; ++x) {
            int i = y * m + x;
            double Gcur = 4.0 * cur[i] - cur[i - 1] - cur[i + 1] - cur[i - m] - cur[i + m];
            pot += Gcur * prev[i];
        }
    return 0.5 * kin + 0.5 * C2 * pot;
}

int main(int argc, char **argv) {
    int m = (argc > 1) ? atoi(argv[1]) : 512;
    int steps = (argc > 2) ? atoi(argv[2]) : 300;
    double C2 = 0.25;                              /* (c*dt/dx)^2, <0.5 for 2D stability */
    size_t N = (size_t)m * m;

    double *prev = calloc(N, sizeof *prev);
    double *cur  = calloc(N, sizeof *cur);
    double *next = calloc(N, sizeof *next);

    /* initial Gaussian pulse, zero velocity (prev = cur) */
    double cx = m / 2.0, cy = m / 2.0, sig = m / 16.0;
    for (int y = 0; y < m; ++y)
        for (int x = 0; x < m; ++x) {
            double dx = x - cx, dy = y - cy;
            double val = exp(-(dx * dx + dy * dy) / (2.0 * sig * sig));
            cur[(size_t)y * m + x] = val;
            prev[(size_t)y * m + x] = val;
        }

    double E0 = wave_energy(m, C2, cur, prev);    /* conserved discrete energy at t=0 */

    clock_t t0 = clock();
    for (int s = 0; s < steps; ++s) {
        step(m, C2, prev, cur, next);
        double *t = prev; prev = cur; cur = next; next = t;
    }
    clock_t t1 = clock();

    double Ef = wave_energy(m, C2, cur, prev);
    double drift = fabs(E0) > 0.0 ? fabs(Ef - E0) / fabs(E0) : fabs(Ef - E0);

    double energy = 0.0;
    for (size_t i = 0; i < N; ++i) energy += cur[i] * cur[i];

    printf("wave2d m=%d steps=%d  energy=%.4f  E_drift=%.2e (conserved)  time=%.3f s\n",
           m, steps, energy, drift, (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(prev); free(cur); free(next);
    return 0;
}
