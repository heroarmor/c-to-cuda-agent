/* heat2d.c -- 2D heat equation by explicit finite differences
 *
 *     du/dt = kappa (d2u/dx2 + d2u/dy2),   Dirichlet (u = 0) boundaries
 *
 * Five-point stencil, double-buffered explicit time stepping.
 *
 * Pattern: structured grid / stencil (Berkeley dwarf) -- the bread-and-butter of
 * explicit PDE solvers. Memory-bound.
 * GPU conversion: 2D thread grid, one thread per interior cell; shared-memory
 * tiling to reuse the halo. The time loop stays on the host; the two grids stay
 * resident on the device with a pointer swap each step.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void step(int m, double r, const double *u, double *un) {
    for (int y = 1; y < m - 1; ++y)
        for (int x = 1; x < m - 1; ++x) {
            int i = y * m + x;
            un[i] = u[i] + r * (u[i - 1] + u[i + 1] + u[i - m] + u[i + m] - 4.0 * u[i]);
        }
}

int main(int argc, char **argv) {
    int m = (argc > 1) ? atoi(argv[1]) : 512;     /* grid side */
    int steps = (argc > 2) ? atoi(argv[2]) : 200;
    double r = 0.2;                               /* kappa*dt/dx^2, <0.25 for stability */
    size_t N = (size_t)m * m;

    double *u  = calloc(N, sizeof *u);
    double *un = calloc(N, sizeof *un);

    /* initial condition: hot square in the middle */
    for (int y = m / 4; y < 3 * m / 4; ++y)
        for (int x = m / 4; x < 3 * m / 4; ++x)
            u[(size_t)y * m + x] = 1.0;

    clock_t t0 = clock();
    for (int s = 0; s < steps; ++s) {
        step(m, r, u, un);
        double *tmp = u; u = un; un = tmp;        /* interior-only write keeps boundaries = 0 */
    }
    clock_t t1 = clock();

    double total = 0.0;
    for (size_t i = 0; i < N; ++i) total += u[i];

    printf("heat2d m=%d steps=%d  total_heat=%.4f  time=%.3f s\n",
           m, steps, total, (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(u); free(un);
    return 0;
}
