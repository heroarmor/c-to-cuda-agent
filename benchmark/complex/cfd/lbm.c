/* lbm.c -- Lattice Boltzmann method (D2Q9, BGK collision) on a periodic domain,
 *          evolving a decaying Taylor-Green vortex.
 *
 * Instead of solving Navier-Stokes directly, LBM evolves 9 particle-distribution
 * functions per cell through two steps: a local BGK *collision* relaxing toward
 * equilibrium, and a *streaming* shift to neighbors. Macroscopic density and
 * velocity are moments of the distributions.
 *
 * Pattern: computational fluid dynamics via kinetic theory -- a memory-bound
 * stencil that is famously GPU-scalable (the streaming step is a pure shift, the
 * collision is fully local). GPU conversion: one thread per cell, two buffers
 * (collide-then-stream), neighbor wrap for the periodic shift.
 *
 * Verification: on a periodic domain both streaming (a permutation) and BGK
 * collision (sum_i f_i = sum_i feq_i = rho) conserve mass *exactly*, so total
 * mass must stay constant to machine precision. Kinetic energy should decay
 * (viscous dissipation) -- the physical signature of the Taylor-Green vortex.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define PI 3.14159265358979323846

static const int    ex[9] = {0, 1, 0,-1, 0, 1,-1,-1, 1};
static const int    ey[9] = {0, 0, 1, 0,-1, 1, 1,-1,-1};
static const double wt[9] = {4.0/9, 1.0/9,1.0/9,1.0/9,1.0/9, 1.0/36,1.0/36,1.0/36,1.0/36};

int main(int argc, char **argv) {
    int nx    = (argc > 1) ? atoi(argv[1]) : 256;
    int ny    = (argc > 2) ? atoi(argv[2]) : 256;
    int steps = (argc > 3) ? atoi(argv[3]) : 2000;
    double tau = 0.8;                 /* relaxation time; viscosity nu=(tau-0.5)/3 */
    double U0  = 0.05;                /* vortex amplitude (keep << 1 for low Mach) */
    size_t N = (size_t)nx * ny;

    double *f  = malloc(N * 9 * sizeof *f);
    double *ft = malloc(N * 9 * sizeof *ft);

    /* initialize equilibrium from a Taylor-Green velocity field, rho = 1 */
    double kx = 2 * PI / nx, ky = 2 * PI / ny;
    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
            double ux = -U0 * cos(kx * x) * sin(ky * y);
            double uy =  U0 * sin(kx * x) * cos(ky * y);
            double u2 = ux * ux + uy * uy;
            double *fc = &f[((size_t)y * nx + x) * 9];
            for (int i = 0; i < 9; ++i) {
                double eu = ex[i] * ux + ey[i] * uy;
                fc[i] = wt[i] * (1.0 + 3 * eu + 4.5 * eu * eu - 1.5 * u2);
            }
        }

    double mass0 = 0.0, ke0 = 0.0;
    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
            double *fc = &f[((size_t)y * nx + x) * 9];
            double rho = 0, mx = 0, my = 0;
            for (int i = 0; i < 9; ++i) { rho += fc[i]; mx += ex[i] * fc[i]; my += ey[i] * fc[i]; }
            mass0 += rho;
            ke0 += 0.5 * (mx * mx + my * my) / rho;
        }

    clock_t t0 = clock();
    for (int s = 0; s < steps; ++s) {
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                double *fc = &f[((size_t)y * nx + x) * 9];
                double rho = 0, ux = 0, uy = 0;
                for (int i = 0; i < 9; ++i) { rho += fc[i]; ux += ex[i] * fc[i]; uy += ey[i] * fc[i]; }
                ux /= rho; uy /= rho;
                double u2 = ux * ux + uy * uy;
                for (int i = 0; i < 9; ++i) {
                    double eu = ex[i] * ux + ey[i] * uy;
                    double feq = wt[i] * rho * (1.0 + 3 * eu + 4.5 * eu * eu - 1.5 * u2);
                    double post = fc[i] - (fc[i] - feq) / tau;     /* BGK collision */
                    int xn = (x + ex[i] + nx) % nx;                /* periodic stream */
                    int yn = (y + ey[i] + ny) % ny;
                    ft[((size_t)yn * nx + xn) * 9 + i] = post;
                }
            }
        double *tmp = f; f = ft; ft = tmp;
    }
    clock_t t1 = clock();

    double mass = 0.0, ke = 0.0;
    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
            double *fc = &f[((size_t)y * nx + x) * 9];
            double rho = 0, mx = 0, my = 0;
            for (int i = 0; i < 9; ++i) { rho += fc[i]; mx += ex[i] * fc[i]; my += ey[i] * fc[i]; }
            mass += rho;
            ke += 0.5 * (mx * mx + my * my) / rho;
        }

    printf("lbm %dx%d steps=%d  mass_drift=%.2e (conserved)  KE: %.4e -> %.4e (decays %.1f%%)  time=%.3f s\n",
           nx, ny, steps, fabs(mass - mass0) / mass0, ke0, ke,
           100.0 * (1.0 - ke / ke0), (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(f); free(ft);
    return 0;
}
