/* lorenz_ensemble.c -- RK4 integration of an ensemble of Lorenz systems
 *
 * Each ensemble member integrates the chaotic Lorenz ODE
 *     x' = sigma (y - x)
 *     y' = x (rho - z) - y
 *     z' = x y - beta z
 * from a slightly different initial condition.
 *
 * Pattern: ODE integration. A single trajectory is sequential (a time
 * recurrence), but an *ensemble* is embarrassingly parallel -- the classic way
 * to saturate a GPU with ODE work (ensemble forecasting, parameter sweeps,
 * uncertainty quantification).
 * GPU conversion: one thread per ensemble member; the time loop runs inside the
 * thread with the 3-component state held in registers.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static const double SIGMA = 10.0, RHO = 28.0, BETA = 8.0 / 3.0;

static void deriv(const double s[3], double d[3]) {
    d[0] = SIGMA * (s[1] - s[0]);
    d[1] = s[0] * (RHO - s[2]) - s[1];
    d[2] = s[0] * s[1] - BETA * s[2];
}

static void rk4_step(double s[3], double h) {
    double k1[3], k2[3], k3[3], k4[3], tmp[3];
    deriv(s, k1);
    for (int i = 0; i < 3; ++i) tmp[i] = s[i] + 0.5 * h * k1[i];
    deriv(tmp, k2);
    for (int i = 0; i < 3; ++i) tmp[i] = s[i] + 0.5 * h * k2[i];
    deriv(tmp, k3);
    for (int i = 0; i < 3; ++i) tmp[i] = s[i] + h * k3[i];
    deriv(tmp, k4);
    for (int i = 0; i < 3; ++i)
        s[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
}

int main(int argc, char **argv) {
    int ens   = (argc > 1) ? atoi(argv[1]) : 20000;   /* ensemble members */
    int steps = (argc > 2) ? atoi(argv[2]) : 2000;
    double h  = 0.005;

    double *state = malloc((size_t)ens * 3 * sizeof *state);
    for (int e = 0; e < ens; ++e) {
        state[3 * e + 0] = 1.0 + 1e-6 * e;   /* nearby initial conditions */
        state[3 * e + 1] = 1.0;
        state[3 * e + 2] = 1.0;
    }

    clock_t t0 = clock();
    for (int e = 0; e < ens; ++e) {
        double s[3] = { state[3 * e], state[3 * e + 1], state[3 * e + 2] };
        for (int t = 0; t < steps; ++t) rk4_step(s, h);
        state[3 * e + 0] = s[0];
        state[3 * e + 1] = s[1];
        state[3 * e + 2] = s[2];
    }
    clock_t t1 = clock();

    double sum = 0.0;
    for (int e = 0; e < ens; ++e)
        sum += state[3 * e] + state[3 * e + 1] + state[3 * e + 2];

    /* invariant: a fixed point must stay stationary, which validates deriv() and
       the RK4 step (all four k's vanish there). The nontrivial fixed point C+ is
       unstable for rho=28, so we check over a short horizon before the chaotic
       dynamics amplify roundoff. */
    double fp = sqrt(BETA * (RHO - 1.0));
    double fs[3]  = { fp, fp, RHO - 1.0 };
    double fs0[3] = { fp, fp, RHO - 1.0 };
    for (int t = 0; t < 100; ++t) rk4_step(fs, h);
    double fp_drift = 0.0;
    for (int i = 0; i < 3; ++i) {
        double d = fabs(fs[i] - fs0[i]);
        if (d > fp_drift) fp_drift = d;
    }

    printf("lorenz ensemble=%d steps=%d  checksum=%.6f  fixed_pt_drift=%.2e (100 steps)  time=%.3f s\n",
           ens, steps, sum, fp_drift, (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(state);
    return 0;
}
