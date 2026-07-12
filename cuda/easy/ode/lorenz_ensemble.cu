/* lorenz_ensemble.cu -- CUDA conversion of benchmark/easy/ode/lorenz_ensemble.c
 * One thread per ensemble member; the time loop runs inside the
 * thread with the 3-component state held in registers.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                 \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

#define SIGMA 10.0
#define RHO   28.0
#define BETA  (8.0 / 3.0)

__host__ __device__ void deriv(const double s[3], double d[3]) {
    d[0] = SIGMA * (s[1] - s[0]);
    d[1] = s[0] * (RHO - s[2]) - s[1];
    d[2] = s[0] * s[1] - BETA * s[2];
}

__host__ __device__ void rk4_step(double s[3], double h) {
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

__global__ void lorenz_kernel(int ens, int steps, double h, double *state) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= ens) return;
    double s[3] = { state[3 * e], state[3 * e + 1], state[3 * e + 2] };
    for (int t = 0; t < steps; ++t) rk4_step(s, h);
    state[3 * e + 0] = s[0];
    state[3 * e + 1] = s[1];
    state[3 * e + 2] = s[2];
}

int main(int argc, char **argv) {
    int ens   = (argc > 1) ? atoi(argv[1]) : 20000;
    int steps = (argc > 2) ? atoi(argv[2]) : 2000;
    double h  = 0.005;

    size_t bytes = (size_t)ens * 3 * sizeof(double);
    double *h_state = (double *)malloc(bytes);
    if (!h_state) { fprintf(stderr, "alloc failed\n"); return 1; }

    for (int e = 0; e < ens; ++e) {
        h_state[3 * e + 0] = 1.0 + 1e-6 * e;
        h_state[3 * e + 1] = 1.0;
        h_state[3 * e + 2] = 1.0;
    }

    double *d_state;
    CUDA_CHECK(cudaMalloc(&d_state, bytes));
    CUDA_CHECK(cudaMemcpy(d_state, h_state, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (ens + threads - 1) / threads;

    clock_t t0 = clock();
    lorenz_kernel<<<blocks, threads>>>(ens, steps, h, d_state);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_t t1 = clock();

    CUDA_CHECK(cudaMemcpy(h_state, d_state, bytes, cudaMemcpyDeviceToHost));

    double sum = 0.0;
    for (int e = 0; e < ens; ++e)
        sum += h_state[3 * e] + h_state[3 * e + 1] + h_state[3 * e + 2];

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

    CUDA_CHECK(cudaFree(d_state));
    free(h_state);
    return 0;
}
