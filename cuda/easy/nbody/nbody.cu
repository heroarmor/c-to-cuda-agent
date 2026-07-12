/* nbody.cu -- CUDA conversion of benchmark/easy/nbody/nbody.c
 * Direct (all-pairs) gravitational N-body with one thread per body
 * for force accumulation, velocity-Verlet integration.
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

__global__ void nbody_force_kernel(int n, double eps2,
                                   const double *x, const double *y, const double *z,
                                   const double *m,
                                   double *ax, double *ay, double *az) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

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

__global__ void nbody_update_kernel(int n, double dt,
                                    const double *ax, const double *ay, const double *az,
                                    double *vx, double *vy, double *vz,
                                    double *x, double *y, double *z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    vx[i] += dt * ax[i];
    vy[i] += dt * ay[i];
    vz[i] += dt * az[i];
    x[i]  += dt * vx[i];
    y[i]  += dt * vy[i];
    z[i]  += dt * vz[i];
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 4096;
    int steps = (argc > 2) ? atoi(argv[2]) : 10;
    double dt = 0.001, eps2 = 1e-6;

    size_t sz = (size_t)n * sizeof(double);

    double *h_x  = (double *)malloc(sz);
    double *h_y  = (double *)malloc(sz);
    double *h_z  = (double *)malloc(sz);
    double *h_vx = (double *)calloc((size_t)n, sizeof(double));
    double *h_vy = (double *)calloc((size_t)n, sizeof(double));
    double *h_vz = (double *)calloc((size_t)n, sizeof(double));
    double *h_m  = (double *)malloc(sz);

    if (!h_x || !h_y || !h_z || !h_vx || !h_vy || !h_vz || !h_m) {
        fprintf(stderr, "host alloc failed\n"); return 1;
    }

    unsigned long s = 1234567ul;
    for (int i = 0; i < n; ++i) {
        s = s * 1103515245ul + 12345ul; h_x[i] = ((s >> 16) & 0x7fff) / 32768.0 - 0.5;
        s = s * 1103515245ul + 12345ul; h_y[i] = ((s >> 16) & 0x7fff) / 32768.0 - 0.5;
        s = s * 1103515245ul + 12345ul; h_z[i] = ((s >> 16) & 0x7fff) / 32768.0 - 0.5;
        h_m[i] = 1.0 / n;
    }

    double *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz;
    double *d_ax, *d_ay, *d_az, *d_m;
    CUDA_CHECK(cudaMalloc(&d_x,  sz));
    CUDA_CHECK(cudaMalloc(&d_y,  sz));
    CUDA_CHECK(cudaMalloc(&d_z,  sz));
    CUDA_CHECK(cudaMalloc(&d_vx, sz));
    CUDA_CHECK(cudaMalloc(&d_vy, sz));
    CUDA_CHECK(cudaMalloc(&d_vz, sz));
    CUDA_CHECK(cudaMalloc(&d_ax, sz));
    CUDA_CHECK(cudaMalloc(&d_ay, sz));
    CUDA_CHECK(cudaMalloc(&d_az, sz));
    CUDA_CHECK(cudaMalloc(&d_m,  sz));

    CUDA_CHECK(cudaMemcpy(d_x,  h_x,  sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y,  h_y,  sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z,  h_z,  sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, h_vx, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, h_vy, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vz, h_vz, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m,  h_m,  sz, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    clock_t t0 = clock();
    for (int step = 0; step < steps; ++step) {
        nbody_force_kernel<<<blocks, threads>>>(n, eps2, d_x, d_y, d_z, d_m, d_ax, d_ay, d_az);
        CUDA_CHECK(cudaDeviceSynchronize());
        nbody_update_kernel<<<blocks, threads>>>(n, dt, d_ax, d_ay, d_az, d_vx, d_vy, d_vz, d_x, d_y, d_z);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    clock_t t1 = clock();

    CUDA_CHECK(cudaMemcpy(h_x,  d_x,  sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y,  d_y,  sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_z,  d_z,  sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vx, d_vx, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vy, d_vy, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vz, d_vz, sz, cudaMemcpyDeviceToHost));

    double cx = 0.0;
    double px = 0.0, py = 0.0, pz = 0.0;
    for (int i = 0; i < n; ++i) {
        cx += h_x[i];
        px += h_m[i] * h_vx[i];
        py += h_m[i] * h_vy[i];
        pz += h_m[i] * h_vz[i];
    }
    double pmag = sqrt(px * px + py * py + pz * pz);
    double secs = (double)(t1 - t0) / CLOCKS_PER_SEC;
    printf("nbody n=%d steps=%d  sum_x=%.6f  |momentum|=%.2e (conserved ~0)  time=%.3f s  (%.2f Gpair/s)\n",
           n, steps, cx, pmag, secs,
           secs > 0 ? ((double)n * n * steps) / secs / 1e9 : 0.0);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_vx));
    CUDA_CHECK(cudaFree(d_vy));
    CUDA_CHECK(cudaFree(d_vz));
    CUDA_CHECK(cudaFree(d_ax));
    CUDA_CHECK(cudaFree(d_ay));
    CUDA_CHECK(cudaFree(d_az));
    CUDA_CHECK(cudaFree(d_m));
    free(h_x); free(h_y); free(h_z);
    free(h_vx); free(h_vy); free(h_vz);
    free(h_m);
    return 0;
}
