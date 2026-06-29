/* lbm.cu -- CUDA conversion of benchmark/complex/cfd/lbm.c
 * Lattice Boltzmann D2Q9 (BGK collision, periodic streaming).
 * One thread per cell; collide+stream fused into a single kernel.
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

#define PI 3.14159265358979323846

/* D2Q9 lattice vectors and weights -- host copies for init / reduction */
static const int    ex[9] = {0, 1, 0,-1, 0, 1,-1,-1, 1};
static const int    ey[9] = {0, 0, 1, 0,-1, 1, 1,-1,-1};
static const double wt[9] = {4.0/9, 1.0/9,1.0/9,1.0/9,1.0/9, 1.0/36,1.0/36,1.0/36,1.0/36};

/* Device constant copies */
__constant__ int    d_ex[9];
__constant__ int    d_ey[9];
__constant__ double d_wt[9];

__global__ void collide_stream(int nx, int ny, double tau,
                               const double *f, double *ft) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    size_t cell = (size_t)y * nx + x;
    const double *fc = &f[cell * 9];

    /* macroscopic moments */
    double rho = 0, ux = 0, uy = 0;
#pragma unroll
    for (int i = 0; i < 9; ++i) {
        rho += fc[i];
        ux  += d_ex[i] * fc[i];
        uy  += d_ey[i] * fc[i];
    }
    ux /= rho;
    uy /= rho;
    double u2 = ux * ux + uy * uy;

    /* BGK collision + periodic streaming */
#pragma unroll
    for (int i = 0; i < 9; ++i) {
        double eu  = d_ex[i] * ux + d_ey[i] * uy;
        double feq = d_wt[i] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2);
        double post = fc[i] - (fc[i] - feq) / tau;
        int xn = (x + d_ex[i] + nx) % nx;
        int yn = (y + d_ey[i] + ny) % ny;
        ft[((size_t)yn * nx + xn) * 9 + i] = post;
    }
}

int main(int argc, char **argv) {
    int nx    = (argc > 1) ? atoi(argv[1]) : 256;
    int ny    = (argc > 2) ? atoi(argv[2]) : 256;
    int steps = (argc > 3) ? atoi(argv[3]) : 2000;
    double tau = 0.8;
    double U0  = 0.05;
    size_t N = (size_t)nx * ny;
    size_t bytes = N * 9 * sizeof(double);

    double *h_f = (double *)malloc(bytes);
    if (!h_f) { fprintf(stderr, "alloc failed\n"); return 1; }

    /* initialise equilibrium from Taylor-Green vortex */
    double kx = 2.0 * PI / nx, ky = 2.0 * PI / ny;
    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
            double ux0 = -U0 * cos(kx * x) * sin(ky * y);
            double uy0 =  U0 * sin(kx * x) * cos(ky * y);
            double u2  = ux0 * ux0 + uy0 * uy0;
            double *fc = &h_f[((size_t)y * nx + x) * 9];
            for (int i = 0; i < 9; ++i) {
                double eu = ex[i] * ux0 + ey[i] * uy0;
                fc[i] = wt[i] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2);
            }
        }

    /* initial invariants (host, exact same reduction as baseline) */
    double mass0 = 0.0, ke0 = 0.0;
    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
            double *fc = &h_f[((size_t)y * nx + x) * 9];
            double rho = 0, mx = 0, my = 0;
            for (int i = 0; i < 9; ++i) {
                rho += fc[i];
                mx  += ex[i] * fc[i];
                my  += ey[i] * fc[i];
            }
            mass0 += rho;
            ke0   += 0.5 * (mx * mx + my * my) / rho;
        }

    /* device buffers */
    double *d_f, *d_ft;
    CUDA_CHECK(cudaMalloc(&d_f,  bytes));
    CUDA_CHECK(cudaMalloc(&d_ft, bytes));

    /* copy constants to device */
    CUDA_CHECK(cudaMemcpyToSymbol(d_ex, ex, 9 * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_ey, ey, 9 * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_wt, wt, 9 * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_f, h_f, bytes, cudaMemcpyHostToDevice));

    dim3 threads(16, 16);
    dim3 blocks((nx + threads.x - 1) / threads.x,
                (ny + threads.y - 1) / threads.y);

    clock_t t0 = clock();
    for (int s = 0; s < steps; ++s) {
        collide_stream<<<blocks, threads>>>(nx, ny, tau, d_f, d_ft);
        double *tmp = d_f; d_f = d_ft; d_ft = tmp;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_t t1 = clock();

    CUDA_CHECK(cudaMemcpy(h_f, d_f, bytes, cudaMemcpyDeviceToHost));

    /* final reduction on host (same order as baseline) */
    double mass = 0.0, ke = 0.0;
    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
            double *fc = &h_f[((size_t)y * nx + x) * 9];
            double rho = 0, mx = 0, my = 0;
            for (int i = 0; i < 9; ++i) {
                rho += fc[i];
                mx  += ex[i] * fc[i];
                my  += ey[i] * fc[i];
            }
            mass += rho;
            ke   += 0.5 * (mx * mx + my * my) / rho;
        }

    printf("lbm %dx%d steps=%d  mass_drift=%.2e (conserved)  KE: %.4e -> %.4e (decays %.1f%%)  time=%.3f s\n",
           nx, ny, steps, fabs(mass - mass0) / mass0, ke0, ke,
           100.0 * (1.0 - ke / ke0), (double)(t1 - t0) / CLOCKS_PER_SEC);

    CUDA_CHECK(cudaFree(d_f));
    CUDA_CHECK(cudaFree(d_ft));
    free(h_f);
    return 0;
}
