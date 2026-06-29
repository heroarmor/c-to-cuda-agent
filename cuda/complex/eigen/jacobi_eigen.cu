/* jacobi_eigen.cu -- CUDA conversion of benchmark/complex/eigen/jacobi_eigen.c
 *
 * Cyclic Jacobi eigenvalue solver for a dense symmetric matrix.  The sweep
 * and pair iteration run sequentially on one thread-block; each rotation's
 * row/column update is parallelised across threads via grid-stride loops.
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

/* One block performs one cyclic Jacobi sweep on the GPU.
 * Shared memory layout:
 *   [0]                 rotation cosine  (c)
 *   [1]                 rotation sine    (s)
 */
__global__ void jacobi_sweep_kernel(int n, double *A) {
    extern __shared__ double shared[];
    int tid = threadIdx.x;
    int nt  = blockDim.x;

    for (int p = 0; p < n; ++p) {
        for (int q = p + 1; q < n; ++q) {
            /* Thread 0 computes the Jacobi rotation (c,s). */
            if (tid == 0) {
                double apq = A[p * n + q];
                if (fabs(apq) > 1e-300) {
                    double app = A[p * n + p];
                    double aqq = A[q * n + q];
                    double theta = (aqq - app) / (2.0 * apq);
                    double t = (theta >= 0.0 ? 1.0 : -1.0)
                               / (fabs(theta) + sqrt(theta * theta + 1.0));
                    shared[0] = 1.0 / sqrt(t * t + 1.0);
                    shared[1] = t * shared[0];
                } else {
                    shared[0] = 1.0;
                    shared[1] = 0.0;
                }
            }
            __syncthreads();

            double c = shared[0];
            double s = shared[1];

            /* rotate columns p,q */
            for (int i = tid; i < n; i += nt) {
                double aip = A[i * n + p];
                double aiq = A[i * n + q];
                A[i * n + p] = c * aip - s * aiq;
                A[i * n + q] = s * aip + c * aiq;
            }
            __syncthreads();

            /* rotate rows p,q */
            for (int i = tid; i < n; i += nt) {
                double api = A[p * n + i];
                double aqi = A[q * n + i];
                A[p * n + i] = c * api - s * aqi;
                A[q * n + i] = s * api + c * aqi;
            }
            __syncthreads();
        }
    }
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 128;
    double tol = 1e-12;
    int max_sweeps = 100;

    size_t bytes = (size_t)n * n * sizeof(double);
    double *h_A = (double *)calloc(n * n, sizeof(double));
    for (int i = 0; i < n; ++i) {
        h_A[i * n + i] = 2.0;
        if (i > 0)     h_A[i * n + (i - 1)] = -1.0;
        if (i < n - 1) h_A[i * n + (i + 1)] = -1.0;
    }
    double trace0 = 0.0;
    for (int i = 0; i < n; ++i) trace0 += h_A[i * n + i];

    double *d_A;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    size_t shmem = 2 * sizeof(double);

    int sweeps = 0;
    clock_t t0 = clock();
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        jacobi_sweep_kernel<<<1, threads, shmem>>>(n, d_A);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost));

        double off = 0.0;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q)
                off += h_A[p * n + q] * h_A[p * n + q];
        if (sqrt(off) < tol) { sweeps = sweep + 1; break; }
    }
    clock_t t1 = clock();
    if (sweeps == 0) sweeps = max_sweeps;

    double tr = 0.0, emin = h_A[0], emax = h_A[0];
    for (int i = 0; i < n; ++i) {
        double e = h_A[i * n + i];
        tr += e;
        if (e < emin) emin = e;
        if (e > emax) emax = e;
    }
    const double PI = 3.14159265358979323846;
    double emin_exact = 2.0 - 2.0 * cos(PI / (n + 1));
    double emax_exact = 2.0 - 2.0 * cos(n * PI / (n + 1));

    printf("jacobi_eigen n=%d  sweeps=%d  sum(eig)=%.6f (trace=%.6f)\n",
           n, sweeps, tr, trace0);
    printf("            lambda_min=%.6f (exact %.6f)  lambda_max=%.6f (exact %.6f)  time=%.3f s\n",
           emin, emin_exact, emax, emax_exact, (double)(t1 - t0) / CLOCKS_PER_SEC);

    CUDA_CHECK(cudaFree(d_A));
    free(h_A);
    return 0;
}
