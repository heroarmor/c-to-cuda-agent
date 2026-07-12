/* statevector.cu -- CUDA conversion of benchmark/complex/quantum/statevector.c
 * One thread per amplitude pair; state vector resident on GPU across all gates.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                 \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

typedef cuDoubleComplex cx;

__global__ void apply1_kernel(cx *psi, long dim, int q,
                               cx u00, cx u01, cx u10, cx u11) {
    long bit = 1L << q;
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < dim / 2) {
        long low  = (tid & (bit - 1)) | ((tid & ~(bit - 1)) << 1);
        long high = low | bit;
        cx a = psi[low];
        cx b = psi[high];
        psi[low]  = cuCadd(cuCmul(u00, a), cuCmul(u01, b));
        psi[high] = cuCadd(cuCmul(u10, a), cuCmul(u11, b));
    }
}

__global__ void cnot_kernel(cx *psi, long dim, int ctrl, int tgt) {
    long tb = 1L << tgt;
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < dim / 4) {
        long low;
        if (ctrl < tgt) {
            long cb = 1L << ctrl;
            low  = tid & (cb - 1L);
            low |= cb;
            low |= ((tid >> ctrl) & ((1L << (tgt - ctrl - 1)) - 1L)) << (ctrl + 1);
            low |= (tid >> (tgt - 1)) << (tgt + 1);
        } else {
            low  = tid & ((1L << tgt) - 1L);
            low |= ((tid >> tgt) & ((1L << (ctrl - tgt - 1)) - 1L)) << (tgt + 1);
            low |= (1L << ctrl);
            low |= (tid >> (ctrl - 1)) << (ctrl + 1);
        }
        long high = low | tb;
        cx tmp = psi[low];
        psi[low] = psi[high];
        psi[high] = tmp;
    }
}

int main(int argc, char **argv) {
    int n      = (argc > 1) ? atoi(argv[1]) : 20;
    int layers = (argc > 2) ? atoi(argv[2]) : 6;
    long dim = 1L << n;

    cx *h_psi = (cx *)malloc((size_t)dim * sizeof(cx));
    if (!h_psi) { fprintf(stderr, "host alloc failed\n"); return 1; }
    for (long k = 0; k < dim; ++k)
        h_psi[k] = make_cuDoubleComplex(0.0, 0.0);
    h_psi[0] = make_cuDoubleComplex(1.0, 0.0);

    cx *d_psi;
    CUDA_CHECK(cudaMalloc(&d_psi, (size_t)dim * sizeof(cx)));
    CUDA_CHECK(cudaMemcpy(d_psi, h_psi, (size_t)dim * sizeof(cx), cudaMemcpyHostToDevice));

    int threads = 256;

    clock_t t0 = clock();
    for (int L = 0; L < layers; ++L) {
        for (int q = 0; q < n; ++q) {
            double th = 0.1 * (q + 1) + 0.3 * L;
            double c = cos(th / 2), s = sin(th / 2);
            cx u00 = make_cuDoubleComplex( c, 0.0);
            cx u01 = make_cuDoubleComplex(-s, 0.0);
            cx u10 = make_cuDoubleComplex( s, 0.0);
            cx u11 = make_cuDoubleComplex( c, 0.0);
            int blocks = (int)((dim / 2 + threads - 1) / threads);
            apply1_kernel<<<blocks, threads>>>(d_psi, dim, q, u00, u01, u10, u11);
            CUDA_CHECK(cudaGetLastError());
        }
        for (int q = 0; q + 1 < n; ++q) {
            int blocks = (int)((dim / 4 + threads - 1) / threads);
            cnot_kernel<<<blocks, threads>>>(d_psi, dim, q, q + 1);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_t t1 = clock();

    CUDA_CHECK(cudaMemcpy(h_psi, d_psi, (size_t)dim * sizeof(cx), cudaMemcpyDeviceToHost));

    double norm2 = 0.0, p0 = 0.0;
    double checksum_re = 0.0, checksum_im = 0.0;
    for (long k = 0; k < dim; ++k) {
        double pr = h_psi[k].x * h_psi[k].x + h_psi[k].y * h_psi[k].y;
        norm2 += pr;
        double w = (double)((k % 7) + 1);
        checksum_re += h_psi[k].x * w;
        checksum_im += h_psi[k].y * w;
    }
    p0 = h_psi[0].x * h_psi[0].x + h_psi[0].y * h_psi[0].y;

    printf("statevector n=%d layers=%d (dim=%ld)  ||psi||^2=%.12f (unit)  P(|0>)=%.6f  re(chk)=%.6f  time=%.3f s\n",
           n, layers, dim, norm2, p0, checksum_re, (double)(t1 - t0) / CLOCKS_PER_SEC);

    CUDA_CHECK(cudaFree(d_psi));
    free(h_psi);
    return 0;
}
