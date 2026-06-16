/* mandelbrot.cu -- CUDA conversion of benchmark/easy/rendering/mandelbrot.c
 * 2D thread grid, one thread per pixel.
 */
#include <stdio.h>
#include <stdlib.h>
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

__global__ void mandelbrot_kernel(int w, int h, int maxit,
                                  double xmin, double xmax,
                                  double ymin, double ymax,
                                  unsigned char *img) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    double x0 = xmin + (xmax - xmin) * px / (w - 1);
    double y0 = ymin + (ymax - ymin) * py / (h - 1);
    double x = 0.0, y = 0.0;
    int it = 0;
    while (x * x + y * y <= 4.0 && it < maxit) {
        double xt = x * x - y * y + x0;
        y = 2.0 * x * y + y0;
        x = xt;
        ++it;
    }
    img[(size_t)py * w + px] = (unsigned char)(255.0 * it / maxit);
}

int main(int argc, char **argv) {
    int w     = (argc > 1) ? atoi(argv[1]) : 1024;
    int h     = (argc > 2) ? atoi(argv[2]) : 1024;
    int maxit = (argc > 3) ? atoi(argv[3]) : 1000;
    double xmin = -2.5, xmax = 1.0, ymin = -1.75, ymax = 1.75;

    size_t img_bytes = (size_t)w * h;
    unsigned char *h_img = (unsigned char *)malloc(img_bytes);
    if (!h_img) { fprintf(stderr, "alloc failed\n"); return 1; }

    unsigned char *d_img;
    CUDA_CHECK(cudaMalloc(&d_img, img_bytes));

    dim3 blockDim(16, 16);
    dim3 gridDim((w + blockDim.x - 1) / blockDim.x,
                 (h + blockDim.y - 1) / blockDim.y);

    clock_t t0 = clock();
    mandelbrot_kernel<<<gridDim, blockDim>>>(w, h, maxit,
                                             xmin, xmax, ymin, ymax, d_img);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_t t1 = clock();

    CUDA_CHECK(cudaMemcpy(h_img, d_img, img_bytes, cudaMemcpyDeviceToHost));

    long long sum = 0;
    long long in_set = 0;
    for (size_t i = 0; i < img_bytes; ++i) {
        sum += h_img[i];
        if (h_img[i] == 255) ++in_set;
    }

    FILE *f = fopen("mandelbrot.pgm", "wb");
    if (f) {
        fprintf(f, "P5\n%d %d\n255\n", w, h);
        fwrite(h_img, 1, img_bytes, f);
        fclose(f);
    }

    double area = (xmax - xmin) * (ymax - ymin) * (double)in_set / ((double)w * h);
    printf("mandelbrot %dx%d maxit=%d  checksum=%lld  area=%.4f (set ~1.5066)  time=%.3f s  -> mandelbrot.pgm\n",
           w, h, maxit, sum, area, (double)(t1 - t0) / CLOCKS_PER_SEC);

    CUDA_CHECK(cudaFree(d_img));
    free(h_img);
    return 0;
}
