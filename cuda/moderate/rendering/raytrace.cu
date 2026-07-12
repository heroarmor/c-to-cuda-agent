/* raytrace.cu -- CUDA conversion of benchmark/moderate/rendering/raytrace.c
 *
 * Two-pass (deferred) ray tracer: two distinct kernels collaborate via a
 * per-pixel G-buffer.
 *   1) primary_kernel -- cast the primary ray, find nearest sphere -> (id, t),
 *                        count hits
 *   2) shade_kernel   -- consume the G-buffer: shadow test + shading -> RGB,
 *                        count shadows
 * One thread per pixel; scene in constant memory.
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

typedef struct { double x, y, z; } V;
__device__ static V vadd(V a, V b)        { return (V){ a.x + b.x, a.y + b.y, a.z + b.z }; }
__device__ static V vsub(V a, V b)        { return (V){ a.x - b.x, a.y - b.y, a.z - b.z }; }
__device__ static V vscl(V a, double s)   { return (V){ a.x * s, a.y * s, a.z * s }; }
__device__ static double vdot(V a, V b)   { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ static V vnorm(V a)            { return vscl(a, 1.0 / sqrt(vdot(a, a))); }

typedef struct { V c; double r; V color; } Sphere;

#define NS 3
__constant__ Sphere d_scene[NS];

__device__ static double hit(V o, V d, Sphere s) {
    V oc = vsub(o, s.c);
    double b = vdot(oc, d);
    double c = vdot(oc, oc) - s.r * s.r;
    double disc = b * b - c;
    if (disc < 0.0) return -1.0;
    double t = -b - sqrt(disc);
    return (t > 1e-4) ? t : -1.0;
}

/* per-pixel primary ray direction (identical formula in both passes) */
__device__ static V ray_dir(int px, int py, int w, int h) {
    double u = (2.0 * (px + 0.5) / w - 1.0) * ((double)w / h);
    double v = (1.0 - 2.0 * (py + 0.5) / h);
    return vnorm((V){ u, v, -1.0 });
}

/* pass 1: nearest-hit G-buffer */
__global__ void primary_kernel(int w, int h, int *gid, double *gt,
                               unsigned long long *hits) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    V eye = { 0.0, 0.0, 0.0 };
    V dir = ray_dir(px, py, w, h);
    double tbest = 1e30;
    int id = -1;
    for (int i = 0; i < NS; ++i) {
        double t = hit(eye, dir, d_scene[i]);
        if (t > 0.0 && t < tbest) { tbest = t; id = i; }
    }
    size_t idx = (size_t)py * w + px;
    gid[idx] = id;
    gt[idx]  = tbest;
    if (id >= 0) atomicAdd(hits, 1ULL);
}

/* pass 2: shading + shadow test from the G-buffer */
__global__ void shade_kernel(int w, int h, V light, const int *gid, const double *gt,
                             unsigned char *img, unsigned long long *shadows) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    size_t idx = (size_t)py * w + px;
    int id = gid[idx];
    V eye = { 0.0, 0.0, 0.0 };
    V dir = ray_dir(px, py, w, h);

    V col = { 0.1, 0.1, 0.15 };
    if (id >= 0) {
        double tbest = gt[idx];
        V p = vadd(eye, vscl(dir, tbest));
        V nrm = vnorm(vsub(p, d_scene[id].c));
        int shadow = 0;
        V po = vadd(p, vscl(nrm, 1e-3));
        for (int i = 0; i < NS; ++i)
            if (hit(po, light, d_scene[i]) > 0.0) { shadow = 1; break; }
        if (shadow) atomicAdd(shadows, 1ULL);
        double diff = shadow ? 0.0 : fmax(0.0, vdot(nrm, light));
        col = vscl(d_scene[id].color, 0.15 + 0.85 * diff);
    }

    size_t o = idx * 3;
    img[o + 0] = (unsigned char)(255.0 * fmin(1.0, col.x));
    img[o + 1] = (unsigned char)(255.0 * fmin(1.0, col.y));
    img[o + 2] = (unsigned char)(255.0 * fmin(1.0, col.z));
}

int main(int argc, char **argv) {
    int w = (argc > 1) ? atoi(argv[1]) : 800;
    int h = (argc > 2) ? atoi(argv[2]) : 600;

    unsigned char *h_img = (unsigned char *)malloc((size_t)w * h * 3);
    if (!h_img) { fprintf(stderr, "alloc failed\n"); return 1; }

    unsigned char *d_img;
    int *d_gid;
    double *d_gt;
    unsigned long long *d_hits, *d_shadows;
    unsigned long long h_hits = 0, h_shadows = 0;

    CUDA_CHECK(cudaMalloc(&d_img, (size_t)w * h * 3));
    CUDA_CHECK(cudaMalloc(&d_gid, (size_t)w * h * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gt,  (size_t)w * h * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_hits, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_shadows, sizeof(unsigned long long)));

    Sphere h_scene[NS] = {
        { { -1.0,  0.0,    -4.0 },    1.0, { 1.0, 0.3, 0.3 } },
        { {  1.0,  0.0,    -4.5 },    1.0, { 0.3, 1.0, 0.3 } },
        { {  0.0, -1001.0, -4.0 }, 1000.0, { 0.6, 0.6, 0.6 } },
    };
    CUDA_CHECK(cudaMemcpyToSymbol(d_scene, h_scene, sizeof(Sphere) * NS));

    V light;
    {
        V a = { -1.0, 1.0, 0.5 };
        double inv = 1.0 / sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
        light.x = a.x * inv; light.y = a.y * inv; light.z = a.z * inv;
    }

    CUDA_CHECK(cudaMemset(d_hits, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_shadows, 0, sizeof(unsigned long long)));

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    clock_t t0 = clock();
    primary_kernel<<<grid, block>>>(w, h, d_gid, d_gt, d_hits);
    shade_kernel<<<grid, block>>>(w, h, light, d_gid, d_gt, d_img, d_shadows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_t t1 = clock();

    CUDA_CHECK(cudaMemcpy(h_img, d_img, (size_t)w * h * 3, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_hits, d_hits, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_shadows, d_shadows, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    long long sum = 0;
    for (size_t i = 0; i < (size_t)w * h * 3; ++i) sum += h_img[i];

    FILE *f = fopen("raytrace.ppm", "wb");
    if (f) {
        fprintf(f, "P6\n%d %d\n255\n", w, h);
        fwrite(h_img, 1, (size_t)w * h * 3, f);
        fclose(f);
    }

    printf("raytrace %dx%d  checksum=%lld  hits=%lld shadows=%lld (exact)  time=%.3f s  -> raytrace.ppm\n",
           w, h, sum, (long long)h_hits, (long long)h_shadows, (double)(t1 - t0) / CLOCKS_PER_SEC);

    CUDA_CHECK(cudaFree(d_img));
    CUDA_CHECK(cudaFree(d_gid));
    CUDA_CHECK(cudaFree(d_gt));
    CUDA_CHECK(cudaFree(d_hits));
    CUDA_CHECK(cudaFree(d_shadows));
    free(h_img);
    return 0;
}
