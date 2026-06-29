/* pathtrace.cu -- CUDA conversion of benchmark/complex/rendering/pathtrace.c
 * Each pixel is one thread; the recursive radiance() is converted to an
 * iterative bounce loop with a small stack for Fresnel branches.
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

typedef struct { double x, y, z; } V;
static inline __host__ __device__ V vec(double x, double y, double z) { V v = { x, y, z }; return v; }
static inline __host__ __device__ V add(V a, V b)  { return vec(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline __host__ __device__ V sub(V a, V b)  { return vec(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline __host__ __device__ V scl(V a, double s) { return vec(a.x * s, a.y * s, a.z * s); }
static inline __host__ __device__ V mul(V a, V b)  { return vec(a.x * b.x, a.y * b.y, a.z * b.z); }
static inline __host__ __device__ double dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline __host__ __device__ V cross(V a, V b) {
    return vec(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline __host__ __device__ V norm(V a) { return scl(a, 1.0 / sqrt(dot(a, a))); }

enum { DIFF, SPEC, REFR };

typedef struct { double rad; V p, e, c; int refl; } Sphere;

enum { NSPH = 9 };
__constant__ Sphere d_spheres[NSPH];

static inline __device__ double rng(unsigned long long *s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(*s >> 11) / 9007199254740992.0;
}

static inline __device__ double sphere_hit(const Sphere *s, V o, V d) {
    V op = sub(s->p, o);
    double b = dot(op, d);
    double det = b * b - dot(op, op) + s->rad * s->rad;
    if (det < 0.0) return 0.0;
    det = sqrt(det);
    double eps = 1e-4, t;
    if ((t = b - det) > eps) return t;
    if ((t = b + det) > eps) return t;
    return 0.0;
}

static inline __device__ int intersect(V o, V d, double *t) {
    double best = 1e20;
    int id = -1;
    for (int i = 0; i < NSPH; ++i) {
        double h = sphere_hit(&d_spheres[i], o, d);
        if (h > 0.0 && h < best) { best = h; id = i; }
    }
    *t = best;
    return id;
}

struct StackEntry { V o, d, thr; int depth; };

static inline __device__ V radiance_iter(V o, V d, unsigned long long *st) {
    V result = vec(0, 0, 0);
    V throughput = vec(1, 1, 1);
    int depth = 0;
    struct StackEntry stack[16];
    int sp = 0;

    while (1) {
        double t;
        int id = intersect(o, d, &t);
        if (id < 0) {
            if (sp == 0) break;
            struct StackEntry e = stack[--sp];
            o = e.o; d = e.d; throughput = e.thr; depth = e.depth;
            continue;
        }

        const Sphere *obj = &d_spheres[id];
        V x = add(o, scl(d, t));
        V n = norm(sub(x, obj->p));
        V nl = dot(n, d) < 0 ? n : scl(n, -1.0);
        V f = obj->c;
        double p = f.x > f.y && f.x > f.z ? f.x : (f.y > f.z ? f.y : f.z);

        result = add(result, mul(throughput, obj->e));

        depth++;
        if (depth > 5) {
            if (depth < 50 && rng(st) < p) {
                f = scl(f, 1.0 / p);
            } else {
                if (sp == 0) break;
                struct StackEntry e = stack[--sp];
                o = e.o; d = e.d; throughput = e.thr; depth = e.depth;
                continue;
            }
        }

        throughput = mul(throughput, f);

        if (obj->refl == DIFF) {
            double r1 = 2 * PI * rng(st), r2 = rng(st), r2s = sqrt(r2);
            V w = nl;
            V u = norm(cross(fabs(w.x) > 0.1 ? vec(0, 1, 0) : vec(1, 0, 0), w));
            V v = cross(w, u);
            V dir = norm(add(add(scl(u, cos(r1) * r2s), scl(v, sin(r1) * r2s)),
                             scl(w, sqrt(1 - r2))));
            o = x; d = dir;
            continue;
        }
        if (obj->refl == SPEC) {
            V refl = sub(d, scl(n, 2 * dot(n, d)));
            o = x; d = refl;
            continue;
        }

        /* REFR: dielectric (glass) */
        V refldir = sub(d, scl(n, 2 * dot(n, d)));
        int into = dot(n, nl) > 0;
        double nc = 1.0, nt = 1.5, nnt = into ? nc / nt : nt / nc;
        double ddn = dot(d, nl), cos2t = 1 - nnt * nnt * (1 - ddn * ddn);
        if (cos2t < 0) {
            o = x; d = refldir;
            continue;
        }
        V tdir = norm(sub(scl(d, nnt), scl(n, (into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))));
        double a = nt - nc, b = nt + nc, R0 = a * a / (b * b);
        double co = 1 - (into ? -ddn : dot(tdir, n));
        double Re = R0 + (1 - R0) * co * co * co * co * co, Tr = 1 - Re;

        if (depth > 2) {
            double P = 0.25 + 0.5 * Re, RP = Re / P, TP = Tr / (1 - P);
            if (rng(st) < P) {
                throughput = mul(throughput, scl(vec(1,1,1), RP));
                o = x; d = refldir;
            } else {
                throughput = mul(throughput, scl(vec(1,1,1), TP));
                o = x; d = tdir;
            }
        } else {
            if (sp < 16) {
                stack[sp].o = x;
                stack[sp].d = refldir;
                stack[sp].thr = mul(throughput, scl(vec(1,1,1), Re));
                stack[sp].depth = depth;
                sp++;
            }
            throughput = mul(throughput, scl(vec(1,1,1), Tr));
            o = x; d = tdir;
        }
    }

    return result;
}

__global__ void pathtrace_kernel(V *img, int w, int h, int spp,
                                  double ox, double oy, double oz,
                                  double cxx, double cxy, double cxz,
                                  double cyx, double cyy, double cyz,
                                  double cdx, double cdy, double cdz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = w * h;
    if (idx >= total) return;
    int y = idx / w;
    int x = idx % w;

    unsigned long long st = 0x9E3779B97F4A7C15ULL ^
        ((unsigned long long)(idx + 1) * 0xD1B54A32D192ED03ULL);

    V cam_o = vec(ox, oy, oz);
    V cx = vec(cxx, cxy, cxz);
    V cy = vec(cyx, cyy, cyz);
    V cam_d = vec(cdx, cdy, cdz);

    V r = vec(0, 0, 0);
    for (int s = 0; s < spp; ++s) {
        double rnd_x = (x + rng(&st)) / w - 0.5;
        double rnd_y = -((y + rng(&st)) / h - 0.5);
        V draw = add(add(scl(cx, rnd_x), scl(cy, rnd_y)), cam_d);
        V d = norm(draw);
        V ro = add(cam_o, scl(draw, 140));
        r = add(r, scl(radiance_iter(ro, d, &st), 1.0 / spp));
    }

    r.x = r.x < 0 ? 0 : (r.x > 1 ? 1 : r.x);
    r.y = r.y < 0 ? 0 : (r.y > 1 ? 1 : r.y);
    r.z = r.z < 0 ? 0 : (r.z > 1 ? 1 : r.z);
    img[idx] = r;
}

static inline double clampd(double x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }
static inline int toInt(double x) { return (int)(pow(clampd(x), 1 / 2.2) * 255 + .5); }

int main(int argc, char **argv) {
    int w   = (argc > 1) ? atoi(argv[1]) : 320;
    int h   = (argc > 2) ? atoi(argv[2]) : 240;
    int spp = (argc > 3) ? atoi(argv[3]) : 16;

    V cam_o = vec(50, 52, 295.6), cam_d = norm(vec(0, -0.042612, -1));
    V cx = vec(w * .5135 / h, 0, 0);
    V cy = scl(norm(cross(cx, cam_d)), .5135);

    Sphere h_spheres[NSPH] = {
        {1e5, { 1e5 + 1, 40.8, 81.6},   {0,0,0},    {.75,.25,.25},    DIFF},
        {1e5, {-1e5 + 99, 40.8, 81.6},  {0,0,0},    {.25,.25,.75},    DIFF},
        {1e5, {50, 40.8, 1e5},          {0,0,0},    {.75,.75,.75},    DIFF},
        {1e5, {50, 40.8, -1e5 + 170},   {0,0,0},    {0,0,0},          DIFF},
        {1e5, {50, 1e5, 81.6},          {0,0,0},    {.75,.75,.75},    DIFF},
        {1e5, {50, -1e5 + 81.6, 81.6},  {0,0,0},    {.75,.75,.75},    DIFF},
        {16.5,{27, 16.5, 47},           {0,0,0},    {.999,.999,.999}, SPEC},
        {16.5,{73, 16.5, 78},           {0,0,0},    {.999,.999,.999}, REFR},
        {600, {50, 681.6 - .27, 81.6},  {12,12,12}, {0,0,0},          DIFF}
    };
    CUDA_CHECK(cudaMemcpyToSymbol(d_spheres, h_spheres, sizeof(h_spheres)));

    size_t npix = (size_t)w * h;
    size_t img_bytes = npix * sizeof(V);

    V *d_img;
    CUDA_CHECK(cudaMalloc(&d_img, img_bytes));

    int threads = 256;
    int blocks = (int)((npix + threads - 1) / threads);

    clock_t t0 = clock();
    pathtrace_kernel<<<blocks, threads>>>(d_img, w, h, spp,
                                           cam_o.x, cam_o.y, cam_o.z,
                                           cx.x, cx.y, cx.z,
                                           cy.x, cy.y, cy.z,
                                           cam_d.x, cam_d.y, cam_d.z);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_t t1 = clock();

    V *img = (V *)malloc(img_bytes);
    CUDA_CHECK(cudaMemcpy(img, d_img, img_bytes, cudaMemcpyDeviceToHost));

    unsigned char *bytes = (unsigned char *)malloc(npix * 3);
    long long sum = 0;
    double lum = 0.0;
    for (size_t i = 0; i < npix; ++i) {
        int R = toInt(img[i].x), G = toInt(img[i].y), B = toInt(img[i].z);
        bytes[3 * i] = (unsigned char)R;
        bytes[3 * i + 1] = (unsigned char)G;
        bytes[3 * i + 2] = (unsigned char)B;
        sum += R + G + B;
        lum += 0.2126 * img[i].x + 0.7152 * img[i].y + 0.0722 * img[i].z;
    }

    FILE *fp = fopen("pathtrace.ppm", "wb");
    if (fp) {
        fprintf(fp, "P6\n%d %d\n255\n", w, h);
        fwrite(bytes, 1, (size_t)w * h * 3, fp);
        fclose(fp);
    }

    printf("pathtrace %dx%d spp=%d  checksum=%lld  mean_luminance=%.5f (converges)  time=%.3f s  -> pathtrace.ppm\n",
           w, h, spp, sum, lum / (double)npix, (double)(t1 - t0) / CLOCKS_PER_SEC);

    CUDA_CHECK(cudaFree(d_img));
    free(img); free(bytes);
    return 0;
}
