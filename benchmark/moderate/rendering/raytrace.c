/* raytrace.c -- minimal ray tracer (spheres, diffuse shading + hard shadows)
 *
 * Casts one primary ray per pixel against a few spheres lit by a single
 * directional light, with shadow rays.
 *
 * Pattern: rendering (embarrassingly parallel per pixel) with branchy control
 * flow (hit/miss, shadow rays) -- a good test of divergence handling.
 * GPU conversion: one thread per pixel; the scene lives in constant/global
 * memory. Writes a PPM color image.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct { double x, y, z; } V;
static V vadd(V a, V b)        { return (V){ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V vsub(V a, V b)        { return (V){ a.x - b.x, a.y - b.y, a.z - b.z }; }
static V vscl(V a, double s)   { return (V){ a.x * s, a.y * s, a.z * s }; }
static double vdot(V a, V b)   { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V vnorm(V a)            { return vscl(a, 1.0 / sqrt(vdot(a, a))); }

typedef struct { V c; double r; V color; } Sphere;

#define NS 3
static Sphere scene[NS] = {
    { { -1.0,  0.0,    -4.0 },    1.0, { 1.0, 0.3, 0.3 } },
    { {  1.0,  0.0,    -4.5 },    1.0, { 0.3, 1.0, 0.3 } },
    { {  0.0, -1001.0, -4.0 }, 1000.0, { 0.6, 0.6, 0.6 } },   /* ground */
};

/* nearest positive ray-sphere intersection, or -1 on miss */
static double hit(V o, V d, Sphere s) {
    V oc = vsub(o, s.c);
    double b = vdot(oc, d);
    double c = vdot(oc, oc) - s.r * s.r;
    double disc = b * b - c;
    if (disc < 0.0) return -1.0;
    double t = -b - sqrt(disc);
    return (t > 1e-4) ? t : -1.0;
}

int main(int argc, char **argv) {
    int w = (argc > 1) ? atoi(argv[1]) : 800;
    int h = (argc > 2) ? atoi(argv[2]) : 600;
    V eye = { 0.0, 0.0, 0.0 };
    V light = vnorm((V){ -1.0, 1.0, 0.5 });   /* directional light */

    unsigned char *img = malloc((size_t)w * h * 3);
    if (!img) { fprintf(stderr, "alloc failed\n"); return 1; }

    long long hits = 0, shadows = 0;
    clock_t t0 = clock();
    for (int py = 0; py < h; ++py)
        for (int px = 0; px < w; ++px) {
            double u = (2.0 * (px + 0.5) / w - 1.0) * ((double)w / h);
            double v = (1.0 - 2.0 * (py + 0.5) / h);
            V dir = vnorm((V){ u, v, -1.0 });

            double tbest = 1e30;
            int id = -1;
            for (int i = 0; i < NS; ++i) {
                double t = hit(eye, dir, scene[i]);
                if (t > 0.0 && t < tbest) { tbest = t; id = i; }
            }

            V col = { 0.1, 0.1, 0.15 };          /* background */
            if (id >= 0) {
                ++hits;
                V p = vadd(eye, vscl(dir, tbest));
                V nrm = vnorm(vsub(p, scene[id].c));
                int shadow = 0;
                V po = vadd(p, vscl(nrm, 1e-3));  /* offset to avoid self-hit */
                for (int i = 0; i < NS; ++i)
                    if (hit(po, light, scene[i]) > 0.0) { shadow = 1; break; }
                if (shadow) ++shadows;
                double diff = shadow ? 0.0 : fmax(0.0, vdot(nrm, light));
                col = vscl(scene[id].color, 0.15 + 0.85 * diff);
            }

            size_t o = ((size_t)py * w + px) * 3;
            img[o + 0] = (unsigned char)(255.0 * fmin(1.0, col.x));
            img[o + 1] = (unsigned char)(255.0 * fmin(1.0, col.y));
            img[o + 2] = (unsigned char)(255.0 * fmin(1.0, col.z));
        }
    clock_t t1 = clock();

    long long sum = 0;
    for (size_t i = 0; i < (size_t)w * h * 3; ++i) sum += img[i];

    FILE *f = fopen("raytrace.ppm", "wb");
    if (f) {
        fprintf(f, "P6\n%d %d\n255\n", w, h);
        fwrite(img, 1, (size_t)w * h * 3, f);
        fclose(f);
    }
    /* invariant: hit and shadow pixel counts are exact integers, so they must
       match bit-for-bit across implementations regardless of float reordering. */
    printf("raytrace %dx%d  checksum=%lld  hits=%lld shadows=%lld (exact)  time=%.3f s  -> raytrace.ppm\n",
           w, h, sum, hits, shadows, (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(img);
    return 0;
}
