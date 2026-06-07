/* pathtrace.c -- Monte Carlo path tracer with global illumination
 *                (a clean C11 port of Beason's "smallpt" Cornell box).
 *
 * Renders the classic Cornell box: diffuse colored walls, a perfectly specular
 * (mirror) sphere, a refractive (glass) sphere, and an area light on the ceiling.
 * Each pixel averages many random light paths that bounce through the scene,
 * giving physically based global illumination -- soft shadows from the area
 * light, color bleeding between walls, mirror reflections, and Fresnel
 * refraction.
 *
 * Pattern: realistic rendering = per-pixel Monte Carlo integration of the
 * rendering equation. Placed in the complex tier because it combines several
 * GPU-hard traits at once:
 *   - per-pixel / per-sample independent RNG streams (like Monte Carlo),
 *   - strongly *divergent* control flow: variable path length, Russian-roulette
 *     termination, and three material branches (diffuse / specular / refractive),
 *   - recursion that a GPU port must convert into an iterative bounce loop
 *     carrying a "throughput" weight (GPUs dislike deep recursion).
 *
 * GPU conversion: one thread per pixel (or per sample), a private RNG, the
 * recursive radiance() rewritten as a loop, and the scene in constant memory.
 *
 * Verification: the estimate is stochastic, so the byte checksum is reproducible
 * for this RNG but not bit-identical across implementations. The meaningful
 * invariant is the image mean luminance -- a Monte Carlo estimate that converges
 * to a fixed value as samples-per-pixel grow (cf. montecarlo/mc_pi.c).
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define PI 3.14159265358979323846

typedef struct { double x, y, z; } V;
static inline V vec(double x, double y, double z) { V v = { x, y, z }; return v; }
static inline V add(V a, V b)  { return vec(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline V sub(V a, V b)  { return vec(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline V scl(V a, double s) { return vec(a.x * s, a.y * s, a.z * s); }
static inline V mul(V a, V b)  { return vec(a.x * b.x, a.y * b.y, a.z * b.z); } /* color */
static inline double dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V cross(V a, V b) {
    return vec(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline V norm(V a) { return scl(a, 1.0 / sqrt(dot(a, a))); }

enum { DIFF, SPEC, REFR };                 /* material: diffuse / mirror / glass */

typedef struct { double rad; V p, e, c; int refl; } Sphere;

/* The Cornell box: walls are huge spheres; two spheres + a ceiling light. */
static const Sphere spheres[] = {
    {1e5, { 1e5 + 1, 40.8, 81.6},   {0,0,0},    {.75,.25,.25},    DIFF}, /* left  (red)  */
    {1e5, {-1e5 + 99, 40.8, 81.6},  {0,0,0},    {.25,.25,.75},    DIFF}, /* right (blue) */
    {1e5, {50, 40.8, 1e5},          {0,0,0},    {.75,.75,.75},    DIFF}, /* back         */
    {1e5, {50, 40.8, -1e5 + 170},   {0,0,0},    {0,0,0},          DIFF}, /* front (open) */
    {1e5, {50, 1e5, 81.6},          {0,0,0},    {.75,.75,.75},    DIFF}, /* floor        */
    {1e5, {50, -1e5 + 81.6, 81.6},  {0,0,0},    {.75,.75,.75},    DIFF}, /* ceiling      */
    {16.5,{27, 16.5, 47},           {0,0,0},    {.999,.999,.999}, SPEC}, /* mirror ball  */
    {16.5,{73, 16.5, 78},           {0,0,0},    {.999,.999,.999}, REFR}, /* glass  ball  */
    {600, {50, 681.6 - .27, 81.6},  {12,12,12}, {0,0,0},          DIFF}  /* area light   */
};
static const int NSPH = (int)(sizeof spheres / sizeof spheres[0]);

/* per-stream 64-bit LCG returning a double in [0, 1) */
static inline double rng(unsigned long long *s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(*s >> 11) / 9007199254740992.0;
}

static double sphere_hit(const Sphere *s, V o, V d) {
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

static int intersect(V o, V d, double *t) {
    double best = 1e20;
    int id = -1;
    for (int i = 0; i < NSPH; ++i) {
        double h = sphere_hit(&spheres[i], o, d);
        if (h > 0.0 && h < best) { best = h; id = i; }
    }
    *t = best;
    return id;
}

static V radiance(V o, V d, int depth, unsigned long long *st) {
    double t;
    int id = intersect(o, d, &t);
    if (id < 0) return vec(0, 0, 0);                       /* missed -> black */

    const Sphere *obj = &spheres[id];
    V x = add(o, scl(d, t));
    V n = norm(sub(x, obj->p));
    V nl = dot(n, d) < 0 ? n : scl(n, -1.0);              /* normal toward viewer */
    V f = obj->c;
    double p = f.x > f.y && f.x > f.z ? f.x : (f.y > f.z ? f.y : f.z);  /* max reflectance */

    if (++depth > 5) {                                     /* Russian roulette */
        if (depth < 50 && rng(st) < p) f = scl(f, 1.0 / p);
        else return obj->e;
    }

    if (obj->refl == DIFF) {                               /* diffuse: cosine-weighted */
        double r1 = 2 * PI * rng(st), r2 = rng(st), r2s = sqrt(r2);
        V w = nl;
        V u = norm(cross(fabs(w.x) > .1 ? vec(0, 1, 0) : vec(1, 0, 0), w));
        V v = cross(w, u);
        V dir = norm(add(add(scl(u, cos(r1) * r2s), scl(v, sin(r1) * r2s)),
                         scl(w, sqrt(1 - r2))));
        return add(obj->e, mul(f, radiance(x, dir, depth, st)));
    }
    if (obj->refl == SPEC) {                               /* perfect mirror */
        V refl = sub(d, scl(n, 2 * dot(n, d)));
        return add(obj->e, mul(f, radiance(x, refl, depth, st)));
    }

    /* REFR: dielectric (glass) with Fresnel via Schlick's approximation */
    V refldir = sub(d, scl(n, 2 * dot(n, d)));
    int into = dot(n, nl) > 0;                             /* ray entering the glass? */
    double nc = 1.0, nt = 1.5, nnt = into ? nc / nt : nt / nc;
    double ddn = dot(d, nl), cos2t = 1 - nnt * nnt * (1 - ddn * ddn);
    if (cos2t < 0)                                         /* total internal reflection */
        return add(obj->e, mul(f, radiance(x, refldir, depth, st)));
    V tdir = norm(sub(scl(d, nnt), scl(n, (into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))));
    double a = nt - nc, b = nt + nc, R0 = a * a / (b * b);
    double co = 1 - (into ? -ddn : dot(tdir, n));
    double Re = R0 + (1 - R0) * co * co * co * co * co, Tr = 1 - Re;
    double P = .25 + .5 * Re, RP = Re / P, TP = Tr / (1 - P);
    V res;
    if (depth > 2) {                                       /* one branch after a few bounces */
        if (rng(st) < P) res = scl(radiance(x, refldir, depth, st), RP);
        else             res = scl(radiance(x, tdir,    depth, st), TP);
    } else {                                               /* both branches near the eye */
        res = add(scl(radiance(x, refldir, depth, st), Re),
                  scl(radiance(x, tdir,    depth, st), Tr));
    }
    return add(obj->e, mul(f, res));
}

static inline double clampd(double x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }
static inline int toInt(double x) { return (int)(pow(clampd(x), 1 / 2.2) * 255 + .5); }

int main(int argc, char **argv) {
    int w   = (argc > 1) ? atoi(argv[1]) : 320;
    int h   = (argc > 2) ? atoi(argv[2]) : 240;
    int spp = (argc > 3) ? atoi(argv[3]) : 16;            /* samples per pixel */

    V cam_o = vec(50, 52, 295.6), cam_d = norm(vec(0, -0.042612, -1));
    V cx = vec(w * .5135 / h, 0, 0);                      /* horizontal sensor axis */
    V cy = scl(norm(cross(cx, cam_d)), .5135);           /* vertical sensor axis   */

    V *img = malloc((size_t)w * h * sizeof *img);
    for (size_t i = 0; i < (size_t)w * h; ++i) img[i] = vec(0, 0, 0);

    clock_t t0 = clock();
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            unsigned long long st = 0x9E3779B97F4A7C15ULL ^
                ((unsigned long long)(y * w + x + 1) * 0xD1B54A32D192ED03ULL);
            V r = vec(0, 0, 0);
            for (int s = 0; s < spp; ++s) {
                double dx = (x + rng(&st)) / w - .5;
                double dy = -((y + rng(&st)) / h - .5);  /* flip y -> upright image */
                V draw = add(add(scl(cx, dx), scl(cy, dy)), cam_d);
                V d = norm(draw);
                r = add(r, scl(radiance(add(cam_o, scl(draw, 140)), d, 0, &st), 1.0 / spp));
            }
            img[(size_t)y * w + x] = vec(clampd(r.x), clampd(r.y), clampd(r.z));
        }
    clock_t t1 = clock();

    unsigned char *bytes = malloc((size_t)w * h * 3);
    long long sum = 0;
    double lum = 0.0;
    for (size_t i = 0; i < (size_t)w * h; ++i) {
        int R = toInt(img[i].x), G = toInt(img[i].y), B = toInt(img[i].z);
        bytes[3 * i] = (unsigned char)R;
        bytes[3 * i + 1] = (unsigned char)G;
        bytes[3 * i + 2] = (unsigned char)B;
        sum += R + G + B;
        lum += 0.2126 * img[i].x + 0.7152 * img[i].y + 0.0722 * img[i].z;  /* linear luma */
    }
    FILE *fp = fopen("pathtrace.ppm", "wb");
    if (fp) {
        fprintf(fp, "P6\n%d %d\n255\n", w, h);
        fwrite(bytes, 1, (size_t)w * h * 3, fp);
        fclose(fp);
    }

    /* invariant: mean luminance is a Monte Carlo estimate that converges as spp grows */
    printf("pathtrace %dx%d spp=%d  checksum=%lld  mean_luminance=%.5f (converges)  time=%.3f s  -> pathtrace.ppm\n",
           w, h, spp, sum, lum / ((double)w * h), (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(img); free(bytes);
    return 0;
}
