/* sobel.c -- 2D Sobel edge detector on a synthetic grayscale image
 *
 * Generates a deterministic image and applies the 3x3 Sobel-Feldman kernels.
 *
 * Pattern: image processing / 2D stencil with fixed-radius neighborhood.
 * GPU conversion: one thread per output pixel; boundary pixels are zero.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static unsigned char pixel(int x, int y, int w, int h) {
    double cx = x - 0.52 * w;
    double cy = y - 0.48 * h;
    double r = sqrt(cx * cx + cy * cy);
    int wave = (int)(60.0 + 55.0 * sin(0.045 * x) + 45.0 * cos(0.061 * y));
    int ring = ((int)r / 9) % 2 ? 70 : 0;
    int stripe = ((x * 13 + y * 7) & 63) < 18 ? 38 : 0;
    int v = wave + ring + stripe;
    if (v < 0) v = 0;
    if (v > 255) v = 255;
    return (unsigned char)v;
}

int main(int argc, char **argv) {
    int w = (argc > 1) ? atoi(argv[1]) : 1024;
    int h = (argc > 2) ? atoi(argv[2]) : 768;
    if (w < 3 || h < 3) {
        fprintf(stderr, "usage: %s [width>=3] [height>=3]\n", argv[0]);
        return 2;
    }

    size_t n = (size_t)w * h;
    unsigned char *img = (unsigned char *)malloc(n);
    unsigned short *edge = (unsigned short *)calloc(n, sizeof(unsigned short));
    if (!img || !edge) { fprintf(stderr, "alloc failed\n"); return 1; }

    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            img[(size_t)y * w + x] = pixel(x, y, w, h);

    clock_t t0 = clock();
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            int p00 = img[(size_t)(y - 1) * w + (x - 1)];
            int p01 = img[(size_t)(y - 1) * w + x];
            int p02 = img[(size_t)(y - 1) * w + (x + 1)];
            int p10 = img[(size_t)y * w + (x - 1)];
            int p12 = img[(size_t)y * w + (x + 1)];
            int p20 = img[(size_t)(y + 1) * w + (x - 1)];
            int p21 = img[(size_t)(y + 1) * w + x];
            int p22 = img[(size_t)(y + 1) * w + (x + 1)];
            int gx = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22;
            int gy = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;
            int mag = (int)lrint(sqrt((double)gx * gx + (double)gy * gy));
            if (mag > 65535) mag = 65535;
            edge[(size_t)y * w + x] = (unsigned short)mag;
        }
    }
    clock_t t1 = clock();

    unsigned long long checksum = 1469598103934665603ULL;
    unsigned long long sum = 0;
    unsigned int maxv = 0;
    long long strong = 0;
    for (size_t i = 0; i < n; ++i) {
        unsigned int v = edge[i];
        sum += v;
        if (v > maxv) maxv = v;
        if (v >= 180) ++strong;
        checksum ^= (unsigned long long)(v + 31U * (unsigned int)(i & 1023U));
        checksum *= 1099511628211ULL;
    }

    printf("sobel %dx%d checksum=%llu sum=%llu max=%u strong=%lld time=%.3f s\n",
           w, h, checksum, sum, maxv, strong, (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(img);
    free(edge);
    return 0;
}
