/* mandelbrot.c -- Mandelbrot set escape-time rendering
 *
 * Pattern: rendering / map (embarrassingly parallel). Every pixel is fully
 * independent -- the textbook GPU image kernel -- but the iteration count varies
 * per pixel, so it also exercises thread divergence.
 * GPU conversion: 2D thread grid, one thread per pixel. Writes a grayscale PGM.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    int w     = (argc > 1) ? atoi(argv[1]) : 1024;
    int h     = (argc > 2) ? atoi(argv[2]) : 1024;
    int maxit = (argc > 3) ? atoi(argv[3]) : 1000;
    double xmin = -2.5, xmax = 1.0, ymin = -1.75, ymax = 1.75;

    unsigned char *img = malloc((size_t)w * h);
    if (!img) { fprintf(stderr, "alloc failed\n"); return 1; }

    long long in_set = 0;
    clock_t t0 = clock();
    for (int py = 0; py < h; ++py)
        for (int px = 0; px < w; ++px) {
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
            if (it == maxit) ++in_set;                 /* point assumed inside the set */
            img[(size_t)py * w + px] = (unsigned char)(255.0 * it / maxit);
        }
    clock_t t1 = clock();

    long long sum = 0;
    for (size_t i = 0; i < (size_t)w * h; ++i) sum += img[i];

    FILE *f = fopen("mandelbrot.pgm", "wb");
    if (f) {
        fprintf(f, "P5\n%d %d\n255\n", w, h);
        fwrite(img, 1, (size_t)w * h, f);
        fclose(f);
    }
    /* invariant: in-set fraction x viewport area estimates the Mandelbrot set
       area, whose accepted value is ~1.5066 (converges as resolution grows). */
    double area = (xmax - xmin) * (ymax - ymin) * (double)in_set / ((double)w * h);
    printf("mandelbrot %dx%d maxit=%d  checksum=%lld  area=%.4f (set ~1.5066)  time=%.3f s  -> mandelbrot.pgm\n",
           w, h, maxit, sum, area, (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(img);
    return 0;
}
