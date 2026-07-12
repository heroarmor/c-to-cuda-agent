/* fft1d.c -- iterative radix-2 Cooley-Tukey complex FFT
 *
 * Builds a deterministic complex signal, runs forward and inverse FFT, and
 * reports reconstruction error plus a frequency-domain checksum.
 *
 * Pattern: staged signal-processing transform with bit reversal and butterfly
 * dependencies.
 * GPU conversion: one kernel per FFT stage, or a block-local FFT for small N.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct { double re, im; } C;

static C cadd(C a, C b) { return (C){ a.re + b.re, a.im + b.im }; }
static C csub(C a, C b) { return (C){ a.re - b.re, a.im - b.im }; }
static C cmul(C a, C b) { return (C){ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re }; }

static int is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

static unsigned revbits(unsigned x, int bits) {
    unsigned r = 0;
    for (int i = 0; i < bits; ++i) {
        r = (r << 1) | (x & 1U);
        x >>= 1;
    }
    return r;
}

static void fft(C *a, int n, int inverse) {
    int bits = 0;
    while ((1 << bits) < n) ++bits;
    for (int i = 0; i < n; ++i) {
        int j = (int)revbits((unsigned)i, bits);
        if (j > i) {
            C tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
    }

    const double pi = acos(-1.0);
    for (int len = 2; len <= n; len <<= 1) {
        double ang = (inverse ? 2.0 : -2.0) * pi / len;
        C wlen = { cos(ang), sin(ang) };
        for (int i = 0; i < n; i += len) {
            C w = { 1.0, 0.0 };
            for (int j = 0; j < len / 2; ++j) {
                C u = a[i + j];
                C v = cmul(a[i + j + len / 2], w);
                a[i + j] = cadd(u, v);
                a[i + j + len / 2] = csub(u, v);
                w = cmul(w, wlen);
            }
        }
    }
    if (inverse)
        for (int i = 0; i < n; ++i) {
            a[i].re /= n;
            a[i].im /= n;
        }
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 4096;
    if (!is_pow2(n) || n < 2) {
        fprintf(stderr, "usage: %s [power_of_two_n>=2]\n", argv[0]);
        return 2;
    }

    C *x = (C *)malloc((size_t)n * sizeof(C));
    C *orig = (C *)malloc((size_t)n * sizeof(C));
    if (!x || !orig) { fprintf(stderr, "alloc failed\n"); return 1; }

    const double pi = acos(-1.0);
    for (int i = 0; i < n; ++i) {
        double t = (double)i / n;
        x[i].re = sin(2.0 * pi * 7.0 * t) + 0.5 * cos(2.0 * pi * 37.0 * t) + 0.001 * (i % 17);
        x[i].im = 0.25 * sin(2.0 * pi * 13.0 * t) - 0.002 * (i % 11);
        orig[i] = x[i];
    }

    clock_t t0 = clock();
    fft(x, n, 0);
    double energy = 0.0;
    long long checksum = 0;
    for (int i = 0; i < n; ++i) {
        double mag2 = x[i].re * x[i].re + x[i].im * x[i].im;
        energy += mag2;
        checksum += (long long)llround((x[i].re + 2.0 * x[i].im) * 1000000.0) * (long long)((i % 97) + 1);
    }
    fft(x, n, 1);
    clock_t t1 = clock();

    double maxerr = 0.0;
    for (int i = 0; i < n; ++i) {
        double er = x[i].re - orig[i].re;
        double ei = x[i].im - orig[i].im;
        double e = sqrt(er * er + ei * ei);
        if (e > maxerr) maxerr = e;
    }

    printf("fft1d n=%d checksum=%lld energy=%.9f max_recon_error=%.3e time=%.3f s\n",
           n, checksum, energy, maxerr, (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(x);
    free(orig);
    return 0;
}
