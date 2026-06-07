/* mc_pi.c -- Monte Carlo estimation of pi
 *
 * Throws random darts at the unit square and counts how many land inside the
 * quarter circle:  pi ~= 4 * (inside / total).
 *
 * Pattern: Monte Carlo (Berkeley dwarf) -- embarrassingly parallel. The only
 * subtlety for the GPU is parallel random-number generation: give each thread an
 * independent RNG stream, then reduce the per-stream counts.
 * GPU conversion: one thread per chunk with a private LCG state; final reduction
 * over the per-thread "inside" counts.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* per-stream 64-bit LCG returning a double in [0, 1) */
static inline double lcg(unsigned long long *s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(*s >> 11) / 9007199254740992.0;   /* divide by 2^53 */
}

int main(int argc, char **argv) {
    long long samples = (argc > 1) ? atoll(argv[1]) : 100000000LL;
    int chunks        = (argc > 2) ? atoi(argv[2]) : 1024;   /* independent streams */
    long long per = samples / chunks;
    long long total = 0;

    clock_t t0 = clock();
    for (int c = 0; c < chunks; ++c) {
        unsigned long long state =
            0x9E3779B97F4A7C15ULL ^ ((unsigned long long)(c + 1) * 0xD1B54A32D192ED03ULL);
        long long inside = 0;
        for (long long i = 0; i < per; ++i) {
            double x = lcg(&state), y = lcg(&state);
            if (x * x + y * y <= 1.0) ++inside;
        }
        total += inside;
    }
    clock_t t1 = clock();

    long long n = per * chunks;
    double pi = 4.0 * (double)total / (double)n;
    printf("mc_pi samples=%lld chunks=%d  pi=%.6f  err=%.2e  time=%.3f s\n",
           n, chunks, pi, pi - 3.14159265358979323846,
           (double)(t1 - t0) / CLOCKS_PER_SEC);
    return 0;
}
