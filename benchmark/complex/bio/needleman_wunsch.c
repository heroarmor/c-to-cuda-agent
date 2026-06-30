/* needleman_wunsch.c -- global sequence alignment dynamic programming
 *
 * Generates two deterministic DNA sequences and fills the full
 * Needleman-Wunsch score matrix.
 *
 * Pattern: 2D dynamic programming with north/west/northwest dependencies.
 * GPU conversion: anti-diagonal wavefront kernels or tiled wavefronts.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static char dna_char(int i, int salt) {
    static const char bases[4] = { 'A', 'C', 'G', 'T' };
    unsigned x = (unsigned)(i + 1) * 1103515245U + (unsigned)salt * 12345U;
    x ^= x >> 16;
    return bases[x & 3U];
}

static int max3(int a, int b, int c) {
    int m = (a > b) ? a : b;
    return (m > c) ? m : c;
}

int main(int argc, char **argv) {
    int m = (argc > 1) ? atoi(argv[1]) : 2048;
    int n = (argc > 2) ? atoi(argv[2]) : 2048;
    if (m < 1 || n < 1) {
        fprintf(stderr, "usage: %s [len_a>=1] [len_b>=1]\n", argv[0]);
        return 2;
    }

    char *a = (char *)malloc((size_t)m);
    char *b = (char *)malloc((size_t)n);
    int *dp = (int *)malloc((size_t)(m + 1) * (n + 1) * sizeof(int));
    if (!a || !b || !dp) { fprintf(stderr, "alloc failed\n"); return 1; }

    for (int i = 0; i < m; ++i) a[i] = dna_char(i, 7);
    for (int j = 0; j < n; ++j) b[j] = dna_char(j, 19);
    for (int i = 0; i <= m; ++i) dp[(size_t)i * (n + 1)] = -2 * i;
    for (int j = 0; j <= n; ++j) dp[j] = -2 * j;

    clock_t t0 = clock();
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            int match = (a[i - 1] == b[j - 1]) ? 2 : -1;
            int diag = dp[(size_t)(i - 1) * (n + 1) + (j - 1)] + match;
            int up = dp[(size_t)(i - 1) * (n + 1) + j] - 2;
            int left = dp[(size_t)i * (n + 1) + (j - 1)] - 2;
            dp[(size_t)i * (n + 1) + j] = max3(diag, up, left);
        }
    }
    clock_t t1 = clock();

    long long checksum = 0;
    long long diag_sum = 0;
    int minv = dp[0], maxv = dp[0];
    for (int i = 0; i <= m; ++i) {
        int j = (int)((long long)i * n / m);
        diag_sum += dp[(size_t)i * (n + 1) + j];
    }
    for (int i = 0; i <= m; i += (m / 64) + 1) {
        for (int j = 0; j <= n; j += (n / 64) + 1) {
            int v = dp[(size_t)i * (n + 1) + j];
            if (v < minv) minv = v;
            if (v > maxv) maxv = v;
            checksum = checksum * 1315423911LL + v * 31LL + i * 17LL + j;
        }
    }

    int score = dp[(size_t)m * (n + 1) + n];
    printf("needleman_wunsch %dx%d score=%d checksum=%lld diag_sum=%lld min=%d max=%d time=%.3f s\n",
           m, n, score, checksum, diag_sum, minv, maxv, (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(a);
    free(b);
    free(dp);
    return 0;
}
