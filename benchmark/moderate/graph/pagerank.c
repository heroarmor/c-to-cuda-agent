/* pagerank.c -- PageRank power iteration on a synthetic CSR graph
 *
 * Generates a deterministic directed graph, then performs a fixed number of
 * PageRank iterations with damping and dangling-node redistribution.
 *
 * Pattern: graph analytics on CSR with sparse gather/scatter and reductions.
 * GPU conversion: one kernel to clear ranks, one edge relaxation kernel, and a
 * reduction for dangling rank per iteration.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int edge_dst(int v, int k, int n) {
    unsigned x = (unsigned)(v + 1) * 2654435761U + (unsigned)(k + 17) * 2246822519U;
    x ^= x >> 13;
    int hop = 1 + (int)(x % (unsigned)(n - 1));
    return (v + hop) % n;
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 20000;
    int iters = (argc > 2) ? atoi(argv[2]) : 30;
    int outdeg = (argc > 3) ? atoi(argv[3]) : 8;
    if (n < 2 || iters < 1 || outdeg < 0) {
        fprintf(stderr, "usage: %s [nodes>=2] [iters>=1] [outdeg>=0]\n", argv[0]);
        return 2;
    }
    if (outdeg > n - 1) outdeg = n - 1;

    int *row = (int *)malloc((size_t)(n + 1) * sizeof(int));
    double *rank = (double *)malloc((size_t)n * sizeof(double));
    double *next = (double *)malloc((size_t)n * sizeof(double));
    if (!row || !rank || !next) { fprintf(stderr, "alloc failed\n"); return 1; }

    row[0] = 0;
    for (int v = 0; v < n; ++v) {
        int d = ((v % 19) == 0) ? 0 : outdeg;
        row[v + 1] = row[v] + d;
    }
    int m = row[n];
    int *col = (int *)malloc((size_t)m * sizeof(int));
    if (m && !col) { fprintf(stderr, "alloc failed\n"); return 1; }
    for (int v = 0; v < n; ++v)
        for (int e = row[v], k = 0; e < row[v + 1]; ++e, ++k)
            col[e] = edge_dst(v, k, n);

    for (int i = 0; i < n; ++i) rank[i] = 1.0 / n;

    clock_t t0 = clock();
    const double damp = 0.85;
    for (int it = 0; it < iters; ++it) {
        double dangling = 0.0;
        for (int v = 0; v < n; ++v)
            if (row[v] == row[v + 1]) dangling += rank[v];
        double base = (1.0 - damp) / n + damp * dangling / n;
        for (int i = 0; i < n; ++i) next[i] = base;
        for (int v = 0; v < n; ++v) {
            int d = row[v + 1] - row[v];
            if (d == 0) continue;
            double contrib = damp * rank[v] / d;
            for (int e = row[v]; e < row[v + 1]; ++e)
                next[col[e]] += contrib;
        }
        double *tmp = rank;
        rank = next;
        next = tmp;
    }
    clock_t t1 = clock();

    double sum = 0.0, l2 = 0.0, maxr = 0.0;
    long long checksum = 0;
    for (int i = 0; i < n; ++i) {
        sum += rank[i];
        l2 += rank[i] * rank[i];
        if (rank[i] > maxr) maxr = rank[i];
        checksum += (long long)llround(rank[i] * 1000000000000.0) * (long long)((i % 131) + 1);
    }

    printf("pagerank nodes=%d edges=%d iters=%d checksum=%lld sum=%.12f l2=%.12f max=%.12f time=%.3f s\n",
           n, m, iters, checksum, sum, l2, maxr, (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(row);
    free(col);
    free(rank);
    free(next);
    return 0;
}
