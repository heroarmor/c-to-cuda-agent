

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define N_NODES   8192
#define MAX_EDGES (N_NODES * 10)
#define DAMPING   0.85f
#define TOL       1e-6f
#define MAX_ITER  200




typedef struct {
    int  n;           
    int  m;           
    int *row_ptr;     
    int *col_idx;     
    int *out_deg;     
} CSRGraph;


static unsigned lcg(unsigned *s) { *s = *s * 1664525u + 1013904223u; return *s; }

static CSRGraph build_graph(int n, unsigned seed) {
    unsigned s = seed;
    int *tmp_row = calloc(n + 1, sizeof(int));
    
    int *esrc = malloc(sizeof(int) * MAX_EDGES);
    int *edst = malloc(sizeof(int) * MAX_EDGES);
    int m = 0;

    for (int u = 0; u < n && m < MAX_EDGES - n; u++) {
        
        unsigned r = lcg(&s);
        int deg = (r % 100 < 10) ? 0          
                : 1 + (int)((r >> 8) % 15);
        for (int e = 0; e < deg && m < MAX_EDGES - 1; e++) {
            int v = (int)(lcg(&s) % (unsigned)n);
            if (v == u) { v = (v + 1) % n; }
            esrc[m] = u; edst[m] = v; m++;
        }
        tmp_row[u + 1] = deg;
    }

    
    int *row_ptr = calloc(n + 1, sizeof(int));
    for (int u = 0; u < n; u++)
        row_ptr[u + 1] = row_ptr[u] + tmp_row[u + 1];
    int real_m = row_ptr[n];

    
    int *col_idx = malloc(sizeof(int) * (real_m + 1));
    int *pos     = calloc(n, sizeof(int));
    for (int e = 0; e < m; e++) {
        int u = esrc[e];
        if (row_ptr[u] + pos[u] < row_ptr[u + 1]) {
            col_idx[row_ptr[u] + pos[u]] = edst[e];
            pos[u]++;
        }
    }

    int *out_deg = malloc(sizeof(int) * n);
    for (int u = 0; u < n; u++) out_deg[u] = row_ptr[u + 1] - row_ptr[u];

    free(tmp_row); free(esrc); free(edst); free(pos);
    CSRGraph g = { n, real_m, row_ptr, col_idx, out_deg };
    return g;
}

static void free_graph(CSRGraph *g) {
    free(g->row_ptr); free(g->col_idx); free(g->out_deg);
}





typedef struct {
    int  n, m;
    int *col_ptr;   
    int *row_idx;   
} CSCGraph;

static CSCGraph transpose(const CSRGraph *g) {
    int n = g->n, m = g->m;
    int *col_ptr = calloc(n + 1, sizeof(int));
    int *row_idx = malloc(sizeof(int) * m);

    
    for (int e = 0; e < m; e++) col_ptr[g->col_idx[e] + 1]++;
    for (int v = 0; v < n; v++) col_ptr[v + 1] += col_ptr[v];

    
    int *pos = calloc(n, sizeof(int));
    for (int u = 0; u < n; u++) {
        for (int e = g->row_ptr[u]; e < g->row_ptr[u + 1]; e++) {
            int v = g->col_idx[e];
            row_idx[col_ptr[v] + pos[v]] = u;
            pos[v]++;
        }
    }
    free(pos);
    CSCGraph c = { n, m, col_ptr, row_idx };
    return c;
}

static void free_csc(CSCGraph *c) { free(c->col_ptr); free(c->row_idx); }





static float pagerank_iter(const CSCGraph *csc, const int *out_deg,
                            const float *r, float *r_new, int n) {
    
    double dangling_sum = 0.0;
    for (int u = 0; u < n; u++)
        if (out_deg[u] == 0) dangling_sum += r[u];

    float teleport    = (1.0f - DAMPING) / (float)n;
    float dangle_contrib = DAMPING * (float)(dangling_sum / n);

    
    for (int v = 0; v < n; v++) {
        double gather = 0.0;
        
        for (int e = csc->col_ptr[v]; e < csc->col_ptr[v + 1]; e++) {
            int u = csc->row_idx[e];
            gather += r[u] / (double)out_deg[u];
        }
        r_new[v] = teleport + dangle_contrib + DAMPING * (float)gather;
    }

    
    double res = 0.0;
    for (int v = 0; v < n; v++) res += fabsf(r_new[v] - r[v]);
    return (float)res;
}




int main(void) {
    CSRGraph g   = build_graph(N_NODES, 0xCAFEu);
    CSCGraph csc = transpose(&g);

    float *r     = malloc(sizeof(float) * N_NODES);
    float *r_new = malloc(sizeof(float) * N_NODES);
    if (!r || !r_new) return 1;

    
    float init = 1.0f / N_NODES;
    for (int i = 0; i < N_NODES; i++) r[i] = init;

    int   n_dangling = 0;
    for (int i = 0; i < N_NODES; i++) if (g.out_deg[i] == 0) n_dangling++;

    int   iter;
    float res = 0.0f;
    for (iter = 0; iter < MAX_ITER; iter++) {
        res = pagerank_iter(&csc, g.out_deg, r, r_new, N_NODES);
        
        float *tmp = r; r = r_new; r_new = tmp;
        if (res < TOL) break;
    }

    
    int top[5] = {0,1,2,3,4};
    for (int i = 5; i < N_NODES; i++) {
        int min_idx = 0;
        for (int j = 1; j < 5; j++) if (r[top[j]] < r[top[min_idx]]) min_idx = j;
        if (r[i] > r[top[min_idx]]) top[min_idx] = i;
    }
    
    for (int i = 0; i < 4; i++)
        for (int j = i+1; j < 5; j++)
            if (r[top[j]] > r[top[i]]) { int t=top[i]; top[i]=top[j]; top[j]=t; }

    printf("Task 5 PageRank: nodes=%d  edges=%d  dangling=%d  iters=%d  residual=%.2e\n",
           N_NODES, g.m, n_dangling, iter + 1, res);
    printf("Top-5 nodes: ");
    for (int i = 0; i < 5; i++) printf("[%d]=%.6f ", top[i], r[top[i]]);
    printf("\n");

    
    double cs = 0.0;
    for (int i = 0; i < N_NODES; i++) cs += r[i] * (i + 1);
    printf("Task 5 PageRank checksum: %.6f\n", cs);

    free(r); free(r_new);
    free_graph(&g);
    free_csc(&csc);
    return 0;
}
