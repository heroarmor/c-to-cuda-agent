

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

#define N_POINTS    1024
#define INPUT_DIM   16
#define K_NN        10
#define K_EIGEN     8
#define SIGMA       1.0f
#define POWER_ITER  30
#define KMEANS_ITER 40




static unsigned rng_state = 0xABCD1234u;
static unsigned lcg(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}
static float lcg_f(void) {
    return ((float)(lcg() >> 8) / (float)(1 << 24)) * 2.0f - 1.0f;
}





static void build_knn_weight_matrix(const float *X, float *W) {
    float inv2s2 = 1.0f / (2.0f * SIGMA * SIGMA);

    
    int   *knn     = malloc(sizeof(int)   * N_POINTS * K_NN);
    float *knn_d2  = malloc(sizeof(float) * N_POINTS * K_NN);

    
    for (int i = 0; i < N_POINTS * K_NN; i++) {
        knn[i]    = -1;
        knn_d2[i] = FLT_MAX;
    }

    
    for (int i = 0; i < N_POINTS; i++) {
        float *worst    = knn_d2 + i * K_NN;
        int   *nn_list  = knn    + i * K_NN;
        float  max_d2   = FLT_MAX;
        int    max_slot = 0;

        for (int j = 0; j < N_POINTS; j++) {
            if (j == i) continue;
            
            float d2 = 0.0f;
            for (int d = 0; d < INPUT_DIM; d++) {
                float diff = X[i * INPUT_DIM + d] - X[j * INPUT_DIM + d];
                d2 += diff * diff;
            }
            if (d2 < max_d2) {
                nn_list[max_slot] = j;
                worst[max_slot]   = d2;
                
                max_d2   = worst[0]; max_slot = 0;
                for (int k = 1; k < K_NN; k++) {
                    if (worst[k] > max_d2) { max_d2 = worst[k]; max_slot = k; }
                }
            }
        }
    }

    
    memset(W, 0, sizeof(float) * N_POINTS * N_POINTS);
    for (int i = 0; i < N_POINTS; i++) {
        for (int k = 0; k < K_NN; k++) {
            int j = knn[i * K_NN + k];
            if (j < 0) continue;
            float d2 = knn_d2[i * K_NN + k];
            float w  = expf(-d2 * inv2s2);
            W[i * N_POINTS + j] = w;
            W[j * N_POINTS + i] = w;   
        }
    }

    free(knn); free(knn_d2);
}





static void build_laplacian(const float *W, float *L, float *deg_inv_sqrt) {
    
    for (int i = 0; i < N_POINTS; i++) {
        float d = 0.0f;
        for (int j = 0; j < N_POINTS; j++) d += W[i * N_POINTS + j];
        deg_inv_sqrt[i] = (d > 0.0f) ? 1.0f / sqrtf(d) : 0.0f;
    }
    
    for (int i = 0; i < N_POINTS; i++) {
        for (int j = 0; j < N_POINTS; j++) {
            float l = (i == j) ? 1.0f
                               : -W[i * N_POINTS + j]
                                 * deg_inv_sqrt[i] * deg_inv_sqrt[j];
            L[i * N_POINTS + j] = l;
        }
    }
}





static void matmul_NxN_NxK(const float *A, const float *B, float *C,
                             int n, int k) {
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            float s = 0.0f;
            for (int p = 0; p < n; p++) s += A[i * n + p] * B[p * k + j];
            C[i * k + j] = s;
        }
    }
}

static void gram_schmidt(float *V, int n, int k) {
    
    for (int j = 0; j < k; j++) {
        
        for (int pj = 0; pj < j; pj++) {
            float dot = 0.0f;
            for (int i = 0; i < n; i++) dot += V[i * k + pj] * V[i * k + j];
            for (int i = 0; i < n; i++) V[i * k + j] -= dot * V[i * k + pj];
        }
        
        float norm2 = 0.0f;
        for (int i = 0; i < n; i++) norm2 += V[i * k + j] * V[i * k + j];
        float inv_norm = (norm2 > 1e-12f) ? 1.0f / sqrtf(norm2) : 0.0f;
        for (int i = 0; i < n; i++) V[i * k + j] *= inv_norm;
    }
}

static void power_iteration(const float *L, float *V, int n, int k, int iters) {
    
    float *tmp = malloc(sizeof(float) * n * k);

    
    for (int i = 0; i < n * k; i++) V[i] = lcg_f();
    gram_schmidt(V, n, k);

    for (int iter = 0; iter < iters; iter++) {
        
        matmul_NxN_NxK(L, V, tmp, n, k);
        
        for (int i = 0; i < n * k; i++) V[i] = 2.0f * V[i] - tmp[i];
        
        gram_schmidt(V, n, k);
    }

    free(tmp);
}




static void row_normalize(float *U, int n, int k) {
    for (int i = 0; i < n; i++) {
        float norm2 = 0.0f;
        for (int j = 0; j < k; j++) norm2 += U[i * k + j] * U[i * k + j];
        float inv = (norm2 > 1e-12f) ? 1.0f / sqrtf(norm2) : 0.0f;
        for (int j = 0; j < k; j++) U[i * k + j] *= inv;
    }
}





static void kmeans(const float *U, int *labels, int n, int k_clusters, int dim,
                   int max_iter) {
    float *centroids = malloc(sizeof(float) * k_clusters * dim);
    float *min_d2    = malloc(sizeof(float) * n);

    
    int first = (int)(lcg() % (unsigned)n);
    memcpy(centroids, U + first * dim, sizeof(float) * dim);
    for (int i = 0; i < n; i++) min_d2[i] = FLT_MAX;

    for (int c = 1; c < k_clusters; c++) {
        
        for (int i = 0; i < n; i++) {
            float d2 = 0.0f;
            for (int d = 0; d < dim; d++) {
                float diff = U[i*dim+d] - centroids[(c-1)*dim+d];
                d2 += diff * diff;
            }
            if (d2 < min_d2[i]) min_d2[i] = d2;
        }
        
        double total = 0.0;
        for (int i = 0; i < n; i++) total += min_d2[i];
        double thr = ((double)lcg() / (double)0xFFFFFFFFu) * total;
        double running = 0.0;
        int chosen = 0;
        for (int i = 0; i < n; i++) {
            running += min_d2[i];
            if (running >= thr) { chosen = i; break; }
        }
        memcpy(centroids + c * dim, U + chosen * dim, sizeof(float) * dim);
    }
    free(min_d2);

    float *sum   = malloc(sizeof(float) * k_clusters * dim);
    int   *count = malloc(sizeof(int)   * k_clusters);

    for (int iter = 0; iter < max_iter; iter++) {
        
        for (int i = 0; i < n; i++) {
            float best = FLT_MAX; int best_c = 0;
            for (int c = 0; c < k_clusters; c++) {
                float d2 = 0.0f;
                for (int d = 0; d < dim; d++) {
                    float diff = U[i*dim+d] - centroids[c*dim+d];
                    d2 += diff * diff;
                }
                if (d2 < best) { best = d2; best_c = c; }
            }
            labels[i] = best_c;
        }
        
        memset(sum,   0, sizeof(float) * k_clusters * dim);
        memset(count, 0, sizeof(int)   * k_clusters);
        for (int i = 0; i < n; i++) {
            int c = labels[i]; count[c]++;
            for (int d = 0; d < dim; d++)
                sum[c*dim+d] += U[i*dim+d];
        }
        for (int c = 0; c < k_clusters; c++) {
            if (count[c] > 0)
                for (int d = 0; d < dim; d++)
                    centroids[c*dim+d] = sum[c*dim+d] / (float)count[c];
        }
    }

    free(centroids); free(sum); free(count);
}




int main(void) {
    
    float *X   = malloc(sizeof(float) * N_POINTS * INPUT_DIM);
    float *W   = malloc(sizeof(float) * N_POINTS * N_POINTS);
    float *L   = malloc(sizeof(float) * N_POINTS * N_POINTS);
    float *dis = malloc(sizeof(float) * N_POINTS);   
    float *V   = malloc(sizeof(float) * N_POINTS * K_EIGEN);
    int   *lbl = malloc(sizeof(int)   * N_POINTS);

    if (!X || !W || !L || !dis || !V || !lbl) return 1;

    
    for (int i = 0; i < N_POINTS * INPUT_DIM; i++) X[i] = lcg_f();
    
    float centers[K_EIGEN * INPUT_DIM];
    for (int i = 0; i < K_EIGEN * INPUT_DIM; i++) centers[i] = lcg_f() * 3.0f;
    for (int i = 0; i < N_POINTS; i++) {
        int c = (int)(lcg() % K_EIGEN);
        for (int d = 0; d < INPUT_DIM; d++)
            X[i * INPUT_DIM + d] = centers[c * INPUT_DIM + d]
                                 + lcg_f() * 0.3f;
    }

    printf("Task 6 Spectral Clustering: building k-NN graph (%d pts, dim=%d, k=%d)...\n",
           N_POINTS, INPUT_DIM, K_NN);

    
    build_knn_weight_matrix(X, W);

    
    build_laplacian(W, L, dis);

    
    printf("  Running power iteration (K_eigen=%d, iters=%d)...\n",
           K_EIGEN, POWER_ITER);
    power_iteration(L, V, N_POINTS, K_EIGEN, POWER_ITER);

    
    row_normalize(V, N_POINTS, K_EIGEN);

    
    printf("  Running K-Means on spectral embedding (k=%d, iters=%d)...\n",
           K_EIGEN, KMEANS_ITER);
    kmeans(V, lbl, N_POINTS, K_EIGEN, K_EIGEN, KMEANS_ITER);

    
    int hist[K_EIGEN]; memset(hist, 0, sizeof(hist));
    for (int i = 0; i < N_POINTS; i++) hist[lbl[i]]++;
    printf("  Cluster sizes:");
    for (int c = 0; c < K_EIGEN; c++) printf(" %d", hist[c]);
    printf("\n");

    
    double cs = 0.0;
    for (int i = 0; i < N_POINTS * K_EIGEN; i++) cs += V[i] * (i % 13 + 1);
    printf("Task 6 Spectral Clustering eigenvector checksum: %.6f\n", cs);

    
    long long lcs = 0;
    for (int i = 0; i < N_POINTS; i++) lcs += (long long)lbl[i] * (i + 1);
    printf("Task 6 Spectral Clustering label checksum: %lld\n", lcs);

    free(X); free(W); free(L); free(dis); free(V); free(lbl);
    return 0;
}
