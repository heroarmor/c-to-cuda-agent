

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

#define BATCH    2
#define SEQ_LEN  32
#define D_MODEL  128
#define N_HEADS  4
#define D_HEAD   (D_MODEL / N_HEADS)   
#define D_FF     256





static unsigned lcg_state = 0xdeadbeef;
static float lcg_float(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((float)(lcg_state >> 8) / (float)(1 << 24)) * 2.0f - 1.0f;
}

static void fill_rand(float *a, int n, float scale) {
    for (int i = 0; i < n; i++) a[i] = lcg_float() * scale;
}





static void layer_norm(const float *in, float *out,
                        const float *gamma, const float *beta,
                        int n_rows, int d) {
    const float eps = 1e-5f;
    for (int r = 0; r < n_rows; r++) {
        const float *row = in  + r * d;
        float       *dst = out + r * d;

        
        double mu = 0.0;
        for (int i = 0; i < d; i++) mu += row[i];
        mu /= d;

        
        double var = 0.0;
        for (int i = 0; i < d; i++) {
            double diff = row[i] - mu;
            var += diff * diff;
        }
        var /= d;
        float inv_std = 1.0f / sqrtf((float)var + eps);

        
        for (int i = 0; i < d; i++)
            dst[i] = gamma[i] * ((row[i] - (float)mu) * inv_std) + beta[i];
    }
}




static void linear(const float *in, const float *W, const float *b,
                   float *out, int rows, int d_in, int d_out) {
    for (int r = 0; r < rows; r++) {
        for (int o = 0; o < d_out; o++) {
            float sum = b ? b[o] : 0.0f;
            for (int i = 0; i < d_in; i++)
                sum += in[r * d_in + i] * W[i * d_out + o];
            out[r * d_out + o] = sum;
        }
    }
}





static void gelu_inplace(float *x, int n) {
    const float c0 = 0.7978845608f;   
    const float c1 = 0.044715f;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float inner = c0 * (v + c1 * v * v * v);
        x[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}





static void causal_attention(const float *Q, const float *K, const float *V,
                              float *ctx, int n_heads, int seq, int dh) {
    float scale = 1.0f / sqrtf((float)dh);
    float *scores = malloc(sizeof(float) * n_heads * seq * seq);

    
    for (int h = 0; h < n_heads; h++) {
        for (int q = 0; q < seq; q++) {
            for (int k = 0; k < seq; k++) {
                float dot = 0.0f;
                for (int d = 0; d < dh; d++) {
                    dot += Q[h * seq * dh + q * dh + d]
                         * K[h * seq * dh + k * dh + d];
                }
                
                float masked = (k <= q) ? dot * scale : -FLT_MAX;
                scores[h * seq * seq + q * seq + k] = masked;
            }

            
            float *row = scores + h * seq * seq + q * seq;
            float mx = -FLT_MAX;
            for (int k = 0; k <= q; k++) if (row[k] > mx) mx = row[k];
            float sum = 0.0f;
            for (int k = 0; k <= q; k++) { row[k] = expf(row[k] - mx); sum += row[k]; }
            for (int k = 0; k <= q; k++) row[k] /= sum;
            for (int k = q + 1; k < seq; k++) row[k] = 0.0f;
        }
    }

    
    memset(ctx, 0, sizeof(float) * n_heads * seq * dh);
    for (int h = 0; h < n_heads; h++) {
        for (int q = 0; q < seq; q++) {
            for (int k = 0; k <= q; k++) {
                float w = scores[h * seq * seq + q * seq + k];
                for (int d = 0; d < dh; d++) {
                    ctx[h * seq * dh + q * dh + d]
                        += w * V[h * seq * dh + k * dh + d];
                }
            }
        }
    }

    free(scores);
}





static void mha_block(const float *in, float *out,
                       const float *Wq, const float *Wk, const float *Wv,
                       const float *Wo) {
    int bs = BATCH * SEQ_LEN;

    float *Q_flat = malloc(sizeof(float) * bs * D_MODEL);
    float *K_flat = malloc(sizeof(float) * bs * D_MODEL);
    float *V_flat = malloc(sizeof(float) * bs * D_MODEL);
    float *ctx_flat = malloc(sizeof(float) * bs * D_MODEL);

    
    linear(in, Wq, NULL, Q_flat, bs, D_MODEL, D_MODEL);
    linear(in, Wk, NULL, K_flat, bs, D_MODEL, D_MODEL);
    linear(in, Wv, NULL, V_flat, bs, D_MODEL, D_MODEL);

    
    for (int b = 0; b < BATCH; b++) {
        
        float *Qb = malloc(sizeof(float) * N_HEADS * SEQ_LEN * D_HEAD);
        float *Kb = malloc(sizeof(float) * N_HEADS * SEQ_LEN * D_HEAD);
        float *Vb = malloc(sizeof(float) * N_HEADS * SEQ_LEN * D_HEAD);
        float *Cb = malloc(sizeof(float) * N_HEADS * SEQ_LEN * D_HEAD);

        for (int s = 0; s < SEQ_LEN; s++) {
            for (int h = 0; h < N_HEADS; h++) {
                for (int d = 0; d < D_HEAD; d++) {
                    int src = (b * SEQ_LEN + s) * D_MODEL + h * D_HEAD + d;
                    int dst = h * SEQ_LEN * D_HEAD + s * D_HEAD + d;
                    Qb[dst] = Q_flat[src];
                    Kb[dst] = K_flat[src];
                    Vb[dst] = V_flat[src];
                }
            }
        }

        causal_attention(Qb, Kb, Vb, Cb, N_HEADS, SEQ_LEN, D_HEAD);

        
        for (int s = 0; s < SEQ_LEN; s++) {
            for (int h = 0; h < N_HEADS; h++) {
                for (int d = 0; d < D_HEAD; d++) {
                    int src = h * SEQ_LEN * D_HEAD + s * D_HEAD + d;
                    int dst = (b * SEQ_LEN + s) * D_MODEL + h * D_HEAD + d;
                    ctx_flat[dst] = Cb[src];
                }
            }
        }

        free(Qb); free(Kb); free(Vb); free(Cb);
    }

    
    linear(ctx_flat, Wo, NULL, out, bs, D_MODEL, D_MODEL);

    free(Q_flat); free(K_flat); free(V_flat); free(ctx_flat);
}





static void ffn_block(const float *in, float *out,
                       const float *W1, const float *b1,
                       const float *W2, const float *b2) {
    int rows = BATCH * SEQ_LEN;
    float *hidden = malloc(sizeof(float) * rows * D_FF);

    linear(in, W1, b1, hidden, rows, D_MODEL, D_FF);
    gelu_inplace(hidden, rows * D_FF);
    linear(hidden, W2, b2, out, rows, D_FF, D_MODEL);

    free(hidden);
}




static void transformer_encoder_layer(
        const float *x,   
        float       *out, 
        
        const float *ln1_gamma, const float *ln1_beta,
        const float *ln2_gamma, const float *ln2_beta,
        
        const float *Wq, const float *Wk, const float *Wv, const float *Wo,
        
        const float *W1, const float *b1, const float *W2, const float *b2) {

    int total = BATCH * SEQ_LEN;
    float *normed1  = malloc(sizeof(float) * total * D_MODEL);
    float *attn_out = malloc(sizeof(float) * total * D_MODEL);
    float *resid1   = malloc(sizeof(float) * total * D_MODEL);
    float *normed2  = malloc(sizeof(float) * total * D_MODEL);
    float *ffn_out  = malloc(sizeof(float) * total * D_MODEL);

    
    layer_norm(x, normed1, ln1_gamma, ln1_beta, total, D_MODEL);
    mha_block(normed1, attn_out, Wq, Wk, Wv, Wo);
    for (int i = 0; i < total * D_MODEL; i++)
        resid1[i] = x[i] + attn_out[i];

    
    layer_norm(resid1, normed2, ln2_gamma, ln2_beta, total, D_MODEL);
    ffn_block(normed2, ffn_out, W1, b1, W2, b2);
    for (int i = 0; i < total * D_MODEL; i++)
        out[i] = resid1[i] + ffn_out[i];

    free(normed1); free(attn_out); free(resid1); free(normed2); free(ffn_out);
}




static float checksum(const float *x, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i] * (float)((i % 7) + 1);
    return s;
}

int main(void) {
    int total = BATCH * SEQ_LEN;
    float sc = 0.02f;   

    float *x         = malloc(sizeof(float) * total * D_MODEL);
    float *out        = malloc(sizeof(float) * total * D_MODEL);
    float *ln1_gamma  = malloc(sizeof(float) * D_MODEL);
    float *ln1_beta   = malloc(sizeof(float) * D_MODEL);
    float *ln2_gamma  = malloc(sizeof(float) * D_MODEL);
    float *ln2_beta   = malloc(sizeof(float) * D_MODEL);
    float *Wq         = malloc(sizeof(float) * D_MODEL * D_MODEL);
    float *Wk         = malloc(sizeof(float) * D_MODEL * D_MODEL);
    float *Wv         = malloc(sizeof(float) * D_MODEL * D_MODEL);
    float *Wo         = malloc(sizeof(float) * D_MODEL * D_MODEL);
    float *W1         = malloc(sizeof(float) * D_MODEL * D_FF);
    float *b1         = malloc(sizeof(float) * D_FF);
    float *W2         = malloc(sizeof(float) * D_FF * D_MODEL);
    float *b2         = malloc(sizeof(float) * D_MODEL);

    if (!x||!out||!ln1_gamma||!ln1_beta||!ln2_gamma||!ln2_beta||
        !Wq||!Wk||!Wv||!Wo||!W1||!b1||!W2||!b2) return 1;

    fill_rand(x,  total * D_MODEL, 1.0f);
    fill_rand(Wq, D_MODEL*D_MODEL, sc);
    fill_rand(Wk, D_MODEL*D_MODEL, sc);
    fill_rand(Wv, D_MODEL*D_MODEL, sc);
    fill_rand(Wo, D_MODEL*D_MODEL, sc);
    fill_rand(W1, D_MODEL*D_FF,    sc);
    fill_rand(b1, D_FF,            0.0f);
    fill_rand(W2, D_FF*D_MODEL,    sc);
    fill_rand(b2, D_MODEL,         0.0f);

    
    for (int i = 0; i < D_MODEL; i++) {
        ln1_gamma[i] = ln2_gamma[i] = 1.0f;
        ln1_beta[i]  = ln2_beta[i]  = 0.0f;
    }

    transformer_encoder_layer(x, out,
        ln1_gamma, ln1_beta, ln2_gamma, ln2_beta,
        Wq, Wk, Wv, Wo, W1, b1, W2, b2);

    printf("Task 4 Transformer Encoder Layer checksum: %.6f\n",
           checksum(out, total * D_MODEL));

    free(x); free(out);
    free(ln1_gamma); free(ln1_beta); free(ln2_gamma); free(ln2_beta);
    free(Wq); free(Wk); free(Wv); free(Wo);
    free(W1); free(b1); free(W2); free(b2);
    return 0;
}
