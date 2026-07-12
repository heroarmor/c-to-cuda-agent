
#include <stdio.h>
#include <stdlib.h>

#define BATCH 64
#define IN_SIZE 128
#define HIDDEN_SIZE 256
#define OUT_SIZE 10

static void init_data(float *x, float *w1, float *b1, float *w2, float *b2) {
    for (int i = 0; i < BATCH * IN_SIZE; i++) x[i] = (float)((i % 31) - 15) / 31.0f;
    for (int i = 0; i < IN_SIZE * HIDDEN_SIZE; i++) w1[i] = (float)((i % 17) - 8) / 64.0f;
    for (int i = 0; i < HIDDEN_SIZE; i++) b1[i] = (float)((i % 13) - 6) / 32.0f;
    for (int i = 0; i < HIDDEN_SIZE * OUT_SIZE; i++) w2[i] = (float)((i % 23) - 11) / 80.0f;
    for (int i = 0; i < OUT_SIZE; i++) b2[i] = (float)(i - 5) / 16.0f;
}

void dense_relu(const float *x, const float *w, const float *b, float *y,
                int batch, int in_size, int out_size) {
    for (int n = 0; n < batch; n++) {
        for (int o = 0; o < out_size; o++) {
            float sum = b[o];

            for (int i = 0; i < in_size; i++)
                sum += x[n * in_size + i] * w[i * out_size + o];

            y[n * out_size + o] = sum > 0.0f ? sum : 0.0f;
        }
    }
}

void dense_linear(const float *x, const float *w, const float *b, float *y,
                  int batch, int in_size, int out_size) {
    for (int n = 0; n < batch; n++) {
        for (int o = 0; o < out_size; o++) {
            float sum = b[o];

            for (int i = 0; i < in_size; i++)
                sum += x[n * in_size + i] * w[i * out_size + o];

            y[n * out_size + o] = sum;
        }
    }
}

void mlp_forward(const float *x, const float *w1, const float *b1,
                 const float *w2, const float *b2, float *hidden, float *logits) {
    dense_relu(x, w1, b1, hidden, BATCH, IN_SIZE, HIDDEN_SIZE);
    dense_linear(hidden, w2, b2, logits, BATCH, HIDDEN_SIZE, OUT_SIZE);
}

static float checksum(const float *x, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i] * (float)((i % 3) + 1);
    return s;
}

int main(void) {
    float *x = malloc(sizeof(float) * BATCH * IN_SIZE);
    float *w1 = malloc(sizeof(float) * IN_SIZE * HIDDEN_SIZE);
    float *b1 = malloc(sizeof(float) * HIDDEN_SIZE);
    float *w2 = malloc(sizeof(float) * HIDDEN_SIZE * OUT_SIZE);
    float *b2 = malloc(sizeof(float) * OUT_SIZE);
    float *hidden = malloc(sizeof(float) * BATCH * HIDDEN_SIZE);
    float *logits = malloc(sizeof(float) * BATCH * OUT_SIZE);
    if (!x || !w1 || !b1 || !w2 || !b2 || !hidden || !logits) return 1;

    init_data(x, w1, b1, w2, b2);
    mlp_forward(x, w1, b1, w2, b2, hidden, logits);
    printf("Task 3 Two-Layer MLP checksum: %.6f\n", checksum(logits, BATCH * OUT_SIZE));

    free(x); free(w1); free(b1); free(w2); free(b2); free(hidden); free(logits);
    return 0;
}
