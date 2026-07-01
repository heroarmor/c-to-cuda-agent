
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define T 32
#define INPUT_SIZE 64
#define HIDDEN_SIZE 128

static void init_data(float *x, float *Wxh, float *Whh, float *b, float *h0) {
    for (int i = 0; i < T * INPUT_SIZE; i++) x[i] = (float)((i % 19) - 9) / 19.0f;
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) Wxh[i] = (float)((i % 23) - 11) / 80.0f;
    for (int i = 0; i < HIDDEN_SIZE * HIDDEN_SIZE; i++) Whh[i] = (float)((i % 29) - 14) / 120.0f;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        b[i] = (float)((i % 11) - 5) / 20.0f;
        h0[i] = (float)((i % 7) - 3) / 30.0f;
    }
}

void rnn_forward(const float *x, const float *Wxh, const float *Whh,
                 const float *b, const float *h0, float *h_all) {
    float h_prev[HIDDEN_SIZE];

    for (int j = 0; j < HIDDEN_SIZE; j++) h_prev[j] = h0[j];

    for (int t = 0; t < T; t++) {
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            float sum = b[h];

            for (int i = 0; i < INPUT_SIZE; i++)
                sum += x[t * INPUT_SIZE + i] * Wxh[i * HIDDEN_SIZE + h];

            for (int j = 0; j < HIDDEN_SIZE; j++)
                sum += h_prev[j] * Whh[j * HIDDEN_SIZE + h];

            h_all[t * HIDDEN_SIZE + h] = tanhf(sum);
        }

        for (int h = 0; h < HIDDEN_SIZE; h++) h_prev[h] = h_all[t * HIDDEN_SIZE + h];
    }
}

static float checksum(const float *x, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i] * (float)((i % 5) + 1);
    return s;
}

int main(void) {
    float *x = malloc(sizeof(float) * T * INPUT_SIZE);
    float *Wxh = malloc(sizeof(float) * INPUT_SIZE * HIDDEN_SIZE);
    float *Whh = malloc(sizeof(float) * HIDDEN_SIZE * HIDDEN_SIZE);
    float *b = malloc(sizeof(float) * HIDDEN_SIZE);
    float *h0 = malloc(sizeof(float) * HIDDEN_SIZE);
    float *h_all = malloc(sizeof(float) * T * HIDDEN_SIZE);
    if (!x || !Wxh || !Whh || !b || !h0 || !h_all) return 1;

    init_data(x, Wxh, Whh, b, h0);
    rnn_forward(x, Wxh, Whh, b, h0, h_all);
    printf("Task 2 Simple RNN checksum: %.6f\n", checksum(h_all, T * HIDDEN_SIZE));

    free(x); free(Wxh); free(Whh); free(b); free(h0); free(h_all);
    return 0;
}
