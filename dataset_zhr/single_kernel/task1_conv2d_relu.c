
#include <stdio.h>
#include <stdlib.h>

#define IN_C 3
#define OUT_C 8
#define H 32
#define W 32
#define K 3
#define OUT_H (H - K + 1)
#define OUT_W (W - K + 1)

static void init_data(float *input, float *kernel, float *bias) {
    for (int i = 0; i < IN_C * H * W; i++) input[i] = (float)((i % 17) - 8) / 17.0f;
    for (int i = 0; i < OUT_C * IN_C * K * K; i++) kernel[i] = (float)((i % 13) - 6) / 13.0f;
    for (int i = 0; i < OUT_C; i++) bias[i] = (float)(i - 4) / 8.0f;
}

void conv2d_relu(const float *input, const float *kernel, const float *bias, float *output) {
    for (int oc = 0; oc < OUT_C; oc++) {
        for (int oh = 0; oh < OUT_H; oh++) {
            for (int ow = 0; ow < OUT_W; ow++) {
                float sum = bias[oc];
                for (int ic = 0; ic < IN_C; ic++) {
                    for (int kh = 0; kh < K; kh++) {
                        for (int kw = 0; kw < K; kw++) {
                            int in_idx = ic * H * W + (oh + kh) * W + (ow + kw);
                            int k_idx = oc * IN_C * K * K + ic * K * K + kh * K + kw;
                            sum += input[in_idx] * kernel[k_idx];
                        }
                    }
                }
                output[oc * OUT_H * OUT_W + oh * OUT_W + ow] = sum > 0.0f ? sum : 0.0f;
            }
        }
    }
}

static float checksum(const float *x, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i] * (float)((i % 7) + 1);
    return s;
}

int main(void) {
    float *input = malloc(sizeof(float) * IN_C * H * W);
    float *kernel = malloc(sizeof(float) * OUT_C * IN_C * K * K);
    float *bias = malloc(sizeof(float) * OUT_C);
    float *output = malloc(sizeof(float) * OUT_C * OUT_H * OUT_W);
    if (!input || !kernel || !bias || !output) return 1;

    init_data(input, kernel, bias);
    conv2d_relu(input, kernel, bias, output);
    printf("Task 1 Conv2D + ReLU checksum: %.6f\n", checksum(output, OUT_C * OUT_H * OUT_W));

    free(input); free(kernel); free(bias); free(output);
    return 0;
}
