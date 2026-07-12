#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define INPUT_H 8
#define INPUT_W 8
#define IN_CHANNELS 1

#define NUM_FILTERS 2
#define KERNEL_SIZE 3
#define CONV_OUT_H (INPUT_H - KERNEL_SIZE + 1)
#define CONV_OUT_W (INPUT_W - KERNEL_SIZE + 1)

#define POOL_SIZE 2
#define POOL_STRIDE 2
#define POOL_OUT_H (CONV_OUT_H / POOL_SIZE)
#define POOL_OUT_W (CONV_OUT_W / POOL_SIZE)

#define FLATTEN_SIZE (NUM_FILTERS * POOL_OUT_H * POOL_OUT_W)
#define NUM_CLASSES 3

static void initialize_input(double input[INPUT_H][INPUT_W])
{
    for (int i = 0; i < INPUT_H; i++) {
        for (int j = 0; j < INPUT_W; j++) {
            input[i][j] = sin(0.1 * (double)(i * INPUT_W + j))
                        + cos(0.2 * (double)(i + j));
        }
    }
}

static void initialize_conv_weights(
    double weights[NUM_FILTERS][KERNEL_SIZE][KERNEL_SIZE],
    double bias[NUM_FILTERS]
)
{
    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int i = 0; i < KERNEL_SIZE; i++) {
            for (int j = 0; j < KERNEL_SIZE; j++) {
                weights[f][i][j] = 0.05 * (double)(f + 1)
                                 * (double)(i + 1)
                                 - 0.03 * (double)(j + 1);
            }
        }
        bias[f] = 0.1 * (double)(f + 1);
    }
}

static void initialize_fc_weights(
    double weights[NUM_CLASSES][FLATTEN_SIZE],
    double bias[NUM_CLASSES]
)
{
    for (int c = 0; c < NUM_CLASSES; c++) {
        for (int i = 0; i < FLATTEN_SIZE; i++) {
            weights[c][i] = 0.01 * (double)(c + 1) * (double)((i % 5) - 2);
        }
        bias[c] = 0.05 * (double)c;
    }
}

static void conv2d(
    const double input[INPUT_H][INPUT_W],
    const double weights[NUM_FILTERS][KERNEL_SIZE][KERNEL_SIZE],
    const double bias[NUM_FILTERS],
    double output[NUM_FILTERS][CONV_OUT_H][CONV_OUT_W]
)
{
    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int i = 0; i < CONV_OUT_H; i++) {
            for (int j = 0; j < CONV_OUT_W; j++) {
                double sum = bias[f];

                for (int ki = 0; ki < KERNEL_SIZE; ki++) {
                    for (int kj = 0; kj < KERNEL_SIZE; kj++) {
                        sum += input[i + ki][j + kj] * weights[f][ki][kj];
                    }
                }

                output[f][i][j] = sum;
            }
        }
    }
}

static void relu(
    double feature_map[NUM_FILTERS][CONV_OUT_H][CONV_OUT_W]
)
{
    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int i = 0; i < CONV_OUT_H; i++) {
            for (int j = 0; j < CONV_OUT_W; j++) {
                if (feature_map[f][i][j] < 0.0) {
                    feature_map[f][i][j] = 0.0;
                }
            }
        }
    }
}

static void max_pool2d(
    const double input[NUM_FILTERS][CONV_OUT_H][CONV_OUT_W],
    double output[NUM_FILTERS][POOL_OUT_H][POOL_OUT_W]
)
{
    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int i = 0; i < POOL_OUT_H; i++) {
            for (int j = 0; j < POOL_OUT_W; j++) {
                double max_value = -DBL_MAX;

                for (int pi = 0; pi < POOL_SIZE; pi++) {
                    for (int pj = 0; pj < POOL_SIZE; pj++) {
                        int input_i = i * POOL_STRIDE + pi;
                        int input_j = j * POOL_STRIDE + pj;

                        if (input[f][input_i][input_j] > max_value) {
                            max_value = input[f][input_i][input_j];
                        }
                    }
                }

                output[f][i][j] = max_value;
            }
        }
    }
}

static void flatten(
    const double input[NUM_FILTERS][POOL_OUT_H][POOL_OUT_W],
    double output[FLATTEN_SIZE]
)
{
    int index = 0;

    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int i = 0; i < POOL_OUT_H; i++) {
            for (int j = 0; j < POOL_OUT_W; j++) {
                output[index] = input[f][i][j];
                index++;
            }
        }
    }
}

static void fully_connected(
    const double input[FLATTEN_SIZE],
    const double weights[NUM_CLASSES][FLATTEN_SIZE],
    const double bias[NUM_CLASSES],
    double output[NUM_CLASSES]
)
{
    for (int c = 0; c < NUM_CLASSES; c++) {
        double sum = bias[c];

        for (int i = 0; i < FLATTEN_SIZE; i++) {
            sum += input[i] * weights[c][i];
        }

        output[c] = sum;
    }
}

static int argmax(const double values[NUM_CLASSES])
{
    int best_index = 0;
    double best_value = values[0];

    for (int i = 1; i < NUM_CLASSES; i++) {
        if (values[i] > best_value) {
            best_value = values[i];
            best_index = i;
        }
    }

    return best_index;
}

static void print_logits(const double logits[NUM_CLASSES])
{
    printf("Logits:\n");
    for (int i = 0; i < NUM_CLASSES; i++) {
        printf("  class %d: %.8f\n", i, logits[i]);
    }
}

int main(int argc, char **argv)
{
    int iterations = 2000000;

    if (argc >= 2) {
        iterations = atoi(argv[1]);
    }

    if (iterations <= 0) {
        fprintf(stderr, "Error: iterations must be positive.\n");
        fprintf(stderr, "Usage: %s [iterations]\n", argv[0]);
        return 1;
    }

    double input[INPUT_H][INPUT_W];
    double conv_weights[NUM_FILTERS][KERNEL_SIZE][KERNEL_SIZE];
    double conv_bias[NUM_FILTERS];
    double conv_out[NUM_FILTERS][CONV_OUT_H][CONV_OUT_W];
    double pool_out[NUM_FILTERS][POOL_OUT_H][POOL_OUT_W];
    double flattened[FLATTEN_SIZE];
    double fc_weights[NUM_CLASSES][FLATTEN_SIZE];
    double fc_bias[NUM_CLASSES];
    double logits[NUM_CLASSES];

    initialize_input(input);
    initialize_conv_weights(conv_weights, conv_bias);
    initialize_fc_weights(fc_weights, fc_bias);

    for (int iter = 0; iter < iterations; iter++) {
        conv2d(input, conv_weights, conv_bias, conv_out);
        relu(conv_out);
        max_pool2d(conv_out, pool_out);
        flatten(pool_out, flattened);
        fully_connected(flattened, fc_weights, fc_bias, logits);
    }

    print_logits(logits);

    int predicted_class = argmax(logits);

    printf("Predicted class: %d\n", predicted_class);

    return 0;
}