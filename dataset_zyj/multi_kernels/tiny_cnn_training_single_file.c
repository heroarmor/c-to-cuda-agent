#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define IMAGE_H 28
#define IMAGE_W 28
#define IMAGE_C 1
#define NUM_CLASSES 2

#define DEFAULT_NUM_SAMPLES 128
#define DEFAULT_EPOCHS 15
#define DEFAULT_LEARNING_RATE 0.01f

typedef struct {
    int n, c, h, w, size, requires_grad;
    float *data;
    float *grad;
} Tensor;

typedef struct {
    int in_channels, out_channels, kernel_size, stride, padding;
    Tensor *weights;
    Tensor *bias;
} Conv2D;

typedef enum { LAYER_CONV = 0, LAYER_RELU = 1, LAYER_SOFTMAX = 2 } LayerType;

typedef struct {
    LayerType type;
    void *layer_data;
    Tensor *input_cache;
    Tensor *output_cache;
} Layer;

typedef struct { Layer *layers; int num_layers; } Model;

typedef struct { Tensor **images; int *labels; int size; } Dataset;

static void fail_if_null(void *ptr, const char *message) {
    if (ptr == NULL) { fprintf(stderr, "%s\n", message); exit(1); }
}

static Tensor *tensor_create(int n, int c, int h, int w, int requires_grad) {
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    fail_if_null(t, "Error: failed to allocate tensor.");
    t->n = n; t->c = c; t->h = h; t->w = w;
    t->size = n * c * h * w;
    t->requires_grad = requires_grad;
    t->data = (float *)calloc((size_t)t->size, sizeof(float));
    fail_if_null(t->data, "Error: failed to allocate tensor data.");
    t->grad = requires_grad ? (float *)calloc((size_t)t->size, sizeof(float)) : NULL;
    if (requires_grad) fail_if_null(t->grad, "Error: failed to allocate tensor gradients.");
    return t;
}

static void tensor_free(Tensor *t) { if (t) { free(t->data); free(t->grad); free(t); } }

static void tensor_zero_grad(Tensor *t) {
    if (t && t->grad) for (int i = 0; i < t->size; i++) t->grad[i] = 0.0f;
}

static void tensor_randomize(Tensor *t, float scale) {
    for (int i = 0; i < t->size; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        t->data[i] = (2.0f * r - 1.0f) * scale;
    }
}

static Conv2D *conv2d_create(int in_c, int out_c, int k, int s, int p) {
    Conv2D *layer = (Conv2D *)malloc(sizeof(Conv2D));
    fail_if_null(layer, "Error: failed to allocate Conv2D layer.");
    layer->in_channels = in_c; layer->out_channels = out_c;
    layer->kernel_size = k; layer->stride = s; layer->padding = p;
    layer->weights = tensor_create(out_c, in_c, k, k, 1);
    layer->bias = tensor_create(1, out_c, 1, 1, 1);
    tensor_randomize(layer->weights, 1.0f / (float)(in_c * k * k));
    return layer;
}

static void conv2d_free(Conv2D *layer) {
    if (layer) { tensor_free(layer->weights); tensor_free(layer->bias); free(layer); }
}

static void conv2d_forward(Conv2D *layer, Tensor *input, Tensor *output) {
    int N = input->n, H_in = input->h, W_in = input->w;
    int H_out = output->h, W_out = output->w;
    int K = layer->kernel_size, S = layer->stride, P = layer->padding;
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < layer->out_channels; oc++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    float sum = layer->bias->data[oc];
                    for (int ic = 0; ic < layer->in_channels; ic++) {
                        for (int kh = 0; kh < K; kh++) {
                            for (int kw = 0; kw < K; kw++) {
                                int ih = oh * S + kh - P;
                                int iw = ow * S + kw - P;
                                if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                                    int in_idx = ((n * layer->in_channels + ic) * H_in + ih) * W_in + iw;
                                    int w_idx = ((oc * layer->in_channels + ic) * K + kh) * K + kw;
                                    sum += input->data[in_idx] * layer->weights->data[w_idx];
                                }
                            }
                        }
                    }
                    int out_idx = ((n * layer->out_channels + oc) * H_out + oh) * W_out + ow;
                    output->data[out_idx] = sum;
                }
            }
        }
    }
}

static void conv2d_backward(Conv2D *layer, Tensor *input, Tensor *output) {
    int N = input->n, H_in = input->h, W_in = input->w;
    int H_out = output->h, W_out = output->w;
    int K = layer->kernel_size, S = layer->stride, P = layer->padding;
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < layer->out_channels; oc++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    int out_idx = ((n * layer->out_channels + oc) * H_out + oh) * W_out + ow;
                    float dout = output->grad[out_idx];
                    layer->bias->grad[oc] += dout;
                    for (int ic = 0; ic < layer->in_channels; ic++) {
                        for (int kh = 0; kh < K; kh++) {
                            for (int kw = 0; kw < K; kw++) {
                                int ih = oh * S + kh - P;
                                int iw = ow * S + kw - P;
                                if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                                    int in_idx = ((n * layer->in_channels + ic) * H_in + ih) * W_in + iw;
                                    int w_idx = ((oc * layer->in_channels + ic) * K + kh) * K + kw;
                                    layer->weights->grad[w_idx] += input->data[in_idx] * dout;
                                    if (input->requires_grad && input->grad) {
                                        input->grad[in_idx] += layer->weights->data[w_idx] * dout;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

static void relu_forward(Tensor *input, Tensor *output) {
    for (int i = 0; i < input->size; i++) output->data[i] = input->data[i] > 0.0f ? input->data[i] : 0.0f;
}

static void relu_backward(Tensor *input, Tensor *output) {
    if (!input->grad) return;
    for (int i = 0; i < input->size; i++) input->grad[i] += (input->data[i] > 0.0f ? 1.0f : 0.0f) * output->grad[i];
}

static void softmax_forward(Tensor *input, Tensor *output) {
    int N = input->n;
    int C = input->c * input->h * input->w;
    for (int n = 0; n < N; n++) {
        float max_value = input->data[n * C];
        for (int c = 1; c < C; c++) if (input->data[n * C + c] > max_value) max_value = input->data[n * C + c];
        float sum = 0.0f;
        for (int c = 0; c < C; c++) { output->data[n * C + c] = expf(input->data[n * C + c] - max_value); sum += output->data[n * C + c]; }
        for (int c = 0; c < C; c++) output->data[n * C + c] /= sum;
    }
}

static void softmax_backward(Tensor *input, Tensor *output) {
    int N = input->n;
    int C = input->c * input->h * input->w;
    if (!input->grad) return;
    for (int n = 0; n < N; n++) {
        for (int j = 0; j < C; j++) {
            float sum = 0.0f;
            for (int i = 0; i < C; i++) {
                float y_i = output->data[n * C + i];
                float y_j = output->data[n * C + j];
                float dp_dz = y_i * ((i == j ? 1.0f : 0.0f) - y_j);
                sum += output->grad[n * C + i] * dp_dz;
            }
            input->grad[n * C + j] += sum;
        }
    }
}

static float cross_entropy_loss(Tensor *pred, int *target) {
    int N = pred->n;
    int C = pred->c * pred->h * pred->w;
    float total_loss = 0.0f;
    for (int n = 0; n < N; n++) {
        int label = target[n];
        float value = pred->data[n * C + label];
        if (value < 1e-7f) value = 1e-7f;
        total_loss += -logf(value);
    }
    return total_loss / (float)N;
}

static void cross_entropy_backward(Tensor *pred, int *target) {
    int N = pred->n;
    int C = pred->c * pred->h * pred->w;
    for (int i = 0; i < pred->size; i++) pred->grad[i] = 0.0f;
    for (int n = 0; n < N; n++) {
        int label = target[n];
        for (int c = 0; c < C; c++) {
            if (c == label) {
                float value = pred->data[n * C + c];
                if (value < 1e-7f) value = 1e-7f;
                pred->grad[n * C + c] = -1.0f / value;
            }
        }
    }
    for (int i = 0; i < pred->size; i++) pred->grad[i] /= (float)N;
}

static void sgd_step(Tensor *weights, Tensor *bias, float learning_rate) {
    if (weights) for (int i = 0; i < weights->size; i++) weights->data[i] -= learning_rate * weights->grad[i];
    if (bias) for (int i = 0; i < bias->size; i++) bias->data[i] -= learning_rate * bias->grad[i];
}

static Model *model_create(void) {
    Model *model = (Model *)malloc(sizeof(Model));
    fail_if_null(model, "Error: failed to allocate model.");
    model->layers = NULL; model->num_layers = 0;
    return model;
}

static void model_add_layer(Model *model, Layer layer) {
    model->num_layers++;
    model->layers = (Layer *)realloc(model->layers, (size_t)model->num_layers * sizeof(Layer));
    fail_if_null(model->layers, "Error: failed to reallocate model layers.");
    model->layers[model->num_layers - 1] = layer;
}

static void model_add_conv(Model *model, int in_c, int out_c, int k, int s, int p) {
    Layer layer;
    layer.type = LAYER_CONV; layer.layer_data = conv2d_create(in_c, out_c, k, s, p);
    layer.input_cache = NULL; layer.output_cache = NULL;
    model_add_layer(model, layer);
}

static void model_add_relu(Model *model) {
    Layer layer;
    layer.type = LAYER_RELU; layer.layer_data = NULL; layer.input_cache = NULL; layer.output_cache = NULL;
    model_add_layer(model, layer);
}

static void model_add_softmax(Model *model) {
    Layer layer;
    layer.type = LAYER_SOFTMAX; layer.layer_data = NULL; layer.input_cache = NULL; layer.output_cache = NULL;
    model_add_layer(model, layer);
}

static void zero_all_cached_gradients(Model *model, Tensor *input) {
    tensor_zero_grad(input);
    for (int i = 0; i < model->num_layers; i++) if (model->layers[i].output_cache) tensor_zero_grad(model->layers[i].output_cache);
}

static void model_forward(Model *model, Tensor *input) {
    Tensor *current = input;
    for (int i = 0; i < model->num_layers; i++) {
        Layer *layer = &model->layers[i];
        layer->input_cache = current;
        int n = current->n, c = current->c, h = current->h, w = current->w;
        if (layer->type == LAYER_CONV) {
            Conv2D *conv = (Conv2D *)layer->layer_data;
            int h_out = (h + 2 * conv->padding - conv->kernel_size) / conv->stride + 1;
            int w_out = (w + 2 * conv->padding - conv->kernel_size) / conv->stride + 1;
            if (layer->output_cache) tensor_free(layer->output_cache);
            layer->output_cache = tensor_create(n, conv->out_channels, h_out, w_out, 1);
            conv2d_forward(conv, current, layer->output_cache);
        } else if (layer->type == LAYER_RELU) {
            if (layer->output_cache) tensor_free(layer->output_cache);
            layer->output_cache = tensor_create(n, c, h, w, 1);
            relu_forward(current, layer->output_cache);
        } else if (layer->type == LAYER_SOFTMAX) {
            if (layer->output_cache) tensor_free(layer->output_cache);
            layer->output_cache = tensor_create(n, c, h, w, 1);
            softmax_forward(current, layer->output_cache);
        }
        current = layer->output_cache;
    }
}

static void model_backward(Model *model) {
    for (int i = model->num_layers - 1; i >= 0; i--) {
        Layer *layer = &model->layers[i];
        Tensor *grad_output = layer->output_cache;
        Tensor *grad_input = layer->input_cache;
        if (layer->type == LAYER_CONV) conv2d_backward((Conv2D *)layer->layer_data, grad_input, grad_output);
        else if (layer->type == LAYER_RELU) relu_backward(grad_input, grad_output);
        else if (layer->type == LAYER_SOFTMAX) softmax_backward(grad_input, grad_output);
    }
}

static void model_step(Model *model, float learning_rate) {
    for (int i = 0; i < model->num_layers; i++) {
        if (model->layers[i].type == LAYER_CONV) {
            Conv2D *conv = (Conv2D *)model->layers[i].layer_data;
            sgd_step(conv->weights, conv->bias, learning_rate);
            tensor_zero_grad(conv->weights);
            tensor_zero_grad(conv->bias);
        }
    }
}

static void model_free(Model *model) {
    if (!model) return;
    for (int i = 0; i < model->num_layers; i++) {
        if (model->layers[i].type == LAYER_CONV) conv2d_free((Conv2D *)model->layers[i].layer_data);
        if (model->layers[i].output_cache) tensor_free(model->layers[i].output_cache);
    }
    free(model->layers);
    free(model);
}

static void generate_synthetic_image(Tensor *image, int label, int sample_id) {
    for (int i = 0; i < image->size; i++) { image->data[i] = 0.0f; if (image->grad) image->grad[i] = 0.0f; }
    for (int row = 0; row < IMAGE_H; row++) {
        for (int col = 0; col < IMAGE_W; col++) {
            int idx = row * IMAGE_W + col;
            float noise = 0.03f * sinf((float)(sample_id * 13 + row * 7 + col * 3));
            int center = 8 + (sample_id % 4);
            if (label == 0) image->data[idx] = (col >= center && col <= center + 3) ? 1.0f + noise : noise;
            else image->data[idx] = (row >= center && row <= center + 3) ? 1.0f + noise : noise;
        }
    }
}

static Dataset *dataset_create_synthetic(int num_samples) {
    Dataset *dataset = (Dataset *)malloc(sizeof(Dataset));
    fail_if_null(dataset, "Error: failed to allocate dataset.");
    dataset->size = num_samples;
    dataset->images = (Tensor **)malloc((size_t)num_samples * sizeof(Tensor *));
    dataset->labels = (int *)malloc((size_t)num_samples * sizeof(int));
    fail_if_null(dataset->images, "Error: failed to allocate dataset images.");
    fail_if_null(dataset->labels, "Error: failed to allocate dataset labels.");
    for (int i = 0; i < num_samples; i++) {
        int label = i % NUM_CLASSES;
        dataset->images[i] = tensor_create(1, IMAGE_C, IMAGE_H, IMAGE_W, 1);
        dataset->labels[i] = label;
        generate_synthetic_image(dataset->images[i], label, i);
    }
    return dataset;
}

static void dataset_free(Dataset *dataset) {
    if (dataset) {
        for (int i = 0; i < dataset->size; i++) tensor_free(dataset->images[i]);
        free(dataset->images); free(dataset->labels); free(dataset);
    }
}

static Model *build_tiny_cnn_model(void) {
    Model *model = model_create();
    model_add_conv(model, 1, 8, 3, 1, 1); model_add_relu(model);
    model_add_conv(model, 8, 16, 3, 2, 1); model_add_relu(model);
    model_add_conv(model, 16, 32, 3, 2, 1); model_add_relu(model);
    model_add_conv(model, 32, 2, 7, 1, 0); model_add_softmax(model);
    return model;
}

static int predict_class(Tensor *output) { return output->data[0] > output->data[1] ? 0 : 1; }

static void train(Model *model, Dataset *dataset, int epochs, float learning_rate) {
    float first_loss = 0.0f, final_loss = 0.0f;
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        int correct = 0;
        for (int i = 0; i < dataset->size; i++) {
            Tensor *input = dataset->images[i];
            int target[1] = {dataset->labels[i]};
            zero_all_cached_gradients(model, input);
            model_forward(model, input);
            Tensor *output = model->layers[model->num_layers - 1].output_cache;
            float loss = cross_entropy_loss(output, target);
            total_loss += loss;
            if (predict_class(output) == target[0]) correct++;
            cross_entropy_backward(output, target);
            model_backward(model);
            model_step(model, learning_rate);
        }
        float avg_loss = total_loss / (float)dataset->size;
        float accuracy = 100.0f * (float)correct / (float)dataset->size;
        if (epoch == 0) first_loss = avg_loss;
        final_loss = avg_loss;
        printf("Epoch %d: Loss = %.6f, Accuracy = %.2f%%\n", epoch, avg_loss, accuracy);
    }
    printf("\nInitial loss: %.6f\n", first_loss);
    printf("Final loss:   %.6f\n", final_loss);
}

int main(int argc, char **argv) {
    int num_samples = DEFAULT_NUM_SAMPLES;
    int epochs = DEFAULT_EPOCHS;
    float learning_rate = DEFAULT_LEARNING_RATE;
    if (argc >= 2) num_samples = atoi(argv[1]);
    if (argc >= 3) epochs = atoi(argv[2]);
    if (argc >= 4) learning_rate = (float)atof(argv[3]);
    if (num_samples <= 0 || epochs <= 0 || learning_rate <= 0.0f) {
        fprintf(stderr, "Error: all arguments must be positive.\n");
        fprintf(stderr, "Usage: %s <num_samples> <epochs> <learning_rate>\n", argv[0]);
        return 1;
    }
    srand(42);
    printf("Program name: tiny_cnn_training_single_file\n");
    printf("Number of samples: %d\n", num_samples);
    printf("Epochs: %d\n", epochs);
    printf("Learning rate: %.6f\n", learning_rate);
    printf("Architecture: Conv-ReLU -> Conv-ReLU -> Conv-ReLU -> Conv-Softmax\n\n");
    Dataset *dataset = dataset_create_synthetic(num_samples);
    Model *model = build_tiny_cnn_model();

    train(model, dataset, epochs, learning_rate);

    model_free(model);
    dataset_free(dataset);
    return 0;
}
