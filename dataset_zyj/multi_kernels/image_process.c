#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define HISTOGRAM_LENGTH 256
#define DEFAULT_IMAGE_WIDTH 1024
#define DEFAULT_IMAGE_HEIGHT 1024
#define IMAGE_CHANNELS 3
#define DEFAULT_NUM_IMAGES 20

static float clamp_float(float x, float start, float end)
{
    if (x < start) {
        return start;
    }
    if (x > end) {
        return end;
    }
    return x;
}

static void generate_rgb_image(float *image, int width, int height, int channels, int image_id)
{
    float image_shift = 0.07f * (float)image_id;

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int pixel_idx = row * width + col;
            int base = pixel_idx * channels;

            float x = (float)col / (float)(width - 1);
            float y = (float)row / (float)(height - 1);

            image[base + 0] = clamp_float(0.45f + 0.35f * sinf(12.0f * x + image_shift) + 0.15f * y, 0.0f, 1.0f);
            image[base + 1] = clamp_float(0.40f + 0.30f * cosf(10.0f * y + image_shift) + 0.20f * x, 0.0f, 1.0f);
            image[base + 2] = clamp_float(0.30f + 0.25f * sinf(8.0f * (x + y) + image_shift), 0.0f, 1.0f);
        }
    }
}

static void convert_rgb_to_grayscale(
    const float *input,
    unsigned char *gray,
    int width,
    int height,
    int channels
)
{
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int pixel_idx = row * width + col;
            int base = pixel_idx * channels;

            unsigned char r = (unsigned char)(255.0f * clamp_float(input[base + 0], 0.0f, 1.0f));
            unsigned char g = (unsigned char)(255.0f * clamp_float(input[base + 1], 0.0f, 1.0f));
            unsigned char b = (unsigned char)(255.0f * clamp_float(input[base + 2], 0.0f, 1.0f));

            gray[pixel_idx] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
        }
    }
}

static void compute_histogram(
    const unsigned char *gray,
    unsigned int histogram[HISTOGRAM_LENGTH],
    int num_pixels
)
{
    for (int i = 0; i < HISTOGRAM_LENGTH; i++) {
        histogram[i] = 0;
    }

    for (int i = 0; i < num_pixels; i++) {
        unsigned char value = gray[i];
        histogram[value]++;
    }
}

static void compute_cdf(
    const unsigned int histogram[HISTOGRAM_LENGTH],
    float cdf[HISTOGRAM_LENGTH],
    int num_pixels
)
{
    float cumulative = 0.0f;

    for (int i = 0; i < HISTOGRAM_LENGTH; i++) {
        cumulative += (float)histogram[i] / (float)num_pixels;
        cdf[i] = cumulative;
    }
}

static float correct_color(unsigned char value, const float cdf[HISTOGRAM_LENGTH])
{
    float denominator = 1.0f - cdf[0];

    if (fabsf(denominator) < 1e-12f) {
        return (float)value;
    }

    return clamp_float(255.0f * (cdf[value] - cdf[0]) / denominator, 0.0f, 255.0f);
}

static void apply_histogram_equalization(
    const float *input,
    float *output,
    const float cdf[HISTOGRAM_LENGTH],
    int width,
    int height,
    int channels
)
{
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int pixel_idx = row * width + col;
            int base = pixel_idx * channels;

            unsigned char r = (unsigned char)(255.0f * clamp_float(input[base + 0], 0.0f, 1.0f));
            unsigned char g = (unsigned char)(255.0f * clamp_float(input[base + 1], 0.0f, 1.0f));
            unsigned char b = (unsigned char)(255.0f * clamp_float(input[base + 2], 0.0f, 1.0f));

            output[base + 0] = correct_color(r, cdf) / 255.0f;
            output[base + 1] = correct_color(g, cdf) / 255.0f;
            output[base + 2] = correct_color(b, cdf) / 255.0f;
        }
    }
}

static void print_image_statistics(const float *image, int width, int height, int channels)
{
    int total_values = width * height * channels;
    float min_value = image[0];
    float max_value = image[0];
    double sum = 0.0;

    for (int i = 0; i < total_values; i++) {
        if (image[i] < min_value) {
            min_value = image[i];
        }
        if (image[i] > max_value) {
            max_value = image[i];
        }
        sum += image[i];
    }

    printf("Min pixel value:  %.6f\n", min_value);
    printf("Max pixel value:  %.6f\n", max_value);
    printf("Mean pixel value: %.6f\n", sum / (double)total_values);
}

int main(int argc, char **argv)
{
    int width = DEFAULT_IMAGE_WIDTH;
    int height = DEFAULT_IMAGE_HEIGHT;
    int channels = IMAGE_CHANNELS;
    int num_images = DEFAULT_NUM_IMAGES;

    if (argc >= 3) {
        width = atoi(argv[1]);
        height = atoi(argv[2]);
    }

    if (argc >= 4) {
        num_images = atoi(argv[3]);
    }

    if (width <= 1 || height <= 1) {
        fprintf(stderr, "Error: width and height must be greater than 1.\n");
        fprintf(stderr, "Usage: %s [width height num_images]\n", argv[0]);
        return 1;
    }

    if (num_images <= 0) {
        fprintf(stderr, "Error: number of images must be positive.\n");
        fprintf(stderr, "Usage: %s [width height num_images]\n", argv[0]);
        return 1;
    }

    int num_pixels = width * height;
    size_t image_size = (size_t)num_pixels * (size_t)channels * sizeof(float);

    float *input_image = (float *)malloc(image_size);
    float *output_image = (float *)malloc(image_size);
    unsigned char *gray_image = (unsigned char *)malloc((size_t)num_pixels * sizeof(unsigned char));
    unsigned int histogram[HISTOGRAM_LENGTH];
    float cdf[HISTOGRAM_LENGTH];

    if (input_image == NULL || output_image == NULL || gray_image == NULL) {
        fprintf(stderr, "Error: memory allocation failed.\n");
        free(input_image);
        free(output_image);
        free(gray_image);
        return 1;
    }

    printf("Program name: image_histogram_equalization\n");
    printf("Number of images: %d\n", num_images);
    printf("Image size: %d x %d x %d\n", width, height, channels);
    printf("Histogram bins: %d\n", HISTOGRAM_LENGTH);

    for (int image_id = 0; image_id < num_images; image_id++) {
        generate_rgb_image(input_image, width, height, channels, image_id);

        convert_rgb_to_grayscale(input_image, gray_image, width, height, channels);
        compute_histogram(gray_image, histogram, num_pixels);
        compute_cdf(histogram, cdf, num_pixels);
        apply_histogram_equalization(input_image, output_image, cdf, width, height, channels);

        printf("\nImage %d statistics:\n", image_id);
        print_image_statistics(output_image, width, height, channels);
    }

    free(input_image);
    free(output_image);
    free(gray_image);

    return 0;
}