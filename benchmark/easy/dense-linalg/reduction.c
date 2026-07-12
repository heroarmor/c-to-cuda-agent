#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

static double generate_value(int i)
{
    /*
     * Deterministic pseudo-random-like data generation.
     * This avoids file I/O and makes the program reproducible.
     */
    double x = sin(i * 0.001) + cos(i * 0.0007);
    double y = (double)(i % 1000) / 1000.0;
    return x + y;
}

static void initialize_array(double *array, int n)
{
    for (int i = 0; i < n; i++) {
        array[i] = generate_value(i);
    }
}

static double compute_sum(const double *array, int n)
{
    double sum = 0.0;

    for (int i = 0; i < n; i++) {
        sum += array[i];
    }

    return sum;
}

static double compute_minimum(const double *array, int n)
{
    double min_value = DBL_MAX;

    for (int i = 0; i < n; i++) {
        if (array[i] < min_value) {
            min_value = array[i];
        }
    }

    return min_value;
}

static double compute_maximum(const double *array, int n)
{
    double max_value = -DBL_MAX;

    for (int i = 0; i < n; i++) {
        if (array[i] > max_value) {
            max_value = array[i];
        }
    }

    return max_value;
}

static double compute_variance(const double *array, int n, double mean)
{
    double variance_sum = 0.0;

    for (int i = 0; i < n; i++) {
        double diff = array[i] - mean;
        variance_sum += diff * diff;
    }

    return variance_sum / (double)n;
}

int main(int argc, char **argv)
{
    int n = 20000000;

    if (argc >= 2) {
        n = atoi(argv[1]);
    }

    if (n <= 0) {
        fprintf(stderr, "Error: array size must be positive.\n");
        fprintf(stderr, "Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    double *array = (double *)malloc((size_t)n * sizeof(double));

    if (array == NULL) {
        fprintf(stderr, "Error: failed to allocate memory.\n");
        return 1;
    }

    initialize_array(array, n);

    double sum = compute_sum(array, n);
    double min_value = compute_minimum(array, n);
    double max_value = compute_maximum(array, n);
    double mean = sum / (double)n;
    double variance = compute_variance(array, n, mean);

    printf("Array size: %d\n", n);
    printf("Sum:        %.12f\n", sum);
    printf("Minimum:    %.12f\n", min_value);
    printf("Maximum:    %.12f\n", max_value);
    printf("Mean:       %.12f\n", mean);
    printf("Variance:   %.12f\n", variance);

    free(array);

    return 0;
}