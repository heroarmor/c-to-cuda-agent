#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define DEFAULT_NUM_POINTS 20000
#define DEFAULT_DIMENSION 16
#define DEFAULT_BATCH_SIZE 2000
#define DEFAULT_MAX_CENTERS 96
#define DEFAULT_PASSES 5

#define TRUE_CLUSTER_COUNT 12
#define OPEN_COST_SCALE 0.35
#define MIN_GAIN_RATIO 0.15

typedef struct {
    int id;
    double *coord;
} Point;

typedef struct {
    int id;
    int count;
    double *coord;
} Center;

static double deterministic_noise(int point_id, int dim_id)
{
    unsigned int x = (unsigned int)(point_id * 1103515245u + dim_id * 12345u + 67891u);
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return ((double)(x % 10000) / 10000.0) - 0.5;
}

static double true_center_value(int cluster_id, int dim_id)
{
    double a = sin(0.37 * (double)(cluster_id + 1) * (double)(dim_id + 1));
    double b = cos(0.19 * (double)(cluster_id + 3) * (double)(dim_id + 2));
    return 10.0 * a + 6.0 * b;
}

static void generate_point(Point *p, int global_id, int dimension)
{
    int cluster_id = (global_id * 17 + 13) % TRUE_CLUSTER_COUNT;
    double drift = 0.0002 * (double)global_id;

    p->id = global_id;

    for (int d = 0; d < dimension; d++) {
        double base = true_center_value(cluster_id, d);
        double noise = deterministic_noise(global_id, d);
        p->coord[d] = base + noise + drift * sin(0.11 * (double)(d + 1));
    }
}

static double squared_distance(const double *a, const double *b, int dimension)
{
    double sum = 0.0;

    for (int d = 0; d < dimension; d++) {
        double diff = a[d] - b[d];
        sum += diff * diff;
    }

    return sum;
}

static void allocate_points(Point *points, int count, int dimension)
{
    for (int i = 0; i < count; i++) {
        points[i].id = -1;
        points[i].coord = (double *)malloc((size_t)dimension * sizeof(double));
        if (points[i].coord == NULL) {
            fprintf(stderr, "Error: failed to allocate point coordinates.\n");
            exit(1);
        }
    }
}

static void free_points(Point *points, int count)
{
    for (int i = 0; i < count; i++) {
        free(points[i].coord);
    }
}

static void allocate_centers(Center *centers, int count, int dimension)
{
    for (int i = 0; i < count; i++) {
        centers[i].id = -1;
        centers[i].count = 0;
        centers[i].coord = (double *)malloc((size_t)dimension * sizeof(double));
        if (centers[i].coord == NULL) {
            fprintf(stderr, "Error: failed to allocate center coordinates.\n");
            exit(1);
        }
        for (int d = 0; d < dimension; d++) {
            centers[i].coord[d] = 0.0;
        }
    }
}

static void free_centers(Center *centers, int count)
{
    for (int i = 0; i < count; i++) {
        free(centers[i].coord);
    }
}

static void copy_point_to_center(Center *center, const Point *point, int dimension)
{
    center->id = point->id;
    center->count = 0;

    for (int d = 0; d < dimension; d++) {
        center->coord[d] = point->coord[d];
    }
}

static int nearest_center(
    const Point *point,
    const Center *centers,
    int num_centers,
    int dimension,
    double *best_distance
)
{
    int best_index = 0;
    double best = DBL_MAX;

    for (int c = 0; c < num_centers; c++) {
        double dist = squared_distance(point->coord, centers[c].coord, dimension);
        if (dist < best) {
            best = dist;
            best_index = c;
        }
    }

    *best_distance = best;
    return best_index;
}

static double assign_points(
    const Point *points,
    int num_points,
    const Center *centers,
    int num_centers,
    int dimension,
    int *assignment,
    double *point_cost
)
{
    double total_cost = 0.0;

    for (int i = 0; i < num_points; i++) {
        double dist = 0.0;
        int center_id = nearest_center(&points[i], centers, num_centers, dimension, &dist);
        assignment[i] = center_id;
        point_cost[i] = dist;
        total_cost += dist;
    }

    return total_cost;
}

static double candidate_gain(
    const Point *candidate,
    const Point *points,
    int num_points,
    const double *current_cost,
    int dimension,
    double open_cost
)
{
    double gain = -open_cost;

    for (int i = 0; i < num_points; i++) {
        double dist = squared_distance(points[i].coord, candidate->coord, dimension);
        if (dist < current_cost[i]) {
            gain += current_cost[i] - dist;
        }
    }

    return gain;
}

static int try_open_new_centers(
    const Point *points,
    int num_points,
    Center *centers,
    int *num_centers,
    int max_centers,
    int dimension,
    int *assignment,
    double *point_cost,
    double open_cost
)
{
    int opened = 0;

    for (int i = 0; i < num_points && *num_centers < max_centers; i++) {
        if (point_cost[i] <= open_cost * MIN_GAIN_RATIO) {
            continue;
        }

        double gain = candidate_gain(
            &points[i], points, num_points, point_cost, dimension, open_cost
        );

        if (gain > 0.0) {
            copy_point_to_center(&centers[*num_centers], &points[i], dimension);
            (*num_centers)++;
            opened++;

            assign_points(points, num_points, centers, *num_centers, dimension, assignment, point_cost);
        }
    }

    return opened;
}

static void update_centers_from_assignments(
    const Point *points,
    int num_points,
    Center *centers,
    int num_centers,
    int dimension,
    const int *assignment
)
{
    for (int c = 0; c < num_centers; c++) {
        centers[c].count = 0;
        for (int d = 0; d < dimension; d++) {
            centers[c].coord[d] = 0.0;
        }
    }

    for (int i = 0; i < num_points; i++) {
        int c = assignment[i];
        centers[c].count++;
        for (int d = 0; d < dimension; d++) {
            centers[c].coord[d] += points[i].coord[d];
        }
    }

    for (int c = 0; c < num_centers; c++) {
        if (centers[c].count == 0) {
            continue;
        }
        for (int d = 0; d < dimension; d++) {
            centers[c].coord[d] /= (double)centers[c].count;
        }
    }
}

static void compact_empty_centers(Center *centers, int *num_centers, int dimension)
{
    int write_index = 0;

    for (int read_index = 0; read_index < *num_centers; read_index++) {
        if (centers[read_index].count > 0) {
            if (write_index != read_index) {
                centers[write_index].id = centers[read_index].id;
                centers[write_index].count = centers[read_index].count;
                for (int d = 0; d < dimension; d++) {
                    centers[write_index].coord[d] = centers[read_index].coord[d];
                }
            }
            write_index++;
        }
    }

    *num_centers = write_index;
}

static double refine_centers(
    const Point *points,
    int num_points,
    Center *centers,
    int *num_centers,
    int dimension,
    int *assignment,
    double *point_cost,
    int passes
)
{
    double cost = 0.0;

    for (int pass = 0; pass < passes; pass++) {
        cost = assign_points(points, num_points, centers, *num_centers, dimension, assignment, point_cost);
        update_centers_from_assignments(points, num_points, centers, *num_centers, dimension, assignment);
        compact_empty_centers(centers, num_centers, dimension);
    }

    cost = assign_points(points, num_points, centers, *num_centers, dimension, assignment, point_cost);
    return cost;
}

static double process_batch(
    Point *batch_points,
    int batch_count,
    Center *global_centers,
    int *num_global_centers,
    int max_centers,
    int dimension,
    int *assignment,
    double *point_cost,
    int passes,
    double open_cost
)
{
    if (*num_global_centers == 0 && batch_count > 0) {
        copy_point_to_center(&global_centers[0], &batch_points[0], dimension);
        *num_global_centers = 1;
    }

    double before_cost = assign_points(
        batch_points, batch_count, global_centers, *num_global_centers,
        dimension, assignment, point_cost
    );

    int opened = try_open_new_centers(
        batch_points, batch_count, global_centers, num_global_centers,
        max_centers, dimension, assignment, point_cost, open_cost
    );

    double after_cost = refine_centers(
        batch_points, batch_count, global_centers, num_global_centers,
        dimension, assignment, point_cost, passes
    );

    printf(
        "Batch processed: points=%d, opened_centers=%d, centers=%d, cost_before=%.4f, cost_after=%.4f\n",
        batch_count, opened, *num_global_centers, before_cost, after_cost
    );

    return after_cost;
}

static double evaluate_stream_cost(
    int num_points,
    int dimension,
    int batch_size,
    const Center *centers,
    int num_centers
)
{
    Point *batch = (Point *)malloc((size_t)batch_size * sizeof(Point));
    int *assignment = (int *)malloc((size_t)batch_size * sizeof(int));
    double *point_cost = (double *)malloc((size_t)batch_size * sizeof(double));

    if (batch == NULL || assignment == NULL || point_cost == NULL) {
        fprintf(stderr, "Error: memory allocation failed during evaluation.\n");
        exit(1);
    }

    allocate_points(batch, batch_size, dimension);

    double total_cost = 0.0;
    int generated = 0;

    while (generated < num_points) {
        int current_batch_size = batch_size;
        if (generated + current_batch_size > num_points) {
            current_batch_size = num_points - generated;
        }

        for (int i = 0; i < current_batch_size; i++) {
            generate_point(&batch[i], generated + i, dimension);
        }

        total_cost += assign_points(
            batch, current_batch_size, centers, num_centers,
            dimension, assignment, point_cost
        );

        generated += current_batch_size;
    }

    free_points(batch, batch_size);
    free(batch);
    free(assignment);
    free(point_cost);

    return total_cost;
}

static void print_center_summary(const Center *centers, int num_centers, int dimension)
{
    printf("\nFinal centers:\n");

    for (int c = 0; c < num_centers; c++) {
        printf("  Center %d: count=%d, first_dims=[", c, centers[c].count);

        int shown_dims = dimension < 4 ? dimension : 4;
        for (int d = 0; d < shown_dims; d++) {
            printf("%.3f", centers[c].coord[d]);
            if (d + 1 < shown_dims) {
                printf(", ");
            }
        }

        if (dimension > shown_dims) {
            printf(", ...");
        }

        printf("]\n");
    }
}

int main(int argc, char **argv)
{
    int num_points = DEFAULT_NUM_POINTS;
    int dimension = DEFAULT_DIMENSION;
    int batch_size = DEFAULT_BATCH_SIZE;
    int max_centers = DEFAULT_MAX_CENTERS;
    int passes = DEFAULT_PASSES;

    if (argc >= 2) {
        num_points = atoi(argv[1]);
    }
    if (argc >= 3) {
        dimension = atoi(argv[2]);
    }
    if (argc >= 4) {
        batch_size = atoi(argv[3]);
    }
    if (argc >= 5) {
        max_centers = atoi(argv[4]);
    }
    if (argc >= 6) {
        passes = atoi(argv[5]);
    }

    if (num_points <= 0 || dimension <= 0 || batch_size <= 0 || max_centers <= 0 || passes <= 0) {
        fprintf(stderr, "Error: all arguments must be positive.\n");
        fprintf(stderr, "Usage: %s <num_points> <dimension> <batch_size> <max_centers> <passes>\n", argv[0]);
        return 1;
    }

    if (batch_size > num_points) {
        batch_size = num_points;
    }

    Point *batch_points = (Point *)malloc((size_t)batch_size * sizeof(Point));
    Center *global_centers = (Center *)malloc((size_t)max_centers * sizeof(Center));
    int *assignment = (int *)malloc((size_t)batch_size * sizeof(int));
    double *point_cost = (double *)malloc((size_t)batch_size * sizeof(double));

    if (batch_points == NULL || global_centers == NULL || assignment == NULL || point_cost == NULL) {
        fprintf(stderr, "Error: memory allocation failed.\n");
        free(batch_points);
        free(global_centers);
        free(assignment);
        free(point_cost);
        return 1;
    }

    allocate_points(batch_points, batch_size, dimension);
    allocate_centers(global_centers, max_centers, dimension);

    int num_global_centers = 0;
    int generated = 0;
    int batch_id = 0;
    double open_cost = OPEN_COST_SCALE * (double)dimension;

    printf("Program name: stream_cluster_facility_location\n");
    printf("Total points: %d\n", num_points);
    printf("Dimension: %d\n", dimension);
    printf("Batch size: %d\n", batch_size);
    printf("Maximum centers: %d\n", max_centers);
    printf("Refinement passes per batch: %d\n", passes);
    printf("Open cost: %.4f\n\n", open_cost);

    while (generated < num_points) {
        int current_batch_size = batch_size;
        if (generated + current_batch_size > num_points) {
            current_batch_size = num_points - generated;
        }

        for (int i = 0; i < current_batch_size; i++) {
            generate_point(&batch_points[i], generated + i, dimension);
        }

        printf("Processing batch %d, global point range [%d, %d)\n",
               batch_id, generated, generated + current_batch_size);

        process_batch(
            batch_points, current_batch_size, global_centers, &num_global_centers,
            max_centers, dimension, assignment, point_cost, passes, open_cost
        );

        generated += current_batch_size;
        batch_id++;
    }

    double total_cost = evaluate_stream_cost(
        num_points, dimension, batch_size, global_centers, num_global_centers
    );

    print_center_summary(global_centers, num_global_centers, dimension);

    printf("\nFinal number of centers: %d\n", num_global_centers);
    printf("Final total cost: %.6f\n", total_cost);
    printf("Final average cost: %.6f\n", total_cost / (double)num_points);

    free_points(batch_points, batch_size);
    free_centers(global_centers, max_centers);
    free(batch_points);
    free(global_centers);
    free(assignment);
    free(point_cost);

    return 0;
}
