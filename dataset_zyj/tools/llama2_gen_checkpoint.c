/*
 * llama2_gen_checkpoint.c
 *
 * Build-tool, not a benchmark: writes a tiny, fully deterministic
 * llama2.c-format checkpoint (model.bin) and tokenizer (tokenizer.bin) into
 * <out_dir>, so llama2_c_inference.c has something to run without needing a
 * real downloaded model. Every weight is a pure deterministic function of
 * its position (no internet, no license, no shared RNG state to seed),
 * matching the rest of this dataset's synthetic-input convention.
 *
 * The checkpoint always uses shared embedding/classifier weights (positive
 * vocab_size in the Config header), so no separate wcls block is written.
 * The tokenizer is a byte-fallback-only vocab: ids 0/1/2 are <unk>/<s>/</s>,
 * ids 3..258 are the 256 raw byte values, and vocab_size must be exactly
 * 259 -- id 35 (byte 0x20, a space) doubles as the dummy-prefix token that
 * llama2_c_inference.c's encode() looks up for every non-empty prompt.
 *
 * Usage:
 *   ./llama2_gen_checkpoint <out_dir> <dim> <hidden_dim> <n_layers> \
 *       <n_heads> <n_kv_heads> <vocab_size> <seq_len>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Mirrors llama2_c_inference.c's Config struct exactly: 7 ints, same order. */
typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

static float gen_weight(unsigned int a, unsigned int b, unsigned int c, float scale)
{
    unsigned int x = a * 2654435761u + b * 2246822519u + c * 3266489917u + 0x9E3779B9u;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    float u = (float)(x % 1000000u) / 1000000.0f;
    return scale * (2.0f * u - 1.0f);
}

static void write_floats(FILE *f, unsigned int layer, unsigned int block_id, long count, float scale, float bias)
{
    for (long i = 0; i < count; i++) {
        float v = bias + gen_weight(layer, block_id, (unsigned int)i, scale);
        if (fwrite(&v, sizeof(float), 1, f) != 1) {
            fprintf(stderr, "Error: failed to write weight data.\n");
            exit(1);
        }
    }
}

static void write_zero_floats(FILE *f, long count)
{
    float zero = 0.0f;
    for (long i = 0; i < count; i++) {
        if (fwrite(&zero, sizeof(float), 1, f) != 1) {
            fprintf(stderr, "Error: failed to write padding data.\n");
            exit(1);
        }
    }
}

static long total_float_count(const Config *p)
{
    long head_size = p->dim / p->n_heads;
    long total = 0;

    total += (long)p->vocab_size * p->dim;                            /* token_embedding_table */
    total += (long)p->n_layers * p->dim;                              /* rms_att_weight */
    total += (long)p->n_layers * p->dim * (p->n_heads * head_size);    /* wq */
    total += (long)p->n_layers * p->dim * (p->n_kv_heads * head_size); /* wk */
    total += (long)p->n_layers * p->dim * (p->n_kv_heads * head_size); /* wv */
    total += (long)p->n_layers * (p->n_heads * head_size) * p->dim;    /* wo */
    total += (long)p->n_layers * p->dim;                              /* rms_ffn_weight */
    total += (long)p->n_layers * p->dim * p->hidden_dim;              /* w1 */
    total += (long)p->n_layers * p->hidden_dim * p->dim;              /* w2 */
    total += (long)p->n_layers * p->dim * p->hidden_dim;              /* w3 */
    total += p->dim;                                                  /* rms_final_weight */
    total += (long)p->seq_len * head_size / 2;                        /* skipped freq_cis_real */
    total += (long)p->seq_len * head_size / 2;                        /* skipped freq_cis_imag */
    /* shared weights: wcls aliases token_embedding_table, no extra block */

    return total;
}

static void write_checkpoint(const char *path, const Config *p)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Error: could not open %s for writing.\n", path);
        exit(1);
    }

    if (fwrite(p, sizeof(Config), 1, f) != 1) {
        fprintf(stderr, "Error: failed to write Config header.\n");
        exit(1);
    }

    long head_size = p->dim / p->n_heads;

    write_floats(f, 0, 1, (long)p->vocab_size * p->dim, 0.02f, 0.0f); /* token_embedding_table */

    for (int l = 0; l < p->n_layers; l++) {
        write_floats(f, (unsigned int)l, 2, p->dim, 0.1f, 1.0f); /* rms_att_weight */
    }
    for (int l = 0; l < p->n_layers; l++) {
        write_floats(f, (unsigned int)l, 3, (long)p->dim * p->n_heads * head_size, 0.02f, 0.0f); /* wq */
    }
    for (int l = 0; l < p->n_layers; l++) {
        write_floats(f, (unsigned int)l, 4, (long)p->dim * p->n_kv_heads * head_size, 0.02f, 0.0f); /* wk */
    }
    for (int l = 0; l < p->n_layers; l++) {
        write_floats(f, (unsigned int)l, 5, (long)p->dim * p->n_kv_heads * head_size, 0.02f, 0.0f); /* wv */
    }
    for (int l = 0; l < p->n_layers; l++) {
        write_floats(f, (unsigned int)l, 6, (long)p->n_heads * head_size * p->dim, 0.02f, 0.0f); /* wo */
    }
    for (int l = 0; l < p->n_layers; l++) {
        write_floats(f, (unsigned int)l, 7, p->dim, 0.1f, 1.0f); /* rms_ffn_weight */
    }
    for (int l = 0; l < p->n_layers; l++) {
        write_floats(f, (unsigned int)l, 8, (long)p->dim * p->hidden_dim, 0.02f, 0.0f); /* w1 */
    }
    for (int l = 0; l < p->n_layers; l++) {
        write_floats(f, (unsigned int)l, 9, (long)p->hidden_dim * p->dim, 0.02f, 0.0f); /* w2 */
    }
    for (int l = 0; l < p->n_layers; l++) {
        write_floats(f, (unsigned int)l, 10, (long)p->dim * p->hidden_dim, 0.02f, 0.0f); /* w3 */
    }

    write_floats(f, 0, 11, p->dim, 0.1f, 1.0f); /* rms_final_weight */

    write_zero_floats(f, (long)p->seq_len * head_size / 2); /* legacy freq_cis_real, unused but skipped on read */
    write_zero_floats(f, (long)p->seq_len * head_size / 2); /* legacy freq_cis_imag, unused but skipped on read */

    long written_bytes = ftell(f);
    long expected_bytes = (long)sizeof(Config) + total_float_count(p) * (long)sizeof(float);

    fclose(f);

    if (written_bytes != expected_bytes) {
        fprintf(stderr, "Error: checkpoint size mismatch: wrote %ld bytes, expected %ld bytes.\n",
                written_bytes, expected_bytes);
        exit(1);
    }
}

static void write_tokenizer(const char *path, int vocab_size)
{
    if (vocab_size != 259) {
        fprintf(stderr, "Error: this generator only supports vocab_size == 259 (byte-fallback-only vocab).\n");
        exit(1);
    }

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Error: could not open %s for writing.\n", path);
        exit(1);
    }

    int max_token_length = 5; /* length of "<unk>", the longest string actually in the vocab */
    if (fwrite(&max_token_length, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Error: failed to write max_token_length.\n");
        exit(1);
    }

    static const char *specials[3] = {"<unk>", "<s>", "</s>"};

    for (int id = 0; id < vocab_size; id++) {
        float score = 0.0f;
        char byte_str[2];
        const char *str;
        int len;

        if (id < 3) {
            str = specials[id];
            len = (int)strlen(str);
        } else {
            byte_str[0] = (char)(id - 3);
            byte_str[1] = '\0';
            str = byte_str;
            len = 1;
        }

        if (fwrite(&score, sizeof(float), 1, f) != 1
            || fwrite(&len, sizeof(int), 1, f) != 1
            || fwrite(str, 1, (size_t)len, f) != (size_t)len) {
            fprintf(stderr, "Error: failed to write vocab entry %d.\n", id);
            exit(1);
        }
    }

    fclose(f);
}

int main(int argc, char **argv)
{
    if (argc != 9) {
        fprintf(stderr,
            "Usage: %s <out_dir> <dim> <hidden_dim> <n_layers> <n_heads> <n_kv_heads> <vocab_size> <seq_len>\n",
            argv[0]);
        return 1;
    }

    const char *out_dir = argv[1];
    Config config;
    config.dim = atoi(argv[2]);
    config.hidden_dim = atoi(argv[3]);
    config.n_layers = atoi(argv[4]);
    config.n_heads = atoi(argv[5]);
    config.n_kv_heads = atoi(argv[6]);
    config.vocab_size = atoi(argv[7]);
    config.seq_len = atoi(argv[8]);

    if (config.dim <= 0 || config.hidden_dim <= 0 || config.n_layers <= 0
        || config.n_heads <= 0 || config.n_kv_heads <= 0 || config.vocab_size <= 0
        || config.seq_len <= 0) {
        fprintf(stderr, "Error: all dimensions must be positive.\n");
        return 1;
    }
    if (config.dim % config.n_heads != 0) {
        fprintf(stderr, "Error: dim must be divisible by n_heads.\n");
        return 1;
    }
    if (config.n_heads % config.n_kv_heads != 0) {
        fprintf(stderr, "Error: n_heads must be divisible by n_kv_heads.\n");
        return 1;
    }
    if ((config.dim / config.n_heads) % 2 != 0) {
        fprintf(stderr, "Error: head_size (dim / n_heads) must be even for RoPE.\n");
        return 1;
    }

    char model_path[4096];
    char tokenizer_path[4096];
    snprintf(model_path, sizeof(model_path), "%s/model.bin", out_dir);
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.bin", out_dir);

    write_checkpoint(model_path, &config);
    write_tokenizer(tokenizer_path, config.vocab_size);

    printf("Wrote %s and %s (dim=%d, hidden_dim=%d, n_layers=%d, n_heads=%d, n_kv_heads=%d, vocab_size=%d, seq_len=%d)\n",
           model_path, tokenizer_path, config.dim, config.hidden_dim, config.n_layers,
           config.n_heads, config.n_kv_heads, config.vocab_size, config.seq_len);

    return 0;
}
