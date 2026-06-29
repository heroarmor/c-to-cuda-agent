# Build rules for the sequential C/C++ benchmark dataset.
#
# Usage:
#   make            build every benchmark binary into bin/, plus the
#                   synthetic llama2 checkpoint/tokenizer data into data/
#   make quick      build, then run measure.py in quick (correctness) mode
#   make perf       build, then run measure.py in perf (timing) mode
#   make clean      remove build output
#
# llama2_c_inference.c needs a model checkpoint + tokenizer.bin to run. This
# dataset does not ship a real pretrained model; instead, llama2_gen_checkpoint
# (built from tools/llama2_gen_checkpoint.c -- a build tool, not part of the
# dataset itself) writes a tiny, fully deterministic checkpoint + tokenizer
# pair into data/quick and data/perf, so the benchmark stays offline and
# reproducible like every other entry in this dataset.

CC    ?= cc
STD    = -std=c11
WARN   = -Wall -Wextra
OPT    = -O2
LIBS   = -lm

BIN_DIR := bin
DATA_DIR := data
VPATH   := single_kernel:multi_kernels:tools

NAMES := \
	reduction \
	CNN \
	image_process \
	stream_cluster \
	tiny_cnn_training_single_file \
	llama2_c_inference

BINARIES := $(addprefix $(BIN_DIR)/,$(NAMES))
GEN_TOOL := $(BIN_DIR)/llama2_gen_checkpoint
QUICK_DATA := $(DATA_DIR)/quick/model.bin $(DATA_DIR)/quick/tokenizer.bin
PERF_DATA  := $(DATA_DIR)/perf/model.bin  $(DATA_DIR)/perf/tokenizer.bin

.PHONY: all quick perf clean

all: $(BINARIES) $(QUICK_DATA) $(PERF_DATA)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(DATA_DIR)/quick:
	mkdir -p $(DATA_DIR)/quick

$(DATA_DIR)/perf:
	mkdir -p $(DATA_DIR)/perf

$(BIN_DIR)/%: %.c | $(BIN_DIR)
	$(CC) $(STD) $(WARN) $(OPT) $< -o $@ $(LIBS)

# Dimensions: dim hidden_dim n_layers n_heads n_kv_heads vocab_size seq_len.
# vocab_size must stay 259 (the generator's byte-fallback-only vocab).
$(QUICK_DATA): $(GEN_TOOL) | $(DATA_DIR)/quick
	$(GEN_TOOL) $(DATA_DIR)/quick 32 64 2 4 2 259 64

$(PERF_DATA): $(GEN_TOOL) | $(DATA_DIR)/perf
	$(GEN_TOOL) $(DATA_DIR)/perf 128 256 4 8 4 259 256

quick: all
	python3 measure.py --mode quick

perf: all
	python3 measure.py --mode perf

clean:
	rm -rf $(BIN_DIR) $(DATA_DIR)
