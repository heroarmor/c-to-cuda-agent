# Build rules for ZHR's sequential C benchmark dataset.
#
# Usage:
#   make            build every benchmark binary into bin/
#   make quick      build, then run measure.py in quick correctness mode
#   make perf       build, then run measure.py in perf timing mode
#   make clean      remove build output

CC    ?= cc
STD    = -std=c11
WARN   = -Wall -Wextra
OPT    = -O2
LIBS   = -lm

BIN_DIR := bin
VPATH   := single_kernel:multi_kernels

NAMES := \
	task1_conv2d_relu \
	task2_simple_rnn \
	task3_mlp_two_layer \
	task4_multihead_attention \
	task5_bfs_graph \
	task6_kmeans

BINARIES := $(addprefix $(BIN_DIR)/,$(NAMES))

.PHONY: all quick perf clean

all: $(BINARIES)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/%: %.c | $(BIN_DIR)
	$(CC) $(STD) $(WARN) $(OPT) $< -o $@ $(LIBS)

quick: all
	python3 measure.py --mode quick

perf: all
	python3 measure.py --mode perf

clean:
	rm -rf $(BIN_DIR)
