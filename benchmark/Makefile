# Benchmark suite build
#
#   make            build every benchmark into bin/ (mirroring the source tree)
#   make clean      remove bin/
#   make CC=clang   override the compiler
#
# Each .c file is a self-contained, single-translation-unit reference program --
# the input form expected by the C->CUDA conversion agent.

CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra
LDLIBS  ?= -lm

SRCS := $(shell find . -name '*.c')
BINS := $(patsubst ./%.c,bin/%,$(SRCS))

all: $(BINS)

bin/%: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

clean:
	rm -rf bin

.PHONY: all clean
