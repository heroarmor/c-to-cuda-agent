# Benchmark Metadata

This file holds the analysis-oriented metadata that used to live in each benchmark's header comment: description, architecture/pipeline, parallelizable regions, parallelism patterns, and difficulty notes.

It is deliberately kept **out of the `.c` files themselves**. The `.c` files in `single_kernel/` and `multi_kernels/` are meant to be handed to the conversion agent as plain sequential C programs, with **zero comments at all** — no description, no architecture, no parallelizable-region hints, not even a comment explaining the output format. The agent's job is to find the hotspots and parallelize them itself, and to figure out from the code (not from a comment) which observable behavior it must preserve. If hints like "parallelizable regions: matmul(), rmsnorm()..." or "recommended task: only parallelize matmul()" were sitting in the source comments, the agent would be reading the answer key instead of doing the analysis. So: this file is for humans and for evaluation/reporting tooling (e.g. bucketing results by difficulty), never for the agent's conversion input.

## Measurement and verification are external, not embedded

The `.c` files contain no timing code and no self-check: they are plain sequential C programs that just compute and print their results (sums, logits, final loss, generated tokens, etc.), nothing else. Timing and correctness checking both live in `measure.py` instead:

- **Timing:** wall-clock around the whole process (`time.perf_counter()` in Python around the subprocess), not an internal `clock_gettime()` region. This matches the project's actual goal — end-to-end program speedup, not an isolated kernel benchmark — and means a future CUDA/OpenMP translation doesn't need to reproduce any internal timing instrumentation either, just be faster to run.
- **Verification:** `measure.py --update-golden` captures a known-good run's full stdout per benchmark into `golden/<mode>.json`. Every subsequent run (including, eventually, a parallel translation's run) is diffed against that golden output line-by-line, with numbers compared using float tolerance so harmless platform/compiler floating-point noise doesn't register as a regression.

A correct CUDA/OpenMP translation needs to keep producing the same stdout (within that float tolerance) for the same reason it needs to preserve any other observable behavior — this isn't a special-cased contract to satisfy, it falls out of "translate the program correctly." Each entry below still notes informally what the heaviest part of each benchmark's computation is, for context.

---

## reduction.c

- **Path:** `single_kernel/reduction.c`
- **Also known as:** `reduction_stats.c`
- **Description:** Generates an array of floating-point values and computes sum, minimum, maximum, mean, and variance. The main computation is sequential and contains clear reduction patterns.
- **Heaviest computation:** sum/min/max/variance over the array (array initialization is comparatively cheap but is included in the measured wall-clock time, see "Measurement and verification are external" above).
- **Build/run reference:** `gcc -O2 reduction.c -o reduction -lm`; `./reduction 1000000` (array size as the only argument).

## CNN.c

- **Path:** `multi_kernels/CNN.c`
- **Description:** A small, complete CNN inference pipeline.
- **Architecture:**
  - Input: 1 x 8 x 8 grayscale image
  - Conv2D: 2 filters, kernel size 3 x 3, stride 1, no padding → output 2 x 6 x 6
  - ReLU: elementwise activation
  - MaxPool2D: 2 x 2 pooling, stride 2 → output 2 x 3 x 3
  - Flatten: 18 features
  - FullyConnected: 18 → 3 classes
- **Parallelizable regions:** convolution loops, ReLU loops, max pooling loops, fully connected loops.
- **Heaviest computation:** the forward pass (conv → relu → maxpool → flatten → fc), repeated `iterations` times so total wall-clock divided by `iterations` is measurable above process-startup noise.
- **Build/run reference:** `gcc -O2 CNN.c -o CNN -lm`; `./CNN [iterations]` (optional repeat count, default 1, for stable timing on this very small/fast kernel — does not change the printed result).

## image_process.c

- **Path:** `multi_kernels/image_process.c`
- **Also known as:** `image_histogram_equalization`
- **Description:** RGB image histogram equalization, converted from a CUDA-style histogram equalization pipeline into a complete sequential C program. Does not depend on CUDA or external image libraries; input images are generated deterministically.
- **Pipeline:**
  1. Generate synthetic RGB images with values in [0, 1]
  2. Convert RGB image to grayscale unsigned char image
  3. Build a 256-bin grayscale histogram
  4. Compute the cumulative distribution function (CDF)
  5. Apply histogram equalization to each RGB channel
  6. Print output statistics for each image
- **Parallelizable regions:** RGB-to-grayscale conversion; histogram construction (reduction or privatized histograms); CDF scan / prefix sum; histogram equalization over pixels.
- **Heaviest computation:** the core pipeline (grayscale → histogram → CDF → equalization) across all generated images.
- **Build/run reference:** `gcc -O2 image_process.c -o image_process -lm`; `./image_process [width height num_images]` (defaults: 512 512 10).

## stream_cluster.c

- **Path:** `multi_kernels/stream_cluster.c`
- **Also known as:** `stream_cluster_facility_location`
- **Description:** A hard streaming clustering workload. Simulates a stream of high-dimensional points arriving in batches; for each batch, dynamically decides whether to open new cluster centers, assigns points to the nearest center, updates center locations, and computes clustering quality metrics.
- **Pipeline:**
  1. Generate streaming points, split into batches
  2. For each batch: assign points to nearest existing center → evaluate candidate points as possible new centers → dynamically open new centers if the gain is positive → reassign points after opening centers → refine centers for several passes (assign → accumulate → update → remove empty centers)
  3. After all batches: re-evaluate total clustering cost over the whole stream, print final centers and statistics
- **Heaviest computation:** generating/clustering the stream and re-evaluating the final cost.
- **Build/run reference:** `gcc -O2 stream_cluster.c -o stream_cluster -lm`; `./stream_cluster [num_points dimension batch_size max_centers passes]` (defaults: 20000 8 2000 64 3).

## tiny_cnn_training_single_file.c

- **Path:** `multi_kernels/tiny_cnn_training_single_file.c`
- **Source:** Merged and adapted from the training-related code structure in https://github.com/rajarshidattapy/CNN (originally split across `src/tensor.c`, `src/conv.c`, `src/activations.c`, `src/loss.c`, `src/optim.c`, `src/model.c`, `src/train.c`, `main.c`). This single-file version merges the training path into one standalone sequential C program and generates a synthetic 28x28 binary-class image dataset directly in C instead of loading one from external files.
- **Description:** A CNN training program (forward pass, cross-entropy loss, backward pass, SGD update) — harder than plain CNN inference because training adds backward propagation and gradient accumulation; convolution backward is especially involved since it accumulates gradients into weights, bias, and input tensors through deeply nested loops.
- **Architecture:**
  - Input: 1 x 28 x 28 synthetic grayscale image
  - Conv1: 1 → 8, kernel 3x3, stride 1, padding 1, ReLU
  - Conv2: 8 → 16, kernel 3x3, stride 2, padding 1, ReLU
  - Conv3: 16 → 32, kernel 3x3, stride 2, padding 1, ReLU
  - Conv4: 32 → 2, kernel 7x7, stride 1, padding 0, Softmax
- **Pipeline:** generate synthetic images/labels → build model → for each epoch: forward pass, cross-entropy loss, softmax backward, convolution/ReLU backward, SGD update, loss/accuracy reporting.
- **Parallelizable regions:** convolution forward loops, convolution backward loops, ReLU forward/backward loops, softmax reduction and normalization, cross-entropy loss over samples, SGD parameter update loops.
- **Heaviest computation:** training itself (forward pass, backward pass, SGD update across all epochs).
- **Build/run reference:** `gcc -O2 tiny_cnn_training_single_file.c -o tiny_cnn_training_single_file -lm`; `./tiny_cnn_training_single_file [num_samples epochs learning_rate]` (defaults: 64 10 0.01).

## llama2_c_inference.c

- **Path:** `multi_kernels/llama2_c_inference.c`
- **Source:** Adapted from llama2.c by Andrej Karpathy, https://github.com/karpathy/llama2.c
- **Description:** Complete C program for Llama-2 Transformer inference. Loads a model checkpoint and tokenizer, encodes an input prompt, runs autoregressive Transformer inference, samples the next token, decodes tokens back to text, and prints the generated output.
- **Input/output:** model checkpoint, tokenizer file, prompt string, generation options → generated text (a tokens-per-second diagnostic from the upstream code is printed to stderr, separate from the stdout that `measure.py` checks).
- **Main pipeline:** parse CLI args → load Transformer checkpoint → memory-map model weights → allocate runtime activation buffers and KV cache → load tokenizer vocabulary → build sampler → encode input prompt into tokens → autoregressive generation loop (forward pass for current token/position, sample or force the next token, decode and print) → free tokenizer/sampler/model state/memory mapping.
- **Transformer forward pipeline:** token embedding lookup → per layer: RMSNorm before attention, Q/K/V projections, write K/V into KV cache, RoPE positional encoding, multi-head attention scores, softmax over previous positions, weighted sum over value vectors, output projection, residual connection, RMSNorm before FFN, FFN projections, SwiGLU activation, FFN output projection, residual connection → final RMSNorm → classifier projection to logits.
- **Parallelizable regions:** `matmul()`, `rmsnorm()`, `softmax()`, attention score computation, attention weighted value accumulation, SwiGLU activation, residual vector updates, logits projection.
- **Parallelism patterns:** dense matrix-vector multiplication, vector map, reduction, softmax reduction and normalization, independent multi-head attention computation, autoregressive sequential dependency across generated tokens, layer-by-layer dependency inside the Transformer, KV-cache update and reuse.
- **Difficulty label:** program complexity: very hard. Basic OpenMP conversion difficulty: medium. Full OpenMP conversion difficulty: hard. CUDA end-to-end conversion difficulty: very hard.
- **Important difficulty note:** this program should not automatically be treated as a very hard task if the expected transformation is only to add a single OpenMP pragma to `matmul()` — its outer loop is naturally parallel since each iteration writes a different output element, so a basic OpenMP version is easy to obtain. It becomes a genuinely hard task when the target is end-to-end parallel Transformer inference rather than only parallel matrix-vector multiplication: the parallelization must then preserve autoregressive token dependency, layer dependency, KV-cache semantics, attention behavior, softmax normalization, and numerical correctness while parallelizing multiple regions of the inference pipeline.
- **Recommended dataset usage:**
  - Medium task: only parallelize `matmul()` with OpenMP.
  - Hard task: parallelize multiple CPU regions, including matmul, RMSNorm, softmax, attention score computation, attention value aggregation, SwiGLU, and residual updates.
  - Very hard task: convert the full sequential C Transformer inference pipeline into an end-to-end CUDA or heterogeneous CPU/GPU program, including GPU memory management, kernel launches, KV-cache handling, softmax/reduction kernels, and correctness verification.
- **Heaviest computation:** the `generate()` call (the autoregressive generation loop).
- **Build/run reference:** `gcc -O2 llama2_c_inference.c -o llama2_c_inference -lm`; `./llama2_c_inference model.bin -n 256 -i "Once upon a time"`.
- **Synthetic checkpoint/tokenizer:** this dataset does not ship a real pretrained model. `tools/llama2_gen_checkpoint.c` writes a tiny, fully deterministic checkpoint + tokenizer pair instead (see `dataset_zyj/Makefile` for the exact dimensions used to produce `data/quick/` and `data/perf/`).
