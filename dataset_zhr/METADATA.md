# Benchmark Metadata

This dataset mirrors the structure used by `dataset_zyj`: agent-facing C programs live under `single_kernel/` and `multi_kernels/`, while descriptions, CUDA-parallelization notes, build commands, and verification details live outside the source files.

The `.c` files are plain sequential C programs with their instructional comments removed. They compute deterministic synthetic workloads and print checksums or summary values. Timing and correctness checking are handled externally by `measure.py`.

## Measurement and Verification

- **Timing:** wall-clock time around the whole benchmark process, measured from Python with `time.perf_counter()`.
- **Verification:** full stdout is compared against `golden/<mode>.json` line-by-line. Numeric values use a small floating-point tolerance to avoid false failures from harmless compiler/platform differences.
- **Modes:** the programs do not expose size arguments, so quick and perf currently run the same binaries with no CLI arguments. Perf mode still uses more repetitions for timing stability.

## Source

Imported from sibling directory:

`/Users/zhanghaoran/Desktop/zhr/26summer/450/cuda_convertible_c_tasks`

The original English and Chinese READMEs were copied to `SOURCE_README.txt` and `SOURCE_README_zh.md` for reference.

---

## task1_conv2d_relu.c

- **Path:** `single_kernel/task1_conv2d_relu.c`
- **Description:** 2D convolution over a 32 x 32 x 3 synthetic image followed by ReLU.
- **Parallelizable regions:** output channels and output pixels are independent.
- **CUDA pattern:** one thread per output element.
- **Build/run reference:** `gcc -O2 task1_conv2d_relu.c -lm -o task1_conv2d_relu`; `./task1_conv2d_relu`.

## task2_simple_rnn.c

- **Path:** `single_kernel/task2_simple_rnn.c`
- **Description:** simple RNN forward pass over 32 time steps with 64 input features and 128 hidden units.
- **Parallelizable regions:** hidden-unit computations within each time step.
- **Sequential dependency:** time steps remain ordered because each step depends on the previous hidden state.
- **Build/run reference:** `gcc -O2 task2_simple_rnn.c -lm -o task2_simple_rnn`; `./task2_simple_rnn`.

## task3_mlp_two_layer.c

- **Path:** `single_kernel/task3_mlp_two_layer.c`
- **Description:** two-layer MLP inference over a batch of 64 samples, with dimensions 128 -> 256 -> 10.
- **Parallelizable regions:** dense-layer output elements and batch items.
- **CUDA pattern:** one thread or tiled block strategy per dense-layer output element.
- **Build/run reference:** `gcc -O2 task3_mlp_two_layer.c -lm -o task3_mlp_two_layer`; `./task3_mlp_two_layer`.

## task4_multihead_attention.c

- **Path:** `multi_kernels/task4_multihead_attention.c`
- **Description:** full Transformer encoder layer using pre-norm, multi-head causal attention, residual connections, and a GELU feed-forward network.
- **Architecture:** batch size 2, sequence length 32, model dimension 128, 4 heads, FFN width 256.
- **Parallelizable regions:** linear projections, layer normalization reductions, attention score computation, causal softmax, value aggregation, GELU, FFN projections, residual updates.
- **Difficulty notes:** layer normalization needs two-pass reductions; causal softmax has variable valid lengths per row; the pipeline interleaves several GEMM-like operations with reductions and nonlinearities.
- **Build/run reference:** `gcc -O2 task4_multihead_attention.c -lm -o task4_multihead_attention`; `./task4_multihead_attention`.

## task5_bfs_graph.c

- **Path:** `multi_kernels/task5_bfs_graph.c`
- **Description:** PageRank-style sparse graph power iteration over a deterministic directed CSR graph.
- **Pipeline:** build CSR graph, transpose to CSC, run damped PageRank iterations until convergence, print top nodes and checksum.
- **Parallelizable regions:** graph construction, CSR-to-CSC histogram/scatter, dangling-node reduction, sparse gather, rank updates, L1 residual reduction.
- **Difficulty notes:** sparse row lengths are irregular, each iteration has global reductions, and convergence checking creates synchronization points.
- **Build/run reference:** `gcc -O2 task5_bfs_graph.c -lm -o task5_bfs_graph`; `./task5_bfs_graph`.

## task6_kmeans.c

- **Path:** `multi_kernels/task6_kmeans.c`
- **Description:** spectral clustering pipeline over 1024 synthetic points.
- **Pipeline:** brute-force k-NN graph, Gaussian similarity weights, normalized graph Laplacian, simultaneous power iteration with Gram-Schmidt, row normalization, K-Means clustering.
- **Parallelizable regions:** all-pairs distance computation, matrix construction, dense matrix-vector products, row normalization, K-Means assignment/update loops.
- **Difficulty notes:** k-NN top-k maintenance is sequential within each row in the C program, Gram-Schmidt has cross-column dependencies, and the workload crosses multiple algorithm families.
- **Build/run reference:** `gcc -O2 task6_kmeans.c -lm -o task6_kmeans`; `./task6_kmeans`.
