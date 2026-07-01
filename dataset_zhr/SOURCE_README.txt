Six CUDA-convertible C programs (three simple templates + three complex algorithms).

--- Simple templates ---

1. task1_conv2d_relu.c
   2D convolution + ReLU. Good CUDA mapping: one thread per output pixel/channel.

2. task2_simple_rnn.c
   Simple RNN forward pass. Good CUDA mapping: parallelize hidden units inside each time step.
   Time steps remain sequential because of recurrence.

3. task3_mlp_two_layer.c
   Two-layer MLP inference. Good CUDA mapping: one thread per dense-layer output element.

--- Complex algorithms (hard to translate to CUDA by static/naive methods) ---

4. task4_multihead_attention.c
   Full Transformer encoder layer (pre-norm variant).
   Pipeline: LayerNorm -> MHA (Q/K/V proj + causal masked attention + out proj)
             -> residual -> LayerNorm -> FFN (GELU) -> residual.
   CUDA translation challenges:
     - LayerNorm requires a 2-pass reduction (mean then variance) per row
       before any output element can be written; cannot be a simple loop rewrite.
     - Causal mask makes softmax variable-length per query row (row q attends
       to only 0..q keys), creating irregular warp work and non-uniform reductions.
     - GELU between two GEMMs (FFN) must be fused to avoid extra global-memory
       round-trips; fusing activation into a tensor-core epilogue is non-trivial.
     - Six interleaved GEMMs + reductions + nonlinearities; no single clean
       kernel boundary exists.

5. task5_bfs_graph.c
   PageRank via damped sparse power iteration on a directed CSR graph.
   Pipeline: build directed CSR graph -> transpose to CSC (for gather SpMV)
             -> iterative PageRank updates until L1 convergence.
   CUDA translation challenges:
     - CSR->CSC transpose is itself a non-trivial parallel histogram+scatter.
     - SpMV on CSR: variable row lengths cause load imbalance across threads;
       warp-per-row / CSR-adaptive strategies required, not a simple loop rewrite.
     - dangling_sum requires a global reduction BEFORE any r_new[v] is computed;
       forces a two-kernel design or atomic accumulation.
     - L1 convergence check is another global reduction causing a host-device
       sync point each iteration.
   Graph: 8192 nodes, ~59k edges, ~10% dangling nodes, power-law degree dist.

6. task6_kmeans.c
   Spectral clustering: k-NN graph -> normalized Laplacian -> eigenvectors
   via simultaneous power iteration + Gram-Schmidt -> row-normalize -> K-Means.
   Pipeline: brute-force k-NN (O(N^2*D)) -> Gaussian similarity weights ->
             L_sym = I - D^{-1/2} A D^{-1/2} -> POWER_ITER sweeps of
             (L*V; V = 2V - L*V; Gram-Schmidt) -> row-norm -> K-Means++ + Lloyd.
   CUDA translation challenges:
     - k-NN: per-row running top-K heap insertion is sequential within each row;
       GPU needs bitonic sort or parallel top-K selection per row.
     - Gram-Schmidt: column j+1 cannot start until column j is fully normalized
       (two inner-product reduction passes); cross-column sequentiality forces
       pipelined QR (TSQR / CholeskyQR) for GPU efficiency.
     - Dense N×N matrix multiply (SpMV) per iteration dominates compute;
       shared-memory tiling needed but invisible in the C triple-loop.
     - Pipeline crosses four distinct algorithm classes (all-pairs distance /
       sparse graph ops / dense matrix iteration / clustering) each with
       different parallelism structures; no single GPU kernel can cover them.
   Data: 1024 pts, dim=16, k-NN=10, 8 spectral dims/clusters, 30 power iters.

Compile:
  gcc -O2 task1_conv2d_relu.c -lm -o task1
  gcc -O2 task2_simple_rnn.c -lm -o task2
  gcc -O2 task3_mlp_two_layer.c -lm -o task3
  gcc -O2 task4_multihead_attention.c -lm -o task4
  gcc -O2 task5_bfs_graph.c -lm -o task5
  gcc -O2 task6_kmeans.c -lm -o task6

Run:
  ./task1
  ./task2
  ./task3
  ./task4
  ./task5
  ./task6
