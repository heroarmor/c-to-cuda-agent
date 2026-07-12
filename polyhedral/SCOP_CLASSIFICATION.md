# Phase 0 — SCoP classification of the benchmark suite

Goal of this pass: **before** investing in a PPCG-based polyhedral backend, measure
how much of the benchmark suite it can actually address. A polyhedral tool only
works on **SCoPs** (Static Control Parts): loop nests with affine bounds and
affine array subscripts, no data-dependent control flow steering the
iteration/access, no indirection, no pointer-chasing / recursion-defined
structure. Loop-carried dependences are fine — polyhedral *scheduling* is exactly
what resolves them (skewing, wavefronting, tiling).

Each program below is classified by its **hot computational kernel** (the region a
GPU port must parallelize), not its setup/verification code.

## Buckets

- **A · PPCG** — the hot kernel is a clean affine SCoP. PPCG can own the whole
  parallelization + tiling, correct by construction. → route to the polyhedral backend.
- **B · Hybrid** — a real affine sub-kernel worth tiling, wrapped in
  sequential / recursive / pivoted / periodic-modulo control PPCG can't own. →
  PPCG on the sub-kernel, host (or LLM) on the glue.
- **C · LLM** — not polyhedral-addressable. Two sub-kinds:
  - **C-easy** — embarrassingly parallel *map/reduction*, but the body is a data-
    dependent loop / RNG recurrence / branchy struct code. Trivial to parallelize,
    but PPCG adds nothing; the LLM path already nails these.
  - **C-hard** — genuinely irregular: sparse/indirect (gather/scatter), bit-flip
    addressing, `<complex.h>`, tree traversal, convergence-dependent iteration.

## Result

| Program | Tier / field | Bucket | Why |
|---|---|---|---|
| `saxpy` | easy/dense-linalg | **A** | single affine loop, unit stride, no deps |
| `gemm` | easy/dense-linalg | **A** | textbook affine triple loop; the canonical tiling case |
| `heat2d` | easy/pde | **A** | 5-pt stencil, affine; sequential *time* loop stays on host |
| `wave2d` | moderate/pde | **A** | 5-pt stencil (3-level time recurrence); per-step kernel affine, host time loop |
| `sobel` | easy/image | **A** | fixed 3×3 stencil, affine (FNV checksum tail is a sequential host reduction — keep it there) |
| `tensor_contraction` | easy/tensor | **A** | affine in (i,j,m,kl); may need explicit subscripts / delinearization (written with base-pointer offsets) |
| `nbody` | easy/nbody | **A** | dense all-pairs O(N²), affine `x[i]`/`x[j]`; reduction inner; host time loop |
| `cholesky` | complex/factorization | **A** | PolyBench-class affine factorization (no pivoting); caveat: SPD early-`return` + `sqrt` need handling |
| `needleman_wunsch` | complex/bio | **A** | 2D DP, affine, *uniform* N/W/NW deps → **needs skewing/wavefront** (a polyhedral specialty; LLM gets it wrong naively) |
| `lu` | complex/factorization | **B** | trailing update is affine GEMM, but **partial pivoting** (argmax + row swap) is data-dependent |
| `qr` | complex/factorization | **B** | reflector loops affine, but sequential column dependency + data-dependent sign/`continue` |
| `multigrid` | complex/multigrid | **B** | each smoother/restrict/prolong is an affine stencil, but the **recursive V-cycle** + per-level alloc is host control |
| `lbm` | complex/cfd | **B** | structured D2Q9 stencil, but **periodic modulo** streaming + q-direction table lookups; affine only after unroll-q + index-set wrap-split |
| `rgf` | complex/physics | **B** | per-block B×B GEMMs are affine (batchable), but the block sweep is a sequential recurrence + pivoted inverse |
| `mc_pi` | easy/montecarlo | **C-easy** | per-stream LCG **recurrence** + reduction; no affine array nest |
| `mandelbrot` | easy/rendering | **C-easy** | affine parallel map, but **data-dependent escape-time `while`** per pixel |
| `lorenz_ensemble` | easy/ode | **C-easy** | embarrassingly-parallel ensemble map; inner **time recurrence** on private scalars + helper calls |
| `raytrace` | moderate/rendering | **C-easy** | per-pixel map, but branchy body (hit/shadow), structs, function calls — not a straight-line statement |
| `pathtrace` | complex/rendering | **C-easy** | per-pixel Monte Carlo; **recursion**, Russian roulette, 3 material branches |
| `spmv` | easy/sparse-linalg | **C-hard** | CSR: data-dependent loop bound `rp[i]..rp[i+1]` + gather `x[col_idx[k]]` |
| `spgemm` | complex/sparse-linalg | **C-hard** | sparse×sparse, symbolic pass, unknown output structure, SPA indirection |
| `pagerank` | moderate/graph | **C-hard** | CSR graph; scatter `next[col[e]] += …` |
| `conjugate_gradient` | moderate/solver | **C-hard** | host-orchestrated multi-kernel; core is SpMV (indirect) + dot/axpy |
| `jacobi_eigen` | complex/eigen | **C-hard** | convergence-dependent sweeps; rotations depend on matrix values |
| `fft1d` | moderate/signal | **C-hard** | bit-reversal permutation + non-affine stage strides (`len <<= 1`) |
| `statevector` | complex/quantum | **C-hard** | `<complex.h>`; bitwise gather/scatter `psi[k|bit]` (non-affine index) |
| `lanczos` | complex/manybody | **C-hard** | matrix-free 2^N operator; bit-flip scatter `out[s^mask]`; reorth + Sturm bisection |
| `dslash` | complex/lattice-qcd | **C-hard** | `<complex.h>`; 4D periodic-modulo neighbors + SU(3) blocks + CG solve |
| `pqp` | complex/geometry | **C-hard** | BVH build (recursive + qsort) + stack-based tree-pair traversal |

## Coverage summary

| Bucket | Count | Share |
|---|---|---|
| **A · PPCG owns it** | 9 | 31% |
| **B · Hybrid (PPCG sub-kernel)** | 5 | 17% |
| **C · LLM** (5 easy + 10 hard) | 15 | 52% |

**PPCG touches ~14/29 (~48%) in some capacity; owns ~9/29 (~31%) outright.**

## Group-member datasets (classified in place)

The group datasets (`dataset_zyj/`, `dataset_zhr/`) are treated as part of the same
unified benchmark set for classification, but are **left where they are** — they
follow a deliberately different contract (comment-free source so the agent gets no
hints, external `measure.py` + `golden/` verification instead of embedded
checksums/timing). The tier/field below is a *logical* label, not a physical path;
no files were moved or rewritten. `dataset_zyj/tools/llama2_gen_checkpoint.c` is a
data generator, not a workload, and is excluded.

| Program (path) | Owner | Logical tier/field | Bucket | Why |
|---|---|---|---|---|
| `dataset_zyj/single_kernel/reduction.c` | zyj | easy/dense-linalg | **A** | affine sum/min/max/variance reductions |
| `dataset_zhr/single_kernel/task1_conv2d_relu.c` | zhr | easy/dnn | **A** | dense conv, affine 6-nest + ReLU |
| `dataset_zhr/single_kernel/task3_mlp_two_layer.c` | zhr | easy/dnn | **A** | two dense (GEMM) layers, affine |
| `dataset_zhr/single_kernel/task2_simple_rnn.c` | zhr | moderate/dnn | **A** | per-step matvec affine; host-driven time recurrence (stencil-shaped) |
| `dataset_zyj/multi_kernels/CNN.c` | zyj | moderate/dnn | **A** | conv→relu→pool→fc, every stage affine (argmax tail is host) |
| `dataset_zyj/multi_kernels/image_process.c` | zyj | moderate/image | **B** | grayscale/equalize passes affine, but **histogram scatter** + **CDF scan** are not |
| `dataset_zhr/multi_kernels/task4_multihead_attention.c` | zhr | moderate/dnn | **B** | GEMMs + (triangular) softmax reductions affine; multi-head reshape/batch is host glue |
| `dataset_zyj/multi_kernels/tiny_cnn_training_single_file.c` | zyj | complex/dnn | **B** | affine conv/dense/backprop kernels buried in a struct-based `Tensor*` framework + sequential training loop |
| `dataset_zhr/multi_kernels/task5_bfs_graph.c` | zhr | moderate/graph | **C-hard** | CSR/CSC PageRank; gather/scatter indirection |
| `dataset_zhr/multi_kernels/task6_kmeans.c` | zhr | complex/clustering | **C-hard** | kNN top-k + Gram-Schmidt deps + data-dependent k-means |
| `dataset_zyj/multi_kernels/stream_cluster.c` | zyj | complex/clustering | **C-hard** | online facility-location; data-dependent center opening |
| `dataset_zyj/multi_kernels/llama2_c_inference.c` | zyj | complex/dnn | **C-hard** | autoregressive token loop; **not self-contained** (needs generated `model.bin`/`tokenizer.bin`) |

Dataset totals: **A ×5, B ×3, C ×4** (12 programs).

## Combined coverage (benchmark + datasets)

| Bucket | benchmark | datasets | **total** | share |
|---|---|---|---|---|
| **A · PPCG** | 9 | 5 | **14** | 34% |
| **B · Hybrid** | 5 | 3 | **8** | 20% |
| **C · LLM** | 15 | 4 | **19** | 46% |
| total | 29 | 12 | **41** | |

Adding the datasets **raises the PPCG-addressable share** (A+B) from 48% to 54% —
the datasets skew toward dense NN kernels (conv/GEMM/MLP), exactly PPCG's strength.

> Integration note: because the dataset programs print no embedded `time=` region,
> the `evaluation/` execution-time metric (which parses that field) does not apply
> to them as-is; their timing comes from `measure.py`'s external wall-clock. The
> polyhedral/LLM *conversion* path is unaffected — it consumes the `.c` and diffs
> stdout, which they do produce (checksums).

## Verdict

The result is clean and favorable to the hybrid plan:

- PPCG cleanly covers the **dense / stencil / affine third** (`A`) — precisely where
  tiling and correctness-by-construction matter, and where `needleman_wunsch`
  (skewing) is something the LLM path reliably gets *wrong*.
- The **hybrid tier** (`B`) is where the most interesting engineering is: dense
  factorizations, multigrid, LBM — an affine kernel PPCG should generate, wrapped
  in host control it shouldn't.
- The **irregular/complex long tail** (`C-hard`) and the **trivial maps** (`C-easy`)
  stay with the LLM pipeline, which already handles them well.

This validates spending Phase 1 effort on PPCG: it addresses a substantial,
well-defined subset rather than everything or nothing. See `DESIGN.md` for the
PPCG→LLM relay + dispatcher plan this classification feeds.

## Caveats on this classification

- Static, source-reading judgement — not yet confirmed against a real PPCG/`pet`
  build. A few `A`/`B` calls hinge on `pet` accepting the exact source form
  (`tensor_contraction`'s base-pointer offsets, `cholesky`'s early-`return`,
  `nbody`'s in-`main` region needing a `#pragma scop`). Phase 1's first job is to
  confirm these on real `pet`.
- Order-sensitive **integer** checksums (`sobel` FNV, `pagerank`, `needleman_wunsch`,
  `fft1d`) must stay host-sequential to reproduce exact stdout — PPCG naturally
  leaves that non-SCoP code on the host. Float reductions may reorder; the
  evaluation's tolerance-aware compare absorbs that.
