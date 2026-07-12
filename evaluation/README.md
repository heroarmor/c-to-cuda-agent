# Evaluation

Metrics for the **C → CUDA conversion agent**: how good are the generated `.cu`
programs (correct? fast? well-utilised? tidy?) and how good is the *agent* at
producing them (fast/cheap to generate? universal across domains?).

The design follows **KernelBench** (Ouyang et al., *Can LLMs Write Efficient GPU
Kernels?*, ICML 2025, [arXiv:2502.10517](https://arxiv.org/abs/2502.10517)) and
adapts it to this project.

### How this differs from KernelBench
| | KernelBench | This suite |
|---|---|---|
| Task | optimise one kernel from a PyTorch `nn.Module` | generate a **whole** host+device CUDA program from C |
| Baseline | PyTorch eager (GPU) | single-thread `gcc -O2` C reference (CPU) |
| Correctness input | 5 **random** tensors, `allclose` | **deterministic** self-checking program; numeric result-field compare (`rtol 1e-3 / atol 1e-2`) |
| Difficulty axis | Levels 1–4 (op → fused → arch → HF) | tiers easy / moderate / complex |
| Timing | CUDA events, 3 warmup + 100 trials | program-internal `time=` region (wall clock), CPU vs GPU |

Because our references are deterministic (fixed seeds / initial conditions),
one run with field-by-field numeric comparison is equivalent to KernelBench's
random-trial `allclose` — there is no input distribution to sample.

## The six metrics

### 1. Functional Correctness  → KernelBench `fast_0`
Does the conversion reproduce the reference's result line within tolerance?
`cuda/build_run.sh` compiles both, runs both, strips timing/parenthetical
fields, and compares every remaining number. **`fast_0` = correctness rate =
(#PASS)/N.** Collected by `run_eval.sh`; per-tier breakdown in the report.

### 2. Execution Time  → KernelBench `fast_p`, speedup
Speedup `= T_cpu / T_gpu` from each program's internal `time=` region.
We report the KernelBench metric
**`fast_p = (1/N) Σ 𝟙(correct_i ∧ speedup_i > p)`** — `fast_1` (correct *and*
faster than the CPU baseline), `fast_2`, and the **geometric-mean speedup** over
correct conversions. Collected by `run_eval.sh`.

### 3. Code Generation Time  *(agent metric; beyond KernelBench)*
Wall-clock + output tokens for opencode's `/cudaify` to generate each
conversion. `scripts/codegen_time.sh` times a regeneration and reads token
usage from opencode's SQLite store, **snapshotting and restoring** the verified
`.cu` so nothing is clobbered.

### 4. Hardware Utilization
`scripts/hw_util.sh <rel> <args>` runs the GPU binary while polling
`nvidia-smi`, reporting avg/peak GPU utilisation, peak memory, and peak power.
Coarse (device-wide) — for achieved occupancy and DRAM/compute throughput,
profile with Nsight Compute: `ncu --set full ./cuda/bin/<rel>.gpu <args>`
(not installed here). Best at `full`-mode sizes so the sampler catches the kernels.

### 5. Scalability & Universality
- **Scalability:** how speedup/correctness hold as the problem grows —
  `scripts/scalability.sh <rel> "args1" "args2" ...`. Iterative workloads scale
  by launch count (steps/iters); `full` mode targets ~10 s GPU.
- **Universality:** breadth the agent handles correctly — pass rate across the
  3 tiers and the distinct problem domains (dense-linalg, pde, rendering,
  lattice-qcd, …), the analogue of KernelBench's Levels. Summarised in the report.

### 6. Code Tidiness & Lines of Code
`scripts/loc.sh` reports, per workload: generated `.cu` vs reference `.c` LoC and
ratio, comment density, presence of CUDA error-checking (`CUDA_CHECK` /
`cudaGetLastError`), and distinct kernel kinds. A tidy conversion stays close to
the reference size, comments the parallel strategy, and checks every CUDA call.

## Running

```sh
cd ~/ME450
./evaluation/run_eval.sh check     # fast: correctness + exec time + LoC -> results/eval_check.md
./evaluation/run_eval.sh full      # heavy (~10s-GPU sizes, hours): results/eval_full.md

# individual / heavier metrics
./evaluation/scripts/hw_util.sh      complex/cfd/lbm 512 512 20000
./evaluation/scripts/scalability.sh  complex/cfd/lbm "128 128 500" "256 256 2000" "512 512 8000"
./evaluation/scripts/codegen_time.sh easy/dense-linalg/saxpy        # uses the model; restores the .cu
```

Reports and raw TSVs land in `evaluation/results/`. Requires the `faiss` conda
env (CUDA 12.4 + `nvcc`); `codegen_time.sh` additionally needs `opencode`.

> Caveat carried from the suite: `jacobi_eigen`'s reported `sweeps` count is not
> bit-portable across gcc/nvcc near its 1e-12 convergence threshold, so it can
> miss `fast_0` at large `n` even though its eigenvalues are exact.
