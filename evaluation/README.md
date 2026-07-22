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
Wall-clock + output tokens for zyj's agent pipeline
(`agent_pipeline/run_pipeline.py`) to produce each conversion.
`scripts/codegen_time.sh` times the pipeline and reads token usage from
opencode's SQLite store. The pipeline writes to `generated/`, so the tracked
`cuda/*.cu` conversions are never clobbered.

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

## Running on the server (UMich Great Lakes, from a clean checkout)

The GPUs here are Slurm-managed and `Exclusive_Process`, so anything that
compiles/runs a `.cu` (verify, profile, timing) must run **inside a GPU job**,
not on a login node. The steps below walk a fresh `git clone` through to a full
evaluation. The example Slurm flags (`--account=nbleier_owned1`,
`--partition=gpu-rtx6000`, `--gres=gpu:1`) mirror `polyhedral/verify_gpu.sbatch`
— swap in your own account/partition.

### 1. Build the environment

```sh
cd ~/ME450

# a) CUDA toolchain (nvcc) — load on any GPU job / interactive GPU shell.
#    This is what the server uses (not the laptop's 'faiss' conda env that
#    cuda/env.sh assumes); prefer the module on Great Lakes.
module load cuda/12.8.2

# b) Polyhedral (PPCG) backend — needed for --backend ppcg|hybrid|auto and the
#    compiler retile optimize move. One-shot, ~15 min, run on a LOGIN node
#    (it uses the llvm/14.0.6 module). Skips to a smoke test if already built.
./polyhedral/build_ppcg.sh                     # re-run with --check to verify

# c) LLM stages — need `bun` and an opencode checkout at the repo root. The free
#    zen model below needs NO API key. run_dataset.py ALWAYS needs this: even
#    --backend ppcg --optimizer compiler still runs the verify + profile stages
#    as opencode agents (only generate + optimize go zero-token). The only fully
#    opencode-free path is standalone PPCG codegen (step 2), not the full loop.
curl -fsSL https://bun.sh/install | bash        # if bun isn't already on PATH
git clone https://github.com/sst/opencode        # -> ./opencode (gitignored)
( cd opencode && bun install --ignore-scripts )
```

Python is stdlib-only (3.10+) — no `pip install` needed for the eval scripts.
`cc`/`gcc` (for the C baselines) and `nvcc` (from the module) are the only
compilers required. llama2's synthetic checkpoint/tokenizer are built on demand
by `run_dataset.py` into `evaluation/data/` (gitignored) — nothing to fetch.

### 2. Sanity-check the deterministic path (no model, cheap)

```sh
./polyhedral/build_ppcg.sh --check                          # ppcg emits CUDA
python3 polyhedral/ppcg_to_cu.py benchmark/easy/dense-linalg/gemm.c
sbatch polyhedral/verify_gpu.sbatch easy/dense-linalg/gemm  # golden-diff on a GPU node
```

### 3. Run the evaluation

**End-to-end pipeline over all benchmarks** (`run_dataset.py`, this suite's
driver): generates each `.cu` from scratch via generate → verify → profile →
optimize and reports compiled/ran/correct/speedup per workload. Needs a GPU +
`bun`/opencode + the model, so submit it as a GPU job. Exports land in
`generated/` (tracked `cuda/*.cu` untouched); the table + a KernelBench-style
rollup land in `evaluation/results/`.

```sh
# a whole run is long (the LLM path is minutes–tens-of-minutes per benchmark),
# so scope with --filter and/or route the affine subset to the zero-token
# deterministic backend with --backend auto.
sbatch --account=nbleier_owned1 --partition=gpu-rtx6000 --gres=gpu:1 \
       --cpus-per-task=4 --mem=16G --time=08:00:00 \
       --job-name=eval-all --output=evaluation/results/run_dataset.log \
       --wrap 'module load cuda/12.8.2 && export PATH="$HOME/.bun/bin:$PATH" && \
               python3 evaluation/run_dataset.py \
                 --model opencode/deepseek-v4-flash-free \
                 --backend auto --optimizer hybrid'

# smaller slices (same --wrap body, shorter --time):
#   python3 evaluation/run_dataset.py --filter easy               # one tier
#   python3 evaluation/run_dataset.py --filter easy/dnn,saxpy     # domain/name
#   python3 evaluation/run_dataset.py --backend ppcg --optimizer compiler  # zero generate/optimize
#         tokens on the affine subset (verify/profile still run as agents, so opencode is still needed)
```

**Post-process the exports** (no GPU / no model needed for the rollup):

```sh
python3 evaluation/scripts/backend_report.py generated/     # per-backend rollup -> results/backend_report.md
# large-size timing (clears the ~80 ms CUDA driver-init floor) — GPU job:
sbatch --account=nbleier_owned1 --partition=gpu-rtx6000 --gres=gpu:1 --time=02:00:00 \
       --wrap 'module load cuda/12.8.2 && python3 evaluation/scripts/scale_exports.py'
```

**Score the tracked reference conversions instead of the pipeline's**
(`run_eval.sh` → `cuda/bench.sh`): correctness (`fast_0`), speedup
(`fast_1`/geomean), LoC/tidiness over `cuda/*.cu`. It sources `cuda/env.sh`,
which is written for the laptop `faiss` conda env; on Great Lakes run it inside
a GPU job and override the toolchain vars (`CUDA_ARCH` for the RTX 6000, `NVCC`
from the module) or adapt `env.sh` to `module load` instead of `conda activate`.

```sh
./evaluation/run_eval.sh check     # fast: correctness + exec time + LoC -> results/eval_check.md
./evaluation/run_eval.sh full      # heavy (~10s-GPU sizes, hours): results/eval_full.md
# heavier per-metric probes:
./evaluation/scripts/hw_util.sh      complex/cfd/lbm 512 512 20000
./evaluation/scripts/scalability.sh  complex/cfd/lbm "128 128 500" "256 256 2000" "512 512 8000"
./evaluation/scripts/codegen_time.sh easy/dense-linalg/saxpy        # uses the model; restores the .cu
```

### 4. Where results land

All reports and raw TSVs land in `evaluation/results/` (gitignored except
`.gitkeep`): `run_dataset.py` → `report_<timestamp>.txt` (+ `--json`),
`run_eval.sh` → `eval_{check,full}.md` + `bench_*.tsv`/`loc.tsv`,
`backend_report.py` → `backend_report.md`, `scale_exports.py` → `scale_report.md`.

> Caveat carried from the suite: `jacobi_eigen`'s reported `sweeps` count is not
> bit-portable across gcc/nvcc near its 1e-12 convergence threshold, so it can
> miss `fast_0` at large `n` even though its eigenvalues are exact.
