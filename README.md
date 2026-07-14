# C-to-CUDA Conversion Agent

A capstone design project building an agent that **automatically and efficiently
converts C programs into CUDA programs**. Unlike tools that only optimize an
existing kernel, this agent generates the *complete* CUDA program — both the
**host** code (memory management, data transfer, kernel launch configuration)
and the **device** code (kernels) — from plain C source.

## How it works

One sequential C file goes through a fixed, externally-orchestrated loop of
four opencode LLM agents (`agent_pipeline/run_pipeline.py`), with two
deterministic compiler paths plugged into its slots:

```
              ┌─ SCoP? ──────►  PPCG generate  (--backend ppcg) ──┐
  C source ───┤  bucket B? ──►  PPCG partial + LLM stitch (hybrid)├─►  verify → profile → optimize ─► best .cu
              └─ otherwise ──►  LLM generate ─────────────────────┘        ▲            │
                                                                           └────────────┘
                                                 optimize slot per iteration (--optimizer):
                                                 nvcc flag search / PPCG re-tiling (compiler)
                                                 or the cuda-optimize agent (llm); hybrid = both
```

- **Correctness first** — every iteration recompiles, reruns, and diffs the
  program against a one-time mechanically-captured C reference (tolerance-aware,
  never judged by eye). Polyhedral (PPCG) conversions are correctness-preserving
  by construction.
- **Objective signals over self-report** — stopping (patience over best-ever
  time), best-version tracking, and speedup numbers all come from mechanical
  measurement, never from an agent's own claims.
- **Zero-token deterministic subset** — `--backend ppcg --optimizer compiler`
  converts and tunes the affine subset (~1/3 of the suite) without any model
  calls; the LLM covers the irregular long tail.

## Repository layout

```
agent_pipeline/    the 4-agent loop: orchestrator, JSON schemas, compiler
                   optimize moves, opencode agent + skill definitions
                   (see ARCHITECTURE.md for the full design)
polyhedral/        deterministic PPCG backend: toolchain build, C→.cu wrapper
                   with delinearization, SCoP prefilter/targets, GPU
                   validation jobs (see DESIGN.md + README.md)
benchmark/         42 self-contained C workloads across easy/moderate/complex
                   tiers (dense/sparse linear algebra, PDE, graphs, rendering,
                   physics, DNN, ...)
cuda/              reference conversions (cuda/<rel>.cu)
evaluation/        KernelBench-style metric suite: correctness (fast_0),
                   speedup (fast_1/geomean), HW utilization, codegen cost,
                   LoC, scalability (see evaluation/README.md)
generated/         pipeline exports: <name>/<name>.cu + nvcc_flags.txt +
                   pipeline_result.json (gitignored, reproducible)
```

## Getting started

Deterministic paths (no model needed):

```sh
./polyhedral/build_ppcg.sh                                   # once: PPCG+isl+pet toolchain
python3 polyhedral/ppcg_to_cu.py benchmark/easy/dense-linalg/gemm.c   # C → .cu via PPCG
sbatch polyhedral/verify_gpu.sbatch easy/dense-linalg/gemm   # golden-diff on a GPU node
```

Full agent pipeline (LLM stages need `bun` and an [opencode](https://github.com/sst/opencode)
clone at the repo root — `bun install --ignore-scripts` — plus any model;
`opencode/deepseek-v4-flash-free` on opencode's zen gateway works with **no
API key**):

```sh
python3 agent_pipeline/run_pipeline.py benchmark/complex/multigrid/multigrid.c \
    --backend hybrid --optimizer hybrid --model opencode/deepseek-v4-flash-free
```

On a Slurm cluster, run the full pipeline inside its own GPU job (GPUs here are
`Exclusive_Process`; see `polyhedral/README.md` for the sbatch shapes used).

## Status

Everything below has been exercised on real hardware (RTX PRO 6000, CUDA 12.8):

- **Polyhedral backend (Phases 0–2)** — SCoP classification of all 41
  workloads; PPCG codegen with flat-pointer delinearization; `saxpy`/`gemm`
  full conversions and `multigrid`/`rgf` hybrid partials all pass the golden
  diff; prefilter-routed dispatch with graceful LLM fallthrough.
- **Compiler optimize moves (Phase 2.5)** — profile-guided nvcc flag search
  and PPCG `--sizes` re-tiling take optimize-loop slots deterministically,
  gated by mechanical compile/run/diff/time acceptance.
- **Hybrid relay (Phase 3) + full loop** — end-to-end run on `multigrid` with
  a free model: PPCG partial → LLM stitch, 5 iterations, every verify PASS,
  compiler and LLM moves alternating (2 flags accepted, `KernelFusion` the
  big win), 0.436s → 0.272s, best iteration exported with its flags.
- **Per-backend evaluation over the easy tier + bucket-B** — 12/14 pipeline
  runs converted cleanly (`--backend auto --optimizer hybrid`, free model);
  auto-routing sent `saxpy`/`gemm`/`conv2d_relu`/`nbody`/`lorenz_ensemble` to
  the ppcg backend (mean codegen 686 s vs the LLM path's 1247 s). At large
  problem sizes (`evaluation/scripts/scale_exports.py`) the exported
  conversions reach **135× (gemm n=2048)**, 20× (lorenz), 12× (mandelbrot),
  8× (reduction), 5× (heat2d).
- **Honest limits** — at the suite's small default sizes the CUDA driver-init
  floor (~80 ms) exceeds many whole C runtimes, so default-size speedups
  mostly read <1×; the evaluation reports this rather than hiding it.
  `nbody`'s conversion mismatches at 4× its default size (under
  investigation); `saxpy`-class streaming stays copy-bound at any size;
  `heat2d`/`lu`/`qr`/`lbm` await region-level SCoP markers.

## Documentation map

| Doc | Contents |
|---|---|
| `agent_pipeline/ARCHITECTURE.md` | the 4-agent loop, stop conditions, skills, compiler moves |
| `polyhedral/DESIGN.md` | why PPCG, the relay architecture, phase plan, risks |
| `polyhedral/SCOP_CLASSIFICATION.md` | per-program SCoP buckets (A/B/C) with reasons |
| `polyhedral/README.md` | toolchain usage + per-phase validation status |
| `evaluation/README.md` | the six KernelBench-adapted metrics |

## License

TBD
