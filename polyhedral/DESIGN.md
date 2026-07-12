# Polyhedral backend — design

A **deterministic, polyhedral codegen path** for the affine subset of the C→CUDA
conversion task, integrated with the existing LLM pipeline (`agent_pipeline/`) and
measured by the existing framework (`evaluation/`). The goal is a **stronger
converter**, not a new compiler: we *integrate* PPCG rather than build a
polyhedral engine from scratch.

> Prerequisite reading: `SCOP_CLASSIFICATION.md` (Phase 0) — which programs this
> backend can address, and why. Across the 29 benchmark + 12 group-dataset
> programs (41 total): 14 PPCG-own, 8 hybrid, 19 stay LLM.

## Why PPCG, not a hand-rolled IR

Polyhedral analysis, scheduling (Pluto), tiling, and CUDA emission are a
multi-person-year body of work already realized in **PPCG** (built on `isl` +
`pet`/clang). Re-implementing even a slice of it is out of scope for this project.
Our contribution is the **orchestration**: detecting what is affine, routing it to
PPCG, and *relaying* PPCG's output into the LLM optimize loop — plus the
cross-backend comparison the `evaluation/` metrics make possible.

## Architecture: PPCG→LLM relay, not either/or

The naive design is a dispatcher that sends affine programs to PPCG and everything
else to the LLM (mutually exclusive). We can do better for the affine subset:

```
             ┌─ SCoP? ──yes──►  PPCG generate  ─┐
  C source ──┤  (classifier)                    ├─►  verify → profile → optimize  ─► .cu
             └─ no ──────────►  LLM  generate  ─┘        (zyj's existing loop)
```

For a SCoP, **PPCG produces the initial `.cu` that would otherwise come from the
`cuda-generate` agent**, then that `.cu` enters zyj's existing
`verify → profile → optimize` loop unchanged. Benefits stack:

- **Correct starting point** — polyhedral transforms are correctness-preserving by
  construction; the `verify` stage's repair budget is essentially free.
- **Strong starting point** — PPCG's auto-tiling is a real kernel, not an LLM
  first draft; the `optimize` loop climbs from a higher floor.
- **Zero-token generation** — the affine subset costs no model calls to generate
  (great for the codegen-cost metric).
- **Everything downstream is reused** — `compare_outputs.py`, golden diff,
  `verify`/`profile`/`optimize`, and `evaluation/` need no changes.

The non-SCoP path is exactly today's pipeline.

## The dispatcher = the Phase 0 classifier, automated

The routing decision is SCoP detection. Phase 0 did this by hand; the dispatcher
promotes it to an automated check. Two viable implementations, cheapest first:

1. **Let `pet` decide** — run `pet`/PPCG on the source; if it extracts a SCoP
   covering the hot region and emits code, route A. This is the ground truth and
   avoids re-implementing SCoP detection.
2. **Lightweight pre-filter** — a static scan (via `pycparser` or clang AST) for
   disqualifiers: indirect subscripts (`x[idx[k]]`), data-dependent loop bounds,
   `<complex.h>`, recursion, `while` with data-dependent condition. Cheap triage
   before paying for a PPCG attempt.

Recommended: (2) as a fast pre-filter, (1) as the authority. A program that passes
the pre-filter but PPCG then rejects simply falls through to the LLM path — the
relay degrades gracefully.

## Integration seams (the parts that actually bite)

1. **Output conventions differ.** `/cudaify`-style output lives at
   `cuda/<rel>.cu`; zyj's pipeline writes `generated/<name>/`. PPCG emits its own
   host+device file with its runtime naming. The PPCG path must **normalize its
   output** into whichever location the verify harness reads, and keep the host
   I/O + result-line printing intact so the golden diff still matches.
2. **Order-sensitive integer checksums stay on the host.** `sobel` (FNV),
   `pagerank`, `needleman_wunsch`, `fft1d` print integer checksums whose value
   depends on reduction order. PPCG parallelizes only the SCoP and leaves that
   non-SCoP tail on the host — which is exactly what preserves the checksum. Float
   reductions may reorder; the evaluation's tolerance-aware compare absorbs that.
3. **`<complex.h>` is out of scope for PPCG anyway.** `statevector`/`dslash` are
   `C-hard` (LLM path); no special handling needed here.
4. **`pet` needs a clean SCoP region.** Some `A` programs may need a `#pragma scop`
   marker or minor source normalization (explicit subscripts instead of
   base-pointer offsets in `tensor_contraction`; the SPD early-`return` in
   `cholesky`). Phase 1 confirms each against real `pet`.
5. **Build cost.** PPCG + `isl` + `pet` (+ clang/LLVM) is a non-trivial toolchain
   build, and it plus `nvcc` only exist on a GPU box — not in this dev environment.

## Compiler backend in the *optimize* stage

The relay above puts the compiler on the **generate** side only. The same idea
extends into the `verify → profile → optimize` loop itself, because the optimize
stage's contract is just "current `.cu` + profile data in, edited `.cu` +
`optimize_result.json` out" — nothing requires that edit to come from an LLM.
Three insertion points, cheapest first:

1. **`nvcc` flag search (every program, both backends).** A deterministic
   sub-step that picks compiler knobs from the measured bottleneck:
   `--maxrregcount` / `__launch_bounds__` when occupancy is register-limited,
   `-Xptxas -dlcm=cg` for scattered access, `-use_fast_math` where the
   tolerance-aware compare permits it. Zero tokens; applies to LLM- and
   PPCG-generated `.cu` alike. The existing best-version tracking + patience
   stop already absorb any candidate that regresses.
2. **PPCG re-tiling for the affine subset (bucket A, later B).** PPCG exposes
   its schedule through `--sizes` (tile/block/grid), shared-/private-memory
   flags, and unrolling. For a SCoP program, an "optimize" iteration can mean:
   pick new parameters from the measured bottleneck (occupancy low → smaller
   tiles; DRAM throughput low → different block shape), **re-run PPCG from the
   original C source**, and re-enter `verify`. That is an autotuning loop over
   a compiler — every candidate is correctness-preserving by construction, so
   the verify repair budget stays essentially free. The shaping constraint:
   **PPCG consumes C, not CUDA** — it cannot post-optimize an arbitrary
   LLM-written `.cu`, only regenerate from the `.c` (which the workdir keeps).
3. **Hybrid dispatch (compiler move vs. LLM move).** Per iteration the
   orchestrator chooses: current `.cu` is PPCG-produced and the bottleneck is
   tiling/launch-shaped → compiler move (2); bottleneck is structural
   (algorithm choice, kernel fusion, host-transfer overhead) or the program is
   bucket C → today's LLM move. `evaluation/`'s per-backend reporting then
   compares the two optimizers for free.

Wiring constraints (from the existing pipeline, both deliberate):

- **`cuda-optimize` has `bash: deny`** — the LLM optimize agent cannot invoke a
  compiler. Compiler moves therefore live in the **orchestrator**
  (`run_pipeline.py` dispatches that iteration to a Python function instead of
  an opencode agent) — consistent with the pipeline's "mechanical signal over
  agent self-report" philosophy. The function writes the same
  `optimize_result.json` shape (e.g. `"technique_applied":
  "ppcg_retile --sizes=..."`), so `schemas.py` and the loop need almost no
  changes.
- **The search must be bounded** — a small profile-guided candidate set per
  iteration, not a grid sweep, or the patience stop (3 iterations without a 5%
  win) ends the run before a tuner converges.

## Phases

- **Phase 0 — SCoP classification** ✅ *(this directory)* — coverage measured
  before investing. Result over 41 programs: 34% own (A), 20% hybrid (B), 46% LLM (C).
- **Phase 1 — PPCG codegen path** ✅ — `build_ppcg.sh` (PPCG 0.09.3 + isl 0.28 +
  pet 0.11.9 against the `llvm/14.0.6` module) and the `ppcg_to_cu.py` wrapper
  emitting one self-contained `.cu`; `saxpy`, `heat2d`, `gemm` PASS the golden
  diff on an RTX PRO 6000 (`verify_gpu.sbatch`). See `README.md` for usage.
- **Phase 2 — Relay + dispatcher** ✅ *(pipeline side)* — `run_pipeline.py
  --backend ppcg|auto`: `prefilter.py` triage → PPCG attempt → LLM fallthrough;
  `pipeline_result.json` carries a `backend` tag. Still open: `evaluation/`
  reporting per-backend `fast_1` / geomean / codegen-cost from that tag.
- **Phase 2.5 — Compiler moves in the optimize loop** — first the `nvcc` flag
  search (backend-agnostic, cheapest), then PPCG re-tiling for bucket-A
  programs (re-invoking `ppcg_to_cu.py` with new `--sizes`); orchestrator-side
  dispatch per "Compiler backend in the *optimize* stage" above. Report tuner
  moves vs. LLM moves separately in `evaluation/`.
- **Phase 3 — Hybrid (bucket B)** — for `lu`/`qr`/`multigrid`/`lbm`/`rgf`, let
  PPCG generate the affine sub-kernel and the LLM stitch the host/irregular glue;
  optionally feed PPCG's dependence/tiling analysis to the LLM as hints.

## Risks

- **Coverage is a third, not all** — frame PPCG as "deterministic backend for the
  affine subset," never "replaces the LLM." Phase 0 numbers keep this honest.
- **PPCG speed ≠ always fastest** — its win is correctness-by-construction +
  determinism + zero token cost, and being a strong *starting point* for the
  optimize loop; it won't necessarily beat cuBLAS on `gemm`. Measure, don't assume.
- **Science-project risk** — timebox Phase 1 to "one verified `saxpy.cu` from
  PPCG" before broadening. Don't let toolchain-building consume the schedule.
- **Tuner-vs-patience interaction** — a compiler-knob search that explores too
  slowly reads as stagnation to the patience stop. Keep per-iteration candidate
  sets small and profile-guided; if a real sweep is ever needed, run it inside
  one optimize iteration (best-of-K measured mechanically), not across K
  iterations.
- **`-use_fast_math` changes numerics** — only admissible because `verify`
  re-runs the tolerance-aware diff on every iteration; never apply it to the
  integer-checksum programs (`sobel`, `pagerank`, `needleman_wunsch`, `fft1d`)
  without checking the checksum still matches.

## How it plugs into evaluation

No new harness. Tag each conversion with the backend that produced it
(`ppcg` / `llm`) and report the existing metrics **per backend**: correctness
rate, `fast_1`/geomean speedup, and codegen cost (wall-clock + tokens; PPCG =
0 tokens). The claim the project can then make with data: *"the polyhedral backend
converts the affine subset with provable correctness at zero model cost, and the
LLM pipeline covers the irregular long tail — together stronger than either
alone."*
