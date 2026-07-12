# Polyhedral backend — usage

The deterministic PPCG codegen path designed in `DESIGN.md` (read that first;
`SCOP_CLASSIFICATION.md` is the Phase 0 coverage analysis behind it). This
directory now contains the working implementation:

| File | Role |
|---|---|
| `build_ppcg.sh` | One-shot toolchain build: libyaml + PPCG 0.09.3 (bundled isl 0.28 + pet 0.11.9) against the `llvm/14.0.6` module, installed into `toolchain/` (gitignored). `--check` re-runs the smoke test. |
| `prefilter.py` | Cheap static SCoP triage (`scop-likely` / `needs-review` / `reject`) before paying for a PPCG attempt. `--summary benchmark/` prints a TSV over the whole tree. |
| `ppcg_to_cu.py` | The backend's generate stage: runs PPCG on one benchmark C file and merges its 3-file output into **one self-contained `.cu`** (host I/O + result line intact), plus a C→C++ compatibility shim so nvcc accepts it. |
| `scop_targets.json` | Hot-function map (basename → `--fn`) from Phase 0, consulted automatically; unmapped files fall back to pet autodetect. |
| `verify_gpu.sbatch` | Short Slurm job (gpu-rtx6000): compile + run generated `.cu` vs the C reference, tolerance-aware golden diff with timing fields stripped. |

## Quick start

```sh
./polyhedral/build_ppcg.sh                                  # once, ~15 min on a login node
python3 polyhedral/ppcg_to_cu.py benchmark/easy/pde/heat2d.c   # -> polyhedral/generated/easy/pde/heat2d.cu
sbatch polyhedral/verify_gpu.sbatch                         # golden-diff on a GPU node
```

Through the agent pipeline (the PPCG→LLM relay — PPCG supplies the initial
`.cu`, the existing verify → profile → optimize loop takes it from there):

```sh
python3 agent_pipeline/run_pipeline.py benchmark/easy/dense-linalg/gemm.c --backend ppcg
python3 agent_pipeline/run_pipeline.py some.c --backend auto   # prefilter-routed, LLM fallthrough
python3 agent_pipeline/run_pipeline.py benchmark/complex/multigrid/multigrid.c --backend hybrid
```

`hybrid` is the bucket-B relay (Phase 3): PPCG GPU-ifies only the affine
sub-kernels named in `scop_targets.json`'s `mode=hybrid` entries (e.g.
`multigrid`'s smoother/restrict/prolong, `rgf`'s block GEMMs), the partial
`.cu` lands in the workdir as `<name>_ppcg_partial.cu`, and the generate
*agent* then builds the full translation on top of it — keeping PPCG's
kernels, hoisting the per-call copies so device data stays resident across
the recursive/pivoted host control, and stitching the rest. In `auto` mode a
`mode=hybrid` entry routes there before the prefilter (whose `reject` on
recursion is exactly what bucket B looks like); a PPCG reject falls through
to the plain LLM prompt.

`pipeline_result.json` records `"backend": "ppcg" | "hybrid" | "llm"` (and,
in auto mode, `ppcg_fallthrough`/`hybrid_fallthrough` with the reason PPCG
passed) so `evaluation/` can report metrics per backend; a PPCG generate
costs zero model tokens, a hybrid generate costs one agent call over a much
smaller stitching problem.

## Status

- **Phase 0 — SCoP classification**: done (`SCOP_CLASSIFICATION.md`).
- **Phase 1 — PPCG codegen path**: done. `saxpy`, `heat2d`, `gemm` generate and
  nvcc-compile; GPU golden-diff via `verify_gpu.sbatch`.
- **Phase 2 — relay + dispatcher**: done. `--backend ppcg|auto` in
  `run_pipeline.py`; `auto` = prefilter triage → PPCG attempt → LLM fallthrough
  (pet is the authority: a `scop-likely` file PPCG rejects falls through cleanly).
- **Phase 3 — hybrid (bucket B)**: pipeline side done. `--backend hybrid`
  (or `auto` via `mode=hybrid` entries in `scop_targets.json`) runs
  `_ppcg_partial` → hybrid generate prompt; `multigrid` and `rgf` are the
  first two targets (`lu`/`qr`/`lbm` need region-level scop markers inside a
  single function — `--fn` can only mark whole bodies — and stay LLM for
  now). Not yet GPU-validated, and the two `scop_targets.json` hybrid entries
  are unverified against real pet (a reject falls through to LLM, so they're
  safe to list).

Known limits: the merged `.cu` reallocs/copies device buffers on every hot-
function call (visible in `heat2d`'s per-step H2D/D2H) — a correct but naive
starting point by design; hoisting transfers is exactly the optimize loop's
first move. `--fn` marking assumes the hot function contains no `return`.
