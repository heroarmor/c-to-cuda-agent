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
| `cublas_to_cu.py` | The library optimize move's generator: replaces a square row-major GEMM hot function (declared by a `blas` entry in `scop_targets.json`) with a `cublasSgemm`/`cublasDgemm` call; output links with `-lcublas`. |
| `moves_smoke.py` | GPU smoke test for the Phase 2.5 compiler optimize moves against the real toolchain (cuBLAS substitution + nvcc flag search + PPCG re-tiling, no opencode needed); submit with the `sbatch --wrap` line in its docstring. |

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
- **Phase 1 — PPCG codegen path**: done, with a correction found during GPU
  validation: pet cannot delinearize flat-pointer subscripts (`p[i*n+j]` on a
  `T *p` — bilinear, undecidable extent), so the early "PASS" for `heat2d` and
  `gemm` was host-only passthrough trivially matching the golden diff.
  `ppcg_to_cu.py` now (a) delinearizes flat params into 2D VLA form for pet
  (per-entry `arrays` in `scop_targets.json`), reflattening afterwards for
  nvcc's C++ mode, and (b) **fails loudly when no `__global__` kernel comes
  out**, so a passthrough can never again count as a conversion. GPU-verified
  (RTX PRO 6000, golden diff): `saxpy` (2 kernels), `gemm` (2 kernels,
  finally real). `heat2d` now falls through honestly — its derived flat index
  (`int i = y*m+x`) defeats textual delinearization (needs region-level
  normalization, same club as `lu`/`qr`/`lbm`).
- **Phase 2 — relay + dispatcher**: done. `--backend ppcg|auto` in
  `run_pipeline.py`; `auto` = prefilter triage → PPCG attempt → LLM fallthrough
  (pet is the authority: a `scop-likely` file PPCG rejects falls through cleanly).
- **Phase 2.5 — compiler optimize moves**: mechanical half GPU-verified via
  `moves_smoke.py` (see `agent_pipeline/ARCHITECTURE.md` for the design): on
  `gemm`, the flag search compiled/ran/diffed/timed both profile-guided
  candidates and the retile move regenerated through real PPCG `--sizes`
  twice; all four candidates were honestly rejected (< 5% win — at n=512 the
  per-call copies dominate, exactly the known-limits story below). The full
  loop with LLM verify/profile stages still needs an opencode setup.
- **Phase 3 — hybrid (bucket B)**: done and exercised end-to-end on GPU.
  `--backend hybrid` (or `auto` via `mode=hybrid` entries in
  `scop_targets.json`) runs `_ppcg_partial` → hybrid generate prompt. Both
  first targets' partials pass the golden diff as complete programs:
  `multigrid` (`smooth_rb`+`prolong_add`, 10 kernels; `residual` excluded —
  `return` inside the would-be scop; `restrict_fw` excluded — flat memset
  loop) and `rgf` (`mat_mul`, 2 kernels; `mat_sub` excluded — `B*B` bound is
  a parameter product, non-affine). Full-pipeline run on `multigrid`
  (`--backend hybrid --optimizer hybrid`, free zen model
  `opencode/deepseek-v4-flash-free`, 5 iterations, 46 min, zero cost): all
  verifies PASS, optimize slots alternated compiler flags (2 accepted:
  `maxrregcount=64`, `dlcm=cg`) with LLM moves (`KernelFusion` was the big
  win), 0.436s → 0.272s (best-version tracking exported iteration 3, not 5).
  Final honest bottleneck: GPU compute is 1.15% of wall — CUDA driver init
  (~80 ms) alone exceeds the whole 29 ms C run at this problem size.
  `lu`/`qr`/`lbm` need region-level scop markers and stay LLM for now.

Known limits: the merged `.cu` reallocs/copies device buffers on every hot-
function call — a correct but naive starting point by design; hoisting
transfers is exactly what the optimize loop / hybrid stitching is for. It is
also why every GPU-verified conversion currently *loses* to the C baseline
end-to-end (`gemm` 0.15s vs 0.13s; `rgf`'s 256 tiny per-block launches: 2.6s
vs 0.02s). `--fn` marking assumes the hot function contains no `return`.
