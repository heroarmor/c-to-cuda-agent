# Agent pipeline architecture

This document records the current design of `agent_pipeline/` — the real, multi-stage, looping C-to-CUDA conversion pipeline. See `CLAUDE.md` for a shorter summary; this is the detailed version, kept here so it doesn't bloat the always-loaded context file.

## Goal and shape

Given one sequential C source file, produce a verified, profiled, iteratively-optimized CUDA translation. The pipeline is four opencode agents run in a fixed loop by an external Python orchestrator:

```
generate --> verify --> profile --> optimize --> (loop back to verify)
                |                       |
              fail?              3 iterations without
                |                 >=5% improvement over
                v                  best time ever seen?
              stop                       v
                                       stop
                                  (export the BEST iteration's
                                   .cu, not the last one's)
```

- **`cuda-generate`**: reads the C file, identifies parallelizable regions, writes a complete `.cu` translation.
- **`cuda-verify`**: compiles and runs the generated CUDA, diffs its output against `baseline_output.txt` (the original C program's reference stdout, captured exactly once by the orchestrator before generate ever ran — see "One-time C baseline" below) with a tolerance-aware tool. If anything fails (compile error, crash, numerical mismatch, timeout), it repairs the `.cu` itself and retries — up to 3 cycles — before reporting `pass`/`fail`.
- **`cuda-profile`**: times the CUDA binary end-to-end (wall-clock, for the speedup number, against the same one-time-captured C baseline timing) and profiles it with Nsight Compute (GPU utilization, memory bandwidth, occupancy), plus Nsight Systems for multi-kernel pipelines.
- **`cuda-optimize`**: reads the profiling data, classifies how much headroom is left, picks exactly one optimization technique targeting the dominant measured bottleneck, and applies it.

The loop repeats verify → profile → optimize, stopping when:
- `cuda-verify` reports `status: "fail"` (couldn't fix it within the repair budget),
- 3 consecutive iterations (`PATIENCE`) each fail to beat the best profiled wall-clock time seen so far by at least 5% (`MIN_DELTA`) — patience-based early stopping, the same shape as ML training early stopping, computed by the orchestrator from each iteration's real `profile_result.json["time_sec"]` — **not** from `cuda-optimize`'s own claim; see "Why the stop condition reads profile, not optimize" below. A single noisy or slightly-regressed iteration can't end a run that's still genuinely improving, since the comparison is always against the best-ever time, not just the previous iteration,
- a hard iteration cap is hit, or
- a stage's JSON output is missing or malformed.

Whichever iteration actually had the best measured time is what gets exported as the final `.cu` — not necessarily the last iteration `cuda-optimize` touched, since later edits aren't guaranteed to be improvements (see "Best-version tracking" below).

## One-time C baseline

The original C program is compiled and run **exactly once**, mechanically, by the orchestrator (`run_pipeline.py`'s `_build_and_run_baseline`) — before `cuda-generate` even starts, no LLM involved, since this is fully deterministic. It writes `baseline_output.txt` (the captured stdout, used as `cuda-verify`'s fixed correctness reference) and `baseline.json` (`{"time_sec": ...}`, the mean of 5 reps via `time_binary.py`, used as `cuda-profile`'s fixed timing reference) into the shared workdir. Every iteration's `cuda-verify`/`cuda-profile` call reads these instead of recompiling/rerunning C itself. This replaced an earlier design where C was rebuilt and retimed every iteration — re-running it repeatedly introduced noise unrelated to the CUDA translation (e.g. cache/warm-up effects making a later run look faster), so comparing each iteration's CUDA run against a *freshly remeasured* C run wasn't a stable reference. The orchestrator also overrides whatever `baseline_time_sec`/`speedup` the profile agent reports with values recomputed from this mechanical baseline, the same "trust the mechanical signal over agent self-report" pattern used everywhere else in this pipeline.

## Best-version tracking

After each iteration's `profile` call, the orchestrator tracks `best_time_sec`/`best_iteration`: an iteration becomes the new "best" if its `time_sec` beats the best seen so far by at least `MIN_DELTA` (5%). Whenever that happens, the current `.cu` plus that iteration's `verify_result`/`profile_result` are snapshotted into `run_dir/best/`, overwriting any earlier snapshot. At the end of the run (whatever the exit reason), `_export_output` copies from `run_dir/best/` instead of the workdir's current `.cu` — so a run that stops on `max_iterations_reached` or `stagnated` after a few non-improving iterations still exports the genuinely-fastest version it found, not whatever `cuda-optimize` happened to leave behind last.

## Why this is built as an external orchestrator, not opencode skills or a source patch

opencode's skill system is model-invoked, reusable context — the model decides whether to load a skill, so skills cannot guarantee stage order, looping, or exit conditions. opencode's plugin system only exposes `catalog.transform`/`aisdk.language`/`aisdk.sdk` hooks, nothing for intercepting between agent turns. Patching opencode's core to add a control-flow engine would be a deep, hard-to-rebase fork divergence for something fully achievable externally. So the fixed ordering lives in `agent_pipeline/run_pipeline.py`, outside opencode entirely — opencode is used only for what it's good at (running one agent turn with tools and a system prompt), and the orchestrator handles sequencing, stop conditions, and data handoff.

## How each stage is wired up

- **Isolation**: each pipeline run gets a fresh `/tmp` workdir containing only the input `.c` file (plus the bundled tool scripts, below) — the same agent-blind trick `agent_baseline/run_baseline.py` already used, so an agent can't see `METADATA.md`, other benchmarks, or anything else in this repo.
- **Agent/skill discovery from an unrelated `/tmp` path**: opencode resolves `opencode.json`/`agent/*.md`/`skill/*/SKILL.md` by walking *up* from `--dir`'s target. A repo-root config would never be found from an isolated `/tmp` workdir, and nesting the workdir inside the repo instead (so the walk-up *would* find it) would let an escaped agent stumble onto `METADATA.md`. The fix: `run_pipeline.py` sets the `OPENCODE_CONFIG_DIR` environment variable (honored unconditionally regardless of `--dir`) to `agent_pipeline/opencode_config` on every `bun dev run` call. Agent and skill definitions are always found, isolation stays intact.
- **Session continuity**: all 4 stages of one pipeline run share a single opencode session (`--session <id>`, captured from the first stage's `--format json` output stream). Each stage sees the prior stages' conversation history, not just their JSON output files.
- **Stage handoff is files, not transcripts**: each stage writes a fixed-schema JSON file (`generate_result.json`, `verify_result.json`, `profile_result.json`, `optimize_result.json`) into the shared workdir. The orchestrator only ever reads these files to decide control flow — never parses agent prose. Schemas are validated in `agent_pipeline/schemas.py`; a malformed response fails loudly instead of silently breaking the loop.
- **Bundled tool scripts**: skill *text* is discovered via `OPENCODE_CONFIG_DIR`, but skill-bundled *files* an agent invokes via Bash are not — so `run_pipeline.py` separately copies `compare_outputs.py`, `time_binary.py`, and `ncu_metrics.cfg` into each workdir at the start of a run.
- **Generic vs. dataset-specific**: `run_pipeline.py` takes a plain file path and knows nothing about "the dataset" — `agent_pipeline/` stays purely the agent's own code. `dataset_eval/run_dataset.py` is the thin, separate driver (its own top-level folder, not nested inside `agent_pipeline/`) that loops it over every benchmark in `dataset_zyj/` — the same separation `agent_baseline/run_baseline.py` and `dataset_zyj/measure.py` already keep.

## Why the stop condition reads profile, not optimize

`cuda-optimize` only ever sees the **pre-edit** profile when it runs — it cannot know how much its own edit actually helped, since that's only knowable once the *next* iteration re-profiles the result. Early versions of this pipeline read `optimize_result["improvement"]` (a self-reported number) for the stop check, which was a structurally unsound placeholder. This project consistently prefers an objective, mechanically-measured signal over agent self-report (`cuda-verify`'s real compile/run/diff instead of static self-judgment; `cuda-profile`'s `time_binary.py` instead of LLM-computed statistics) — so the orchestrator computes the patience/best-time comparison itself from each iteration's real `profile_result.json["time_sec"]`, and checks it right after `profile`, *before* deciding whether to call `optimize` again. A diminishing-returns iteration stops one `cuda-optimize` call earlier than a design reading optimize's own claim would have.

An earlier version of this stop check compared only against the *immediately previous* iteration's time (10% threshold, no patience). That made a single noisy or slightly-regressed iteration end a run that was still genuinely improving overall, and exported whatever the *last* iteration's `.cu` happened to be rather than the fastest one actually found. The current patience-based design (3 consecutive iterations without a 5% improvement over the best-ever time, exporting that best iteration's `.cu` — see "Best-version tracking" above) fixes both problems at once.

## Compiler optimize moves (`--optimizer compiler|hybrid`)

An iteration's optimize slot can be taken by a deterministic, orchestrator-side **compiler move** instead of the `cuda-optimize` agent — the implementation of `polyhedral/DESIGN.md`'s "Compiler backend in the *optimize* stage" (Phase 2.5), in `agent_pipeline/compiler_moves.py`. Two moves, dispatched in this order while untried profile-guided candidates remain:

- **`flags`** — an nvcc flag search, backend-agnostic. A small candidate set (≤3 per iteration) is proposed from the parsed `ncu_summary` (`--maxrregcount` when occupancy is low / registers high, `-Xptxas -dlcm=cg` on memory-latency stalls, `-use_fast_math` only when the `.cu` actually calls transcendentals), then each candidate is compiled, run, diffed against `baseline_output.txt`, and timed — mechanically, inside this one optimize slot (best-of-K, so a sweep never spreads across iterations and reads as stagnation to the patience stop). A winner must beat the iteration's profiled time by `MIN_DELTA`; it's persisted to **`nvcc_flags.txt`**, which the verify/profile skills include in every later nvcc invocation and which travels with best-snapshots and the exported `.cu` (the tuned time isn't reproducible from the `.cu` alone).
- **`retile`** — PPCG re-tiling, only when the `.cu` came from the ppcg backend. Re-runs `polyhedral/ppcg_to_cu.py` on the *original C source* (PPCG consumes C, not CUDA) with a new `--sizes` string chosen from the measured signal (smaller blocks for low occupancy, bigger tiles for memory-bound), behind the same mechanical accept/reject gate. A losing candidate leaves the workdir's `.cu` untouched. A hybrid-backend `.cu` (Phase 3's PPCG-partial + LLM-stitched glue) is deliberately *not* retile-eligible: regenerating from the C source would reproduce only the partial translation and clobber the agent's stitching — only the flags move applies there.

Both moves write the same schema-validated `optimize_result.json` the agent writes, so the loop, patience stop, and best-version tracking are move-agnostic; each iteration's `pipeline_result.json` entry records which optimizer took the slot (`optimize_move`). In `hybrid` mode the `cuda-optimize` agent takes over once no compiler candidate remains; in `compiler` mode the run instead stops with exit_reason `optimizer_exhausted` (a normal exit, not a hard error) — combined with `--backend ppcg` that's a fully deterministic, zero-token pipeline for the affine subset. The moves live in the orchestrator deliberately: `cuda-optimize` has `bash: deny`, and the mechanical gate (real compile/run/diff/time, not model judgment) is the same "objective signal over agent self-report" preference used everywhere else here — it's also what makes a numerics-changing candidate like `-use_fast_math` safe to even try, since an output mismatch just rejects that candidate. `compiler_moves.py`'s planning half is pure and covered by `test_compiler_moves.py` (which fakes the toolchain to exercise the accept/reject gate end-to-end without a GPU).

## Skills (shared knowledge, not control flow)

Skills are a shared pool — any agent can load any skill it has `allow` permission for (`permission.skill` in that agent's frontmatter). They hold domain knowledge the model decides whether to pull into context; they never decide *when* a stage runs.

| Skill | Used by | Content |
|---|---|---|
| `cuda-correctness-review` | `cuda-generate`, `cuda-verify` | Kernel-structure classification (S0 streaming / S1 reuse-friendly / S2 irregular-stencil / S3 reduction-scan / S4 multi-kernel pipeline), a correctness checklist (thread bounds, sync placement, type/precision, memory safety, numerical stability), compile-time risk patterns, and compilation/correctness failure-pattern tables. |
| `cuda-verification-procedure` | `cuda-verify` | The mechanical compile → run → compare procedure: compile and run only the CUDA binary (no arguments, wrapped in a timeout), diff against the one-time-captured `baseline_output.txt` with the bundled `compare_outputs.py` instead of judging by eye. |
| `cuda-repair` | `cuda-verify` | Failure-triage discipline once verification fails: priority order (compile failure > runtime crash > numerical mismatch > timeout), evidence-grounded diagnosis, one minimal concrete fix at a time, a 3-attempt repair budget. |
| `cuda-profiling-procedure` | `cuda-profile` | End-to-end timing of the CUDA binary with the bundled `time_binary.py` against the one-time-captured `baseline.json` timing, then Nsight Compute profiling with the bundled `ncu_metrics.cfg` (plus Nsight Systems for S4/multi-kernel programs), with graceful degradation if `ncu` isn't available. |
| `cuda-optimization-patterns` | `cuda-generate`, `cuda-optimize` | A 3-tier optimization priority order (algorithmic > hardware utilization > fine-tuning), a checklist of concrete techniques, a performance-issue-pattern table, and an iteration discipline (state expected effect before applying, don't silently revert on regression). |
| `cuda-bottleneck-playbook` | `cuda-optimize` | Headroom triage (Tier-H/M/L from measured GPU utilization), a bottleneck-symptom-to-technique taxonomy, and a method catalog of ~17 named techniques (intent, applicability, expected metric change, anti-patterns), each tied to a specific measured symptom rather than general advice. |

## Tools bundled with skills

- `cuda-verification-procedure/compare_outputs.py` — tolerance-aware line/number diff between two program outputs (the same philosophy `dataset_zyj/measure.py` uses for golden-output diffing, reimplemented standalone so `agent_pipeline` stays decoupled from `dataset_zyj`).
- `cuda-profiling-procedure/time_binary.py` — runs a binary N times, reports min/mean/median wall-clock seconds as JSON.
- `cuda-profiling-procedure/ncu_metrics.cfg` — an `ncu --config-file-path` config listing the SM/DRAM throughput, occupancy, register, and warp-stall metrics to collect.

Mechanical tools were deliberately used instead of asking the LLM to compute statistics or eyeball a diff — consistent with how `dataset_zyj/measure.py` already does tolerance-based output comparison externally rather than trusting embedded self-checks.

## Acknowledgments

Two external repositories' prompt/skill content informed parts of this pipeline (read, evaluated for applicability, and substantially rewritten — never copied verbatim — since both target a different task than ours):

- **[BytedTsinghua-SIA/CUDA-Agent](https://github.com/BytedTsinghua-SIA/CUDA-Agent)** — its `agent_workdir/SKILL.md` (written for optimizing an existing PyTorch model via a custom CUDA extension) supplied the original shape of `cuda-correctness-review`'s checklist and `cuda-optimization-patterns`'s priority order/checklist/iteration-discipline sections, generalized away from the PyTorch-extension-specific workspace conventions (`binding.cpp`/`kernels/`/`torch.compile` baseline) that don't apply to translating a whole standalone C program.
- **[0satan0/KernelMem](https://github.com/0satan0/KernelMem)** — a much larger influence, since several of its `prompts/` files and its `memorybank/bottleneck_headroom_kernelstructure.yaml` are generic CUDA performance-engineering knowledge underneath PyTorch-specific framing:
  - `prompts/judge_gate.py`'s kernel-structure taxonomy (S0-S4) -> `cuda-correctness-review`.
  - `prompts/judger_compilation_timeout.py`'s compile-risk checklist -> `cuda-correctness-review`.
  - `prompts/judger_repair_memory.py` / `error_memory.py`'s failure-triage priority order and evidence/fix discipline -> `cuda-repair`.
  - `run_ncu_memory.py` / `run_nsys.py` / `config_metrics.ncu-cfg`'s Nsight invocation shapes and metric list -> `cuda-profiling-procedure` and `ncu_metrics.cfg`.
  - `memorybank/bottleneck_headroom_kernelstructure.yaml`'s `headroom_tiers`, `bottleneck_taxonomy`, and `method_catalog` -> `cuda-bottleneck-playbook` (17 of ~19 catalog entries ported; 2 skipped because their `implementation_case` is literally PyTorch `nn.Module`/`cublasLt` binding C++).

  Not migrated from either repo: PyTorch-extension workspace conventions, `torch.compile`-baseline grading criteria, NCU-metric-threshold deterministic gating tables (`machine_check`), GPU-spec lookup tables, and anything depending on a real GPU's NCU/nsys output that this pipeline can't yet exercise.

## Current status

All 4 stages and all 6 skills are real (no stubs remain) and the orchestration plumbing (agent/skill discovery from an isolated workdir, session continuity across stage/agent switches, JSON schema validation, the stop-condition arithmetic) has been verified without a GPU. The actual compile/run/profile/optimize *behavior* has not yet been exercised end-to-end on real hardware — there is no `nvcc`/`ncu`/`nsys` in this development environment. First thing to check once on a GPU machine: run `python3 agent_pipeline/run_pipeline.py <some .c file>` and confirm `verify_result.json`/`profile_result.json`/`optimize_result.json` contain sane real values, not just well-formed JSON.

The compiler optimize moves' mechanical half **has** now been exercised on real hardware (RTX PRO 6000, `polyhedral/moves_smoke.py` under a 1-GPU Slurm job): on `gemm`, the flag search compiled/ran/diffed/timed both profile-guided candidates against real nvcc (arch auto-detected as `sm_120`) and the retile move regenerated twice through real PPCG `--sizes` — all four candidates were honestly rejected (no ≥5% win; at n=512 the per-call copies dominate, which flags and tiling can't fix). Two practical notes from that run: benchmarks print their own `time=`/`GFLOP/s` fields, which `_measure_candidate` now strips before diffing (they'd otherwise mismatch on every candidate), and GPUs in this cluster run in `Exclusive_Process` mode — a busy allocation can't host a second CUDA process, so validation runs go through their own short `sbatch` jobs. What still needs a first run is the *full* loop with the LLM verify/profile stages around the compiler moves — that needs the opencode setup this environment doesn't have.
