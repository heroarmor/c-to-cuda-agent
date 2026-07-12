---
name: cuda-profiling-procedure
description: The two-step procedure for profiling a verified CUDA translation -- end-to-end wall-clock timing against the original C program, then Nsight Compute (and, for multi-kernel pipelines, Nsight Systems) profiling for GPU utilization, memory bandwidth, and occupancy. Use after a translation has passed verification, to produce real performance data.
---

## Step 1: End-to-end timing

`baseline_time_sec` is **not** measured here -- the pipeline orchestrator already compiled and ran the original C program exactly once, before generate ever started, and recorded its mean timing in `baseline.json` (already present in the current directory: `{"time_sec": ...}`). Read it from there directly. Never recompile or re-time the C binary yourself -- re-running it repeatedly was found to introduce noise (e.g. cache/warm-up effects) unrelated to the CUDA translation, which is exactly why this is now a single fixed reference instead of a per-iteration measurement.

The `<name>_cuda` binary should already exist in the current directory from the verify stage (same session, same working directory). If it's missing, rebuild it with the same command the verify stage uses:
```
nvcc -O2 <name>.cu -o <name>_cuda -lm
```

Time it with `time_binary.py` (already present in this directory) rather than computing min/mean/median by hand:
```
python3 time_binary.py ./<name>_cuda --reps 5
```
It prints `{"min": ..., "mean": ..., "median": ...}` as JSON. Runs with **no arguments** (same convention as the verify stage -- rely on the program's built-in defaults). Use the `mean` as `time_sec`; `speedup = baseline_time_sec / time_sec` (the orchestrator recomputes both `baseline_time_sec` and `speedup` from `baseline.json` itself after this stage returns, so it's fine if these are only approximately right here -- just don't skip reporting them).

## Step 2: Nsight Compute profiling

Run `ncu` against the CUDA binary using the bundled config file:
```
ncu --config-file-path ncu_metrics.cfg --csv --log-file=ncu_metrics.csv ./<name>_cuda
```
Read `ncu_metrics.csv` directly (it's plain CSV -- no special tooling needed) and summarize the key columns in plain text: SM throughput %, DRAM throughput %, achieved occupancy %, registers per thread, and the dominant warp-issue-stall reason. That summary is what goes in `profile_result.json`'s `ncu_summary` field.

**If the kernel structure is S4** (multi-kernel pipeline -- check `verify_result.json`'s `kernel_structure` field, already written by the prior stage in this session): also run Nsight Systems to see the per-kernel launch breakdown across the whole pipeline, which `ncu`'s per-kernel deep-dive doesn't show as clearly:
```
nsys profile -t cuda,nvtx,osrt -o nsys_report ./<name>_cuda
nsys stats --report cuda_gpu_kern_sum nsys_report.nsys-rep
```
Fold any notable findings (e.g. one kernel dominating total GPU time, or a surprisingly high launch count) into `ncu_summary` as well -- there's no separate field for it.

## Graceful degradation

If `ncu` isn't installed, or fails with a permissions error (profiling counters commonly require elevated privileges -- look for messages mentioning `ERR_NVGPUCTRPERM` or needing `NVreg_RestrictProfilingToAdminUsers=0`), don't fail the whole stage. Set `ncu_available: false`, put the reason in `ncu_summary`, and `ncu_csv_file: null` -- but still report the Step 1 timing, since it doesn't depend on `ncu` at all. A profiling stage that reports "timing succeeded, detailed GPU metrics unavailable: <reason>" is far more useful than one that fails outright and loses the timing data too.

## Adjusting for repeated-kernel workloads

The bundled `ncu_metrics.cfg` profiles every kernel launch with no skip/limit, which is right for a single-shot program. **This is not just a metrics-quality choice -- it's a budget problem.** `ncu`'s per-launch replay overhead is substantial (often far slower than the unprofiled kernel itself), so a binary that launches kernels many times in a loop (a training loop's epochs, autoregressive token generation, a multi-pass clustering/batch pipeline) can make full-launch `ncu` profiling take many times longer than the same binary's plain Step 1 timing run -- easily enough to blow through this stage's whole timeout and lose the timing data too, not just the GPU metrics.

Before running `ncu` on a binary that you know (from `verify_result.json`'s `kernel_structure` being `S4`, or from having read the source during generate/verify) loops over many batches/epochs/tokens/passes calling the same kernel(s) repeatedly: **don't profile every launch.** Add `--launch-skip=<n>` and `--launch-count=<n>` to the `ncu` invocation to sample a handful of steady-state launches instead (skip enough to get past warmup, then count a small number, e.g. 5-10) -- there's no need to edit the config file for this, just pass them as extra command-line flags, which override the config file's values. Estimate the expected launch count from the source's own loop bounds (epoch count, token count, batch count, any CLI default controlling how many times the main loop runs) rather than finding out empirically by letting an unrestricted run take however long it takes.
