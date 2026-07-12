---
description: Profiles a verified CUDA translation -- end-to-end timing against the original C program, plus Nsight Compute (and, for multi-kernel pipelines, Nsight Systems) GPU metrics.
mode: primary
permission:
  edit: allow
  bash: allow
  skill:
    cuda-profiling-procedure: allow
---

The CUDA translation in the current directory has already passed verification. Follow the `cuda-profiling-procedure` skill exactly:

1. Time the generated CUDA binary end-to-end with `time_binary.py`. Read `baseline_time_sec` from `baseline.json` (already in the current directory, captured once by the pipeline before generate ran) -- do not recompile or re-time the original C binary yourself.
2. Profile the CUDA binary with `ncu` using the bundled `ncu_metrics.cfg`, and additionally with `nsys` if `verify_result.json`'s `kernel_structure` is `S4`.

If `ncu` isn't available or fails (e.g. a permissions error), don't fail the whole stage -- still report the timing from step 1, and explain why GPU metrics are unavailable.

When you are done, write a file named `profile_result.json` in the current directory with this exact shape:
```json
{
  "stubbed": false,
  "time_sec": <float, mean wall-clock seconds for the CUDA binary>,
  "baseline_time_sec": <float, mean wall-clock seconds for the C binary>,
  "speedup": <float, baseline_time_sec / time_sec>,
  "ncu_available": true or false,
  "ncu_summary": "<key GPU metrics (SM/DRAM throughput %, occupancy %, registers/thread, dominant stall reason), or why ncu wasn't available>",
  "ncu_csv_file": "ncu_metrics.csv" or null
}
```
