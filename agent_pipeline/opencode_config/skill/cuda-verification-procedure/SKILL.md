---
name: cuda-verification-procedure
description: The mechanical procedure for objectively verifying a CUDA translation -- compile the original C and the generated CUDA, run both, and diff their output with a tolerance-aware tool instead of eyeballing it. Use whenever you need to determine whether a CUDA translation is actually correct, not just whether it looks correct.
---

## Why mechanical, not by-eye

A static code review (see the `cuda-correctness-review` skill) can catch obvious mistakes, but it cannot tell you whether a translation is *actually* correct -- only running both programs and comparing their real output can. When a compiler and GPU are available, always prefer this procedure over judging correctness from reading the code alone.

`baseline_output.txt` is already present in the current directory -- the pipeline orchestrator compiled and ran the original C program exactly once, before generate ever started, and that captured stdout is the one fixed correctness reference for every iteration. You never compile or run the original C program yourself; only the CUDA translation.

## Step 1: Compile

Only the generated CUDA needs compiling here -- the C reference is already built and run once by the orchestrator (`baseline_output.txt`).

If a file named `nvcc_flags.txt` exists in the current directory, it holds extra nvcc flags chosen by the pipeline's mechanical compiler tuner (an orchestrator-side optimize move). Those flags are part of the artifact being verified -- the tuned time isn't reproducible without them -- so include them in every nvcc invocation, and never edit or delete the file yourself:
```
nvcc -O2 $(cat nvcc_flags.txt 2>/dev/null) <name>.cu -o <name>_cuda -lm
```

If `nvcc` fails with "no kernel image is available for execution on the device", the architecture flags don't match the GPU. Detect the actual compute capability and retry with it:
```
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
nvcc -O2 $(cat nvcc_flags.txt 2>/dev/null) -arch=sm_<cc-without-dot> <name>.cu -o <name>_cuda -lm
```

If compilation fails for any other reason, this is a **compile failure** -- stop here and go to the `cuda-repair` skill rather than trying to run anything.

## Step 2: Run

Run the CUDA binary with **no arguments** -- every program in this dataset has a sane built-in default when `argc` is too small, so this is deterministic and doesn't require guessing meaningful inputs. Wrap the run in a timeout so a hang doesn't consume the whole verification budget:
```
timeout 30s ./<name>_cuda > cuda_output.txt
```

If the command times out or exits with a crash (non-zero exit from a signal, "illegal memory access", a device-side assert, a segfault), this is a **runtime crash** -- stop here and go to the `cuda-repair` skill rather than trying to compare output that was never produced.

## Step 3: Compare

Programs in this dataset print their own wall-clock fields (`time=0.123 s`, sometimes `(12.3 GFLOP/s)`), which legitimately differ between the C baseline and the CUDA run -- strip them from both sides first, or every comparison reports a false mismatch:
```
sed -E 's/ *time=[0-9.eE+-]+ s//; s/ *\([0-9.eE+-]+ GFLOP\/s\)//' baseline_output.txt > baseline_stripped.txt
sed -E 's/ *time=[0-9.eE+-]+ s//; s/ *\([0-9.eE+-]+ GFLOP\/s\)//' cuda_output.txt > cuda_stripped.txt
```
Then use the comparison tool rather than reading the two output files yourself:
```
python3 compare_outputs.py baseline_stripped.txt cuda_stripped.txt
```
It prints `MATCH` and exits 0 if the outputs agree (numbers compared with tolerance, not exact text), or `MISMATCH: <reason>` and exits 1 if they don't.

If it reports `MISMATCH`, this is a **numerical mismatch** -- go to the `cuda-repair` skill. Small differences in the last few significant digits of floating-point output are expected from parallel reduction reordering on the GPU, not a bug -- that's exactly why this tool uses tolerance instead of an exact diff. Don't treat a result the tool already accepted as a problem to "fix."

## Outcome

- Compile succeeds, run succeeds, output matches -> the translation is verified correct. Report `status: "pass"`.
- Any step fails -> go to the `cuda-repair` skill, fix the issue, and retry from Step 1. This whole procedure (steps 1-3) is one "attempt" -- track how many attempts you've made.
