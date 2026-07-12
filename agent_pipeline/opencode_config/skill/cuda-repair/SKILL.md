---
name: cuda-repair
description: Failure-triage discipline and fix guidance for a CUDA translation that fails to compile, crashes, produces wrong output, or hangs. Use after the cuda-verification-procedure skill's compile/run/compare steps report a failure, to diagnose and fix it before retrying.
---

## Triage order (evaluate top to bottom, stop at the first match)

Don't try to diagnose everything at once. Determine which single category applies, then only apply that category's guidance.

1. **Compile failure** -- `nvcc` exited non-zero. Fix signatures, includes, `extern "C"` linkage mismatches, or syntax errors. Do not touch algorithmic logic at this stage -- a compile failure is never an algorithm problem.
2. **Runtime crash** -- the binary ran but crashed, timed out, or reported an illegal memory access / device-side assert / segfault. Focus on bounds checks and grid/block launch configuration relative to the problem size. Do not change algorithmic semantics while fixing a crash.
3. **Numerical mismatch** -- both binaries ran to completion, but `compare_outputs.py` reported `MISMATCH`. Check indexing and reduction order, missing `__syncthreads()` or atomics, and type/precision mismatches (see `cuda-correctness-review`'s checklist, weighted by the kernel's structure tier).
4. **Timeout/hang** -- the run (or the compile itself) exceeded its timeout with no crash. Check for a missing loop termination condition, a deadlocked synchronization (e.g. a `__syncthreads()` inside a conditional that not all threads in the block reach), or a compile-time blowup per `cuda-correctness-review`'s "Compile-Time Risk Patterns" section.

## Discipline

- **Apply the smallest change that addresses the identified issue.** Don't refactor unrelated code while repairing one problem.
- **Cite verbatim evidence**: the exact compiler error line, the exact crash message, or the exact `compare_outputs.py` mismatch line -- not a paraphrase.
- **State one concrete fix**: "change X from A to B" or "add a bounds check `if (idx < N)` before line 42" -- not generic advice like "improve bounds checking."
- **One category at a time**: if fixing the compile failure surfaces a different category of problem on the next attempt (e.g. it now compiles but crashes), that's a new, separate diagnosis -- re-enter the triage order from the top for the new failure, don't try to anticipate it in advance.

## Repair budget

You have a limited number of repair attempts (the calling agent's prompt specifies the exact number). If the issue isn't resolved within that budget, stop, report `status: "fail"`, and describe the still-unresolved issue with the same evidence discipline as above -- don't keep attempting fixes past the budget, and don't report a misleading "pass" to avoid reporting failure.
