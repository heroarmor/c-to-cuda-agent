---
name: cuda-correctness-review
description: Checklist and debugging guide for verifying a CUDA kernel translation is correct -- kernel structure classification, thread/bounds safety, sync placement, compile-time risk patterns, and common compilation/correctness failure patterns and their fixes. Use when writing or reviewing CUDA code for correctness, or when diagnosing a compile failure or wrong output.
---

## Kernel Structure Classification (triage first)

Before checking anything else, classify the kernel(s) you're looking at. The structure determines which checklist items below matter most -- don't apply every check with equal weight to every kernel.

- **S0 -- Streaming/No-Reuse**: simple elementwise load-compute-store, each output depends only on the corresponding input(s), no shared memory, no reduction. *Weight*: bounds checks and type/precision matter most; sync/reduction checks are irrelevant here.
- **S1 -- Reuse-Friendly**: explicit data reuse via shared-memory tiling, register blocking, or warp-shuffle exchange (e.g. matrix multiply, convolution with tile caching). *Weight*: shared-memory sizing and synchronization around the tile matter most.
- **S2 -- Irregular/Stencil**: nested loops over multiple spatial/window dimensions with bounds checks and multi-dimensional indexing (convolution, pooling, stencils). *Weight*: indexing math and boundary handling matter most -- this is where off-by-one errors hide.
- **S3 -- Reduction/Scan**: cross-thread aggregation via shared-memory tree reduction, atomics, or warp-shuffle reduce/scan. *Weight*: synchronization placement and the reduction's identity/neutral values for out-of-range threads matter most.
- **S4 -- Multi-Kernel Pipeline**: the host code launches multiple distinct kernels in sequence with intermediate buffers between them (e.g. conv -> relu -> pool -> fc). *Weight*: host-side kernel ordering, intermediate buffer sizing, and data dependencies between launches matter most -- a kernel can be individually correct and the pipeline still wrong if buffers are reused too early or sized for the wrong stage.

A single program can mix tiers across its kernels (e.g. a multi-kernel CNN pipeline is S4 overall, but its conv kernel is S2 and its pooling kernel might be S0). Classify per-kernel when it changes what to look for, and note the overall pipeline shape separately.

## Correctness Checklist

- **Thread bounds**: every kernel guards its work with `tid < N` (or the equivalent per-dimension check) before touching memory. A thread launched past the problem size must do nothing.
- **Synchronization**: `__syncthreads()` is placed before any shared-memory location is read after being written by a different thread in the block. A missing or misplaced barrier is one of the most common sources of silently wrong results.
- **Type and precision conversions**: conversions between `float`/`double`/integer types are explicit and intentional, not accidental truncation from a default type choice.
- **Memory safety**: no out-of-bounds access to global or shared memory, in either direction (reads or writes), including off-by-one errors at array boundaries.
- **Numerical stability**: division by zero is guarded; NaN/Inf cannot silently propagate through the computation without being noticed.

## Compile-Time Risk Patterns

Catchable by reading, without ever invoking a compiler -- useful both as a fast pre-check and when diagnosing a real compile failure or an unexpectedly slow compile:

- Heavy or recursive template instantiation (deeply nested template types, many template parameters).
- `#pragma unroll` applied to a loop with a large or runtime-dependent bound (forced unrolling of a non-trivial loop blows up compile time and code size).
- Excessive `__forceinline__` on large device functions.
- Large macro-generated sets of near-identical kernel variants (many specializations multiplying compile time).

## Compilation Issues

| Symptom | Likely cause / fix |
|---|---|
| Undefined symbol at link time | Mismatched `extern "C"` linkage between a function's declaration and definition (host vs. device, or across translation units). |
| "no kernel image is available for execution on the device" | The compiled architecture flags (`-arch`/`-gencode`) don't match the target GPU's compute capability. |

## Correctness Failure Patterns

| Symptom | Debug steps |
|---|---|
| Wrong output values | Re-check kernel indexing and math by hand; test with a minimal input (e.g. N=4) before trusting the result on the real input size. |
| NaN/Inf in the output | Check for division by zero and out-of-bounds reads before assuming the underlying algorithm is wrong. |
| Mismatched array sizes/shapes | Log the actual dimensions at kernel-launch boundaries; re-derive the index math from the original sequential loop's bounds rather than guessing. |

## Review Method

Classify kernel structure first (above), then work through the checklist weighted by what that tier flags as high-risk -- don't spend equal attention on every item regardless of kernel shape.

When no compiler/GPU is available (pure static review): pick a tiny concrete example (e.g. N=4, a single block) and trace through the kernel's indexing by hand. Compare that trace explicitly against the original sequential loop's iteration order and bounds -- don't just read the parallel version in isolation and judge whether it "looks right." A translation that's correct in spirit but off by one in the loop-to-thread mapping will look plausible on a skim and wrong on a trace.

For every finding, whether from a hand trace or a real compiler/runtime error: quote the exact offending code (or error line) as evidence, and state one concrete minimal fix -- not a restated description of the symptom. "The kernel might have a sync issue" is not a finding; "missing `__syncthreads()` after the shared-memory write on line 42, before the reduction read on line 45" is.
