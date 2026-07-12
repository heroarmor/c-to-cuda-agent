---
name: cuda-bottleneck-playbook
description: Headroom triage and a catalog of named optimization techniques (intent, when it applies, expected metric change, anti-patterns), each tied to a specific measured bottleneck symptom from Nsight Compute metrics. Use after profiling, to decide whether it's worth optimizing at all and, if so, exactly which one technique to apply.
---

## Headroom triage (decide if it's worth optimizing at all)

Compute `primary_limiter_pct = max(sm_throughput_pct, dram_throughput_pct)` from `cuda-profile`'s `ncu_summary`:

- **Tier-H** (`primary_limiter_pct < 60`): high headroom, plenty of room to improve.
- **Tier-M** (`60 <= primary_limiter_pct <= 80`): medium headroom.
- **Tier-L** (`primary_limiter_pct > 80`): low headroom -- the kernel is near the ceiling of whatever resource dominates. Say so plainly in `rationale` rather than forcing a technique that won't move the needle much.

If `profile_result.json`'s `ncu_available` is `false`, headroom is `"unknown"` -- fall back to `verify_result.json`'s `kernel_structure` and the general checklist in `cuda-optimization-patterns` instead of precise tiering.

## Bottleneck taxonomy (symptom -> candidate techniques)

**Memory access**
- Global memory bandwidth-bound, or global memory latency-bound -> `Improve_Coalescing_and_TransactionSize`, `SharedMemoryTiling`, `RegisterBlocking`, `KernelFusion`
- Uncoalesced / irregular access -> `Improve_Coalescing_and_TransactionSize`, `KernelFusion`
- Underused wide loads -> `Vectorization_Refinement`, `Vectorized_Math`
- Shared-memory bank conflicts, capacity, or bandwidth pressure -> `SharedMemoryTiling`, `RegisterBlocking`, `SharedMemory_Tradeoff_if_Beneficial`
- Register file capacity / spilling -> `Reduce_Live_Ranges`, `Reduce_Unrolling`, `RegisterBlocking`, `KernelFission`
- Cache conflicts/misses/capacity/bandwidth -> `SharedMemoryTiling`, `RegisterBlocking`, `KernelFusion`

**Irregularity**
- Branch divergence -> `WarpUniformControlFlow`, `KernelFission`
- Load imbalance across threads/blocks -> `Launch_Tuning` (work distribution), `KernelFission`

**Parallelism / scheduling**
- Low SM utilization -> `Increase_ILP_WorkPerThread`, `Launch_Tuning`
- Kernel launch overhead dominating (many small kernels) -> `KernelFusion`, `MultiKernelScheduling`
- Register pressure or spilling -> `Reduce_Live_Ranges`, `Reduce_Unrolling`, `KernelFission`
- Instruction-issue / pipeline latency -> `Increase_ILP_WorkPerThread`, `Reduce_Instruction_Count`

**Host/system**
- Host<->device transfer overhead -> reduce transfer count/size (not a kernel-internal technique; check whether a transfer can be eliminated or batched)
- Insufficient overlap of compute and memory -> `SoftwarePrefetching`, `Increase_Memory_Level_Parallelism`

## Method catalog

Each entry: when it applies, what should move (and which direction) afterward, and what NOT to do while applying it.

**Reduce_Instruction_Count** -- Reduce total dynamic instruction count (remove redundant ops, hoist invariants, simplify expressions). Applies when correctness is preserved by the simplification. Expect: SM throughput up, issued IPC up, kernel duration down. Don't: add extra global loads, or recompute an invariant unnecessarily.

**Improve_Coalescing_and_TransactionSize** -- Improve memory coalescing/alignment and widen load/store transactions. Applies when access strides are contiguous/regular and semantics are preserved. Expect: DRAM throughput up, long-scoreboard stalls down, kernel duration down. Don't: replace vector loads with scalar per-lane loads, add extra global reads/writes, or use misaligned wide loads without safe tail handling.

**Increase_Memory_Level_Parallelism** -- Increase the number of concurrent outstanding memory requests so warps hide latency better. Applies when there's little/no reuse and memory accesses per thread/warp are independent. Expect: long-scoreboard stalls down, kernel duration down. Don't: increase registers so much that occupancy collapses, or unroll so aggressively that it spills.

**Increase_ILP_WorkPerThread** -- Process multiple elements per thread (grid-stride loop, controlled unrolling). Applies when outputs are independent across elements and bounds/tail handling stays safe. Expect: long-scoreboard stalls down, issued IPC up, kernel duration down. Don't: use unroll factors large enough to spill registers, or introduce divergent control flow.

**SoftwarePrefetching** -- Prefetch data that will be reused after a known latency window. Applies only when a genuine reuse source exists. Expect: memory-dependency stalls down, kernel duration down. Don't: apply to a pure streaming/no-reuse kernel, or prefetch without a proven reuse source.

**Increase_Occupancy_if_limited** -- Increase active warps when occupancy is actually limited by registers, shared memory, or warp count. Applies only when one of those is the confirmed occupancy limiter. Expect: achieved occupancy up, kernel duration down (sometimes). Don't: reduce occupancy again via increased register pressure, or change algorithmic work/precision.

**SharedMemoryTiling** -- Use shared-memory tiling to exploit reuse across threads/outputs (stencil/conv spatial reuse, GEMM-like). Applies when reuse is identified, a tile shape is defined, and the shared-memory budget is respected. Expect: L1 throughput up, long-scoreboard stalls down, kernel duration down. Don't: size the tile so large that occupancy collapses, or load a tile with no actual reuse in the inner loop.

**Reduce_Address_Calculation_Overhead** -- Hoist/simplify index math and pointer arithmetic. Applies whenever correctness is preserved. Expect: SM throughput up, issued IPC up, kernel duration down. Don't: introduce extra global loads while simplifying.

**Alignment_and_Tail_Minimization** -- Fix misalignment and reduce tail-handling overhead for wide vector loads/stores. Applies when vector width is preserved and tail handling stays safe. Expect: DRAM/L2 throughput up, kernel duration down. Don't: replace vector loads with scalar ones, assume alignment without proving it, or leave the tail path unsafe.

**Vectorization_Refinement** -- Refine existing vectorized access (same width) for better alignment/coalescing and less remainder branching. Applies when vector width and semantics are preserved. Expect: DRAM throughput up, issued IPC up, kernel duration down. Don't: fake-vectorize (scalarize the lanes internally), or recompute addresses per lane.

**Launch_Tuning** -- Tune block size, grid-stride, unroll factors, and work partitioning without changing the algorithm. Applies broadly; respect any problem-size constraints. Expect: occupancy/IPC up (sometimes), kernel duration down (sometimes). Don't: unroll aggressively enough to spill, or change dtype/layout/precision while "tuning."

**RegisterBlocking** -- Reuse data in registers via micro-tiling (multiple outputs per thread, accumulate in registers). Applies when reuse is identified and the register budget is respected. Expect: SM throughput up, DRAM throughput down (less traffic), kernel duration down. Don't: blow up register usage into spilling, or add extra global traffic.

**TensorCore_or_CUBLASLT** -- Use Tensor Cores or a cuBLAS/cuBLASLt GEMM path when dtype/layout/size constraints permit. Applies to GEMM-shaped compute with supported dtypes and aligned dimensions. Expect: SM throughput up, kernel duration down. Don't: force this path when not eligible, or change numerical precision without it being an explicit, accepted tradeoff.

**KernelFusion** -- Fuse adjacent producer-consumer kernels to cut global round-trips and launch overhead. Applies only across multiple kernels with a fusible intermediate and compatible memory footprint. Expect: kernel launch count down, kernel duration down end-to-end. Don't: fuse across incompatible layouts/dtypes, or add extra global reads/writes in the process.

**KernelFission** -- Split a monolithic kernel into simpler stages to cut divergence/register pressure or isolate heavy math from memory ops. Applies when the benefit is concretely explainable, not just "might help." Expect: registers/thread down (sometimes), branch divergence down (sometimes). Don't: split into so many kernels that launch overhead dominates, or break a synchronization/ordering assumption.

**MultiKernelScheduling** -- Reorder or schedule multiple kernels to reduce confirmed launch overhead or improve pipeline efficiency. Applies to multi-kernel (S4) programs only, when overlap/reordering is actually justified by evidence. Expect: effective launch cost down, end-to-end time down. Don't: change the algorithmic dependency order, or assume overlap is happening without evidence it is.

**WarpUniformControlFlow** -- Restructure branches/indexing to make control flow warp-uniform (predication, branch hoisting, regrouping). Applies when the divergence source is identified and semantics are preserved. Expect: branch divergence down, kernel duration down. Don't: replace divergence with heavy predication that inflates instruction count, or add extra global accesses.

**Approximate_Transcendentals_or_LUT** -- Replace expensive transcendental ops with approximations or a lookup table, when the accuracy budget allows it. Applies only when an explicit error budget or equivalence requirement is satisfied -- don't silently trade away accuracy. Expect: SM throughput up, kernel duration down. Don't: take an unapproved precision loss, or add a large LUT in global memory that creates new bandwidth pressure.

**Vectorized_Math** -- Use vector math/data types (`half2`/`float2`/`float4`) and fuse operations to raise math throughput. Applies when vector types are feasible and alignment is checked. Expect: issued IPC/SM throughput up (sometimes), kernel duration down. Don't: scalarize the vector lanes internally, or use misaligned vector ops with no fallback.

**Reduce_Live_Ranges** -- Reorder computation / reuse registers / limit temporaries to cut register usage. Applies when the register-pressure source is identified and semantics are preserved. Expect: registers/thread down, occupancy up (sometimes). Don't: add extra global loads just to recompute a value you freed a register from.

**Reduce_Unrolling** -- Reduce unroll factors to cut register pressure and avoid spills while keeping enough ILP. Applies when an unroll factor is the actual cause of pressure. Expect: registers/thread down (sometimes), occupancy up (sometimes). Don't: strip out all unrolling and lose ILP dramatically, or change algorithmic behavior.

**SharedMemory_Tradeoff_if_Beneficial** -- Trade shared memory for registers (or vice versa) when register- or shared-memory-limited occupancy is the confirmed issue. Applies when that occupancy limiter is confirmed and the shared-memory budget stays respected. Expect: occupancy up (sometimes), long-scoreboard stalls down (sometimes). Don't: oversize shared memory into a new occupancy collapse, or introduce bank conflicts without mitigating them.

## A technique deliberately not catalogued here

**CUDA Graphs** (`cudaGraph_t` / stream capture) amortize kernel-launch overhead across *repeated* invocations of the same kernel sequence. Most benchmarks in this dataset are single-shot programs (run once), so this doesn't apply yet -- but it's relevant if a future translation has an outer iteration loop (e.g. `CNN.c`'s `iterations` argument) and launch overhead, not compute, turns out to dominate.

## Discipline

Pick exactly one technique addressing the dominant measured bottleneck -- not a bundle of "improvements." State the expected metric change before editing, in `rationale`, citing the actual numbers from `profile_result.json`. Don't apply a technique whose `mechanism_requirements` aren't actually met by this kernel's structure (e.g. don't apply `SharedMemoryTiling` to a kernel with no real reuse). This mirrors `cuda-optimization-patterns`'s existing Iteration Discipline section -- the two skills are meant to be used together, not as alternatives.
