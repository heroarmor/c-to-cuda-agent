---
name: cuda-optimization-patterns
description: CUDA performance optimization knowledge -- a priority-ordered strategy (algorithmic > hardware utilization > fine-tuning), a checklist of concrete techniques, and a table mapping performance symptoms to root causes and fixes. Use when writing CUDA kernels for performance, or when deciding what to optimize next in an existing CUDA translation.
---

## Optimization Priority Order

Apply optimizations in this order -- higher tiers have more impact and should be exhausted before moving to the next:

1. **Algorithmic (highest impact)**: kernel fusion (combine separate kernels that touch the same data, to cut launch overhead and redundant memory traffic), shared-memory tiling (reuse data loaded once across multiple threads instead of re-reading from global memory), memory coalescing (lay out accesses so consecutive threads read/write consecutive addresses).
2. **Hardware utilization (medium impact)**: vectorized loads/stores (`float2`/`float4` instead of scalar accesses where the data layout allows it), warp-level primitives (`__shfl_sync`, `__ballot_sync` for intra-warp communication without shared memory), occupancy tuning (balance block size against register and shared-memory usage per block).
3. **Fine-tuning (lowest impact, diminishing returns)**: instruction-level parallelism, mixed precision (FP16/TF32) where the numerical tolerance allows it, prefetching and double buffering to overlap memory transfers with computation.

## Optimization Checklist

**Essential** (apply first): coalesced memory access; kernel fusion where independent kernels touch the same data; shared-memory staging for data reused across threads; grid-stride loops for problem sizes larger than the launched grid; boundary checks on every thread.

**Performance** (apply as needed): vectorized memory operations; warp-level primitives; occupancy tuning via block size and register pressure; shared-memory padding to avoid bank conflicts; loop unrolling.

**Advanced** (situational, for final tuning): tensor cores/WMMA where the operation is eligible (e.g. GEMM-shaped); mixed precision; persistent kernels (keep data resident across iterations); CUDA graphs (reduce repeated launch overhead); double buffering.

## Performance Issue Patterns

| Symptom | Likely cause | Fix |
|---|---|---|
| Slower than expected | Missing kernel fusion | Combine separate kernels operating on the same data into one. |
| Low compute/SM utilization | Poor occupancy | Tune block size and reduce register/shared-memory pressure per block. |
| Low memory throughput | Uncoalesced access | Restructure indexing so consecutive threads read/write consecutive addresses. |
| High kernel-launch count | Missing fusion opportunity | Look for compound operations that can be merged into fewer kernels. |

## Iteration Discipline

Before applying an optimization, state what effect you expect and why (e.g. "fusing these two kernels should eliminate one round-trip through global memory"). If a change makes things worse or breaks correctness, diagnose and fix forward rather than silently reverting to a known-safe but slower version -- losing the attempted optimization without understanding why it failed just means re-discovering the same dead end later. Keep iterating as long as there's measurable room for improvement; don't stop at the first version that merely works.
