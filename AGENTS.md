# C → CUDA conversion agent — project rules

This repo converts the self-contained C reference programs under `benchmark/`
into CUDA programs under `cuda/`, then verifies each conversion against the CPU
baseline on the local NVIDIA RTX 3070 Laptop GPU.

## What to do when asked to convert `<tier>/<field>/<name>`

| | path |
|---|---|
| Source (read-only baseline) | `benchmark/<tier>/<field>/<name>.c` |
| Output (you create this)    | `cuda/<tier>/<field>/<name>.cu` |
| Verify with                 | `./cuda/build_run.sh <tier>/<field>/<name>` |

## Hard rules

1. **Never edit anything under `benchmark/`.** Those are the ground-truth
   baselines. Only create/modify files under `cuda/`.
2. **Preserve the program's stdout contract exactly.** Reproduce the baseline's
   result line verbatim — same labels, same `printf` formats, same values —
   whatever result token it uses (`checksum=`, `total_heat=`, `pi=`, `trace=`,
   `energy=`, `hits=`, …; it is **not** always `checksum=`). The verifier strips
   the `time=` field and any `(...)` perf-rate/annotation, then compares every
   remaining number, so only those fields may legitimately differ.
3. **Preserve the CLI.** Same `argv` handling and the same default problem size.
4. **Keep the checksum reduction faithful.** If the baseline accumulates the
   checksum in `double` on the host, copy the result array back and reduce on the
   host in `double` the same way — GPU sum reordering would otherwise perturb the
   value. If you must reduce on the GPU, use a numerically stable reduction and
   stay within `build_run.sh`'s tolerance (`TOL`, default 1e-4).
5. **Always check CUDA errors** (wrap `cudaMalloc`/`cudaMemcpy`/launch/sync) and
   call `cudaDeviceSynchronize()` before reading results back.
6. **Target the local GPU:** compute capability **sm_86** (Ampere, RTX 3070),
   CUDA 12.4. The harness already passes `-arch=sm_86`; do not hardcode a
   different arch.

## How to parallelize, by tier

- **easy/** — one kernel, one thread per output element; the host just
  allocates / copies H2D / launches / copies D2H. (saxpy, heat2d, mandelbrot,
  lorenz_ensemble, nbody, mc_pi)
- **moderate/** — shared-memory tiling for reuse (gemm,
  tensor_contraction), per-row/warp work for irregular access (spmv), host-side
  orchestration of several kernels + dot reductions across iterations
  (conjugate_gradient), or a stencil with a resident multi-level time recurrence
  (wave2d, raytrace).
- **complex/** — research-grade: data dependencies, hybrid CPU-panel /
  GPU-trailing-update factorizations, batched tiny dense blocks, recursive
  control (V-cycle). Get the checksum/invariant matching **first**, then optimize.

## Complex numbers

`complex/lattice-qcd/dslash.c`, `complex/quantum/statevector.c`, and
`complex/cfd/lbm.c` use C `<complex.h>` / `_Complex_I`, which CUDA C++ does not
support directly. Map them onto `cuDoubleComplex` (`cuComplex.h`) or
`thrust::complex`.

## Definition of done

Run `./cuda/build_run.sh <tier>/<field>/<name>` and iterate — fix compile errors
first, then correctness — until it prints `RESULT: PASS`. Then report the final
`checksum` compare and the GPU-vs-CPU timing.

A worked, passing reference conversion lives at
`cuda/easy/dense-linalg/saxpy.cu` — match its structure and output discipline.
