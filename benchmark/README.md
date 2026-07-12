# Benchmark Suite

Reference **C** programs used as conversion targets and correctness/performance
baselines for the C→CUDA agent. Each file is a **self-contained, single
translation unit** (no shared headers, no external deps beyond `libm`) — the
input form the agent is designed to consume. Every program sets up its own data,
runs the computation, prints a **deterministic checksum/result** (for verifying
the converted CUDA output against this baseline) and a timing.

## Organization

Programs are grouped by **difficulty of producing an efficient host+device CUDA
conversion** (`easy/`, `moderate/`, `complex/`), and within each tier by
**field** (`<tier>/<field>/<name>.c`). Each tier deliberately spans a spread of
scientific-computing fields rather than being all linear algebra.

```
benchmark/<difficulty>/<field>/<program>.c
```

### easy/ — embarrassingly parallel, one kernel, trivial host

Obvious data parallelism, a single device kernel, and a host that just
allocates / copies / launches. Good first demos.

| Field | File | What converting it exercises |
|-------|------|------------------------------|
| Dense linear algebra | `easy/dense-linalg/saxpy.c` | Element-wise BLAS-1; memory-bound, one thread per element |
| PDE / stencil | `easy/pde/heat2d.c` | 2D heat equation, 5-point stencil, double-buffer swap |
| ODE | `easy/ode/lorenz_ensemble.c` | Ensemble RK4; one thread per trajectory |
| N-body | `easy/nbody/nbody.c` | All-pairs O(N²) force loop; high arithmetic intensity |
| Rendering | `easy/rendering/mandelbrot.c` | Per-pixel escape-time; thread divergence |
| Monte Carlo | `easy/montecarlo/mc_pi.c` | Per-stream RNG + final reduction |
| Image processing | `easy/image/sobel.c` | 2D Sobel edge detector; one thread per pixel with fixed 3×3 stencil |

### moderate/ — tiling / multi-kernel orchestration / recurrence

Needs shared-memory tiling to be efficient, **or** host-side orchestration of
several kernels and reductions, **or** a time recurrence with resident state.

| Field | File | What converting it exercises |
|-------|------|------------------------------|
| Dense linear algebra | `moderate/dense-linalg/gemm.c` | GEMM; shared-memory tiling for reuse |
| Sparse linear algebra | `moderate/sparse-linalg/spmv.c` | CSR SpMV; irregular access, per-row/warp work |
| Iterative solver | `moderate/solver/conjugate_gradient.c` | **Host orchestration** of SpMV + dot reductions + AXPY across iterations |
| PDE / stencil | `moderate/pde/wave2d.c` | Wave equation; stencil with 3-level time recurrence |
| Tensor algebra | `moderate/tensor/tensor_contraction.c` | Multi-index contraction; fuse transpose with the GEMM |
| Rendering | `moderate/rendering/raytrace.c` | Per-pixel with branchy control flow (hits, shadow rays) |
| Signal processing | `moderate/signal/fft1d.c` | Iterative radix-2 FFT; bit reversal and staged butterfly dependencies |
| Graph analytics | `moderate/graph/pagerank.c` | PageRank on CSR graph; sparse traversal plus per-iteration reductions |

### complex/ — data dependencies, irregular structure, hierarchy

Research-grade: data dependencies, unknown output structure, batched dense
blocks, or a multi-level recursive control structure.

| Field | File | What converting it exercises |
|-------|------|------------------------------|
| Factorization | `complex/factorization/lu.c` | LU + pivoting; hybrid CPU-panel / GPU trailing-update (MAGMA-style) |
| Factorization | `complex/factorization/cholesky.c` | A = LLᵀ; panel dependency + trailing SYRK/GEMM; batched tiny variant |
| Factorization | `complex/factorization/qr.c` | Householder QR; sequential reflectors + rank-1/block update |
| Sparse linear algebra | `complex/sparse-linalg/spgemm.c` | Sparse×sparse; symbolic pass + irregular merge, load imbalance |
| Eigen / SVD | `complex/eigen/jacobi_eigen.c` | Symmetric eigenvalues via cyclic Jacobi (one-sided → GPU SVD) |
| Multigrid / AMG | `complex/multigrid/multigrid.c` | Grid hierarchy: smooth + restrict/prolong, recursive V-cycle |
| Geometry / proximity | `complex/geometry/pqp.c` | PQP-style triangle-mesh proximity queries; BVH traversal + irregular pruning |
| Selected inversion / NEGF | `complex/physics/rgf.c` | **Diagonal of a matrix inverse** (Green's function); sequential block sweep + batched B×B GEMM/inverse |
| Rendering (global illum.) | `complex/rendering/pathtrace.c` | Monte Carlo path tracer (Cornell box); divergent paths, Russian roulette, recursion → bounce loop |
| Lattice QCD | `complex/lattice-qcd/dslash.c` | Wilson–Dirac stencil with SU(3) complex links + CG on D†D |
| CFD (lattice Boltzmann) | `complex/cfd/lbm.c` | D2Q9 collide+stream; Taylor–Green vortex on a periodic torus |
| Quantum computing | `complex/quantum/statevector.c` | Gate circuit on a 2ⁿ state vector; stride-by-qubit gather/scatter |
| Quantum many-body | `complex/manybody/lanczos.c` | Matrix-free Lanczos on a 2ᴺ Heisenberg Hamiltonian |
| Bioinformatics | `complex/bio/needleman_wunsch.c` | Global sequence alignment DP; anti-diagonal wavefront dependencies |

## Self-verification

Every program prints a deterministic checksum that the converted CUDA output can
be diffed against. Many also print a **physical / exact / analytic invariant** —
a stronger signal than a float checksum because it has a known target and is
robust to floating-point reordering on the GPU.

> Note on `complex/lattice-qcd/dslash.c`, `complex/quantum/statevector.c`,
> `complex/cfd/lbm.c`: these use complex numbers / `_Complex_I` (`<complex.h>`),
> which CUDA C++ does not support directly — the converter must map them onto
> `cuDoubleComplex` / `thrust::complex`.

Every program prints a deterministic checksum that the converted CUDA output can
be diffed against. The five time-integrators / renderers additionally print a
**physical or exact-integer invariant** — a stronger signal than a float checksum
because it has a known target and is robust to floating-point reordering on the GPU:

| Program | Invariant | Expected |
|---------|-----------|----------|
| `easy/nbody/nbody.c` | total momentum \|Σ m·v\| (starts at 0) | ~0 (roundoff) |
| `moderate/pde/wave2d.c` | discrete leapfrog energy drift | ~0 (roundoff) |
| `easy/ode/lorenz_ensemble.c` | fixed-point stationarity drift | ~0 |
| `easy/rendering/mandelbrot.c` | in-set area estimate | → 1.5066 (set area) |
| `moderate/rendering/raytrace.c` | exact hit / shadow pixel counts | bit-identical |
| `complex/rendering/pathtrace.c` | mean luminance (Monte Carlo) | converges as spp grows |
| `complex/lattice-qcd/dslash.c` | γ₅-Hermiticity \|⟨φ,γ₅Dψ⟩−⟨ψ,γ₅Dφ⟩*\| + CG residual | ~0 |
| `complex/cfd/lbm.c` | total mass drift | ~0 (machine precision) |
| `complex/quantum/statevector.c` | ‖ψ‖² (unitarity) | = 1 |
| `complex/manybody/lanczos.c` | E₀ of built-in N=4 ring | = −2.0 (exact) |

## Build & run

```sh
make                              # builds bin/<difficulty>/<field>/<name>
./bin/easy/dense-linalg/saxpy     # run with defaults
./bin/moderate/dense-linalg/gemm 1024   # most accept size args (see each file's main())
make clean
```

`mandelbrot` and `raytrace` also write an image (`mandelbrot.pgm`, `raytrace.ppm`)
to the current directory.

## Conventions for new benchmarks

- One algorithm per file, self-contained, C11, builds clean under `-Wall -Wextra`.
- Place it under `<difficulty>/<field>/`; reuse an existing field folder or add one.
- Accept problem size via `argv` with sensible defaults.
- Print a deterministic checksum so the converted CUDA output can be diffed
  against this baseline.
- Add a top-of-file comment naming the computational pattern and what the GPU
  conversion should parallelize.
