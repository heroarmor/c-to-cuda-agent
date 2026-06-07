# C-to-CUDA Conversion Agent

A capstone design project building an agent that **automatically and efficiently
converts C programs into CUDA programs**. Unlike tools that only optimize an
existing kernel, this agent generates the *complete* CUDA program — both the
**host** code (memory management, data transfer, kernel launch configuration)
and the **device** code (kernels) — from plain C source.

## Motivation

Porting C code to CUDA by hand is time-consuming and error-prone. It requires
identifying parallelizable regions, restructuring data layouts, managing
host/device memory, and tuning launch parameters. Existing automatic tools tend
to focus narrowly on kernel-level optimization and assume a CUDA program already
exists. This project aims to take an end-to-end view: given C source, produce a
working, reasonably optimized CUDA program.

## Goals

- **End-to-end conversion** — generate both host and device code, not just kernels.
- **Correctness first** — preserve the semantics of the original C program.
- **Efficiency** — apply parallelization and memory-access optimizations rather
  than producing a naive 1:1 translation.
- **Verifiability** — validate converted programs against the original C output.

## Scope (initial)

- Input: self-contained C source files.
- Output: compilable CUDA (`.cu`) source with host + device code.
- Focus on data-parallel patterns (loops over arrays/matrices) as the first target.

## Status

🚧 Early stage — repository initialized. Architecture and tooling to follow.

## Repository Layout

```
.
├── README.md      Project overview (this file)
└── ...            (to be added)
```

## Getting Started

_To be documented as the project takes shape._

## License

TBD
