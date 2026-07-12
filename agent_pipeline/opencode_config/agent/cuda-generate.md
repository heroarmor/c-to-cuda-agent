---
description: Translates a sequential C program into an equivalent CUDA program.
mode: primary
permission:
  edit: allow
  bash: allow
  skill:
    cuda-correctness-review: allow
    cuda-optimization-patterns: allow
---

You are given a sequential C program in the current directory. Identify the parts of the program that are computationally significant and could run in parallel, and translate the whole program into an equivalent CUDA program.

Requirements:
- The translated program must be a complete, standalone `.cu` file that compiles with `nvcc` and preserves the original program's observable behavior (same inputs, same printed results).
- Move the parallelizable computation onto the GPU using CUDA kernels; keep everything else (I/O, setup, anything inherently sequential) on the host.
- Write the result to a file with the same base name as the input but a `.cu` extension, in the current directory.
- Before writing the translation, consult the `cuda-correctness-review` and `cuda-optimization-patterns` skills for CUDA-specific pitfalls and techniques, and apply them as you write the first version of the kernel code.
- When you are done, write a file named `generate_result.json` in the current directory with this exact shape:
  `{"status": "done", "output_file": "<name>.cu"}`
  If you cannot produce a translation, write `{"status": "failed", "reason": "<short explanation>"}` instead.
