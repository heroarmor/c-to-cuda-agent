---
description: Convert benchmark/<path>.c to cuda/<path>.cu and verify it on the GPU
---
Convert the CPU reference program `benchmark/$ARGUMENTS.c` into an efficient CUDA
program at `cuda/$ARGUMENTS.cu`, following every rule in @AGENTS.md.

Steps:
1. Read `benchmark/$ARGUMENTS.c` and name the parallel pattern.
2. Create `cuda/$ARGUMENTS.cu` (make parent directories). Reproduce the
   baseline's exact result line (whatever token it prints), the CLI handling and
   default size, and a faithful host-side `double` reduction so the result is
   comparable.
3. Verify by running:  `./cuda/build_run.sh $ARGUMENTS`
4. If it fails to compile or prints `RESULT: FAIL`, fix `cuda/$ARGUMENTS.cu` and
   re-run until it prints `RESULT: PASS`.

Finish by reporting the checksum compare and the GPU-vs-CPU timing.
