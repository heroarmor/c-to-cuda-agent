---
name: c-to-cuda-multiagent
description: Use this skill when converting a self-contained C benchmark program into a verified CUDA program with multiple specialized agents for analysis, architecture, implementation, verification, and optimization.
---

# C to CUDA Multi-Agent Skill

Use this skill for `benchmark/<path>.c -> cuda/<path>.cu` conversions.

## Workflow

1. **Analyzer agent** reads only the C baseline and writes a conversion brief:
   CLI/defaults, stdout contract, data structures, hotspots, dependencies,
   reductions, expected numerical risks, and the minimum data that must move
   between host and device.
2. **Architect agent** turns the brief into a CUDA plan: kernels, grids/blocks,
   memory ownership, transfer schedule, reduction strategy, and correctness
   invariants. For complex programs, choose a hybrid CPU/GPU boundary before
   writing code.
3. **Implementer agent** creates `cuda/<path>.cu` following the architecture.
   It must preserve CLI/stdout and add CUDA error checks.
4. **Verifier agent** runs `./cuda/build_run.sh <path> [args...]`, classifies
   compile/runtime/output failures, and gives concrete fixes.
5. **Optimizer agent** runs only after correctness passes. It may tune launch
   geometry, shared memory, data reuse, and transfer placement, but must rerun
   verification after every change.
6. **Coordinator** owns final decisions and never accepts a conversion without
   `RESULT: PASS`.

## Role Prompts

Role prompts are stored in `.opencode/agents/`:

- `cuda-analyzer.md`
- `cuda-architect.md`
- `cuda-implementer.md`
- `cuda-verifier.md`
- `cuda-optimizer.md`

If opencode subagents are available, spawn one role per phase and pass only the
previous phase's artifact plus the required file paths. If subagents are not
available, follow the same role boundaries sequentially in the main session.

## Handoff Contract

Each phase produces a short handoff:

```text
path:
stdout_contract:
cli_contract:
hotspots:
dependencies:
cuda_plan:
verification_status:
next_action:
```

The implementer edits only `cuda/`. The verifier may run shell commands but must
not edit source. The optimizer may edit only after the verifier reports PASS.
