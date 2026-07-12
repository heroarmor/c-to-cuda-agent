---
description: Multi-agent C to CUDA conversion with analysis, architecture, implementation, verification, and optimization phases
---

Convert `benchmark/$ARGUMENTS.c` into a verified CUDA program at
`cuda/$ARGUMENTS.cu`.

Use the multi-agent workflow from `skills/c-to-cuda-multiagent/SKILL.md` and
the project rules in `AGENTS.md`.

Role prompts:

- Analyzer: `.opencode/agents/cuda-analyzer.md`
- Architect: `.opencode/agents/cuda-architect.md`
- Implementer: `.opencode/agents/cuda-implementer.md`
- Verifier: `.opencode/agents/cuda-verifier.md`
- Optimizer: `.opencode/agents/cuda-optimizer.md`

Process:

1. Run the Analyzer on `benchmark/$ARGUMENTS.c`. Capture the handoff.
2. Run the Architect using the Analyzer handoff. Capture the CUDA design.
3. Run the Implementer to create `cuda/$ARGUMENTS.cu`.
4. Run the Verifier with `./cuda/build_run.sh $ARGUMENTS`.
5. If verification fails, send the failure log back to the Architect or
   Implementer, edit, and rerun verification until `RESULT: PASS`.
6. After PASS, run the Optimizer only if the conversion is obviously slow or
   wasteful. Rerun verification after every optimization.

If the task/subagent tool is unavailable, emulate these agents sequentially in
the main session and keep the same handoff boundaries.

Finish with:

- final verdict from `cuda/build_run.sh`
- CPU and GPU result lines
- CPU/GPU timing if present
- key kernels and any intentional CPU/GPU hybrid boundaries
