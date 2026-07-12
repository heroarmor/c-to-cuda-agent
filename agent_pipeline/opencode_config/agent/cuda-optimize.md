---
description: Applies one targeted optimization to a verified, profiled CUDA translation, chosen from the actual measured bottleneck rather than guesswork.
mode: primary
permission:
  edit: allow
  bash: deny
  skill:
    cuda-optimization-patterns: allow
    cuda-bottleneck-playbook: allow
    cuda-correctness-review: allow
---

`profile_result.json` and `verify_result.json` are already in the current directory, written by the earlier stages of this iteration. Read both before doing anything else.

1. Consult `cuda-bottleneck-playbook` and classify the headroom tier from `profile_result.json`'s `ncu_summary` (or fall back to `"unknown"` plus `verify_result.json`'s `kernel_structure` if `ncu_available` is `false`).
2. Consult both `cuda-optimization-patterns` and `cuda-bottleneck-playbook` to identify the single dominant bottleneck and pick exactly **one** technique from the method catalog that addresses it -- not a bundle of changes.
3. Apply that one technique to the `.cu` file. Consult `cuda-correctness-review` while editing so the change doesn't reintroduce a bounds/sync bug.
4. Do not try to recompile, run, or re-profile yourself -- `cuda-verify` and `cuda-profile` will do that independently next iteration.

When you are done, write a file named `optimize_result.json` in the current directory with this exact shape:
```json
{
  "stubbed": false,
  "technique_applied": "<method catalog name, e.g. Improve_Coalescing_and_TransactionSize>",
  "bottleneck_addressed": "<taxonomy symptom name, e.g. global_memory_bandwidth>",
  "headroom_tier": "Tier-H" | "Tier-M" | "Tier-L" | "unknown",
  "rationale": "<why this technique, citing the actual numbers from profile_result.json>",
  "expected_metric_change": ["<directional change you expect, e.g. 'dram_throughput_pct up'>"]
}
```
If headroom is `Tier-L` and no technique's `mechanism_requirements` are actually met by this kernel, it's fine to report that explicitly in `rationale` and pick the least-risky applicable technique anyway, or state that no safe change is available -- don't force an edit just to have made one.
