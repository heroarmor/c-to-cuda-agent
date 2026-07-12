"""Fixed JSON shapes each pipeline stage writes into its workdir.

The orchestrator (run_pipeline.py) only ever reads these files to decide
control flow -- never the agent's chat transcript. Validating against these
shapes means a malformed stage response fails loudly instead of silently
breaking the loop.
"""

REQUIRED_KEYS = {
    "generate": {"status"},
    # compile_ok/run_ok describe the CUDA binary only -- the original C
    # program is compiled/run exactly once by the orchestrator, mechanically,
    # before generate ever starts (see run_pipeline.py's
    # _build_and_run_baseline), not re-checked by this stage every iteration.
    "verify": {
        "status",
        "kernel_structure",
        "compile_ok",
        "run_ok",
        "outputs_match",
        "repair_attempts",
        "findings",
        "checked_at",
    },
    "profile": {
        "stubbed",
        "time_sec",
        "baseline_time_sec",
        "speedup",
        "ncu_available",
        "ncu_summary",
        "ncu_csv_file",
    },
    "optimize": {
        "stubbed",
        "technique_applied",
        "bottleneck_addressed",
        "headroom_tier",
        "rationale",
        "expected_metric_change",
    },
}

RESULT_FILENAMES = {
    "generate": "generate_result.json",
    "verify": "verify_result.json",
    "profile": "profile_result.json",
    "optimize": "optimize_result.json",
}


def validate(stage: str, data: dict) -> None:
    missing = REQUIRED_KEYS[stage] - data.keys()
    if missing:
        raise ValueError(f"{stage}_result.json is missing required key(s): {sorted(missing)}")
