#!/usr/bin/env python3
"""GPU smoke test for the Phase 2.5 compiler optimize moves against the real
toolchain (nvcc + PPCG) -- the mechanical half of the optimize loop, no
opencode/LLM required. Run under a 1-GPU Slurm job:

    sbatch --account=nbleier_owned1 --partition=gpu-rtx6000 --gres=gpu:1 \
      --cpus-per-task=8 --mem=16G --time=00:20:00 -o polyhedral/moves-smoke.log \
      --wrap "module load cuda/12.8.2 && python3 polyhedral/moves_smoke.py \
              benchmark/easy/dense-linalg/gemm.c"

Builds the same isolated-workdir shape run_pipeline.py uses (one-time C
baseline + bundled compare/time tools), takes the initial .cu from PPCG, then
exercises run_flag_move and run_retile_move exactly as the orchestrator
would: plan from a profile_result, measure candidates mechanically, print
each move's decision. Exits non-zero if any step breaks; accept/reject
decisions themselves are data, not pass/fail.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "agent_pipeline"))

import compiler_moves as cm

TOOLS = [
    REPO / "agent_pipeline/opencode_config/skill/cuda-verification-procedure/compare_outputs.py",
    REPO / "agent_pipeline/opencode_config/skill/cuda-profiling-procedure/time_binary.py",
]


def sh(cmd, cwd, timeout=300):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", timeout=timeout)


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1
               else REPO / "benchmark/easy/dense-linalg/gemm.c").resolve()
    name = src.stem
    workdir = Path(tempfile.mkdtemp(prefix=f"moves_smoke_{name}_"))
    print(f"workdir: {workdir}")
    shutil.copy(src, workdir / src.name)
    for tool in TOOLS:
        shutil.copy(tool, workdir / tool.name)

    # One-time C baseline, same convention as run_pipeline._build_and_run_baseline.
    proc = sh(["gcc", "-std=c11", "-O2", src.name, "-o", f"{name}_c", "-lm"], workdir)
    assert proc.returncode == 0, proc.stderr
    proc = sh([f"./{name}_c"], workdir, timeout=120)
    assert proc.returncode == 0, proc.stderr
    (workdir / "baseline_output.txt").write_text(proc.stdout, encoding="utf-8")
    baseline = json.loads(sh(
        [sys.executable, "time_binary.py", f"./{name}_c", "--reps", "3"],
        workdir).stdout)
    print(f"C baseline: {baseline['mean']:.4f}s")

    # The ppcg backend's generate product.
    proc = sh([sys.executable, str(HERE / "ppcg_to_cu.py"), src.name,
               "-o", f"{name}.cu"], workdir)
    assert proc.returncode == 0, proc.stderr

    # Initial CUDA measurement -- stands in for the profile stage's time_sec.
    arch = cm._detect_arch_flag()
    mean, why = cm._measure_candidate(
        workdir, f"{name}.cu", [arch] if arch else [], "initial")
    assert mean is not None, why
    print(f"PPCG initial: {mean:.4f}s  (vs C: {baseline['mean'] / mean:.2f}x)"
          f"  arch={arch}")
    profile = {"time_sec": mean, "ncu_summary": "", "ncu_available": False}

    state = cm.new_state()
    cu_text = (workdir / f"{name}.cu").read_text(encoding="utf-8")
    entry = json.loads((HERE / "scop_targets.json").read_text()).get(src.name)

    # Walk the compiler-move dispatch exactly as the orchestrator would:
    # library substitution (if the entry declares one), then flags, then
    # retile, until nothing is left. Decisions are data, not pass/fail.
    for step in range(1, 5):
        plan = cm.plan_move("compiler", "ppcg", profile, cu_text, state,
                            scop_entry=entry)
        if plan.move not in ("library", "flags", "retile"):
            print(f"\nno compiler move left after step {step - 1} "
                  f"(plan: {plan.move})")
            break
        print(f"\n== move {step}: {plan.move} ==")
        if plan.move == "library":
            result = cm.run_library_move(
                workdir, src.name, f"{name}.cu", profile, plan, state,
                HERE / "cublas_to_cu.py", 0.05)
        elif plan.move == "flags":
            result = cm.run_flag_move(
                workdir, f"{name}.cu", profile, plan, state, 0.05)
        else:
            result = cm.run_retile_move(
                workdir, src.name, f"{name}.cu", profile, plan, state,
                HERE / "ppcg_to_cu.py", 0.05)
        print(json.dumps({k: result[k] for k in
                          ("technique_applied", "accepted", "rationale")},
                         indent=2))
        if result["accepted"]:
            # the accepted candidate is the new thing to beat, same way the
            # next profile stage would re-baseline the loop
            best = min(c["mean_sec"] for c in result["candidates"]
                       if c["mean_sec"] is not None)
            profile = dict(profile, time_sec=best)
    print("\nmoves_smoke: all steps completed")


if __name__ == "__main__":
    main()
