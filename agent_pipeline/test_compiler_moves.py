#!/usr/bin/env python3
"""Tests for compiler_moves.py, runnable without a GPU: plain asserts, no
test framework (this repo has none).

The planning half (parse/propose/plan) is pure and tested directly. The
effectful half (run_flag_move / run_retile_move) is tested end-to-end against
a fake toolchain: a `nvcc` shell script on a prepended PATH that emits a tiny
shell-script "binary" whose runtime depends on the flags/sizes chosen -- so
the mechanical compile -> run -> diff -> time -> accept/reject gate is
exercised for real, with the repo's actual compare_outputs.py and
time_binary.py, just not a real compiler.

    python3 agent_pipeline/test_compiler_moves.py
"""

import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compiler_moves as cm
import schemas

HERE = Path(__file__).resolve().parent
COMPARE = HERE / "opencode_config" / "skill" / "cuda-verification-procedure" / "compare_outputs.py"
TIME_BIN = HERE / "opencode_config" / "skill" / "cuda-profiling-procedure" / "time_binary.py"

SUMMARY = ("SM throughput 45.2%, DRAM throughput 81.0%, achieved occupancy "
           "24.5%, 96 registers per thread, dominant stall: long_scoreboard")


def test_parse_ncu_summary():
    m = cm.parse_ncu_summary(SUMMARY)
    assert m["sm_pct"] == 45.2, m
    assert m["dram_pct"] == 81.0, m
    assert m["occupancy_pct"] == 24.5, m
    assert m["regs_per_thread"] == 96, m
    assert m["stall"] == "long_scoreboard", m
    assert cm.headroom_tier(m) == "Tier-L"

    empty = cm.parse_ncu_summary(None)
    assert all(v is None for v in empty.values()), empty
    assert cm.headroom_tier(empty) == "unknown"

    # "smem"/"nvidia-smi" must not satisfy the \bsm\b throughput pattern
    m = cm.parse_ncu_summary("smem bank conflicts observed, occupancy 55%")
    assert m["sm_pct"] is None and m["occupancy_pct"] == 55.0, m
    assert cm.headroom_tier(m) == "unknown"


def test_propose_flag_candidates():
    metrics = cm.parse_ncu_summary(SUMMARY)
    cu = "..." + " __expf(x); sqrtf(y); "
    cands = cm.propose_flag_candidates(metrics, cu, set(), [])
    keys = [k for k, _ in cands]
    # register/occupancy signal first, then the memory stall, then fast math
    assert keys == ["maxrregcount=64", "maxrregcount=32", "dlcm=cg",
                    "use_fast_math"], keys

    # tried filtering + already-accepted-flag filtering
    cands = cm.propose_flag_candidates(metrics, cu, {"maxrregcount=64"},
                                       ["-Xptxas", "-dlcm=cg"])
    keys = [k for k, _ in cands]
    assert keys == ["maxrregcount=32", "use_fast_math"], keys

    # healthy kernel with real metrics but nothing flag-shaped -> empty
    healthy = cm.parse_ncu_summary("SM throughput 92%, occupancy 85%")
    assert cm.propose_flag_candidates(healthy, "a*b+c", set(), []) == []

    # no ncu signal at all -> generic ladder (no fast math without
    # transcendentals in the source)
    cands = cm.propose_flag_candidates(cm.parse_ncu_summary(""), "a*b+c",
                                       set(), [])
    keys = [k for k, _ in cands]
    assert keys == ["maxrregcount=64", "dlcm=cg"], keys


def test_propose_sizes_candidates():
    low_occ = cm.parse_ncu_summary("achieved occupancy 20%")
    assert [k for k, _ in cm.propose_sizes_candidates(low_occ, set())][0] == "small-block"
    membound = cm.parse_ncu_summary("DRAM throughput 85%, occupancy 60%")
    assert [k for k, _ in cm.propose_sizes_candidates(membound, set())][0] == "big-tile"
    neutral = cm.parse_ncu_summary("")
    order = [k for k, _ in cm.propose_sizes_candidates(neutral, {"default-256"})]
    assert order[0] == "big-tile" and "default-256" not in order, order


def test_plan_move():
    profile = {"time_sec": 1.0, "ncu_summary": SUMMARY}
    state = cm.new_state()
    assert cm.plan_move("llm", "llm", profile, "", state).move == "llm"

    plan = cm.plan_move("compiler", "llm", profile, "", state)
    assert plan.move == "flags" and len(plan.flag_candidates) <= cm.MAX_FLAG_CANDIDATES

    # flags exhausted, ppcg backend -> retile; non-ppcg -> none/llm by mode
    state = cm.new_state()
    state["tried_flags"] = set(cm.FLAG_GROUPS)
    assert cm.plan_move("compiler", "ppcg", profile, "", state).move == "retile"
    assert cm.plan_move("compiler", "llm", profile, "", state).move == "none"
    assert cm.plan_move("hybrid", "llm", profile, "", state).move == "llm"
    assert cm.plan_move("compiler", "ppcg", profile, "", state,
                        retile_available=False).move == "none"

    # everything exhausted
    state["tried_sizes"] = set(cm.SIZES_LADDER)
    assert cm.plan_move("compiler", "ppcg", profile, "", state).move == "none"


def _write_executable(path: Path, text: str):
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _fake_toolchain(tmp: Path, nvcc_body: str) -> dict:
    """A bin dir with a fake nvcc (and a failing nvidia-smi, so arch
    detection cleanly returns None) prepended to PATH."""
    bindir = tmp / "bin"
    bindir.mkdir()
    _write_executable(bindir / "nvcc", nvcc_body)
    _write_executable(bindir / "nvidia-smi", "#!/bin/sh\nexit 1\n")
    env_path = str(bindir) + os.pathsep + os.environ["PATH"]
    return {"PATH": env_path}


# Fake nvcc: emits a shell-script "binary" that sleeps SLOW or FAST depending
# on whether --maxrregcount=64 was passed, then prints the expected output.
NVCC_FLAG_SENSITIVE = """#!/bin/sh
out=""; sleep=0.1
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out=$2; shift ;;
    --maxrregcount=64) sleep=0.01 ;;
  esac
  shift
done
cat > "$out" <<EOF
#!/bin/sh
sleep $sleep
echo "result 42"
EOF
chmod +x "$out"
"""

# Fake nvcc for the reject path: every binary prints the wrong answer.
NVCC_WRONG_OUTPUT = """#!/bin/sh
out=""
while [ $# -gt 0 ]; do
  case "$1" in -o) out=$2; shift ;; esac
  shift
done
printf '#!/bin/sh\\necho "result 43"\\n' > "$out"
chmod +x "$out"
"""

# Fake nvcc for retile: fast iff the .cu source (last arg ending in .cu)
# contains the big-tile marker the fake ppcg wrapper embeds.
NVCC_SOURCE_SENSITIVE = """#!/bin/sh
out=""; src=""
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out=$2; shift ;;
    *.cu) src=$1 ;;
  esac
  shift
done
sleep=0.1
grep -q 'tile\\[64,64\\]' "$src" && sleep=0.01
cat > "$out" <<EOF
#!/bin/sh
sleep $sleep
echo "result 42"
EOF
chmod +x "$out"
"""

# Fake ppcg_to_cu.py: records the --sizes it was given into the emitted "cu".
FAKE_PPCG_WRAPPER = """#!/usr/bin/env python3
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("input")
ap.add_argument("-o", "--output")
ap.add_argument("--sizes")
args = ap.parse_args()
with open(args.output, "w") as f:
    f.write("/* fake ppcg output; sizes=%s */\\n" % args.sizes)
print(args.output)
"""


def _workdir(tmp: Path) -> Path:
    workdir = tmp / "work"
    workdir.mkdir()
    shutil.copy(COMPARE, workdir / COMPARE.name)
    shutil.copy(TIME_BIN, workdir / TIME_BIN.name)
    (workdir / "baseline_output.txt").write_text("result 42\n", encoding="utf-8")
    (workdir / "prog.cu").write_text("/* current translation */\n", encoding="utf-8")
    return workdir


def _with_path(env_patch, fn):
    old = os.environ["PATH"]
    os.environ["PATH"] = env_patch["PATH"]
    try:
        return fn()
    finally:
        os.environ["PATH"] = old


def test_run_flag_move_accepts_winner():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        env = _fake_toolchain(tmp, NVCC_FLAG_SENSITIVE)
        workdir = _workdir(tmp)
        profile = {"time_sec": 0.5,
                   "ncu_summary": "achieved occupancy 20%, 128 registers per thread"}
        state = cm.new_state()
        plan = cm.plan_move("compiler", "llm", profile, "", state)
        assert plan.move == "flags"
        result = _with_path(env, lambda: cm.run_flag_move(
            workdir, "prog.cu", profile, plan, state, 0.05))
        schemas.validate("optimize", result)
        assert result["accepted"] is True, result
        assert result["technique_applied"] == "Nvcc_Flag_Search(maxrregcount=64)", result
        assert state["flags"] == ["--maxrregcount=64"], state
        flags_file = workdir / cm.NVCC_FLAGS_FILE
        assert flags_file.read_text().strip() == "--maxrregcount=64"
        on_disk = json.loads((workdir / cm.OPTIMIZE_RESULT_FILE).read_text())
        assert on_disk["technique_applied"] == result["technique_applied"]
        # both candidates were tried and recorded, winner and loser alike
        assert state["tried_flags"] == {"maxrregcount=64", "maxrregcount=32"}
        assert len(result["candidates"]) == 2, result["candidates"]


def test_run_flag_move_rejects_wrong_output():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        env = _fake_toolchain(tmp, NVCC_WRONG_OUTPUT)
        workdir = _workdir(tmp)
        profile = {"time_sec": 0.5, "ncu_summary": "achieved occupancy 20%"}
        state = cm.new_state()
        plan = cm.plan_move("compiler", "llm", profile, "", state)
        result = _with_path(env, lambda: cm.run_flag_move(
            workdir, "prog.cu", profile, plan, state, 0.05))
        schemas.validate("optimize", result)
        assert result["accepted"] is False, result
        assert state["flags"] == []
        assert not (workdir / cm.NVCC_FLAGS_FILE).exists()
        assert all("output mismatch" in c["rejected"]
                   for c in result["candidates"]), result["candidates"]


def test_run_retile_move_accepts_and_rewrites_cu():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        env = _fake_toolchain(tmp, NVCC_SOURCE_SENSITIVE)
        workdir = _workdir(tmp)
        (workdir / "prog.c").write_text("int main(){}\n", encoding="utf-8")
        wrapper = tmp / "fake_ppcg_to_cu.py"
        wrapper.write_text(FAKE_PPCG_WRAPPER, encoding="utf-8")
        # memory-bound signal -> big-tile candidate first -> fake nvcc makes
        # exactly that one fast -> it must win and replace prog.cu
        profile = {"time_sec": 0.5,
                   "ncu_summary": "DRAM throughput 85%, occupancy 60%"}
        state = cm.new_state()
        plan = cm.plan_move("compiler", "ppcg", profile, "", state)
        assert plan.move == "flags" or plan.move == "retile"
        # force the retile move directly (flags may be proposed first)
        plan.move = "retile"
        result = _with_path(env, lambda: cm.run_retile_move(
            workdir, "prog.c", "prog.cu", profile, plan, state, wrapper, 0.05))
        schemas.validate("optimize", result)
        assert result["accepted"] is True, result
        assert "tile[64,64]" in result["technique_applied"], result
        assert "tile[64,64]" in (workdir / "prog.cu").read_text()
        assert "big-tile" in state["tried_sizes"]


def test_run_retile_move_losing_candidate_keeps_cu():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        env = _fake_toolchain(tmp, NVCC_SOURCE_SENSITIVE)
        workdir = _workdir(tmp)
        (workdir / "prog.c").write_text("int main(){}\n", encoding="utf-8")
        wrapper = tmp / "fake_ppcg_to_cu.py"
        wrapper.write_text(FAKE_PPCG_WRAPPER, encoding="utf-8")
        # low-occupancy signal -> small-block/default candidates, neither of
        # which the fake nvcc makes fast enough to beat a 0.05s reference
        profile = {"time_sec": 0.05, "ncu_summary": "achieved occupancy 20%"}
        state = cm.new_state()
        plan = cm.plan_move("compiler", "ppcg", profile, "", state)
        plan.move = "retile"
        before = (workdir / "prog.cu").read_text()
        result = _with_path(env, lambda: cm.run_retile_move(
            workdir, "prog.c", "prog.cu", profile, plan, state, wrapper, 0.05))
        schemas.validate("optimize", result)
        assert result["accepted"] is False, result
        assert (workdir / "prog.cu").read_text() == before


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"\n{len(tests)} test(s) passed.")


if __name__ == "__main__":
    main()
