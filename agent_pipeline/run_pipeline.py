#!/usr/bin/env python3
"""
run_pipeline.py - the real (multi-stage, looping) C-to-CUDA conversion
pipeline: generate -> verify -> profile -> optimize -> (loop back to
verify), stopping when this iteration's profiled wall-clock time isn't at
least 5% better than the best time seen so far, for 3 consecutive
iterations in a row (patience-based early stopping, like ML training --
see PATIENCE/MIN_DELTA below). This is computed from each iteration's real
profile_result.json["time_sec"], not optimize's own self-report -- optimize
only ever sees the pre-edit profile, so it can't know how much its own
edit helped until the next iteration actually re-profiles it. The .cu
exported at the end is whichever iteration actually had the best measured
time, not necessarily the last one optimize touched.

The original C program is compiled and run exactly once, mechanically, by
this orchestrator (see _build_and_run_baseline) before generate even
starts -- not re-measured by verify/profile every iteration. Re-running C
repeatedly was producing noise unrelated to the CUDA translation (e.g.
cache/warm-up effects making a later run look faster), so every iteration's
correctness diff and speedup number is compared against this one fixed,
mechanically-captured reference instead.

This is the agent's pipeline itself, not a dataset-running script: it
takes a sequential C source file and produces a CUDA translation plus a
JSON trail of what each stage decided -- it has no notion of "benchmarks"
or any other dataset-specific concept. See dataset_eval/run_dataset.py for
the script that drives this over every benchmark in dataset_zyj/.

Each of the 4 stages is a separate opencode agent (defined under
opencode_config/agent/) with its own system prompt and tool permissions
(e.g. cuda-verify is read-only). This script is the orchestrator: it
enforces the fixed stage order and the loop/stop logic itself, outside
opencode, since opencode has no built-in multi-agent pipeline mechanism
(see CLAUDE.md's "Agent pipeline architecture" section for why).

The source file is run from an isolated temp workdir containing only that
one file, and a single opencode session that all 4 stages share via
--session, so each stage sees prior stages' conversation history.

If a stage crashes, times out, or writes a malformed result, the pipeline
stops there but still records everything earlier stages produced (e.g. a
verify pass/fail from a completed iteration even if that iteration's
profile call then failed) and still exports the best .cu produced so far
-- see exit_reason values ending in "_error" in the written
pipeline_result.json, and HARD_ERROR_EXIT_REASONS below.

Usage:
    python3 run_pipeline.py path/to/program.c
    python3 run_pipeline.py path/to/a.c path/to/b.c
    python3 run_pipeline.py program.c --model deepseek/deepseek-chat --max-iterations 5
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import schemas

ROOT_DIR = Path(__file__).resolve().parent.parent
OPENCODE_DIR = ROOT_DIR / "opencode"
PIPELINE_DIR = Path(__file__).resolve().parent
OPENCODE_CONFIG_DIR = PIPELINE_DIR / "opencode_config"
RUNS_DIR = PIPELINE_DIR / "runs"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "generated"
COMPARE_SCRIPT = OPENCODE_CONFIG_DIR / "skill" / "cuda-verification-procedure" / "compare_outputs.py"
TIME_BINARY_SCRIPT = OPENCODE_CONFIG_DIR / "skill" / "cuda-profiling-procedure" / "time_binary.py"
NCU_CONFIG_FILE = OPENCODE_CONFIG_DIR / "skill" / "cuda-profiling-procedure" / "ncu_metrics.cfg"
WORKDIR_TOOLS = [COMPARE_SCRIPT, TIME_BINARY_SCRIPT, NCU_CONFIG_FILE]

DEFAULT_MODEL = "deepseek/deepseek-chat"
DEFAULT_TIMEOUT_SEC = 600
DEFAULT_MAX_ITERATIONS = 5

# Patience-based early stopping (same shape as ML training early stopping):
# an iteration only counts as "real progress" if it beats the best time seen
# so far by at least MIN_DELTA. After PATIENCE consecutive iterations without
# that much progress, stop -- the comparison is always against the best time
# ever seen, not just the previous iteration, so a single noisy iteration
# can't prematurely end a run that's still genuinely improving.
PATIENCE = 3
MIN_DELTA = 0.05

# exit_reason values that mean a stage crashed, timed out, or wrote a
# malformed result -- as opposed to a stage running fine and reporting a
# real "fail" (generate_failed/verify_failed), or the loop stopping on its
# own normal conditions (max_iterations_reached/stagnated).
HARD_ERROR_EXIT_REASONS = {
    "error",
    "baseline_error",
    "generate_failed",
    "generate_error",
    "verify_error",
    "profile_error",
    "optimize_error",
}

STAGE_AGENT = {
    "generate": "cuda-generate",
    "verify": "cuda-verify",
    "profile": "cuda-profile",
    "optimize": "cuda-optimize",
}


def stage_prompt(stage: str, c_filename: str) -> str:
    if stage == "generate":
        return f"Translate the C program in {c_filename} into CUDA, following your instructions."
    if stage == "verify":
        return "Review the current CUDA translation now, following your instructions."
    if stage == "profile":
        return "Proceed with profiling, following your instructions."
    if stage == "optimize":
        return "Proceed with optimization, following your instructions."
    raise ValueError(f"unknown stage: {stage}")


def run_stage(stage: str, prompt: str, workdir: Path, model: str, timeout_sec: float, session_id: str | None):
    cmd = [
        "bun", "dev", "run", prompt,
        "--agent", STAGE_AGENT[stage],
        "--dir", str(workdir),
        "--model", model,
        "--format", "json",
        "--dangerously-skip-permissions",
    ]
    if session_id:
        cmd += ["--session", session_id]

    env = os.environ.copy()
    env["OPENCODE_CONFIG_DIR"] = str(OPENCODE_CONFIG_DIR)

    # stdin=DEVNULL is required, not optional: opencode's `run` command does
    # `process.stdin.isTTY ? undefined : await Bun.stdin.text()` to pick up
    # piped context (opencode/packages/opencode/src/cli/cmd/run.ts:351). Any
    # non-interactive launch -- this one included -- has a non-TTY stdin, so
    # without an explicit EOF it hangs forever waiting for a stdin that never
    # closes, instead of resolving immediately to "no piped input".
    # encoding="utf-8" is required, not optional: opencode emits UTF-8 JSON on
    # stdout, but Python's text mode otherwise decodes with
    # locale.getpreferredencoding() -- the system ANSI codepage (e.g. GBK on a
    # Chinese-locale Windows install), which raises UnicodeDecodeError on any
    # non-ASCII byte sequence opencode's output contains.
    proc = subprocess.run(
        cmd,
        cwd=OPENCODE_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout_sec,
        env=env,
        stdin=subprocess.DEVNULL,
    )

    new_session_id = session_id
    for line in proc.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "sessionID" in event:
            new_session_id = event["sessionID"]
            break

    return proc, new_session_id


def read_stage_result(workdir: Path, stage: str) -> dict:
    path = workdir / schemas.RESULT_FILENAMES[stage]
    if not path.exists():
        raise RuntimeError(f"{stage} stage did not write {path.name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    schemas.validate(stage, data)
    return data


def _build_and_run_baseline(workdir: Path, c_filename: str, name: str) -> dict:
    """Compile and run the original C program exactly once, mechanically (no
    LLM involved -- it's deterministic), and capture its stdout + a stable
    timing as the one fixed reference every iteration's verify/profile stage
    compares against. Mirrors the same compile/run convention the verify
    skill already used (cc -std=c11 -O2 ... -lm, no CLI args), but as a
    Python subprocess call instead of shell text in a skill, so a portable
    subprocess timeout can be used instead of the shell `timeout` command.

    This pipeline already assumes a Unix-like toolchain throughout (nvcc,
    ncu/nsys, nvidia-smi) -- this step matches that assumption rather than
    introducing new cross-platform compiler-detection logic.
    """
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        raise RuntimeError("no C compiler found on PATH (tried cc, gcc)")

    binary_name = f"{name}_c"
    subprocess.run(
        [cc, "-std=c11", "-O2", c_filename, "-o", binary_name, "-lm"],
        cwd=workdir,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    proc = subprocess.run(
        [f"./{binary_name}"],
        cwd=workdir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=True,
    )
    output_file = "baseline_output.txt"
    (workdir / output_file).write_text(proc.stdout, encoding="utf-8")

    timing_proc = subprocess.run(
        ["python3", TIME_BINARY_SCRIPT.name, f"./{binary_name}", "--reps", "5"],
        cwd=workdir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    timing = json.loads(timing_proc.stdout)

    baseline = {"time_sec": timing["mean"], "output_file": output_file}
    (workdir / "baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    return baseline


def _snapshot_best(
    run_dir: Path, workdir: Path, output_file: str, iteration: int, verify_result: dict, profile_result: dict
) -> None:
    """Copy the just-profiled .cu plus its verify/profile results into
    run_dir/best/, overwriting any earlier snapshot -- only the most recent
    "best" needs to survive, since each new best fully supersedes the last."""
    best_dir = run_dir / "best"
    best_dir.mkdir(exist_ok=True)
    shutil.copy(workdir / output_file, best_dir / output_file)
    (best_dir / "verify_result.json").write_text(json.dumps(verify_result, indent=2), encoding="utf-8")
    (best_dir / "profile_result.json").write_text(json.dumps(profile_result, indent=2), encoding="utf-8")
    (best_dir / "meta.json").write_text(json.dumps({"iteration": iteration}, indent=2), encoding="utf-8")


def run_pipeline(
    source: Path,
    model: str,
    timeout_sec: float,
    max_iterations: int,
    runs_dir: Path = RUNS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    extra_files: list[Path] | None = None,
) -> dict:
    """Run the full generate->verify->profile->optimize pipeline on one sequential C file.

    extra_files, if given, are copied into the workdir alongside the source
    (keeping their own filenames) before anything runs. This stays generic --
    run_pipeline.py itself has no notion of what these files are for -- but
    lets a caller like dataset_eval/run_dataset.py supply runtime data a
    particular source file's zero-arg default depends on (e.g.
    llama2_c_inference.c's default checkpoint_path/tokenizer_path of
    "model.bin"/"tokenizer.bin", which must physically exist in the
    directory the binary runs in)."""
    name = source.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / f"{name}_{timestamp}"
    run_dir.mkdir(parents=True)

    workdir = Path(tempfile.mkdtemp(prefix=f"opencode_pipeline_{name}_"))
    c_filename = source.name
    shutil.copy(source, workdir / c_filename)
    for tool in WORKDIR_TOOLS:
        shutil.copy(tool, workdir / tool.name)
    for extra in extra_files or []:
        shutil.copy(extra, workdir / extra.name)

    def call(stage: str, session_id: str | None, iteration: int | None = None) -> str | None:
        # iteration suffix keeps each loop pass's log distinct -- verify/profile/
        # optimize each run once per iteration, and without it every iteration
        # after the first would silently overwrite the previous one's log.
        log_name = f"{stage}.log" if iteration is None else f"{stage}_iter{iteration}.log"
        prompt = stage_prompt(stage, c_filename)
        proc, new_session_id = run_stage(stage, prompt, workdir, model, timeout_sec, session_id)
        (run_dir / log_name).write_text(
            f"$ {' '.join(['bun', 'dev', 'run', prompt, '--agent', STAGE_AGENT[stage]])}\n\n"
            f"--- stdout ---\n{proc.stdout}\n\n--- stderr ---\n{proc.stderr}\n",
            encoding="utf-8",
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{stage} stage exited {proc.returncode}, see {run_dir / log_name}")
        return new_session_id

    def call_and_read(stage: str, session_id: str | None, iteration: int | None = None):
        """Run one stage and read its result, turning any failure into an
        (unchanged session_id, None, "<stage>: <reason>") tuple instead of an
        exception -- so the caller can record what happened and still save
        whatever earlier stages already produced, rather than losing it all
        to an unwound stack."""
        try:
            new_session_id = call(stage, session_id, iteration)
            data = read_stage_result(workdir, stage)
            return new_session_id, data, None
        except (RuntimeError, subprocess.TimeoutExpired, ValueError) as exc:
            return session_id, None, f"{stage}: {exc}"

    result = {"name": name, "source": str(source), "workdir": str(workdir), "iterations": []}

    def finish(exit_reason: str, error: str | None = None) -> dict:
        # Single exit path so a summary is always written and the best .cu
        # seen so far is always exported, no matter which stage stopped the
        # pipeline -- partial progress is still useful, not all-or-nothing.
        result["exit_reason"] = exit_reason
        if error is not None:
            result["error"] = error
        _write_summary(run_dir, result)
        _export_output(output_dir, name, workdir, result.get("generate"), run_dir)
        return result

    try:
        baseline = _build_and_run_baseline(workdir, c_filename, name)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError) as exc:
        return finish("baseline_error", str(exc))
    result["baseline"] = baseline

    session_id, generate_result, error = call_and_read("generate", None)
    if error is not None:
        return finish("generate_error", error)
    result["generate"] = generate_result

    if generate_result["status"] != "done":
        return finish("generate_failed")

    output_file = generate_result.get("output_file") or f"{name}.cu"

    best_time_sec = None
    best_iteration = None
    stagnant_count = 0

    for i in range(1, max_iterations + 1):
        iteration_data = {"iteration": i}

        session_id, verify_result, error = call_and_read("verify", session_id, i)
        if error is not None:
            result["iterations"].append(iteration_data)
            return finish("verify_error", error)
        iteration_data["verify"] = verify_result

        if verify_result["status"] == "fail":
            result["iterations"].append(iteration_data)
            return finish("verify_failed")

        session_id, profile_result, error = call_and_read("profile", session_id, i)
        if error is not None:
            result["iterations"].append(iteration_data)
            return finish("profile_error", error)
        # Trust the mechanically-captured baseline over whatever the agent
        # claims for baseline_time_sec/speedup -- same "objective signal over
        # self-report" preference applied everywhere else in this pipeline.
        profile_result["baseline_time_sec"] = baseline["time_sec"]
        profile_result["speedup"] = baseline["time_sec"] / profile_result["time_sec"]
        iteration_data["profile"] = profile_result

        # Patience-based early stopping against the best time ever seen, not
        # just the previous iteration -- a single noisy/regressed iteration
        # can't end a run that's still genuinely improving overall.
        curr_time = profile_result["time_sec"]
        if best_time_sec is None or curr_time < best_time_sec * (1 - MIN_DELTA):
            best_time_sec, best_iteration, stagnant_count = curr_time, i, 0
            _snapshot_best(run_dir, workdir, output_file, i, verify_result, profile_result)
        else:
            stagnant_count += 1
        result["best"] = {"iteration": best_iteration, "time_sec": best_time_sec}

        if stagnant_count >= PATIENCE:
            result["iterations"].append(iteration_data)
            return finish("stagnated")

        session_id, optimize_result, error = call_and_read("optimize", session_id, i)
        if error is not None:
            result["iterations"].append(iteration_data)
            return finish("optimize_error", error)
        iteration_data["optimize"] = optimize_result

        result["iterations"].append(iteration_data)

    return finish("max_iterations_reached")


def _write_summary(run_dir: Path, result: dict) -> None:
    result["completed_at"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "pipeline_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def _export_output(output_dir: Path, name: str, workdir: Path, generate_result: dict | None, run_dir: Path) -> None:
    """Copy the best .cu (and the run summary) out into output_dir/<name>/, if
    one exists -- regardless of whether the pipeline went on to succeed or
    fail in a later stage. Prefers run_dir/best/ (the snapshot taken whenever
    an iteration set a new best profiled time -- see _snapshot_best) over the
    workdir's current .cu, which may already reflect a further optimize edit
    that was never reverified/reprofiled. Falls back to the workdir (and to
    the "<name>.cu" convention cuda-generate is instructed to use) when no
    best snapshot exists yet -- e.g. generate_error/generate_failed/
    verify_failed-on-iteration-1, before any iteration ever got profiled."""
    output_file = (generate_result or {}).get("output_file") or f"{name}.cu"
    best_dir = run_dir / "best"
    src_dir = best_dir if (best_dir / output_file).is_file() else workdir
    produced = src_dir / output_file
    if not produced.is_file():
        return
    dest_dir = output_dir / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(produced, dest_dir / output_file)
    shutil.copy(run_dir / "pipeline_result.json", dest_dir / "pipeline_result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="+", type=Path, help="path(s) to a sequential C source file to convert")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"opencode model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SEC, help="per-stage-call timeout in seconds")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS, help="hard cap on verify/profile/optimize loop iterations")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"where to copy the final .cu (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()

    for src in args.source:
        if not src.exists():
            sys.exit(f"error: no such file: {src}")

    RUNS_DIR.mkdir(exist_ok=True)

    results = []
    for src in args.source:
        print(f"==> {src.name}: running pipeline with {args.model} ...", flush=True)
        start = time.monotonic()
        try:
            result = run_pipeline(src, args.model, args.timeout, args.max_iterations, output_dir=args.output_dir)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            print(f"    FAILED: {exc}")
            results.append({"name": src.stem, "exit_reason": "error", "error": str(exc)})
            continue
        elapsed = time.monotonic() - start
        print(f"    exit_reason={result['exit_reason']} iterations={len(result['iterations'])} in {elapsed:.1f}s")
        results.append(result)

    ok_count = sum(1 for r in results if r.get("exit_reason") not in HARD_ERROR_EXIT_REASONS)
    print(f"\n{ok_count}/{len(results)} file(s) completed the pipeline without a hard error.")

    if ok_count < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
