#!/usr/bin/env python3
"""
run_dataset.py - drives agent_pipeline/run_pipeline.py (the generic
C-to-CUDA agent pipeline) over EVERY benchmark in benchmark/, one
independent pipeline run per benchmark, and reports whether each
translation compiled, ran, stayed correct, and how much faster it is
than the original sequential C.

The benchmark set is discovered automatically from the benchmark/ tree
(benchmark/<tier>/<domain>/<name>.c, tiers easy/moderate/complex) so this
driver stays in sync with the suite as workloads are added -- there is no
hand-maintained list to fall out of date. The same <tier>/<domain>/<name>
relative path the rest of the suite uses (cuda/bench.sh, build_run.sh,
scripts/loc.sh) is the benchmark's name here too.

agent_pipeline/ itself knows nothing about "the dataset" -- it just takes
a sequential C source file and runs it through generate -> verify ->
profile -> optimize. This script is the dataset-specific driver, kept out
of agent_pipeline/ (which stays purely the agent's own code) the same way
the shell drivers (cuda/bench.sh, evaluation/run_eval.sh) keep
dataset-driving logic out of the thing being driven. Where run_eval.sh
scores the tracked cuda/*.cu reference conversions, this scores what the
*pipeline* generates from scratch, per benchmark.

The per-benchmark numbers reported here (compiled/ran/correct, CPU vs.
GPU time, speedup) come from the pipeline's *best* iteration's
verify/profile results -- the iteration whose measured time run_pipeline.py
actually exported as the final .cu, which may be neither the first nor the
last one optimize touched (see summarize() and run_pipeline's patience-based
stop). They're the same compile_ok/run_ok/outputs_match/time_sec/
baseline_time_sec/speedup fields cuda-verify and cuda-profile already write
into verify_result.json/profile_result.json; this script only pulls them
out into one table across the whole dataset.

The printed summary table (plus a compact KernelBench-style rollup) is
always also saved to a text file (default:
evaluation/results/report_<timestamp>.txt, override with --report) so
results from different runs can be compared side by side later instead of
only existing in scrollback.

Usage:
    python3 run_dataset.py                                  # all benchmarks
    python3 run_dataset.py --filter easy                    # a whole tier
    python3 run_dataset.py --filter easy/dnn,complex/cfd    # tier/domain prefixes
    python3 run_dataset.py --filter saxpy,easy/dense-linalg/reduction  # by name
    python3 run_dataset.py --model deepseek/deepseek-chat --max-iterations 5
    python3 run_dataset.py --backend auto --optimizer hybrid
    python3 run_dataset.py --json results.json --report my_report.txt
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "agent_pipeline"))
from run_pipeline import (  # noqa: E402
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TIMEOUT_SEC,
    HARD_ERROR_EXIT_REASONS,
    RUNS_DIR,
    run_pipeline,
)

BENCHMARK_DIR = ROOT_DIR / "benchmark"
# Reports land alongside the suite's other eval outputs (and are gitignored
# there via evaluation/results/*), not in a separate reports/ dir.
DEFAULT_REPORT_DIR = Path(__file__).resolve().parent / "results"
TIER_ORDER = {"easy": 0, "moderate": 1, "complex": 2}

# Sources under benchmark/ that are build/data tools rather than benchmarks in
# their own right -- their output is consumed by an actual benchmark. Excluded
# from discovery so the pipeline is never pointed at a non-benchmark.
# (rel path == <tier>/<domain>/<name>, matching cuda/bench.sh's identifiers.)
NON_BENCHMARK_SOURCES = {"complex/dnn/llama2_gen_checkpoint"}

# llama2_inference is the one benchmark that reads external *data* files rather
# than being sized by argv: its zero-arg default opens "model.bin"/"tokenizer.bin"
# from the current directory. Those are produced by the sibling
# llama2_gen_checkpoint build tool. run_pipeline.py must stay unaware of this
# (it just runs the C source zero-arg in an isolated workdir), so the knowledge
# -- which checkpoint dimensions to build, and that the two .bin files must sit
# next to the binary -- lives here, handed in as extra_files.
LLAMA2_REL = "complex/dnn/llama2_inference"
LLAMA2_GEN_REL = "complex/dnn/llama2_gen_checkpoint"
LLAMA2_DATA_DIR = ROOT_DIR / "evaluation" / "data" / "llama2"
# Tiny, fully deterministic config. Constraints from llama2_gen_checkpoint.c's
# header: vocab_size must be exactly 259, dim divisible by n_heads, and
# head_size (dim/n_heads) even. Small enough that a zero-arg inference run
# stays well under run_pipeline's 30 s baseline timeout.
#                 dim  hidden  n_layers  n_heads  n_kv_heads  vocab  seq_len
LLAMA2_CONFIG = ["64", "176", "4", "8", "8", "259", "256"]


def discover_benchmarks() -> list[dict]:
    """Every benchmark/<tier>/<domain>/<name>.c, minus the non-benchmark build
    tools, ordered easy -> moderate -> complex then by path. Each benchmark's
    name is its <tier>/<domain>/<name> path (the suite-wide identifier)."""
    benchmarks = []
    for src in sorted(BENCHMARK_DIR.rglob("*.c")):
        rel = src.relative_to(BENCHMARK_DIR).with_suffix("").as_posix()
        parts = rel.split("/")
        if parts[0] == "bin" or rel in NON_BENCHMARK_SOURCES:
            continue
        entry = {"name": rel, "source": src, "tier": parts[0]}
        if rel == LLAMA2_REL:
            entry["extra_files"] = [LLAMA2_DATA_DIR / "model.bin", LLAMA2_DATA_DIR / "tokenizer.bin"]
        benchmarks.append(entry)
    benchmarks.sort(key=lambda b: (TIER_ORDER.get(b["tier"], 99), b["name"]))
    return benchmarks


def ensure_llama2_checkpoint() -> None:
    """Build llama2_inference's synthetic checkpoint/tokenizer if missing, by
    compiling and running the sibling llama2_gen_checkpoint tool into
    LLAMA2_DATA_DIR. Both .bin files are gitignored (*.bin), so this
    regenerates them on demand -- the same auto-build-on-first-use pattern the
    rest of the suite uses for its own binaries."""
    model = LLAMA2_DATA_DIR / "model.bin"
    tokenizer = LLAMA2_DATA_DIR / "tokenizer.bin"
    if model.exists() and tokenizer.exists():
        return
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        sys.exit("error: no C compiler (cc/gcc) on PATH to build the llama2 checkpoint generator")
    LLAMA2_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"llama2 checkpoint/tokenizer missing, building into {LLAMA2_DATA_DIR} ...")
    gen_src = BENCHMARK_DIR / f"{LLAMA2_GEN_REL}.c"
    gen_bin = LLAMA2_DATA_DIR / "gen_checkpoint"
    compiled = subprocess.run([cc, "-std=c11", "-O2", str(gen_src), "-o", str(gen_bin), "-lm"],
                              capture_output=True, text=True)
    if compiled.returncode != 0:
        sys.exit(f"error: failed to compile {gen_src}:\n{compiled.stderr}")
    generated = subprocess.run([str(gen_bin), str(LLAMA2_DATA_DIR), *LLAMA2_CONFIG],
                               capture_output=True, text=True)
    if generated.returncode != 0:
        sys.exit(f"error: llama2 checkpoint generator failed:\n{generated.stderr}")
    if not (model.exists() and tokenizer.exists()):
        sys.exit(f"error: llama2 generator did not produce model.bin/tokenizer.bin in {LLAMA2_DATA_DIR}")


def summarize(name: str, result: dict, elapsed: float) -> dict:
    """Pull compiled/ran/correct/timing out of a run_pipeline() result for one benchmark.

    Uses the *best* iteration's verify/profile data (result["best"]["iteration"],
    the same iteration run_pipeline.py snapshotted and exported as the final
    .cu) rather than the last iteration -- the pipeline may have kept
    iterating past its best point before the patience-based stop condition
    triggered, and the exported .cu is the best one, not the last one.
    Falls back to the last iteration if no best was ever recorded (e.g. the
    run never got past generate_failed/verify_failed on iteration 1).
    """
    summary = {
        "name": name,
        "exit_reason": result.get("exit_reason"),
        "iterations": len(result.get("iterations", [])),
        "elapsed_sec": round(elapsed, 1),
        "best_iteration": result.get("best", {}).get("iteration"),
        "compiled": None,
        "ran": None,
        "correct": None,
        "cpu_time_sec": None,
        "gpu_time_sec": None,
        "speedup": None,
    }

    iterations = result.get("iterations", [])
    if not iterations:
        return summary

    best_iteration = result.get("best", {}).get("iteration")
    if best_iteration is not None and 1 <= best_iteration <= len(iterations):
        chosen = iterations[best_iteration - 1]
    else:
        chosen = iterations[-1]

    verify = chosen.get("verify")
    if verify:
        summary["compiled"] = verify.get("compile_ok")
        summary["ran"] = verify.get("run_ok")
        summary["correct"] = verify.get("outputs_match")

    profile = chosen.get("profile")
    if profile and not profile.get("stubbed", True):
        summary["cpu_time_sec"] = profile.get("baseline_time_sec")
        summary["gpu_time_sec"] = profile.get("time_sec")
        summary["speedup"] = profile.get("speedup")

    return summary


def _fmt(value, fmt="{:.4g}"):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return fmt.format(value)
    return str(value)


def build_report(summaries: list[dict]) -> str:
    """Render the same aligned table print_report used to only print to
    stdout, as a string -- so it can be both printed and saved to a file
    (see --report in main()) without duplicating the formatting logic."""
    name_w = max([len(s["name"]) for s in summaries] + [len("Benchmark")]) + 2
    header = (
        f"{'Benchmark':<{name_w}}{'Compiled':>9}{'Ran':>6}{'Correct':>8}"
        f"{'CPU(s)':>10}{'GPU(s)':>10}{'Speedup':>9}{'Iters':>7}  Exit reason"
    )
    lines = [header, "-" * len(header)]
    for s in summaries:
        speedup = f"{s['speedup']:.2f}x" if isinstance(s["speedup"], (int, float)) else "n/a"
        row = (
            f"{s['name']:<{name_w}}{_fmt(s['compiled']):>9}{_fmt(s['ran']):>6}{_fmt(s['correct']):>8}"
            f"{_fmt(s['cpu_time_sec']):>10}{_fmt(s['gpu_time_sec']):>10}{speedup:>9}"
            f"{s['iterations']:>7}  {s['exit_reason']}"
        )
        lines.append(row)
    return "\n".join(lines)


def print_report(summaries: list[dict]) -> None:
    print(build_report(summaries))


def build_aggregate(summaries: list[dict]) -> str:
    """A compact KernelBench-style rollup over the whole run: how many
    compiled/ran/stayed correct, the geomean speedup over correct conversions,
    and per-tier correctness. Complements the per-benchmark table for the
    all-benchmarks pass (mirrors run_eval.sh's fast_0 / geomean framing)."""
    n = len(summaries)
    if n == 0:
        return "No benchmarks run."
    compiled = sum(1 for s in summaries if s["compiled"])
    ran = sum(1 for s in summaries if s["ran"])
    correct = sum(1 for s in summaries if s["correct"])
    sps = [s["speedup"] for s in summaries
           if s["correct"] and isinstance(s["speedup"], (int, float)) and s["speedup"] > 0]
    geo = math.exp(sum(math.log(x) for x in sps) / len(sps)) if sps else None
    fast1 = sum(1 for x in sps if x > 1)

    tiers: dict[str, list[int]] = {}
    for s in summaries:
        t = s["name"].split("/")[0]
        tiers.setdefault(t, [0, 0])
        tiers[t][0] += 1
        tiers[t][1] += 1 if s["correct"] else 0

    lines = [
        f"Totals: {n} benchmarks | compiled {compiled} | ran {ran} | "
        f"correct {correct} ({100 * correct / n:.0f}%)",
        f"fast_1 (correct & >1x): {fast1}/{n}"
        + (f" | geomean speedup over correct: {geo:.2f}x" if geo else ""),
        "Per-tier correct: " + " | ".join(
            f"{t} {tiers[t][1]}/{tiers[t][0]}"
            for t in sorted(tiers, key=lambda t: TIER_ORDER.get(t, 99))),
    ]
    return "\n".join(lines)


def select_benchmarks(benchmarks: list[dict], filter_arg: str | None) -> list[dict]:
    """Subset benchmarks by a comma-separated --filter. A token matches a
    benchmark when it equals the full <tier>/<domain>/<name>, equals the bare
    file stem (e.g. `saxpy`), or is a path prefix of it (`easy`, `easy/dnn`),
    so a whole tier or domain can be selected as easily as one workload. Any
    token that matches nothing is a hard error."""
    tokens = [t.strip().strip("/") for t in filter_arg.split(",") if t.strip()]
    selected, matched = [], set()
    for b in benchmarks:
        stem = b["source"].stem
        for tok in tokens:
            if tok == b["name"] or tok == stem or b["name"].startswith(tok + "/"):
                selected.append(b)
                matched.add(tok)
                break
    unknown = [t for t in tokens if t not in matched]
    if unknown:
        sys.exit(f"unknown benchmark filter(s): {', '.join(unknown)}")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"opencode model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--filter", help="comma-separated subset to run: a full <tier>/<domain>/<name>, "
                                         "a bare name (saxpy), or a path prefix (easy, easy/dnn)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SEC, help="per-stage-call timeout in seconds")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS, help="hard cap on verify/profile/optimize loop iterations")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"where to copy each benchmark's final .cu (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--backend", choices=("llm", "ppcg", "hybrid", "auto"), default="llm",
                        help="who produces the initial .cu (passed through to run_pipeline; default: llm)")
    parser.add_argument("--optimizer", choices=("llm", "compiler", "hybrid"), default="llm",
                        help="who takes each iteration's optimize slot (passed through to run_pipeline; default: llm)")
    parser.add_argument("--json", type=Path, help="write the per-benchmark summary table to this JSON file")
    parser.add_argument("--report", type=Path, help=f"write the printed summary table to this text file (default: {DEFAULT_REPORT_DIR}/report_<timestamp>.txt)")
    args = parser.parse_args()

    benchmarks = discover_benchmarks()
    if not benchmarks:
        sys.exit(f"no benchmarks found under {BENCHMARK_DIR}")
    if args.filter:
        benchmarks = select_benchmarks(benchmarks, args.filter)

    RUNS_DIR.mkdir(exist_ok=True)
    if any(b["name"] == LLAMA2_REL for b in benchmarks):
        ensure_llama2_checkpoint()

    print(f"Running {len(benchmarks)} benchmark(s) through the pipeline "
          f"(backend={args.backend}, optimizer={args.optimizer}).", flush=True)
    summaries = []
    for bench in benchmarks:
        print(f"==> {bench['name']}: running pipeline with {args.model} ...", flush=True)
        start = time.monotonic()
        try:
            result = run_pipeline(
                bench["source"],
                args.model,
                args.timeout,
                args.max_iterations,
                output_dir=args.output_dir,
                extra_files=bench.get("extra_files"),
                backend=args.backend,
                optimizer=args.optimizer,
            )
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            print(f"    FAILED: {exc}")
            summaries.append(summarize(bench["name"], {"exit_reason": "error"}, time.monotonic() - start))
            continue
        elapsed = time.monotonic() - start
        summary = summarize(bench["name"], result, elapsed)
        print(
            f"    exit_reason={summary['exit_reason']} compiled={summary['compiled']} "
            f"ran={summary['ran']} correct={summary['correct']} speedup={summary['speedup']} "
            f"in {elapsed:.1f}s"
        )
        summaries.append(summary)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print()
    report_text = build_report(summaries)
    print(report_text)

    aggregate = build_aggregate(summaries)
    ok_count = sum(1 for s in summaries if s["exit_reason"] not in HARD_ERROR_EXIT_REASONS)
    summary_line = f"{ok_count}/{len(summaries)} benchmarks completed the pipeline without a hard error."
    print(f"\n{aggregate}\n{summary_line}")

    # Always save the printed table to a file, not just stdout, so multiple
    # runs can be directly compared later instead of being lost to scrollback.
    report_path = args.report or (DEFAULT_REPORT_DIR / f"report_{timestamp}.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text + "\n\n" + aggregate + "\n" + summary_line + "\n", encoding="utf-8")
    print(f"\nWrote report to {report_path}")

    if args.json:
        # Timestamp the actual filename written, regardless of what was
        # passed in -- otherwise a fixed --json path gets clobbered every
        # time this is re-run instead of accumulating a history of results.
        json_path = args.json.with_stem(f"{args.json.stem}_{timestamp}")
        json_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"Wrote summary to {json_path}")

    if ok_count < len(summaries):
        sys.exit(1)


if __name__ == "__main__":
    main()
