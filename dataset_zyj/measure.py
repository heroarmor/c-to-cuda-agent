#!/usr/bin/env python3
"""
measure.py - build/run/verify/time the sequential C benchmark dataset.

The .c files in single_kernel/ and multi_kernels/ are plain sequential C:
no embedded timing, no embedded self-check. Both of those are this
script's job instead, so the dataset stays "pure" and realistic (this is
what the conversion agent will be handed) while still letting us confirm
the dataset itself is correct and reproducible:

  - Timing: wall-clock around the whole process (setup + computation),
    matching the project's actual goal of end-to-end program speedup, not
    an isolated kernel benchmark.
  - Verification: each benchmark's stdout is diffed, line by line and
    number by number (with float tolerance), against a "golden" reference
    captured once from a known-good run and stored under golden/<mode>.json.

Usage:
    python3 measure.py                          # quick correctness pass
    python3 measure.py --mode perf              # heavier sizes, 5 reps, timing
    python3 measure.py --mode perf --json baseline.json
    python3 measure.py --filter CNN,reduction
    python3 measure.py --mode perf --baseline baseline.json    # show speedup
    python3 measure.py --update-golden          # (re)capture golden/quick.json
    python3 measure.py --mode perf --update-golden

llama2_c_inference runs against a tiny, fully deterministic checkpoint and
tokenizer synthesized by `make all` (see llama2_gen_checkpoint.c and the
Makefile) -- no real pretrained model or internet access required. Its
upstream "achieved tok/s" diagnostic is printed to stderr (wall-clock
dependent, intentionally not captured here).

Once parallel (OpenMP/CUDA) variants of a benchmark exist, add another
entry to BENCHMARKS pointing at the new binary with the same args, run
this script again, and pass the sequential run's --json output as
--baseline to get an automatic speedup column. The golden diff doubles as
a correctness check for those variants too, since they share the same
stdout-based contract.
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent
BIN_DIR = DATASET_DIR / "bin"
DATA_DIR = DATASET_DIR / "data"
GOLDEN_DIR = DATASET_DIR / "golden"

NUMBER_RE = re.compile(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?")

# name        -> binary under bin/, plus the CLI args used for a fast
# correctness smoke test ("quick") and for a heavier timing run ("perf").
BENCHMARKS = [
    {"name": "reduction", "binary": "reduction", "quick_args": ["1000000"], "perf_args": ["20000000"]},
    {"name": "CNN", "binary": "CNN", "quick_args": ["1000"], "perf_args": ["2000000"]},
    # quick_args are now explicit (rather than relying on argc<2 defaults):
    # the .c files' zero-arg defaults were bumped to perf_args scale so the
    # agent pipeline (which always runs binaries with no CLI args) exercises
    # a GPU-meaningful problem size instead of one dominated by kernel-launch
    # overhead. quick_args restores the original small sizes for measure.py's
    # fast correctness smoke test.
    {"name": "image_process", "binary": "image_process", "quick_args": ["512", "512", "10"], "perf_args": ["1024", "1024", "20"]},
    {"name": "stream_cluster", "binary": "stream_cluster", "quick_args": ["20000", "8", "2000", "64", "3"], "perf_args": ["20000", "16", "2000", "96", "5"]},
    {"name": "tiny_cnn_training_single_file", "binary": "tiny_cnn_training_single_file", "quick_args": ["64", "10", "0.01"], "perf_args": ["128", "15", "0.01"]},
    {
        "name": "llama2_c_inference",
        "binary": "llama2_c_inference",
        "quick_args": [str(DATA_DIR / "quick" / "model.bin"), "-z", str(DATA_DIR / "quick" / "tokenizer.bin"), "-t", "0", "-s", "42", "-n", "8", "-i", "hi"],
        "perf_args": [str(DATA_DIR / "perf" / "model.bin"), "-z", str(DATA_DIR / "perf" / "tokenizer.bin"), "-t", "0", "-s", "42", "-n", "64", "-i", "the quick brown fox"],
    },
]

DEFAULT_REPS = {"quick": 1, "perf": 5}


def ensure_built():
    missing = [b for b in BENCHMARKS if not (BIN_DIR / b["binary"]).exists()]
    data_missing = not (DATA_DIR / "quick" / "model.bin").exists() or not (DATA_DIR / "perf" / "model.bin").exists()
    if not missing and not data_missing:
        return
    if data_missing and not missing:
        print("Synthetic llama2 checkpoint/tokenizer data missing, building via `make all`...")
    else:
        print(f"Building {len(missing)} missing binar{'y' if len(missing) == 1 else 'ies'} via `make all`...")
    result = subprocess.run(["make", "-C", str(DATASET_DIR), "all"])
    if result.returncode != 0:
        sys.exit(
            "error: `make all` failed. Build manually, e.g.:\n"
            "  cc -std=c11 -Wall -Wextra -O2 multi_kernels/CNN.c -o bin/CNN -lm"
        )


def outputs_match(current: str, golden: str, rel_tol=1e-4, abs_tol=1e-6):
    """Line-by-line, number-by-number comparison with float tolerance.

    Non-numeric text on each line must match exactly; numbers (ints or
    floats, found in the same order on each line) are compared with
    math.isclose so harmless platform/compiler floating-point noise
    doesn't look like a regression.
    """
    current_lines = current.splitlines()
    golden_lines = golden.splitlines()

    if len(current_lines) != len(golden_lines):
        return False, f"line count differs: {len(current_lines)} vs golden {len(golden_lines)}"

    for lineno, (a, b) in enumerate(zip(current_lines, golden_lines), start=1):
        template_a = NUMBER_RE.sub("\0", a)
        template_b = NUMBER_RE.sub("\0", b)
        if template_a != template_b:
            return False, f"line {lineno} differs: {a!r} vs golden {b!r}"

        nums_a = NUMBER_RE.findall(a)
        nums_b = NUMBER_RE.findall(b)
        for na, nb in zip(nums_a, nums_b):
            if not _isclose(float(na), float(nb), rel_tol, abs_tol):
                return False, f"line {lineno} value differs: {na} vs golden {nb}"

    return True, None


def _isclose(a: float, b: float, rel_tol: float, abs_tol: float) -> bool:
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def run_once(binary: Path, args, timeout: float):
    try:
        start = time.perf_counter()
        proc = subprocess.run(
            [str(binary), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": f"timed out after {timeout}s", "time": None, "stdout": None}
    except OSError as exc:
        return {"ok": False, "reason": f"failed to launch: {exc}", "time": None, "stdout": None}

    if proc.returncode != 0:
        return {"ok": False, "reason": f"exit code {proc.returncode}", "time": None, "stdout": proc.stdout}

    return {"ok": True, "reason": None, "time": elapsed, "stdout": proc.stdout}


def run_benchmark(bench, mode: str, reps: int, timeout: float, golden: dict):
    binary = BIN_DIR / bench["binary"]
    args = bench["perf_args"] if mode == "perf" else bench["quick_args"]

    times = []
    stdout = None
    failure_reason = None
    for _ in range(reps):
        result = run_once(binary, args, timeout)
        if not result["ok"]:
            failure_reason = result["reason"]
            break
        times.append(result["time"])
        stdout = result["stdout"]

    if failure_reason is not None:
        return {"name": bench["name"], "status": "FAIL", "reason": failure_reason, "args": args, "times": times}

    golden_stdout = golden.get(bench["name"])
    if golden_stdout is None:
        return {
            "name": bench["name"],
            "status": "FAIL",
            "reason": "no golden reference for this mode -- run with --update-golden first",
            "args": args,
            "times": times,
        }

    match, reason = outputs_match(stdout, golden_stdout)
    if not match:
        return {"name": bench["name"], "status": "FAIL", "reason": f"output mismatch: {reason}", "args": args, "times": times}

    return {
        "name": bench["name"],
        "status": "PASS",
        "reason": None,
        "args": args,
        "times": times,
        "min": min(times),
        "mean": statistics.mean(times),
        "median": statistics.median(times),
    }


def golden_path(mode: str) -> Path:
    return GOLDEN_DIR / f"{mode}.json"


def load_golden(mode: str) -> dict:
    path = golden_path(mode)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def update_golden(benchmarks, mode: str, timeout: float):
    golden = {}
    for bench in benchmarks:
        binary = BIN_DIR / bench["binary"]
        args = bench["perf_args"] if mode == "perf" else bench["quick_args"]
        result = run_once(binary, args, timeout)
        if not result["ok"]:
            sys.exit(f"error: could not capture golden output for {bench['name']}: {result['reason']}")
        golden[bench["name"]] = result["stdout"]

    path = golden_path(mode)
    path.parent.mkdir(exist_ok=True)
    existing = load_golden(mode)
    existing.update(golden)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2, sort_keys=True)
    print(f"Wrote golden reference for {len(golden)} benchmark(s) to {path}")


def fmt_seconds(value: float) -> str:
    # 4 significant digits, switching to scientific notation for very small
    # values (e.g. the nanosecond-scale CNN kernel) so the column stays
    # readable across benchmarks that span many orders of magnitude. Full
    # float precision is preserved in --json output regardless.
    return f"{value:.4g}"


def print_report(results, baseline=None):
    name_w = max([len(r["name"]) for r in results] + [len("Benchmark")]) + 2
    header = f"{'Benchmark':<{name_w}}{'Status':<8}{'Min(s)':>12}{'Mean(s)':>12}{'Median(s)':>12}"
    if baseline:
        header += f"{'Speedup':>10}"
    print(header)
    print("-" * len(header))

    for r in results:
        if r["status"] != "PASS":
            print(f"{r['name']:<{name_w}}{'FAIL':<8}  ({r['reason']})")
            continue

        row = (
            f"{r['name']:<{name_w}}{'PASS':<8}"
            f"{fmt_seconds(r['min']):>12}{fmt_seconds(r['mean']):>12}{fmt_seconds(r['median']):>12}"
        )
        if baseline:
            base = baseline.get(r["name"])
            if base and base.get("status") == "PASS" and r["mean"] > 0:
                row += f"{base['mean'] / r['mean']:>9.2f}x"
            else:
                row += f"{'n/a':>10}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["quick", "perf"], default="quick", help="quick correctness pass or heavier perf timing run")
    parser.add_argument("--reps", type=int, default=None, help="override repeat count (default: 1 for quick, 5 for perf)")
    parser.add_argument("--filter", type=str, default=None, help="comma-separated list of benchmark names to run")
    parser.add_argument("--timeout", type=float, default=120.0, help="per-run timeout in seconds")
    parser.add_argument("--json", type=str, default=None, help="write full results to this JSON file")
    parser.add_argument("--baseline", type=str, default=None, help="JSON file from a previous run to compute speedup against")
    parser.add_argument("--no-build", action="store_true", help="skip the automatic `make all` build step")
    parser.add_argument("--update-golden", action="store_true", help="(re)capture golden/<mode>.json from the current binaries instead of checking against it")
    args = parser.parse_args()

    benchmarks = BENCHMARKS
    if args.filter:
        wanted = set(args.filter.split(","))
        benchmarks = [b for b in BENCHMARKS if b["name"] in wanted]
        unknown = wanted - {b["name"] for b in BENCHMARKS}
        if unknown:
            sys.exit(f"error: unknown benchmark name(s): {', '.join(sorted(unknown))}")

    if not args.no_build:
        ensure_built()

    if args.update_golden:
        update_golden(benchmarks, args.mode, args.timeout)
        return

    reps = args.reps if args.reps is not None else DEFAULT_REPS[args.mode]

    baseline = None
    if args.baseline:
        with open(args.baseline) as f:
            baseline = {r["name"]: r for r in json.load(f)}

    golden = load_golden(args.mode)
    results = [run_benchmark(b, args.mode, reps, args.timeout, golden) for b in benchmarks]
    print_report(results, baseline)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote results to {args.json}")

    if any(r["status"] != "PASS" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
