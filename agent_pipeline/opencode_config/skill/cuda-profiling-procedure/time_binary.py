#!/usr/bin/env python3
"""
time_binary.py - wall-clock time a compiled binary over several reps.

Used by the cuda-profile agent (see cuda-profiling-procedure skill) to get
end-to-end timing for both the original C binary and the generated CUDA
binary, instead of asking the LLM to compute min/mean/median by hand.

Usage:
    python3 time_binary.py <binary> [args...] [--reps 5] [--timeout 30]

Prints {"min": ..., "mean": ..., "median": ...} (seconds) as JSON to stdout.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("binary", type=Path)
    parser.add_argument("args", nargs="*", help="arguments to pass to the binary")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    binary = args.binary.resolve()
    if not binary.exists():
        sys.exit(f"error: no such file: {binary}")

    times = []
    for _ in range(args.reps):
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [str(binary), *args.args],
                capture_output=True,
                timeout=args.timeout,
            )
        except subprocess.TimeoutExpired:
            sys.exit(f"error: {args.binary} timed out after {args.timeout}s")
        elapsed = time.perf_counter() - start
        if proc.returncode != 0:
            sys.exit(f"error: {args.binary} exited {proc.returncode}")
        times.append(elapsed)

    print(json.dumps({
        "min": min(times),
        "mean": statistics.mean(times),
        "median": statistics.median(times),
    }))


if __name__ == "__main__":
    main()
