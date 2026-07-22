#!/usr/bin/env python3
"""
compare_outputs.py - tolerance-aware diff for two program output files.

Used by the cuda-verify agent (see cuda-verification-procedure skill) to
objectively compare a CUDA translation's stdout against the original C
program's stdout, instead of an LLM eyeballing the two. Line structure
must match exactly; numbers found on each line are compared with
math.isclose so harmless floating-point reassociation (e.g. from a
different parallel reduction order on the GPU) doesn't register as a
mismatch.

Standalone and self-contained on purpose: agent_pipeline is meant to stay
decoupled from the rest of the repo, so this intentionally re-implements the
same line/number-tolerant comparison the suite's own drivers (cuda/build_run.sh,
evaluation/) use for golden-output diffing, rather than importing across that
boundary.

Usage:
    python3 compare_outputs.py <file_a> <file_b> [--rel-tol 1e-4] [--abs-tol 1e-6]

Exit code 0 and prints "MATCH" if the outputs agree within tolerance.
Exit code 1 and prints "MISMATCH: <reason>" otherwise.
"""

import argparse
import math
import re
import sys
from pathlib import Path

NUMBER_RE = re.compile(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?")


def _isclose(a: float, b: float, rel_tol: float, abs_tol: float) -> bool:
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def outputs_match(a: str, b: str, rel_tol: float, abs_tol: float):
    """Line-by-line, number-by-number comparison with float tolerance.

    Non-numeric text on each line must match exactly; numbers (ints or
    floats, found in the same order on each line) are compared with
    math.isclose-style tolerance so harmless floating-point noise (e.g.
    a different parallel reduction order on the GPU) doesn't look like a
    real mismatch.
    """
    lines_a = a.splitlines()
    lines_b = b.splitlines()

    if len(lines_a) != len(lines_b):
        return False, f"line count differs: {len(lines_a)} vs {len(lines_b)}"

    for lineno, (la, lb) in enumerate(zip(lines_a, lines_b), start=1):
        template_a = NUMBER_RE.sub("\0", la)
        template_b = NUMBER_RE.sub("\0", lb)
        if template_a != template_b:
            return False, f"line {lineno} differs: {la!r} vs {lb!r}"

        nums_a = NUMBER_RE.findall(la)
        nums_b = NUMBER_RE.findall(lb)
        for na, nb in zip(nums_a, nums_b):
            if not _isclose(float(na), float(nb), rel_tol, abs_tol):
                return False, f"line {lineno} value differs: {na} vs {nb}"

    return True, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file_a", type=Path)
    parser.add_argument("file_b", type=Path)
    parser.add_argument("--rel-tol", type=float, default=1e-4)
    parser.add_argument("--abs-tol", type=float, default=1e-6)
    args = parser.parse_args()

    for path in (args.file_a, args.file_b):
        if not path.exists():
            sys.exit(f"error: no such file: {path}")

    match, reason = outputs_match(
        args.file_a.read_text(), args.file_b.read_text(), args.rel_tol, args.abs_tol
    )

    if match:
        print("MATCH")
        sys.exit(0)

    print(f"MISMATCH: {reason}")
    sys.exit(1)


if __name__ == "__main__":
    main()
