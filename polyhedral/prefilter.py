#!/usr/bin/env python3
"""SCoP pre-filter: cheap static triage before paying for a PPCG attempt.

Scans a benchmark C source for the PPCG disqualifiers listed in DESIGN.md
(indirect subscripts, data-dependent loop bounds, <complex.h>, recursion,
data-dependent while) plus a few other cheap ones (goto/setjmp, function
pointers, RNG in the compute path). Text-based on purpose -- zero
dependencies, runs anywhere; `pet` remains the authority (a program that
passes here but PPCG rejects just falls through to the LLM path).

Verdicts:
  scop-likely   no disqualifier found -> worth a PPCG attempt
  needs-review  soft flags (while/break/early-return/RNG) -> attempt PPCG,
                expect it may only cover part of the program
  reject        hard disqualifier -> route straight to the LLM backend

Usage:
  ./prefilter.py file.c [file2.c ...]      one JSON object per file
  ./prefilter.py --summary benchmark/      TSV verdict table for a tree
"""
import json
import os
import re
import sys

HARD = "reject"
SOFT = "needs-review"


def strip_comments_and_strings(src):
    """Remove comments and string/char literals (keeps line structure)."""
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif c == "/" and nxt == "*":
            j = src.find("*/", i + 2)
            seg = src[i : (n if j < 0 else j + 2)]
            out.append("\n" * seg.count("\n"))
            i = n if j < 0 else j + 2
        elif c in "\"'":
            q = c
            j = i + 1
            while j < n and src[j] != q:
                j += 2 if src[j] == "\\" else 1
            out.append(q + q)
            i = min(j + 1, n)
        else:
            out.append(c)
            i += 1
    return "".join(out)


def function_bodies(code):
    """Yield (name, body) for each function definition, by brace matching."""
    for m in re.finditer(r"\b([A-Za-z_]\w+)\s*\([^;{)]*\)\s*\{", code):
        name = m.group(1)
        if name in ("if", "for", "while", "switch", "sizeof"):
            continue
        depth, i = 1, m.end()
        while i < len(code) and depth:
            depth += {"{": 1, "}": -1}.get(code[i], 0)
            i += 1
        yield name, code[m.end() : i - 1]


def scan(path):
    with open(path, errors="replace") as f:
        raw = f.read()
    code = strip_comments_and_strings(raw)
    findings = []  # (severity, reason)

    if re.search(r"#\s*include\s*<complex\.h>", code):
        findings.append((HARD, "<complex.h> (cuDoubleComplex/thrust needed; C-hard)"))
    if re.search(r"\[[^][]*\[", code):
        findings.append((HARD, "indirect/nested subscript x[idx[k]] (gather/scatter)"))
    if re.search(r"\bgoto\b|\bsetjmp\b|\blongjmp\b", code):
        findings.append((HARD, "goto/setjmp control flow"))
    if re.search(r"\(\s*\*\s*\w+\s*\)\s*\(", code):
        findings.append((HARD, "function pointer call"))

    for name, body in function_bodies(code):
        if re.search(r"\b%s\s*\(" % re.escape(name), body):
            findings.append((HARD, "recursion: %s() calls itself" % name))

    # data-dependent loop bounds: an array element inside a for-header
    for m in re.finditer(r"\bfor\s*\(([^)]*)\)", code):
        if "[" in m.group(1):
            findings.append((HARD, "data-dependent for bound: for (%s)"
                             % " ".join(m.group(1).split())))
            break

    if re.search(r"\bwhile\s*\(", code) and not re.search(
            r"\bwhile\s*\(\s*(?:[A-Za-z_]\w*\s*(?:<|<=|>|>=|!=)\s*[A-Za-z_0-9]\w*|\d+)\s*\)", code):
        findings.append((SOFT, "while with non-counter condition (data-dependent?)"))
    if re.search(r"\bbreak\s*;", code):
        findings.append((SOFT, "break in a loop (early exit / convergence test?)"))
    if re.search(r"\b(rand|rand_r|drand48|random)\s*\(", code):
        findings.append((SOFT, "libc RNG call (RNG recurrence is C-easy territory)"))
    if re.search(r"\breturn\b[^;]*;", code) and "while" not in code:
        pass  # returns alone are fine; cholesky-style early return caught by review

    if any(s == HARD for s, _ in findings):
        verdict = "reject"
    elif findings:
        verdict = "needs-review"
    else:
        verdict = "scop-likely"
    return {"file": path, "verdict": verdict,
            "reasons": [r for _, r in findings]}


def main(argv):
    if argv and argv[0] == "--summary":
        root = argv[1] if len(argv) > 1 else "benchmark"
        files = sorted(
            os.path.join(d, f)
            for d, _, fs in os.walk(root) for f in fs if f.endswith(".c"))
        for path in files:
            r = scan(path)
            print("%s\t%s\t%s" % (r["verdict"], path, "; ".join(r["reasons"])))
        return 0
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    for path in argv:
        print(json.dumps(scan(path), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
