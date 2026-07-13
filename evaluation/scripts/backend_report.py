#!/usr/bin/env python3
"""Per-backend report over agent-pipeline conversions.

Scans <dir>/*/pipeline_result.json (default: generated/) -- every run the
pipeline exports carries a `backend` tag (llm / ppcg / hybrid), an `optimizer`
mode, per-iteration `optimize_move` tags, the mechanical C-baseline timing,
and the best measured CUDA time -- and writes the per-backend rollup
polyhedral/DESIGN.md's "How it plugs into evaluation" calls for: correctness
rate, speedup geomean, codegen cost, and the compiler-vs-LLM optimize-move
mix, per backend. Complements run_eval.sh (which scores the tracked cuda/*.cu
reference conversions); this scores what the *pipeline* produced.

    python3 evaluation/scripts/backend_report.py [generated/] \
        [-o evaluation/results/backend_report.md]
"""

import argparse
import json
import math
import sys
from pathlib import Path

# a run is a successful conversion iff it ended on a loop-exit reason, not a
# stage failure (mirrors run_pipeline.HARD_ERROR_EXIT_REASONS plus the
# verify/generate "clean fail" reasons, which are also non-conversions)
CLEAN_EXITS = {"max_iterations_reached", "stagnated", "optimizer_exhausted"}


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def load_runs(root: Path):
    runs = []
    for pj in sorted(root.glob("*/pipeline_result.json")):
        try:
            runs.append(json.loads(pj.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: skipping {pj}: {exc}", file=sys.stderr)
    return runs


def summarize(run):
    best = run.get("best") or {}
    baseline = (run.get("baseline") or {}).get("time_sec")
    speedup = (baseline / best["time_sec"]) if baseline and best.get("time_sec") else None
    moves = {}
    accepted = 0
    for it in run.get("iterations", []):
        move = it.get("optimize_move")
        if move:
            moves[move] = moves.get(move, 0) + 1
        opt = it.get("optimize") or {}
        if str(opt.get("optimizer", "")).startswith("compiler") and opt.get("accepted"):
            accepted += 1
    return {
        "name": run.get("name", "?"),
        "backend": run.get("backend", "llm"),
        "optimizer": run.get("optimizer", "llm"),
        "exit": run.get("exit_reason", "?"),
        "ok": run.get("exit_reason") in CLEAN_EXITS,
        "iterations": len(run.get("iterations", [])),
        "speedup": speedup,
        "elapsed": run.get("elapsed_sec"),
        "moves": moves,
        "accepted_compiler_moves": accepted,
    }


def render(rows):
    o = ["# Per-backend pipeline report\n\n",
         "One row per exported conversion (`generated/*/pipeline_result.json`); "
         "speedup = mechanical C baseline / best measured CUDA time. A run "
         "counts as a conversion only on a clean loop exit.\n"]
    backends = sorted({r["backend"] for r in rows})
    o.append("\n## Rollup by backend\n\n")
    o.append("| backend | runs | converted | geomean speedup | mean codegen s "
             "| compiler moves accepted | optimize-move mix |\n")
    o.append("|---|--:|--:|--:|--:|--:|:--|\n")
    for b in backends:
        rs = [r for r in rows if r["backend"] == b]
        ok = [r for r in rs if r["ok"]]
        gm = geomean([r["speedup"] for r in ok])
        el = [r["elapsed"] for r in rs if r["elapsed"]]
        mix = {}
        for r in rs:
            for m, n in r["moves"].items():
                mix[m] = mix.get(m, 0) + n
        o.append(f"| {b} | {len(rs)} | {len(ok)} | "
                 f"{f'{gm:.2f}x' if gm else '-'} | "
                 f"{f'{sum(el)/len(el):.0f}' if el else '-'} | "
                 f"{sum(r['accepted_compiler_moves'] for r in rs)} | "
                 f"{', '.join(f'{m}:{n}' for m, n in sorted(mix.items())) or '-'} |\n")
    o.append("\n## Per program\n\n")
    o.append("| program | backend | optimizer | exit | iters | speedup | "
             "codegen s | moves |\n|---|---|---|---|--:|--:|--:|:--|\n")
    for r in sorted(rows, key=lambda r: (r["backend"], r["name"])):
        sp = f"{r['speedup']:.2f}x" if r["speedup"] else "-"
        o.append(f"| {r['name']} | {r['backend']} | {r['optimizer']} | "
                 f"{r['exit']} | {r['iterations']} | {sp} | "
                 f"{r['elapsed'] if r['elapsed'] else '-'} | "
                 f"{', '.join(f'{m}:{n}' for m, n in sorted(r['moves'].items())) or '-'} |\n")
    return "".join(o)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", nargs="?", default="generated",
                    help="directory of pipeline exports (default: generated/)")
    ap.add_argument("-o", "--output",
                    default="evaluation/results/backend_report.md")
    args = ap.parse_args()

    runs = load_runs(Path(args.dir))
    if not runs:
        sys.exit(f"no pipeline_result.json found under {args.dir}/")
    rows = [summarize(r) for r in runs]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(rows), encoding="utf-8")
    converted = sum(1 for r in rows if r["ok"])
    print(f"{len(rows)} run(s), {converted} converted -> {out}")


if __name__ == "__main__":
    main()
