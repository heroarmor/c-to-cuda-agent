#!/usr/bin/env python3
"""Large-size timing pass over pipeline exports (the Scalability metric for
what the *pipeline* produced -- scalability.sh covers the tracked cuda/*.cu).

The suite's default sizes sit below the CUDA fixed-cost floor (~80 ms of
driver init alone on this cluster), so every conversion "loses" to the C
baseline regardless of kernel quality. This pass recompiles each
generated/<name>/<name>.cu (with its exported nvcc_flags.txt) plus its C
reference, runs both at a LARGE argv (table below), tolerance-diffs the
outputs (timing fields stripped), and reports the speedup where the GPU can
actually show up. Run under a 1-GPU Slurm job:

    sbatch ... --wrap "module load cuda/12.8.2 && \
        python3 evaluation/scripts/scale_exports.py"

Writes evaluation/results/scale_report.md.
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COMPARE = REPO / ("agent_pipeline/opencode_config/skill/"
                  "cuda-verification-procedure/compare_outputs.py")
TIME_BINARY = REPO / ("agent_pipeline/opencode_config/skill/"
                      "cuda-profiling-procedure/time_binary.py")
TIMING_RE = re.compile(r" *time=[0-9.eE+-]+ s| *\([0-9.eE+-]+ GFLOP/s\)")

# Large argv per program (from each benchmark's own argv handling); absent
# programs are skipped with a note. Sized for seconds-not-minutes C runtimes.
LARGE_ARGS = {
    "saxpy": ["134217728"],            # 1<<27 elements (default 1<<24)
    "gemm": ["2048"],                  # 64x the default FLOPs
    "reduction": ["134217728"],
    "heat2d": ["2048", "400"],         # grid side 512->2048, steps 200->400
    "sobel": ["8192", "8192"],
    "mandelbrot": ["4096", "4096"],
    "nbody": ["16384", "10"],          # all-pairs O(N^2)
    "mc_pi": ["1000000000", "4096"],
    "lorenz_ensemble": ["200000", "2000"],
    "spmv": ["4096"],                  # 16M-row grid Laplacian
    "tensor_contraction": ["96"],
    "conv2d_relu": [],                 # no argv sizing in source
    "mlp_two_layer": [],
    "multigrid": ["11", "12"],         # (2^11+1)^2 grid
    "rgf": ["512", "64"],              # 512 blocks of 64x64
}

RUN_TIMEOUT = 600
REPS = 3


def sh(cmd, cwd, timeout=RUN_TIMEOUT):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", timeout=timeout)


def detect_arch():
    try:
        proc = sh(["nvidia-smi", "--query-gpu=compute_cap",
                   "--format=csv,noheader"], cwd=".", timeout=10)
        cap = proc.stdout.strip().splitlines()[0].strip()
        if proc.returncode == 0 and re.fullmatch(r"\d+\.\d+", cap):
            return "-arch=sm_" + cap.replace(".", "")
    except Exception:
        pass
    return None


def time_mean(work, binary, args):
    proc = sh([sys.executable, str(TIME_BINARY), f"./{binary}", *args,
               "--reps", str(REPS), "--timeout", str(RUN_TIMEOUT)],
              cwd=work, timeout=(RUN_TIMEOUT + 10) * REPS)
    if proc.returncode != 0:
        return None, proc.stderr.strip().splitlines()[-1:]
    return json.loads(proc.stdout)["mean"], None


def run_one(name, export_dir, arch):
    pj = json.loads((export_dir / "pipeline_result.json").read_text())
    src = Path(pj["source"])
    args = LARGE_ARGS.get(name)
    if not args:
        return {"name": name, "skip": "no large-args entry"}
    if not src.is_file():
        return {"name": name, "skip": f"source missing: {src}"}
    cu = export_dir / f"{name}.cu"
    if not cu.is_file():
        return {"name": name, "skip": "no exported .cu"}
    flags_file = export_dir / "nvcc_flags.txt"
    flags = flags_file.read_text().split() if flags_file.is_file() else []

    work = Path(tempfile.mkdtemp(prefix=f"scale_{name}_"))
    row = {"name": name, "backend": pj.get("backend"), "args": " ".join(args)}
    try:
        if sh(["gcc", "-std=c11", "-O2", str(src), "-o", "ref", "-lm"],
              cwd=work).returncode != 0:
            return {**row, "skip": "ref compile failed"}
        nvcc = ["nvcc", "-O2", *flags] + ([arch] if arch else []) \
            + [str(cu), "-o", "gpu", "-lm"]
        proc = sh(nvcc, cwd=work)
        if proc.returncode != 0:
            return {**row, "skip": "nvcc: "
                    + proc.stderr.strip().splitlines()[-1][:80]}
        ref = sh(["./ref", *args], cwd=work)
        gpu = sh(["./gpu", *args], cwd=work)
        if ref.returncode != 0 or gpu.returncode != 0:
            return {**row, "skip": f"run rc: ref={ref.returncode} "
                                   f"gpu={gpu.returncode}"}
        (work / "r.txt").write_text(TIMING_RE.sub("", ref.stdout))
        (work / "g.txt").write_text(TIMING_RE.sub("", gpu.stdout))
        ok = sh([sys.executable, str(COMPARE), "r.txt", "g.txt",
                 "--rel-tol", "1e-3", "--abs-tol", "1e-2"],
                cwd=work).returncode == 0
        row["match"] = ok
        if not ok:
            return {**row, "skip": "output mismatch at large size"}
        row["cpu"], err = time_mean(work, "ref", args)
        row["gpu"], err2 = time_mean(work, "gpu", args)
        if row["cpu"] and row["gpu"]:
            row["speedup"] = row["cpu"] / row["gpu"]
        return row
    except subprocess.TimeoutExpired:
        return {**row, "skip": f"timed out (> {RUN_TIMEOUT}s)"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    gen = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "generated"
    out = REPO / "evaluation/results/scale_report.md"
    arch = detect_arch()
    rows = []
    for export_dir in sorted(gen.iterdir()):
        if (export_dir / "pipeline_result.json").is_file():
            row = run_one(export_dir.name, export_dir, arch)
            rows.append(row)
            print(f"{row['name']}: "
                  + (row.get("skip") or f"{row.get('speedup', 0):.2f}x "
                     f"(C {row['cpu']:.2f}s / GPU {row['gpu']:.2f}s)"),
                  flush=True)
    o = ["# Large-size timing of pipeline exports\n\n",
         "Same exported `.cu` + `nvcc_flags.txt`, larger argv (clear of the "
         "~80 ms CUDA driver-init floor). Outputs tolerance-diffed at the "
         "large size before timing.\n\n",
         "| program | backend | args | correct | C (s) | GPU (s) | speedup |\n",
         "|---|---|---|:--|--:|--:|--:|\n"]
    for r in rows:
        if r.get("skip"):
            o.append(f"| {r['name']} | {r.get('backend', '-')} | "
                     f"{r.get('args', '-')} | skipped: {r['skip']} | - | - | - |\n")
        else:
            o.append(f"| {r['name']} | {r['backend']} | {r['args']} | "
                     f"{'PASS' if r['match'] else 'FAIL'} | {r['cpu']:.2f} | "
                     f"{r['gpu']:.2f} | {r['speedup']:.2f}x |\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(o), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
