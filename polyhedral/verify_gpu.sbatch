#!/bin/bash
# GPU verification of PPCG-generated conversions: compile + run each
# polyhedral/generated/<rel>.cu next to its benchmark/<rel>.c reference and
# golden-diff the outputs (timing/perf fields stripped, tolerance-aware --
# same convention as evaluation/: rtol 1e-3, atol 1e-2).
#
#   sbatch polyhedral/verify_gpu.sbatch [rel ...]     # default: the Phase 1 trio
#   tail -f polyhedral/verify-gpu.log
#
#SBATCH --job-name=ppcg-verify
#SBATCH --account=nbleier_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=polyhedral/verify-gpu.log
set -u

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
module load cuda/12.8.2
COMPARE="agent_pipeline/opencode_config/skill/cuda-verification-procedure/compare_outputs.py"
RELS=("$@")
[ ${#RELS[@]} -gt 0 ] || RELS=(easy/dense-linalg/saxpy easy/pde/heat2d easy/dense-linalg/gemm)

strip_timing() { sed -E 's/ *time=[0-9.eE+-]+ s//; s/ *\([0-9.eE+-]+ GFLOP\/s\)//'; }

nvidia-smi --query-gpu=name --format=csv,noheader | head -1
pass=0; fail=0
for rel in "${RELS[@]}"; do
    src="benchmark/$rel.c"; gen="polyhedral/generated/$rel.cu"
    work="$(mktemp -d)"
    if [ ! -f "$gen" ]; then echo "SKIP  $rel (no generated .cu)"; continue; fi
    gcc -std=c11 -O2 "$src" -o "$work/ref" -lm || { echo "FAIL  $rel (ref compile)"; fail=$((fail+1)); continue; }
    nvcc -O2 -arch=native "$gen" -o "$work/gpu" -lm || { echo "FAIL  $rel (nvcc)"; fail=$((fail+1)); continue; }
    (cd "$work" && ./ref > ref.txt 2>&1) || { echo "FAIL  $rel (ref run)"; fail=$((fail+1)); continue; }
    (cd "$work" && timeout 120 ./gpu > gpu.txt 2>&1) || { echo "FAIL  $rel (gpu run)"; sed 's/^/      /' "$work/gpu.txt" | head -5; fail=$((fail+1)); continue; }
    strip_timing < "$work/ref.txt" > "$work/ref.stripped"
    strip_timing < "$work/gpu.txt" > "$work/gpu.stripped"
    if python3 "$COMPARE" "$work/ref.stripped" "$work/gpu.stripped" --rel-tol 1e-3 --abs-tol 1e-2 > "$work/cmp.txt"; then
        echo "PASS  $rel"
        pass=$((pass+1))
    else
        echo "FAIL  $rel (output mismatch: $(cat "$work/cmp.txt"))"
        fail=$((fail+1))
    fi
    echo "      C:   $(cat "$work/ref.txt" | head -1)"
    echo "      GPU: $(cat "$work/gpu.txt" | head -1)"
    rm -rf "$work"
done
echo "== $pass passed, $fail failed =="
[ "$fail" -eq 0 ]
