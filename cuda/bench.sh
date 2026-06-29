#!/usr/bin/env bash
# Dual-mode e2e benchmark driver for the C->CUDA suite.
#
#   cuda/bench.sh check [<tier>/<field>/<name> ...]   # small sizes, fast full CPU-vs-GPU verify (dev loop)
#   cuda/bench.sh full  [<tier>/<field>/<name> ...]   # >=10s-GPU sizes, full verify (heavy / overnight)
#
# With no explicit list it runs every benchmark. For each it reports the scaled
# args, CPU/GPU time, number of DISTINCT kernel kinds launched, total launch
# sites, and PASS/FAIL.  Verification is done by cuda/build_run.sh (compile both,
# run both, diff every numeric result field).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

MODE="${1:-check}"; shift || true
[ "$MODE" = check ] || [ "$MODE" = full ] || { echo "usage: $0 <check|full> [rel ...]" >&2; exit 2; }

# --- per-workload args -------------------------------------------------------
# CHECK: small & fast (full CPU+GPU verify in seconds).  FULL: aim ~10s GPU.
declare -A CHECK FULL
CHECK[easy/dense-linalg/saxpy]="4194304";              FULL[easy/dense-linalg/saxpy]="536870912"
CHECK[easy/montecarlo/mc_pi]="20000000 1024";          FULL[easy/montecarlo/mc_pi]="20000000000 4096"
CHECK[easy/nbody/nbody]="2048 5";                      FULL[easy/nbody/nbody]="16384 400"
CHECK[easy/ode/lorenz_ensemble]="20000 500";           FULL[easy/ode/lorenz_ensemble]="500000 6000"
CHECK[easy/pde/heat2d]="512 100";                      FULL[easy/pde/heat2d]="2048 20000"
CHECK[easy/rendering/mandelbrot]="1024 1024 200";      FULL[easy/rendering/mandelbrot]="8192 8192 4000"
# easy tier (single-kernel OK; no multi-kernel/10s requirement)
CHECK[easy/dense-linalg/gemm]="512";                   FULL[easy/dense-linalg/gemm]="4096"
CHECK[easy/sparse-linalg/spmv]="1024";                 FULL[easy/sparse-linalg/spmv]="4096"
CHECK[easy/tensor/tensor_contraction]="32";            FULL[easy/tensor/tensor_contraction]="96"
# moderate tier (genuine multi-kernel + ~10s)
CHECK[moderate/pde/wave2d]="512 100";                  FULL[moderate/pde/wave2d]="2048 20000"
CHECK[moderate/rendering/raytrace]="800 600";          FULL[moderate/rendering/raytrace]="12800 9600"
CHECK[moderate/solver/conjugate_gradient]="256 2000";  FULL[moderate/solver/conjugate_gradient]="2048 50000"
CHECK[complex/cfd/lbm]="128 128 500";                  FULL[complex/cfd/lbm]="512 512 20000"
CHECK[complex/eigen/jacobi_eigen]="128";               FULL[complex/eigen/jacobi_eigen]="1024"
CHECK[complex/factorization/cholesky]="512";           FULL[complex/factorization/cholesky]="6144"
CHECK[complex/factorization/lu]="512";                 FULL[complex/factorization/lu]="6144"
CHECK[complex/factorization/qr]="512";                 FULL[complex/factorization/qr]="3072"
CHECK[complex/lattice-qcd/dslash]="6";                 FULL[complex/lattice-qcd/dslash]="16"
CHECK[complex/manybody/lanczos]="16 120";              FULL[complex/manybody/lanczos]="20 600"
CHECK[complex/multigrid/multigrid]="9 12";             FULL[complex/multigrid/multigrid]="12 60"
CHECK[complex/physics/rgf]="256 32";                   FULL[complex/physics/rgf]="2048 96"
CHECK[complex/quantum/statevector]="20 6";             FULL[complex/quantum/statevector]="26 40"
CHECK[complex/rendering/pathtrace]="320 240 16";       FULL[complex/rendering/pathtrace]="1280 960 256"
CHECK[complex/sparse-linalg/spgemm]="512";             FULL[complex/sparse-linalg/spgemm]="8192"

# --- which benchmarks --------------------------------------------------------
if [ $# -gt 0 ]; then
  REQ=("$@")
else
  REQ=(); while IFS= read -r f; do rel="${f#benchmark/}"; REQ+=("${rel%.c}"); done < <(find benchmark -name '*.c' | sort)
fi

# how many DISTINCT kernel kinds does the .cu launch, and how many launch sites?
kkinds() { grep -oE '[A-Za-z_][A-Za-z0-9_]*[[:space:]]*<<<' "cuda/$1.cu" 2>/dev/null | sed 's/[[:space:]]*<<<//' | sort -u | wc -l; }
ksites() { grep -c '<<<' "cuda/$1.cu" 2>/dev/null || echo 0; }

# Optional machine-readable output: set BENCH_TSV=<path> to also write a TSV
# (rel<TAB>args<TAB>kinds<TAB>sites<TAB>cpu<TAB>gpu<TAB>verdict) for tooling.
TSV="${BENCH_TSV:-}"; [ -n "$TSV" ] && : > "$TSV"

[ "$MODE" = full ] && TMO=1800 || TMO=300
pass=0; fail=0; other=0
printf "%-40s %-16s %5s %5s %10s %10s %-8s\n" "BENCHMARK[$MODE]" "ARGS" "kind" "site" "CPU(s)" "GPU(s)" "VERDICT"
for rel in "${REQ[@]}"; do
  declare -n TBL=$([ "$MODE" = full ] && echo FULL || echo CHECK)
  args="${TBL[$rel]:-}"
  if [ ! -f "cuda/$rel.cu" ]; then
    printf "%-40s %-16s %5s %5s %10s %10s %-8s\n" "$rel" "-" "-" "-" "-" "-" "NO-CU"; other=$((other+1))
    [ -n "$TSV" ] && printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$rel" "-" "0" "0" "?" "?" "NO-CU" >> "$TSV"
    continue
  fi
  out="$(timeout "$TMO" ./cuda/build_run.sh "$rel" $args 2>&1)"; rc=$?
  cpu=$(grep -A1 'running CPU baseline'  <<<"$out" | grep -oE 'time=[0-9.]+' | head -1 | cut -d= -f2)
  gpu=$(grep -A1 'running GPU conversion' <<<"$out" | grep -oE 'time=[0-9.]+' | head -1 | cut -d= -f2)
  if   grep -q "RESULT: PASS" <<<"$out"; then v=PASS; pass=$((pass+1))
  elif grep -q "RESULT: FAIL" <<<"$out"; then v=FAIL; fail=$((fail+1))
  elif [ $rc -eq 124 ]; then v=TIMEOUT; other=$((other+1))
  else v=ERROR; other=$((other+1)); fi
  k="$(kkinds "$rel")"; s="$(ksites "$rel")"
  printf "%-40s %-16s %5s %5s %10s %10s %-8s\n" "$rel" "$args" "$k" "$s" "${cpu:-?}" "${gpu:-?}" "$v"
  [ -n "$TSV" ] && printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$rel" "$args" "$k" "$s" "${cpu:-?}" "${gpu:-?}" "$v" >> "$TSV"
done
echo "---- $MODE: PASS=$pass FAIL=$fail OTHER=$other / ${#REQ[@]} ----"
