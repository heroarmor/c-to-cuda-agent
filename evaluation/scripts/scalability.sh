#!/usr/bin/env bash
# Metric: Scalability (one workload across growing problem sizes).
# Runs the workload at each given arg-set and reports CPU/GPU time, speedup and
# correctness, so you can see how the conversion holds up as the problem grows
# (the "Universality" half -- breadth across tiers/domains -- is summarised by
# run_eval.sh across the whole suite).
#
#   evaluation/scripts/scalability.sh <tier>/<field>/<name> "args1" "args2" ...
#   e.g. scalability.sh complex/cfd/lbm "128 128 500" "256 256 2000" "512 512 8000"
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
[ $# -ge 2 ] || { echo "usage: $0 <rel> \"args1\" \"args2\" ..." >&2; exit 2; }
REL="$1"; shift

printf "%-26s %10s %10s %9s %-7s\n" "ARGS" "CPU(s)" "GPU(s)" "speedup" "VERDICT"
for a in "$@"; do
  out="$(./cuda/build_run.sh "$REL" $a 2>&1)"
  cpu=$(grep -A1 'running CPU baseline'  <<<"$out" | grep -oE 'time=[0-9.]+' | head -1 | cut -d= -f2)
  gpu=$(grep -A1 'running GPU conversion' <<<"$out" | grep -oE 'time=[0-9.]+' | head -1 | cut -d= -f2)
  v=FAIL; grep -q "RESULT: PASS" <<<"$out" && v=PASS
  sp="-"; [ -n "${cpu:-}" ] && [ -n "${gpu:-}" ] && awk "BEGIN{exit !($gpu>0)}" && sp=$(awk "BEGIN{printf \"%.1fx\", $cpu/$gpu}")
  printf "%-26s %10s %10s %9s %-7s\n" "$a" "${cpu:-?}" "${gpu:-?}" "$sp" "$v"
done
exit 0