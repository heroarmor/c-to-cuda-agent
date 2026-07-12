#!/usr/bin/env bash
# Metric: Agent pipeline time (agent efficiency).
# Time how long zyj's agent pipeline (agent_pipeline/run_pipeline.py) takes to
# produce a CUDA conversion for each workload, and how many output tokens it
# costs. run_pipeline.py runs the full generate->verify->profile->optimize loop
# and writes its output under generated/<name>/ -- it never touches the tracked
# cuda/<rel>.cu files, so verified conversions can't be clobbered and there is
# no snapshot/restore to do (unlike the old /cudaify-based version).
#
#   evaluation/scripts/codegen_time.sh <tier>/<field>/<name> ...   (default: all)
#   CODEGEN_TIMEOUT=1200  per-workload cap (s)
#
# Heavy: uses the model (via opencode under run_pipeline.py) and a CUDA toolchain
# (nvcc/ncu). Reports wall-clock + output tokens + pipeline outcome per workload.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
DB="$HOME/.local/share/opencode/opencode.db"
TMO="${CODEGEN_TIMEOUT:-1200}"
OUT="${CODEGEN_TSV:-}"; [ -n "$OUT" ] && : > "$OUT"

command -v python3 >/dev/null || { echo "python3 not on PATH" >&2; exit 2; }
[ -f agent_pipeline/run_pipeline.py ] || { echo "agent_pipeline/run_pipeline.py not found" >&2; exit 2; }

if [ $# -gt 0 ]; then REQ=("$@"); else
  REQ=(); while IFS= read -r f; do rel="${f#benchmark/}"; REQ+=("${rel%.c}"); done < <(find benchmark -name '*.c' | sort)
fi

tokens_since() {  # sum assistant output tokens created at/after $1 (ms epoch),
                  # across every opencode stage run_pipeline.py drove for this workload
  sqlite3 "$DB" "SELECT data FROM message WHERE time_created>=$1;" 2>/dev/null | python3 -c '
import sys,json
t=0
for ln in sys.stdin:
    try: d=json.loads(ln)
    except: continue
    if d.get("role")=="assistant": t+=d.get("tokens",{}).get("output",0)
print(t)'
}

printf "%-40s %10s %10s %-8s\n" "WORKLOAD" "pipe_s" "out_tok" "outcome"
for rel in "${REQ[@]}"; do
  src="benchmark/$rel.c"
  [ -f "$src" ] || { printf "%-40s %10s %10s %-8s\n" "$rel" "-" "-" "NOSRC"; continue; }
  start_ms=$(date +%s%3N); t0=$(date +%s.%N)
  timeout "$TMO" python3 agent_pipeline/run_pipeline.py "$src" > /dev/null 2>&1; rc=$?
  t1=$(date +%s.%N)
  wall=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}")
  toks=$(tokens_since "$start_ms")
  # run_pipeline.py exits 0 when the pipeline completed without a hard error,
  # 1 otherwise; 124 is the shell `timeout` cap being hit.
  case $rc in 0) outcome=PASS;; 124) outcome=TIMEOUT;; *) outcome=FAIL;; esac
  printf "%-40s %10s %10s %-8s\n" "$rel" "$wall" "$toks" "$outcome"
  [ -n "$OUT" ] && printf '%s\t%s\t%s\t%s\n' "$rel" "$wall" "$toks" "$outcome" >> "$OUT"
done
exit 0
