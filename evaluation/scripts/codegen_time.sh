#!/usr/bin/env bash
# Metric: Code Generation Time (agent efficiency).
# Time how long opencode's /cudaify takes to (re)generate each conversion, and
# how many output tokens it costs. The existing cuda/<rel>.cu is snapshotted and
# RESTORED afterwards, so verified conversions are never clobbered.
#
#   evaluation/scripts/codegen_time.sh <tier>/<field>/<name> ...   (default: all)
#   CODEGEN_TIMEOUT=600  per-workload cap (s)
#
# Heavy + uses the network/model. Reports wall-clock + output tokens per workload.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
DB="$HOME/.local/share/opencode/opencode.db"
TMO="${CODEGEN_TIMEOUT:-600}"
OUT="${CODEGEN_TSV:-}"; [ -n "$OUT" ] && : > "$OUT"

command -v opencode >/dev/null || { echo "opencode not on PATH" >&2; exit 2; }

if [ $# -gt 0 ]; then REQ=("$@"); else
  REQ=(); while IFS= read -r f; do rel="${f#benchmark/}"; REQ+=("${rel%.c}"); done < <(find benchmark -name '*.c' | sort)
fi

tokens_since() {  # sum assistant output tokens created at/after $1 (ms epoch)
  sqlite3 "$DB" "SELECT data FROM message WHERE time_created>=$1;" 2>/dev/null | python3 -c '
import sys,json
t=0
for ln in sys.stdin:
    try: d=json.loads(ln)
    except: continue
    if d.get("role")=="assistant": t+=d.get("tokens",{}).get("output",0)
print(t)'
}

printf "%-40s %10s %10s %-8s\n" "WORKLOAD" "gen_s" "out_tok" "regen"
for rel in "${REQ[@]}"; do
  snap="$(mktemp)"; had=0
  [ -f "cuda/$rel.cu" ] && { cp "cuda/$rel.cu" "$snap"; had=1; }
  start_ms=$(date +%s%3N); t0=$(date +%s.%N)
  timeout "$TMO" opencode run "/cudaify $rel" > /dev/null 2>&1; rc=$?
  t1=$(date +%s.%N)
  wall=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}")
  toks=$(tokens_since "$start_ms")
  # report whether the regenerated file compiles+passes before we restore it
  regen="?"; [ -f "cuda/$rel.cu" ] && { ./cuda/build_run.sh "$rel" >/dev/null 2>&1 && regen=PASS || regen=FAIL; }
  [ $rc -eq 124 ] && regen=TIMEOUT
  if [ $had -eq 1 ]; then cp "$snap" "cuda/$rel.cu"; else git checkout -- "cuda/$rel.cu" 2>/dev/null || true; fi
  rm -f "$snap"
  printf "%-40s %10s %10s %-8s\n" "$rel" "$wall" "$toks" "$regen"
  [ -n "$OUT" ] && printf '%s\t%s\t%s\t%s\n' "$rel" "$wall" "$toks" "$regen" >> "$OUT"
done
exit 0