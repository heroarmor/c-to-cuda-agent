#!/usr/bin/env bash
# Metric: Code Tidiness & Lines of Code.
# For every converted workload, count code / comment / blank lines in the
# generated cuda/<rel>.cu and the reference benchmark/<rel>.c, plus tidiness
# signals: comment density, presence of CUDA error checking, and #kernel kinds.
#
#   evaluation/scripts/loc.sh [<tier>/<field>/<name> ...]   (default: all)
#   BENCH_TSV-style TSV to stdout; set LOC_TSV=<path> to also save it.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# classify a C/CUDA file -> "code comment blank" line counts (handles // and /* */)
classify() {
  awk '
    BEGIN{code=0;com=0;blank=0;inblk=0}
    {
      line=$0; sub(/^[ \t]+/,"",line); sub(/[ \t]+$/,"",line)
      if (inblk){ com++; if (line ~ /\*\//) inblk=0; next }
      if (line==""){ blank++; next }
      if (line ~ /^\/\//){ com++; next }
      if (line ~ /^\/\*/){ com++; if (line !~ /\*\//) inblk=1; next }
      code++
    }
    END{printf "%d %d %d", code, com, blank}
  ' "$1"
}

OUT="${LOC_TSV:-}"
[ -n "$OUT" ] && : > "$OUT"
printf "%-40s %5s %5s %5s %5s %6s %6s %5s %6s\n" \
  "WORKLOAD" "c_LOC" "cu_LOC" "code" "cmt" "cmt%%" "ratio" "errk" "kinds"

if [ $# -gt 0 ]; then REQ=("$@"); else
  REQ=(); while IFS= read -r f; do rel="${f#benchmark/}"; REQ+=("${rel%.c}"); done < <(find benchmark -name '*.c' | sort)
fi

for rel in "${REQ[@]}"; do
  c="benchmark/$rel.c"; cu="cuda/$rel.cu"
  [ -f "$cu" ] || continue
  cloc=$(wc -l < "$c"); culoc=$(wc -l < "$cu")
  read -r code com blank < <(classify "$cu")
  tot=$((code+com)); pct=0; [ $tot -gt 0 ] && pct=$(awk "BEGIN{printf \"%.0f\", 100*$com/$tot}")
  ratio=$(awk "BEGIN{printf \"%.2f\", $culoc/$cloc}")
  errk=$(grep -c 'CUDA_CHECK\|cudaGetLastError' "$cu")          # error-checking call sites
  kinds=$(grep -oE '[A-Za-z_][A-Za-z0-9_]*[[:space:]]*<<<' "$cu" | sed 's/[[:space:]]*<<<//' | sort -u | wc -l)
  printf "%-40s %5s %5s %5s %5s %5s%% %6s %5s %6s\n" "$rel" "$cloc" "$culoc" "$code" "$com" "$pct" "$ratio" "$errk" "$kinds"
  [ -n "$OUT" ] && printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$rel" "$cloc" "$culoc" "$code" "$com" "$pct" "$ratio" "$errk" "$kinds" >> "$OUT"
done
exit 0