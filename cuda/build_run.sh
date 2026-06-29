#!/usr/bin/env bash
# Build + verify one C->CUDA conversion against its CPU reference baseline.
#
#   cuda/build_run.sh <tier>/<field>/<name> [program-args...]
#
# Example:
#   cuda/build_run.sh easy/dense-linalg/saxpy
#   cuda/build_run.sh moderate/dense-linalg/gemm 1024
#
# Compiles benchmark/<path>.c (CPU baseline) and cuda/<path>.cu (GPU conversion),
# runs both with the same args, and compares their result lines. Programs print
# different result tokens (checksum=, total_heat=, pi=, trace=, energy=, ...), so
# rather than look for one token we strip the fields that legitimately differ --
# the `time=...` and any `(...)` perf-rate / annotation parentheticals -- then
# compare every remaining number with a combined absolute+relative tolerance.
#
# Tolerances (override via env):
#   RTOL  relative tolerance for large values   (default 1e-3)
#   ATOL  absolute tolerance for near-zero ones  (default 1e-2)  e.g. drift/residual
# A field passes if |a-b| <= ATOL OR rel_err <= RTOL.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
RTOL="${RTOL:-1e-3}"
ATOL="${ATOL:-1e-2}"

# shellcheck disable=SC1091
source "$HERE/env.sh"

[ $# -ge 1 ] || { echo "usage: $0 <tier>/<field>/<name> [args...]" >&2; exit 2; }
REL="$1"; shift
ARGS=("$@")

SRC_C="$ROOT/benchmark/$REL.c"
SRC_CU="$HERE/$REL.cu"
BIN_C="$HERE/bin/$REL.cpu"
BIN_CU="$HERE/bin/$REL.gpu"

[ -f "$SRC_C" ]  || { echo "missing CPU reference: $SRC_C" >&2; exit 1; }
[ -f "$SRC_CU" ] || { echo "missing CUDA conversion: $SRC_CU  (run /cudaify first)" >&2; exit 1; }

mkdir -p "$(dirname "$BIN_C")"

echo "==> compiling CPU baseline   ($CC)"
$CC $CC_FLAGS "$SRC_C" -o "$BIN_C" $LDLIBS

echo "==> compiling CUDA conversion ($CUDA_ARCH)"
$NVCC $NVCC_FLAGS "$SRC_CU" -o "$BIN_CU"

echo "==> running CPU baseline"
OUT_C="$("$BIN_C" "${ARGS[@]}")";  echo "    $OUT_C"
echo "==> running GPU conversion"
OUT_CU="$("$BIN_CU" "${ARGS[@]}")"; echo "    $OUT_CU"

# Drop timing fields and any (parenthetical) perf-rate / annotation, then pull
# out every numeric token that remains.  Timing fields are any "<label>=<num> s"
# duration token (time=, D-apply=, ...) -- they legitimately differ CPU vs GPU.
strip_perf() { sed -E 's/[A-Za-z][A-Za-z0-9_.-]*=[-+]?[0-9.]+([eE][-+]?[0-9]+)?[[:space:]]+s\b//g; s/time=[^[:space:]]*[[:space:]]*s?//g; s/\([^)]*\)//g' <<<"$1"; }
nums() { grep -oE '[-+]?[0-9]+\.?[0-9]*([eE][-+]?[0-9]+)?' <<<"$1" || true; }

mapfile -t A < <(nums "$(strip_perf "$OUT_C")")
mapfile -t B < <(nums "$(strip_perf "$OUT_CU")")

if [ "${#A[@]}" -eq 0 ] || [ "${#A[@]}" -ne "${#B[@]}" ]; then
    echo "==> compared ${#A[@]} cpu fields vs ${#B[@]} gpu fields"
    echo "RESULT: FAIL  (result-line shape differs -- GPU output must match the baseline's)"
    exit 1
fi

paste -d' ' <(printf '%s\n' "${A[@]}") <(printf '%s\n' "${B[@]}") \
  | awk -v at="$ATOL" -v rt="$RTOL" '
    { a=$1; b=$2; d=a-b; if(d<0)d=-d; den=(a<0?-a:a); rel=(den>0?d/den:d);
      if (!(d<=at || rel<=rt)) {
        printf "    field %d mismatch: cpu=%s gpu=%s rel_err=%.3e\n", NR-1, a, b, rel; fail=1 } }
    END { printf "==> compared %d numeric fields  (rtol=%s atol=%s)\n", NR, rt, at;
          if (fail) { print "RESULT: FAIL"; exit 1 } else { print "RESULT: PASS" } }'
