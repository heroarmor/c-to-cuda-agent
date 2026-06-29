#!/usr/bin/env bash
# Metric: Hardware Utilization.
# Compile the GPU conversion, run it while polling nvidia-smi, and report
# average/peak GPU utilisation, peak memory, and peak power for the run.
#
#   evaluation/scripts/hw_util.sh <tier>/<field>/<name> [program-args...]
#
# Coarse, device-wide sampling (WSL nvidia-smi). For per-kernel achieved
# occupancy / DRAM & compute throughput install Nsight Compute and use:
#   ncu --set full --target-processes all ./cuda/bin/<rel>.gpu <args>
# Best run at full-mode sizes (>=~10s) so the sampler captures the kernels.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
[ $# -ge 1 ] || { echo "usage: $0 <tier>/<field>/<name> [args...]" >&2; exit 2; }
REL="$1"; shift; ARGS=("$@")

# shellcheck disable=SC1091
source cuda/env.sh
BIN="cuda/bin/$REL.gpu"
mkdir -p "$(dirname "$BIN")"
echo "==> compiling $REL.cu (sm_86)"
# shellcheck disable=SC2086
$NVCC $NVCC_FLAGS "cuda/$REL.cu" -o "$BIN" $LDLIBS

SAMP="$(mktemp)"
echo "==> running + sampling nvidia-smi"
"$BIN" "${ARGS[@]}" > /dev/null 2>&1 &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw \
             --format=csv,noheader,nounits 2>/dev/null >> "$SAMP"
done
wait "$pid"

awk -F',' '
  { g=$1+0; mu=$2+0; mem=$3+0; p=$4+0; n++;
    gsum+=g; if(g>gmax)gmax=g; musum+=mu; if(mem>memmax)memmax=mem; if(p>pmax)pmax=p }
  END{ if(n==0){print "    (no samples captured -- run too short; use full-mode args)"; exit}
       printf "    samples=%d  GPU-util avg=%.0f%% peak=%.0f%%  mem-bw-util avg=%.0f%%  mem-peak=%.0f MiB  power-peak=%.0f W\n",
              n, gsum/n, gmax, musum/n, memmax, pmax }
' "$SAMP"
rm -f "$SAMP"