#!/usr/bin/env bash
# Top-level evaluation driver. Runs the suite once (via cuda/bench.sh) for
# Functional Correctness + Execution Time, collects LoC/Tidiness, and writes a
# KernelBench-style report (fast_p, speedups, universality) to evaluation/results/.
#
#   evaluation/run_eval.sh [check|full]        (default: check)
#
# Hardware Utilization and Code Generation Time are heavier / external and are
# collected separately (see evaluation/scripts/hw_util.sh and codegen_time.sh);
# their outputs can be appended to the report.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-check}"
RES="evaluation/results"; mkdir -p "$RES"
BTSV="$RES/bench_$MODE.tsv"; LTSV="$RES/loc.tsv"; REPORT="$RES/eval_$MODE.md"

echo "==> [1/3] correctness + execution time  (cuda/bench.sh $MODE)"
BENCH_TSV="$BTSV" ./cuda/bench.sh "$MODE" | tail -1
echo "==> [2/3] lines of code / tidiness"
LOC_TSV="$LTSV" ./evaluation/scripts/loc.sh > /dev/null
echo "==> [3/3] aggregating -> $REPORT"

python3 - "$MODE" "$BTSV" "$LTSV" "$REPORT" <<'PY'
import sys, math
mode, btsv, ltsv, report = sys.argv[1:5]

def f(x):
    try: return float(x)
    except: return None

bench={}
for ln in open(btsv):
    p=ln.rstrip("\n").split("\t")
    if len(p)<7: continue
    rel,args,kinds,sites,cpu,gpu,v=p[:7]
    cpu,gpu=f(cpu),f(gpu)
    sp=(cpu/gpu) if (cpu and gpu and gpu>0) else None
    bench[rel]=dict(args=args,kinds=int(kinds) if kinds.isdigit() else 0,
                    sites=int(sites) if sites.isdigit() else 0,cpu=cpu,gpu=gpu,sp=sp,v=v)
loc={}
for ln in open(ltsv):
    p=ln.rstrip("\n").split("\t")
    if len(p)<9: continue
    loc[p[0]]=dict(cloc=int(p[1]),culoc=int(p[2]),code=int(p[3]),com=int(p[4]),
                   pct=int(p[5]),ratio=float(p[6]),errk=int(p[7]),kinds=int(p[8]))

rels=sorted(bench)
N=len(rels)
def tier(r): return r.split("/")[0]
def dom(r):  return r.split("/")[1]

npass=sum(1 for r in rels if bench[r]["v"]=="PASS")
def fast(p): return sum(1 for r in rels if bench[r]["v"]=="PASS" and bench[r]["sp"] and bench[r]["sp"]>p)
sps=[bench[r]["sp"] for r in rels if bench[r]["v"]=="PASS" and bench[r]["sp"]]
geo=math.exp(sum(math.log(s) for s in sps)/len(sps)) if sps else 0

o=[]
o.append(f"# Evaluation report ({mode} mode)\n")
o.append("Metrics for the C→CUDA conversion suite, aligned with KernelBench "
         "(Ouyang et al., 2025, arXiv:2502.10517). Baseline = the single-thread "
         "`gcc -O2` C reference; conversions verified by `cuda/build_run.sh` "
         "(rtol 1e-3 / atol 1e-2 on every result field).\n")
o.append(f"**Suite:** {N} workloads · tiers "
         f"{', '.join(f'{t}={sum(1 for r in rels if tier(r)==t)}' for t in ['easy','moderate','complex'])}\n")

# 1. Functional Correctness (fast_0)
o.append("\n## 1. Functional Correctness\n")
o.append(f"`fast_0` (correctness rate) = **{npass}/{N} = {100*npass/N:.0f}%**. "
         "KernelBench checks 5 random inputs; our references are deterministic "
         "self-checking programs, so a single run with numeric-field comparison suffices.\n")
o.append("\n| tier | workloads | pass | rate |\n|---|--:|--:|--:|\n")
for t in ["easy","moderate","complex"]:
    rs=[r for r in rels if tier(r)==t]
    if rs: o.append(f"| {t} | {len(rs)} | {sum(1 for r in rs if bench[r]['v']=='PASS')} | {100*sum(1 for r in rs if bench[r]['v']=='PASS')/len(rs):.0f}% |\n")

# 2. Execution Time + fast_p
o.append("\n## 2. Execution Time (speedup vs CPU baseline)\n")
o.append(f"`fast_1` (correct & >1×) = **{fast(1)}/{N} = {100*fast(1)/N:.0f}%** · "
         f"`fast_2` = **{fast(2)}/{N}** · geomean speedup (correct) = **{geo:.1f}×**. "
         "`fast_p = (1/N) Σ \U0001d7d9(correct_i ∧ speedup_i > p)`; "
         "speedup = T_cpu / T_gpu from each program's internal `time=` region.\n")
o.append("\n| workload | args | kinds | CPU(s) | GPU(s) | speedup | verdict |\n|---|---|--:|--:|--:|--:|:--|\n")
for r in rels:
    b=bench[r]; sp=f"{b['sp']:.1f}×" if b['sp'] else "-"
    o.append(f"| {r} | `{b['args']}` | {b['kinds']} | {b['cpu'] if b['cpu'] is not None else '?'} | "
             f"{b['gpu'] if b['gpu'] is not None else '?'} | {sp} | {b['v']} |\n")

# 5. Scalability & Universality
o.append("\n## 5. Scalability & Universality\n")
doms=sorted(set(dom(r) for r in rels))
o.append(f"**Universality:** the agent converts **{len(doms)} domains** "
         f"({', '.join(doms)}) across 3 difficulty tiers — analogous to KernelBench's "
         f"Levels 1–3. Per-tier correctness above. **Scalability:** per-workload size "
         f"sweeps via `evaluation/scripts/scalability.sh` (e2e launch counts grow with the "
         f"iteration arg; `full` mode targets ~10s GPU).\n")

# 6. Code Tidiness & LoC
o.append("\n## 6. Code Tidiness & Lines of Code\n")
tot_c=sum(loc[r]['cloc'] for r in rels if r in loc)
tot_cu=sum(loc[r]['culoc'] for r in rels if r in loc)
mk=[loc[r]['kinds'] for r in rels if r in loc and tier(r)!='easy']
o.append(f"Total reference C = **{tot_c}** LoC → generated CUDA = **{tot_cu}** LoC "
         f"(×{tot_cu/tot_c:.2f}). Every conversion carries CUDA error-checking "
         f"(`CUDA_CHECK`/`cudaGetLastError`). Mean distinct kernel kinds in moderate+complex "
         f"= **{sum(mk)/len(mk):.1f}**.\n")
o.append("\n| workload | C LoC | CU LoC | × | comment% | err-checks | kinds |\n|---|--:|--:|--:|--:|--:|--:|\n")
for r in rels:
    if r not in loc: continue
    l=loc[r]
    o.append(f"| {r} | {l['cloc']} | {l['culoc']} | {l['ratio']:.2f} | {l['pct']}% | {l['errk']} | {l['kinds']} |\n")

# 3 & 4 pointers
o.append("\n## 3. Code Generation Time\n")
o.append("Collected separately (uses the model): `evaluation/scripts/codegen_time.sh` "
         "reports the `agent_pipeline/run_pipeline.py` wall-clock + output tokens per "
         "workload (output goes to generated/, so tracked cuda/*.cu is untouched).\n")
o.append("\n## 4. Hardware Utilization\n")
o.append("Collected separately: `evaluation/scripts/hw_util.sh <rel> <full-args>` samples "
         "nvidia-smi (avg/peak GPU util, peak mem, peak power). For achieved occupancy / "
         "DRAM & compute throughput, profile with Nsight Compute (`ncu --set full ...`).\n")

open(report,"w").write("".join(o))
print(f"correctness {npass}/{N}  fast_1={fast(1)}/{N}  geomean={geo:.1f}x  -> {report}")
PY
echo "==> done. report: $REPORT"