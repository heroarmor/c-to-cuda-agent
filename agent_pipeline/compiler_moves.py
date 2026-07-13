"""Deterministic, orchestrator-side optimize moves -- polyhedral/DESIGN.md's
"Compiler backend in the *optimize* stage" (Phase 2.5).

Two mechanical moves can take an iteration's optimize slot instead of the
cuda-optimize agent (run_pipeline.py --optimizer compiler|hybrid):

  flags   nvcc flag search, backend-agnostic. A small profile-guided candidate
          set (at most MAX_FLAG_CANDIDATES per iteration) is compiled, run,
          diffed against baseline_output.txt and timed -- all mechanically,
          inside this one optimize slot (best-of-K, so a sweep never spreads
          across iterations and reads as stagnation to the patience stop). A
          winner must beat the iteration's profiled time by the acceptance
          margin; it is then persisted to nvcc_flags.txt, which the
          verify/profile skills include in every later nvcc invocation and
          which travels with best-snapshots and the exported .cu.
  retile  PPCG re-tiling, only for a .cu the ppcg backend produced. Re-runs
          polyhedral/ppcg_to_cu.py on the *original C source* (PPCG consumes
          C, not CUDA -- it can't post-optimize an arbitrary .cu) with a new
          --sizes string, then the same mechanical compile/run/diff/time
          acceptance gate. A losing candidate leaves the workdir's .cu
          untouched.

Both moves write the same optimize_result.json shape the cuda-optimize agent
writes (schemas.py validates it either way), so the loop, the patience stop,
and best-version tracking neither know nor care which optimizer produced an
iteration. Candidates that merely fail (PPCG reject, compile error, output
mismatch, no speedup) are recorded and skipped, never raised -- the mechanical
acceptance gate is what makes an aggressive candidate like -use_fast_math safe
to even try.

Planning (which move, which candidates, in what order) is pure and
unit-testable without a GPU: parse_ncu_summary / propose_flag_candidates /
propose_sizes_candidates / plan_move. Only run_flag_move / run_retile_move
touch the toolchain, and they find nvcc via PATH so tests can inject a fake.
"""

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

NVCC_FLAGS_FILE = "nvcc_flags.txt"
OPTIMIZE_RESULT_FILE = "optimize_result.json"

MAX_FLAG_CANDIDATES = 3   # bounded per DESIGN.md: small profile-guided set,
MAX_SIZES_CANDIDATES = 2  # not a grid sweep the patience stop would kill

COMPILE_TIMEOUT_SEC = 180
RUN_TIMEOUT_SEC = 30      # same no-args-with-timeout run convention as verify
TIME_REPS = 3             # candidate *ranking* only -- the next profile stage
                          # re-measures the winner authoritatively (reps=5)

# One knob per candidate group, mirroring the cuda-optimize agent's
# one-technique-per-iteration discipline. bottleneck/expect reuse the
# cuda-bottleneck-playbook vocabulary so a compiler-move optimize_result.json
# reads like an agent one.
FLAG_GROUPS = {
    "maxrregcount=64": {
        "flags": ("--maxrregcount=64",),
        "bottleneck": "register_pressure_or_low_occupancy",
        "expect": ["achieved occupancy up", "registers per thread down"],
    },
    "maxrregcount=32": {
        "flags": ("--maxrregcount=32",),
        "bottleneck": "register_pressure_or_low_occupancy",
        "expect": ["achieved occupancy up", "registers per thread down",
                   "possible local-memory spilling -- gated by measured time"],
    },
    "dlcm=cg": {
        "flags": ("-Xptxas", "-dlcm=cg"),
        "bottleneck": "global_memory_latency_scattered_access",
        "expect": ["fewer wasted bytes per transaction on uncoalesced loads"],
    },
    "use_fast_math": {
        "flags": ("-use_fast_math",),
        "bottleneck": "compute_throughput_transcendentals",
        "expect": ["sm_throughput_pct up", "kernel duration down"],
    },
    "dscm=cs": {
        "flags": ("-Xptxas", "-dscm=cs"),
        "bottleneck": "global_memory_store_streaming",
        "expect": ["less L2 pollution from streaming stores"],
    },
    "ptxas-O3": {
        "flags": ("-Xptxas", "-O3", "-Xptxas",
                  "--allow-expensive-optimizations=true"),
        "bottleneck": "instruction_pipeline_generic",
        "expect": ["issued IPC up"],
    },
}

# Retile candidates for the PPCG re-tiling move: each entry is a --sizes
# string, extra raw ppcg arguments, or both (option names verified against
# ppcg --help of the pinned toolchain, PPCG 0.09.3). {kernel[i]->...} applies
# to every kernel PPCG emits; shapes a kernel can't take (e.g. 2D block on a
# 1D nest) just make that candidate fail PPCG or the diff and get skipped
# mechanically.
SIZES_LADDER = {
    "small-block": {"sizes": "{kernel[i]->tile[16,16];kernel[i]->block[16,8]}"},
    "default-256": {"sizes": "{kernel[i]->tile[32,32];kernel[i]->block[32,8]}"},
    "big-tile": {"sizes": "{kernel[i]->tile[64,64];kernel[i]->block[16,16]}"},
    "wide-x": {"sizes": "{kernel[i]->tile[64,16];kernel[i]->block[64,4]}"},
    "tile-64": {"args": ["--tile-size=64"]},
    "no-shared": {"args": ["--no-shared-memory"]},
    "unroll-tile": {"args": ["--unroll-gpu-tile"]},
}

# A launch count at or above this (parsed from ncu_summary) marks the run as
# launch-overhead-shaped: the fix (wrap the launch loop in CUDA Graph capture)
# is a structural host-code rewrite no compiler flag can perform, so plan_move
# attaches a directed hint to the LLM optimize move instead (see plan_move).
LAUNCH_OVERHEAD_THRESHOLD = 1000

CUDA_GRAPHS_HINT = (
    "The profile shows a very large number of kernel launches with CPU-side "
    "launch overhead dominating wall-clock time. Strongly consider the "
    "CUDA_Graph_Capture technique from the cuda-bottleneck-playbook: capture "
    "the repetitive launch sequence into a CUDA Graph once "
    "(cudaStreamBeginCapture / cudaStreamEndCapture / cudaGraphInstantiate) "
    "and replay it with cudaGraphLaunch, keeping allocations and one-time "
    "setup outside the captured region.")

# Benchmarks print their own wall-clock (e.g. "time=0.123 s", gemm adds
# "(12.34 GFLOP/s)"), which legitimately differs between the C baseline and
# any CUDA candidate -- strip those fields before diffing, the same convention
# polyhedral/verify_gpu.sbatch and evaluation/ already use. Without this the
# mechanical gate rejects every candidate of a time=-printing benchmark.
_TIMING_FIELD_RE = re.compile(r" *time=[0-9.eE+-]+ s| *\([0-9.eE+-]+ GFLOP/s\)")

_MEMORY_STALLS = ("long_scoreboard", "lg_throttle", "mio_throttle")
_STALL_KEYWORDS = _MEMORY_STALLS + (
    "short_scoreboard", "barrier", "math_pipe_throttle", "not_selected")
_TRANSCENDENTAL_RE = re.compile(
    r"\b__?(?:expf?|logf?|sqrtf?|rsqrtf?|sinf?|cosf?|tanhf?|powf?|erff?)\s*\(")


def parse_ncu_summary(text: str | None) -> dict:
    """Best-effort scrape of the numbers the profiling skill asks the agent to
    put in ncu_summary (SM %, DRAM %, occupancy %, registers/thread, dominant
    stall). ncu_summary is free text by design, so every field is optional and
    a miss means "signal unavailable", never zero -- candidate proposal treats
    the two very differently (unavailable falls back to a generic ladder)."""
    low = (text or "").lower()

    def pct(*patterns):
        for pattern in patterns:
            m = re.search(pattern, low)
            if m:
                return float(m.group(1))
        return None

    regs = pct(r"(\d+)\s*(?:regs?|registers?)\s*(?:/|per)\s*thread",
               r"(?:regs?|registers?)\s*(?:/|per)\s*thread\D{0,10}?(\d+)")
    launch_counts = [int(m.replace(",", "")) for m in re.findall(
        r"([\d,]+)\s*(?:total\s+)?(?:kernel\s+)?(?:launches|instances)", low)]
    return {
        "sm_pct": pct(r"\bsm\b\D{0,30}?([\d.]+)\s*%"),
        "dram_pct": pct(r"\bdram\b\D{0,30}?([\d.]+)\s*%"),
        "occupancy_pct": pct(r"occupanc\w*\D{0,20}?([\d.]+)\s*%",
                             r"([\d.]+)\s*%\s*(?:achieved\s+)?occupanc"),
        "regs_per_thread": int(regs) if regs is not None else None,
        "stall": next((k for k in _STALL_KEYWORDS if k in low), None),
        "launches": max(launch_counts) if launch_counts else None,
    }


def headroom_tier(metrics: dict) -> str:
    """Same tiering as cuda-bottleneck-playbook: the primary limiter is the
    larger of SM/DRAM throughput %."""
    limiters = [v for v in (metrics.get("sm_pct"), metrics.get("dram_pct"))
                if v is not None]
    if not limiters:
        return "unknown"
    primary = max(limiters)
    if primary < 60:
        return "Tier-H"
    if primary <= 80:
        return "Tier-M"
    return "Tier-L"


def propose_flag_candidates(metrics: dict, cu_text: str,
                            tried: set, current_flags: list) -> list:
    """Ordered [(key, group)] of untried nvcc flag groups the measured (or
    absent) signals justify. An empty list is an honest answer -- e.g. a
    healthy Tier-L kernel with nothing register/memory/transcendental-shaped
    -- and makes plan_move fall through to retile/llm/none."""
    occupancy = metrics.get("occupancy_pct")
    regs = metrics.get("regs_per_thread")
    uses_transcendentals = bool(_TRANSCENDENTAL_RE.search(cu_text or ""))

    order = []
    if (regs is not None and regs >= 64) or \
            (occupancy is not None and occupancy < 50):
        order += ["maxrregcount=64", "maxrregcount=32"]
    if metrics.get("stall") in _MEMORY_STALLS:
        order += ["dlcm=cg", "dscm=cs"]
    if uses_transcendentals:
        order += ["use_fast_math"]
    if not order and all(metrics.get(k) is None for k in
                         ("sm_pct", "dram_pct", "occupancy_pct",
                          "regs_per_thread", "stall")):
        # No ncu signal at all (ncu_available false / unparseable summary):
        # fall back to the generic ladder rather than doing nothing.
        order = (["use_fast_math"] if uses_transcendentals else []) \
            + ["maxrregcount=64", "dlcm=cg", "ptxas-O3"]

    seen, candidates = set(), []
    for key in order:
        group = FLAG_GROUPS[key]
        if key in seen or key in tried:
            continue
        if all(f in current_flags for f in group["flags"]):
            continue  # already part of the accepted flag set
        seen.add(key)
        candidates.append((key, group))
    return candidates


def propose_sizes_candidates(metrics: dict, tried: set) -> list:
    """Ordered [(key, spec)] of untried PPCG retile candidates (spec carries a
    --sizes string, extra ppcg args, or both): smaller blocks first when
    occupancy is the complaint, bigger tiles first when the signal is
    memory-shaped, a neutral order otherwise; the option-flag candidates
    (uniform tile, no shared memory, tile unrolling) trail as diversity."""
    occupancy = metrics.get("occupancy_pct")
    dram = metrics.get("dram_pct")
    memory_bound = (metrics.get("stall") in _MEMORY_STALLS) or \
        (dram is not None and dram > 60)
    if occupancy is not None and occupancy < 50:
        order = ["small-block", "default-256", "wide-x", "big-tile",
                 "no-shared", "unroll-tile"]
    elif memory_bound:
        order = ["big-tile", "tile-64", "wide-x", "default-256",
                 "small-block", "unroll-tile"]
    else:
        order = ["default-256", "big-tile", "wide-x", "small-block",
                 "tile-64", "no-shared", "unroll-tile"]
    seen = set()
    return [(key, SIZES_LADDER[key]) for key in order
            if key not in tried and not (key in seen or seen.add(key))]


@dataclass
class MovePlan:
    move: str  # "library" | "flags" | "retile" | "llm" | "none"
    metrics: dict = field(default_factory=dict)
    tier: str = "unknown"
    flag_candidates: list = field(default_factory=list)
    sizes_candidates: list = field(default_factory=list)
    blas: dict | None = None   # scop_targets "blas" metadata for the library move
    hint: str | None = None    # directed instruction appended to an LLM move's prompt


def new_state() -> dict:
    """Per-run optimizer state the orchestrator threads through iterations:
    which candidates were already tried (accepted or not), and the currently
    accepted nvcc flag set (what nvcc_flags.txt holds)."""
    return {"tried_flags": set(), "tried_sizes": set(), "tried_library": False,
            "flags": []}


def plan_move(mode: str, backend: str, profile_result: dict, cu_text: str,
              state: dict, retile_available: bool = True,
              scop_entry: dict | None = None) -> MovePlan:
    """Decide this iteration's optimize move. Pure -- no toolchain, no I/O.

    Priority: library substitution first when the program's scop_targets
    entry declares a BLAS-shaped hot function (algorithmic tier -- the
    playbook's own priority order puts algorithm changes above hardware
    tuning), then the nvcc flag search (cheapest, backend-agnostic), then
    PPCG re-tiling (ppcg-produced .cu only -- a hybrid/llm .cu would lose
    its LLM work to a regeneration), then the LLM agent (hybrid mode) or a
    stop signal (compiler-only). When the parsed profile is launch-overhead
    shaped, an LLM move carries a directed CUDA-Graphs hint -- that fix is a
    structural host rewrite no compiler knob can make."""
    metrics = parse_ncu_summary(profile_result.get("ncu_summary"))
    plan = MovePlan(move="llm", metrics=metrics, tier=headroom_tier(metrics))
    launch_shaped = (metrics.get("launches") or 0) >= LAUNCH_OVERHEAD_THRESHOLD \
        or "launch overhead" in (profile_result.get("ncu_summary") or "").lower()
    if mode == "llm":
        return plan
    blas = (scop_entry or {}).get("blas")
    plan.flag_candidates = propose_flag_candidates(
        metrics, cu_text, state["tried_flags"], state["flags"]
    )[:MAX_FLAG_CANDIDATES]
    if backend == "ppcg" and retile_available:
        plan.sizes_candidates = propose_sizes_candidates(
            metrics, state["tried_sizes"])[:MAX_SIZES_CANDIDATES]
    if blas and backend == "ppcg" and not state["tried_library"]:
        # ppcg-only for the same reason as retile: the substitution
        # regenerates from the original C source.
        plan.move, plan.blas = "library", blas
    elif plan.flag_candidates:
        plan.move = "flags"
    elif plan.sizes_candidates:
        plan.move = "retile"
    else:
        plan.move = "llm" if mode == "hybrid" else "none"
    if plan.move == "llm" and launch_shaped:
        plan.hint = CUDA_GRAPHS_HINT
    return plan


def _detect_arch_flag() -> str | None:
    """-arch=sm_XX for the local GPU, or None. Tuner compiles only -- the
    accepted flag set written to nvcc_flags.txt never includes it, since the
    verify skill already handles arch mismatch itself."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        proc = subprocess.run(
            [smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, encoding="utf-8", timeout=10)
        cap = proc.stdout.strip().splitlines()[0].strip() if proc.stdout.strip() else ""
        if proc.returncode == 0 and re.fullmatch(r"\d+\.\d+", cap):
            return "-arch=sm_" + cap.replace(".", "")
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _last_line(text: str) -> str:
    lines = [l for l in (text or "").strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def _measure_candidate(workdir: Path, cu_filename: str, flags: list,
                       tag: str) -> tuple[float | None, str | None]:
    """Compile/run/diff/time one candidate mechanically. Returns
    (mean_sec, None) on success or (None, why_rejected). Reuses the exact
    conventions the verify/profile skills use: no-args run under a timeout,
    compare_outputs.py against baseline_output.txt, time_binary.py means --
    all three already present in the workdir."""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return None, "nvcc not on PATH"
    binary = f"{Path(cu_filename).stem}_{tag}"
    try:
        proc = subprocess.run(
            [nvcc, "-O2", *flags, cu_filename, "-o", binary, "-lm"],
            cwd=workdir, capture_output=True, text=True, encoding="utf-8",
            timeout=COMPILE_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        return None, "compile timed out"
    if proc.returncode != 0:
        return None, f"compile failed: {_last_line(proc.stderr)}"
    try:
        proc = subprocess.run(
            [f"./{binary}"], cwd=workdir, capture_output=True, text=True,
            encoding="utf-8", timeout=RUN_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        return None, f"run timed out after {RUN_TIMEOUT_SEC}s"
    if proc.returncode != 0:
        return None, f"run failed (rc={proc.returncode}): {_last_line(proc.stderr)}"
    candidate_output = f"{binary}_output.txt"
    (workdir / candidate_output).write_text(
        _TIMING_FIELD_RE.sub("", proc.stdout), encoding="utf-8")
    baseline_stripped = "baseline_output_stripped.txt"
    (workdir / baseline_stripped).write_text(
        _TIMING_FIELD_RE.sub(
            "", (workdir / "baseline_output.txt").read_text(encoding="utf-8")),
        encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "compare_outputs.py", baseline_stripped,
         candidate_output],
        cwd=workdir, capture_output=True, text=True, encoding="utf-8",
        timeout=60)
    if proc.returncode != 0:
        return None, f"output mismatch: {_last_line(proc.stdout)}"
    proc = subprocess.run(
        [sys.executable, "time_binary.py", f"./{binary}",
         "--reps", str(TIME_REPS)],
        cwd=workdir, capture_output=True, text=True, encoding="utf-8",
        timeout=(RUN_TIMEOUT_SEC + 5) * TIME_REPS)
    if proc.returncode != 0:
        return None, f"timing failed: {_last_line(proc.stderr)}"
    return json.loads(proc.stdout)["mean"], None


def _write_result(workdir: Path, technique: str, bottleneck: str, tier: str,
                  rationale: str, expect: list, move: str,
                  extra: dict | None = None) -> dict:
    """Same required shape as the cuda-optimize agent's optimize_result.json
    (schemas.py validates both paths identically); extra keys record what the
    mechanical search actually measured."""
    result = {
        "stubbed": False,
        "technique_applied": technique,
        "bottleneck_addressed": bottleneck,
        "headroom_tier": tier,
        "rationale": rationale,
        "expected_metric_change": expect,
        "optimizer": f"compiler:{move}",
    }
    result.update(extra or {})
    (workdir / OPTIMIZE_RESULT_FILE).write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def _describe(attempts: list) -> str:
    return "; ".join(
        f"{a['candidate']}: " + (f"{a['mean_sec']:.4f}s" if a["mean_sec"]
                                 is not None else a["rejected"])
        for a in attempts)


def write_noop_result(workdir: Path, plan: MovePlan, reason: str) -> dict:
    return _write_result(
        workdir, "Compiler_NoOp", "none_applicable", plan.tier, reason, [],
        "none")


def run_flag_move(workdir: Path, output_file: str, profile_result: dict,
                  plan: MovePlan, state: dict, min_delta: float) -> dict:
    """Best-of-K nvcc flag search. Accepts a candidate only if its measured
    mean beats this iteration's profiled time by min_delta -- the same bar the
    orchestrator's patience stop uses, so an accepted flag is by construction
    one the next profile stage should confirm as progress."""
    reference = profile_result["time_sec"]
    arch = _detect_arch_flag()
    attempts, best = [], None
    for n, (key, group) in enumerate(plan.flag_candidates):
        state["tried_flags"].add(key)
        candidate_flags = state["flags"] + [f for f in group["flags"]
                                            if f not in state["flags"]]
        mean, rejected = _measure_candidate(
            workdir, output_file,
            candidate_flags + ([arch] if arch else []), f"tune{n}")
        attempts.append({"candidate": key, "flags": candidate_flags,
                         "mean_sec": mean, "rejected": rejected})
        if mean is not None and (best is None or mean < best["mean_sec"]):
            best = {"key": key, "group": group, "flags": candidate_flags,
                    "mean_sec": mean}

    accepted = best is not None and \
        best["mean_sec"] < reference * (1 - min_delta)
    if accepted:
        state["flags"] = best["flags"]
        (workdir / NVCC_FLAGS_FILE).write_text(
            " ".join(state["flags"]) + "\n", encoding="utf-8")
        technique = f"Nvcc_Flag_Search({best['key']})"
        bottleneck = best["group"]["bottleneck"]
        expect = best["group"]["expect"]
    else:
        technique = "Nvcc_Flag_Search(no_candidate_accepted)"
        bottleneck = "none_accepted"
        expect = []
    rationale = (
        f"mechanical best-of-{len(attempts)} nvcc flag search against this "
        f"iteration's profiled {reference:.4f}s (acceptance margin "
        f"{min_delta:.0%}): {_describe(attempts)}")
    return _write_result(
        workdir, technique, bottleneck, plan.tier, rationale, expect, "flags",
        {"accepted": accepted, "candidates": attempts,
         "nvcc_flags": state["flags"]})


def run_library_move(workdir: Path, c_filename: str, output_file: str,
                     profile_result: dict, plan: MovePlan, state: dict,
                     blas_wrapper: Path, min_delta: float) -> dict:
    """Library-substitution move: regenerate the .cu from the original C
    source with the BLAS-shaped hot function replaced by a cuBLAS call
    (polyhedral/cublas_to_cu.py, driven by the entry's blas metadata), behind
    the same mechanical accept/reject gate as every other move. Algorithmic
    tier: when it applies, it usually dwarfs flag/tiling wins -- but the gate
    still decides (at small sizes the cuBLAS handle + copies can lose).
    On acceptance -lcublas joins the persisted nvcc flag set so the verify/
    profile skills' rebuilds keep linking."""
    reference = profile_result["time_sec"]
    state["tried_library"] = True
    arch = _detect_arch_flag()
    blas = plan.blas or {}
    candidate_cu = f"{Path(output_file).stem}_cublas.cu"
    technique = "CuBLAS_Substitution(no_candidate_accepted)"
    accepted, attempts = False, []
    cmd = [sys.executable, str(blas_wrapper), c_filename, "-o", candidate_cu,
           "--fn", blas.get("fn", ""), "--type", blas.get("type", "float"),
           "--dim", blas.get("dim", "n")]
    for param in ("A", "B", "C"):
        if blas.get(param):
            cmd += [f"--{param}", blas[param]]
    try:
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True,
                              text=True, encoding="utf-8",
                              timeout=COMPILE_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        proc = None
    if proc is None or proc.returncode != 0 \
            or not (workdir / candidate_cu).is_file():
        why = "cublas_to_cu timed out" if proc is None else \
            "cublas_to_cu failed: " + _last_line(proc.stderr or proc.stdout)
        attempts.append({"candidate": "cublas", "mean_sec": None,
                         "rejected": why})
    else:
        link_flags = state["flags"] + ["-lcublas"] + ([arch] if arch else [])
        mean, rejected = _measure_candidate(
            workdir, candidate_cu, link_flags, "cublas")
        attempts.append({"candidate": "cublas", "mean_sec": mean,
                         "rejected": rejected})
        accepted = mean is not None and mean < reference * (1 - min_delta)
        if accepted:
            shutil.copy(workdir / candidate_cu, workdir / output_file)
            if "-lcublas" not in state["flags"]:
                state["flags"] = state["flags"] + ["-lcublas"]
            (workdir / NVCC_FLAGS_FILE).write_text(
                " ".join(state["flags"]) + "\n", encoding="utf-8")
            technique = f"CuBLAS_Substitution({blas.get('fn')})"
    rationale = (
        f"mechanical cuBLAS substitution of {blas.get('fn')} (regenerated "
        f"from {c_filename}) against this iteration's profiled "
        f"{reference:.4f}s (acceptance margin {min_delta:.0%}): "
        f"{_describe(attempts)}")
    return _write_result(
        workdir, technique,
        "algorithmic_library_substitution" if accepted else "none_accepted",
        plan.tier, rationale,
        ["kernel duration down (vendor-tuned GEMM)"] if accepted else [],
        "library",
        {"accepted": accepted, "candidates": attempts,
         "nvcc_flags": state["flags"]})


def run_retile_move(workdir: Path, c_filename: str, output_file: str,
                    profile_result: dict, plan: MovePlan, state: dict,
                    ppcg_wrapper: Path, min_delta: float) -> dict:
    """Best-of-K PPCG re-tiling: regenerate from the original C source with a
    new --sizes, gate mechanically, and only overwrite the loop's .cu when a
    candidate actually wins -- a losing retile leaves everything untouched."""
    reference = profile_result["time_sec"]
    arch = _detect_arch_flag()
    flags = state["flags"] + ([arch] if arch else [])
    attempts, best = [], None
    for n, (key, spec) in enumerate(plan.sizes_candidates):
        state["tried_sizes"].add(key)
        candidate_cu = f"{Path(output_file).stem}_retile{n}.cu"
        cmd = [sys.executable, str(ppcg_wrapper), c_filename, "-o", candidate_cu]
        if spec.get("sizes"):
            cmd += ["--sizes", spec["sizes"]]
        for arg in spec.get("args", []):
            cmd += ["--ppcg-arg", arg]
        described = spec.get("sizes") or " ".join(spec.get("args", []))
        try:
            proc = subprocess.run(
                cmd, cwd=workdir, capture_output=True, text=True,
                encoding="utf-8", timeout=COMPILE_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            attempts.append({"candidate": key, "spec": described,
                             "mean_sec": None, "rejected": "ppcg timed out"})
            continue
        if proc.returncode != 0 or not (workdir / candidate_cu).is_file():
            attempts.append({
                "candidate": key, "spec": described, "mean_sec": None,
                "rejected": "ppcg failed: "
                            + _last_line(proc.stderr or proc.stdout)})
            continue
        mean, rejected = _measure_candidate(
            workdir, candidate_cu, flags, f"retile{n}")
        attempts.append({"candidate": key, "spec": described, "mean_sec": mean,
                         "rejected": rejected})
        if mean is not None and (best is None or mean < best["mean_sec"]):
            best = {"key": key, "spec": described, "cu": candidate_cu,
                    "mean_sec": mean}

    accepted = best is not None and \
        best["mean_sec"] < reference * (1 - min_delta)
    if accepted:
        shutil.copy(workdir / best["cu"], workdir / output_file)
        technique = f"PPCG_Retile({best['spec']})"
    else:
        technique = "PPCG_Retile(no_candidate_accepted)"
    rationale = (
        f"mechanical best-of-{len(attempts)} PPCG --sizes re-tiling "
        f"(regenerated from {c_filename}) against this iteration's profiled "
        f"{reference:.4f}s (acceptance margin {min_delta:.0%}): "
        f"{_describe(attempts)}")
    return _write_result(
        workdir, technique,
        "tiling_launch_shape" if accepted else "none_accepted",
        plan.tier, rationale,
        ["achieved occupancy toward the block-size sweet spot",
         "dram_throughput_pct up from tiled reuse"] if accepted else [],
        "retile",
        {"accepted": accepted, "candidates": attempts,
         "nvcc_flags": state["flags"]})
