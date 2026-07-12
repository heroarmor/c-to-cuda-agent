#!/usr/bin/env python3
"""Tests for the bucket-B hybrid relay's pipeline half (DESIGN.md Phase 3):
scop_targets.json routing, the hybrid generate prompt, and _ppcg_partial's
subprocess contract (against a fake wrapper -- the real PPCG toolchain only
exists on a GPU box). Plain asserts, no test framework, same as
test_compiler_moves.py:

    python3 agent_pipeline/test_hybrid_routing.py
"""

import stat
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_pipeline as rp

FAKE_WRAPPER_OK = """#!/usr/bin/env python3
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("input")
ap.add_argument("-o", "--output")
args = ap.parse_args()
with open(args.output, "w") as f:
    f.write("/* fake ppcg partial */\\n")
print(args.output)
"""

FAKE_WRAPPER_REJECT = """#!/usr/bin/env python3
import sys
sys.stderr.write("ppcg_to_cu: ppcg failed (rc=1) -- fall through to the LLM backend\\n")
sys.exit(3)
"""


def test_scop_entry():
    entry = rp._scop_entry("multigrid.c")
    assert entry is not None and entry.get("mode") == "hybrid", entry
    assert "smooth_rb" in entry["fn"], entry
    entry = rp._scop_entry("rgf.c")
    assert entry is not None and entry.get("mode") == "hybrid", entry
    # bucket-A entries carry no mode; unknown files carry no entry
    assert rp._scop_entry("saxpy.c").get("mode") is None
    assert rp._scop_entry("no_such_file.c") is None


def test_wants_hybrid():
    hybrid_entry = {"fn": "mat_mul,mat_sub", "mode": "hybrid"}
    plain_entry = {"fn": "gemm"}
    # forced hybrid applies regardless of entry; auto only via a hybrid entry
    assert rp._wants_hybrid("hybrid", None)
    assert rp._wants_hybrid("hybrid", plain_entry)
    assert rp._wants_hybrid("auto", hybrid_entry)
    assert not rp._wants_hybrid("auto", plain_entry)
    assert not rp._wants_hybrid("auto", None)
    assert not rp._wants_hybrid("ppcg", hybrid_entry)
    assert not rp._wants_hybrid("llm", hybrid_entry)


def test_hybrid_generate_prompt():
    plain = rp.stage_prompt("generate", "rgf.c")
    hybrid = rp.stage_prompt("generate", "rgf.c", {
        "partial_file": "rgf_ppcg_partial.cu", "fns": "mat_mul,mat_sub"})
    assert hybrid.startswith(plain), "hybrid prompt must extend, not replace"
    assert "rgf_ppcg_partial.cu" in hybrid
    assert "mat_mul,mat_sub" in hybrid
    assert "resident" in hybrid  # the copy-hoisting instruction
    # fns may be absent (forced hybrid without a scop_targets entry)
    fallback = rp.stage_prompt("generate", "rgf.c", {
        "partial_file": "rgf_ppcg_partial.cu", "fns": None})
    assert "affine sub-kernels PPCG could extract" in fallback
    # other stages never see hybrid info
    assert rp.stage_prompt("verify", "rgf.c", None) == rp.stage_prompt("verify", "rgf.c")


def _install_fake_wrapper(tmp: Path, body: str) -> Path:
    wrapper = tmp / "fake_wrapper.py"
    wrapper.write_text(body, encoding="utf-8")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    return wrapper


def test_ppcg_partial():
    real_wrapper = rp.PPCG_WRAPPER
    try:
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            workdir = tmp / "work"
            workdir.mkdir()
            (workdir / "rgf.c").write_text("int main(){}\n", encoding="utf-8")

            rp.PPCG_WRAPPER = _install_fake_wrapper(tmp, FAKE_WRAPPER_OK)
            partial, reason = rp._ppcg_partial(workdir, "rgf.c", "rgf", 30)
            assert partial == "rgf_ppcg_partial.cu" and reason is None, (partial, reason)
            assert (workdir / partial).is_file()

            rp.PPCG_WRAPPER = _install_fake_wrapper(tmp, FAKE_WRAPPER_REJECT)
            partial, reason = rp._ppcg_partial(workdir, "rgf.c", "rgf", 30)
            assert partial is None and reason.startswith("ppcg:"), (partial, reason)
            assert "fall through" in reason

            rp.PPCG_WRAPPER = tmp / "does_not_exist.py"
            partial, reason = rp._ppcg_partial(workdir, "rgf.c", "rgf", 30)
            assert partial is None and "not found" in reason, (partial, reason)
    finally:
        rp.PPCG_WRAPPER = real_wrapper


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"\n{len(tests)} test(s) passed.")


if __name__ == "__main__":
    main()
