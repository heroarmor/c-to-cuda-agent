#!/usr/bin/env python3
"""Tests for ppcg_to_cu.py's source transforms -- delinearize (flat pointer ->
2D VLA so pet can extract the SCoP) and reflatten (2D VLA -> flat pointer so
nvcc's C++ mode accepts the merged .cu). Plain asserts, no framework:

    python3 polyhedral/test_ppcg_to_cu.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ppcg_to_cu as w

GEMM = """\
static void gemm(int n, const float *A, const float *B, float *C) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < n; ++k)
                acc += A[i * n + k] * B[k * n + j];
            C[i * n + j] = acc;
        }
}
"""

STENCIL = """\
static void smooth(int n, double *u, const double *f) {
    for (int i = 1; i < n - 1; ++i)
        for (int j = 1; j < n - 1; ++j)
            u[i * n + j] = f[i * n + j] + u[(i - 1) * n + j] + u[i * n + j - 1];
}
"""


def test_delinearize_gemm():
    out = w.delinearize(GEMM, "gemm", {"A": "n", "B": "n", "C": "n"})
    assert "const float (*A)[n]" in out and "float (*C)[n]" in out, out
    assert "A[i][k]" in out and "B[k][j]" in out and "C[i][j]" in out, out
    assert "*A" not in out.replace("(*A)", ""), out  # no flat decl left


def test_delinearize_stencil_offsets():
    out = w.delinearize(STENCIL, "smooth", {"u": "n", "f": "n"})
    # parenthesized row exprs and +/- column offsets must both rewrite
    assert "u[(i - 1)][j]" in out, out
    assert "u[i][j - 1]" in out, out
    assert "f[i][j]" in out, out


def test_delinearize_skips_absent_param():
    # arrays dict is shared across all marked fns -- a param not in this
    # function's signature is skipped, not an error
    out = w.delinearize(GEMM, "gemm", {"A": "n", "uf": "nf"})
    assert "const float (*A)[n]" in out, out


def test_reflatten_roundtrip():
    delin = w.delinearize(GEMM, "gemm", {"A": "n", "B": "n", "C": "n"})
    flat = w.reflatten(delin, {"A": "n", "B": "n", "C": "n"})
    assert "(*A)[n]" not in flat and "(*C)[n]" not in flat, flat
    assert "const float *A" in flat and "float *C" in flat, flat
    assert "A[(i) * (n) + (k)]" in flat and "C[(i) * (n) + (j)]" in flat, flat


def test_find_function_with_vla_params():
    delin = w.delinearize(GEMM, "gemm", {"A": "n"})
    m, ob, cb = w._find_function(delin, "gemm")
    assert delin[ob] == "{" and delin[cb] == "}", (ob, cb)


def test_cublas_substitution():
    import cublas_to_cu as cb
    out = cb.substitute(GEMM, "gemm", "float", "n", "A", "B", "C")
    # signature intact, body replaced by the swapped-operand Sgemm call
    assert "static void gemm(int n, const float *A" in out, out
    assert "cublasSgemm" in out and "dev_b, n, dev_a, n" in out, out
    assert "acc += A[i * n + k]" not in out, out
    out_d = cb.substitute(GEMM.replace("float", "double"), "gemm",
                          "double", "n", "A", "B", "C")
    assert "cublasDgemm" in out_d, out_d


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"\n{len(tests)} test(s) passed.")


if __name__ == "__main__":
    main()
