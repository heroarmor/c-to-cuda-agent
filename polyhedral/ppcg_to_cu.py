#!/usr/bin/env python3
"""PPCG -> single-.cu wrapper: the polyhedral backend's `generate` stage.

Takes one benchmark C file, runs PPCG's CUDA codegen on it, and merges
PPCG's three-file output (<stem>_host.cu, <stem>_kernel.hu, <stem>_kernel.cu)
into ONE self-contained .cu -- the shape the rest of this repo expects
(cuda/<rel>.cu conversions, agent_pipeline verify stage). Host-side I/O and
the result-printing line survive untouched, so the golden diff against the C
reference still matches.

SCoP selection, three modes (first match wins):
  --fn NAME[,NAME...]   insert `#pragma scop`/`endscop` around the body of the
                        named hot function(s); only those are GPU-ified.
                        Preferred: init/checksum loops stay on the host.
  scop_targets.json     when --fn is absent, the input's basename is looked up
                        in polyhedral/scop_targets.json (Phase 0's per-program
                        hot-kernel knowledge, kept out of the generic tools).
  --autodetect / none   `--pet-autodetect`: pet finds maximal SCoPs anywhere.
                        Zero-config, but init loops get offloaded too.

Because nvcc compiles .cu as C++ while the benchmarks are C11, a small
compatibility prelude is prepended (restrict keyword, implicit malloc casts).

Usage:
  ./ppcg_to_cu.py benchmark/easy/dense-linalg/saxpy.c --fn saxpy
  ./ppcg_to_cu.py input.c --fn step -o out.cu --sizes '{kernel[i]->block[16,16]}'
Output defaults to polyhedral/generated/<rel>.cu (mirrors benchmark/ layout).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_PPCG = os.path.join(HERE, "toolchain", "bin", "ppcg")

CXX_PRELUDE = """\
/* C->C++ compatibility shim: nvcc compiles .cu as C++, the source is C11. */
#include <stdlib.h>
#define restrict __restrict__
#ifdef __cplusplus
struct ppcg_voidp { void *p; template <class T> operator T *() const { return (T *)p; } };
static inline ppcg_voidp ppcg_malloc(size_t n) { ppcg_voidp r; r.p = malloc(n); return r; }
static inline ppcg_voidp ppcg_calloc(size_t n, size_t s) { ppcg_voidp r; r.p = calloc(n, s); return r; }
static inline ppcg_voidp ppcg_realloc(void *q, size_t n) { ppcg_voidp r; r.p = realloc(q, n); return r; }
#define malloc(n) ppcg_malloc(n)
#define calloc(n, s) ppcg_calloc(n, s)
#define realloc(p, n) ppcg_realloc(p, n)
#endif
"""


def die(msg, code=1):
    print("ppcg_to_cu: %s" % msg, file=sys.stderr)
    sys.exit(code)


def mark_functions(src, names):
    """Wrap the body statements of each named function in #pragma scop/endscop."""
    for name in names:
        m = re.search(
            r"^[ \t]*(?:static\s+)?(?:inline\s+)?[A-Za-z_][\w \t\*]*\b%s\s*\([^;{)]*\)\s*\{"
            % re.escape(name), src, re.M)
        if not m:
            die("--fn %s: no function definition found" % name)
        open_brace = m.end() - 1
        depth, i = 1, m.end()
        while i < len(src) and depth:
            depth += {"{": 1, "}": -1}.get(src[i], 0)
            i += 1
        close_brace = i - 1
        body = src[open_brace + 1 : close_brace]
        src = (src[: open_brace + 1]
               + "\n#pragma scop\n" + body + "\n#pragma endscop\n"
               + src[close_brace:])
    return src


def merge(stem, tmp):
    """host.cu + kernel.hu + kernel.cu -> one translation unit."""
    def slurp(suffix):
        with open(os.path.join(tmp, stem + suffix)) as f:
            return f.read()

    host = slurp("_host.cu")
    hu = slurp("_kernel.hu")
    ker = slurp("_kernel.cu")
    # PPCG mirrors a `const T *` source parameter into `const T *dev_x;`, but
    # then hands dev_x to cudaMemcpy/cudaFree (void * params) -- fine in C,
    # a hard const-conversion error in C++. Drop const on device-side decls.
    host = re.sub(r"\bconst\s+(\w[\w ]*\*+\s*dev_)", r"\1", host)
    inc = re.compile(r'^[ \t]*#[ \t]*include[ \t]*"%s_kernel\.hu"[ \t]*\n'
                     % re.escape(stem), re.M)
    ker = inc.sub("", ker)
    merged, n = inc.subn(hu.rstrip() + "\n\n" + ker.rstrip() + "\n", host, count=1)
    if not n:
        # host never included the header (e.g. no kernel emitted); append.
        merged = host + "\n" + hu + "\n" + ker
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", help="benchmark C source")
    ap.add_argument("-o", "--output", help="output .cu (default polyhedral/generated/<rel>.cu)")
    ap.add_argument("--fn", help="comma-separated hot function(s) to wrap in #pragma scop")
    ap.add_argument("--autodetect", action="store_true",
                    help="force pet autodetect even if scop_targets.json has an entry")
    ap.add_argument("--ppcg", default=os.environ.get("PPCG", DEFAULT_PPCG))
    ap.add_argument("--sizes", help="PPCG --sizes string (tile/block/grid)")
    ap.add_argument("--ppcg-arg", action="append", default=[],
                    help="extra raw ppcg argument (repeatable)")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        die("no such file: %s" % args.input)
    if not os.access(args.ppcg, os.X_OK):
        die("ppcg not found at %s (run polyhedral/build_ppcg.sh, or set $PPCG)" % args.ppcg)

    out = args.output
    if not out:
        rel = os.path.relpath(os.path.abspath(args.input), REPO)
        parts = rel.split(os.sep)
        rel = os.path.join(*parts[1:]) if parts[0] == "benchmark" else os.path.basename(rel)
        out = os.path.join(HERE, "generated", os.path.splitext(rel)[0] + ".cu")

    with open(args.input) as f:
        src = f.read()

    if not args.fn and not args.autodetect:
        targets_path = os.path.join(HERE, "scop_targets.json")
        if os.path.isfile(targets_path):
            with open(targets_path) as f:
                entry = json.load(f).get(os.path.basename(args.input))
            if entry:
                args.fn = entry["fn"]
                if entry.get("sizes") and not args.sizes:
                    args.sizes = entry["sizes"]

    stem = os.path.splitext(os.path.basename(args.input))[0]
    tmp = tempfile.mkdtemp(prefix="ppcg_%s_" % stem)
    try:
        cmd = [args.ppcg, "--target=cuda"]
        if args.fn:
            src = mark_functions(src, [s.strip() for s in args.fn.split(",") if s.strip()])
        else:
            cmd.append("--pet-autodetect")
        if args.sizes:
            cmd.append("--sizes=%s" % args.sizes)
        cmd += args.ppcg_arg
        marked = os.path.join(tmp, stem + ".c")
        with open(marked, "w") as f:
            f.write(src)
        cmd.append(stem + ".c")

        env = os.environ.copy()
        # ppcg links the LLVM module's shared libclang-cpp; build_ppcg.sh
        # records that libdir in toolchain/ldpath so no module load is needed.
        ldpath = os.path.join(os.path.dirname(os.path.dirname(args.ppcg)), "ldpath")
        if os.path.isfile(ldpath):
            with open(ldpath) as f:
                extra = f.read().strip()
            env["LD_LIBRARY_PATH"] = extra + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        proc = subprocess.run(cmd, cwd=tmp, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, universal_newlines=True,
                              timeout=args.timeout, env=env)
        host_path = os.path.join(tmp, stem + "_host.cu")
        if proc.returncode or not os.path.isfile(host_path):
            sys.stderr.write(proc.stdout or "")
            die("ppcg failed (rc=%d) -- fall through to the LLM backend" % proc.returncode, 3)

        merged = CXX_PRELUDE + "\n" + merge(stem, tmp)
        header = ("/* Generated by the polyhedral backend (PPCG) from %s.\n"
                  " * Mode: %s. Merged host+kernel translation unit; do not hand-edit\n"
                  " * the kernel launch bounds here -- regenerate or let the optimize\n"
                  " * loop iterate on it. */\n"
                  % (os.path.relpath(os.path.abspath(args.input), REPO),
                     ("#pragma scop around: " + args.fn) if args.fn else "pet autodetect"))
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            f.write(header + merged)
        print(out)
    finally:
        if args.keep_tmp:
            print("kept tmp: %s" % tmp, file=sys.stderr)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
