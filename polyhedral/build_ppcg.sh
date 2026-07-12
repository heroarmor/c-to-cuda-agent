#!/usr/bin/env bash
# Build the polyhedral toolchain (libyaml + PPCG w/ bundled isl+pet) into
# polyhedral/toolchain/ (gitignored). Designed for UMich Great Lakes login
# nodes: uses the llvm/14.0.6 module for clang libs, system gcc/gmp/autotools.
#
# Usage:  ./polyhedral/build_ppcg.sh          # full build (~10-20 min)
#         ./polyhedral/build_ppcg.sh --check  # just verify an existing install
#
# Re-runnable: skips straight to the smoke test if bin/ppcg already exists.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PREFIX="$HERE/toolchain"
SRC="$PREFIX/src"
JOBS="${JOBS:-8}"
PPCG_TAG="${PPCG_TAG:-ppcg-0.09.3}"
LIBYAML_URL="https://pyyaml.org/download/libyaml/yaml-0.2.5.tar.gz"
LLVM_MODULE="${LLVM_MODULE:-llvm/14.0.6}"

log() { printf '\n== %s ==\n' "$*"; }

# clang libs come from the LLVM module (full static libclang* + headers)
if command -v module >/dev/null 2>&1 || [ -n "${LMOD_CMD:-}" ]; then
    set +u; source /etc/profile.d/lmod.sh 2>/dev/null || true
    module load "$LLVM_MODULE"; set -u
fi
command -v llvm-config >/dev/null || { echo "llvm-config not found (module $LLVM_MODULE not loaded?)" >&2; exit 1; }
CLANG_PREFIX="$(llvm-config --prefix)"
log "using clang from $CLANG_PREFIX ($(llvm-config --version))"

smoke() {
    log "smoke test"
    "$PREFIX/bin/ppcg" --version
    tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' RETURN
    cat > "$tmp/smoke.c" <<'EOF'
void saxpy(int n, float a, float *x, float *y) {
    for (int i = 0; i < n; i++)
        y[i] = a * x[i] + y[i];
}
EOF
    (cd "$tmp" && "$PREFIX/bin/ppcg" --target=cuda smoke.c && ls smoke_host.cu smoke_kernel.cu smoke_kernel.hu)
    grep -q '__global__' "$tmp/smoke_kernel.cu"
    log "OK: ppcg emits CUDA for a trivial SCoP"
}

if [ "${1:-}" = "--check" ] || [ -x "$PREFIX/bin/ppcg" ]; then
    smoke; exit 0
fi

mkdir -p "$SRC"

if [ ! -f "$PREFIX/lib/libyaml.a" ]; then
    log "building libyaml (static, PIC) for pet"
    cd "$SRC"
    curl -fsSL "$LIBYAML_URL" | tar xz
    cd yaml-0.2.5
    ./configure --prefix="$PREFIX" --disable-shared CFLAGS=-fPIC
    make -j"$JOBS" && make install
fi

if [ ! -d "$SRC/ppcg" ]; then
    log "cloning PPCG $PPCG_TAG (+ isl/pet submodules)"
    cd "$SRC"
    git clone --branch "$PPCG_TAG" --depth 1 --recurse-submodules \
        --shallow-submodules https://repo.or.cz/ppcg.git
fi

log "configuring + building PPCG"
cd "$SRC/ppcg"
[ -x configure ] || ./autogen.sh
./configure --prefix="$PREFIX" \
    --with-clang-prefix="$CLANG_PREFIX" \
    CPPFLAGS="-I$PREFIX/include" LDFLAGS="-L$PREFIX/lib"
make -j"$JOBS"
make install

# ppcg dynamically links the LLVM module's libclang-cpp; record its libdir so
# ppcg_to_cu.py can inject LD_LIBRARY_PATH without needing the module loaded.
llvm-config --libdir > "$PREFIX/ldpath"

smoke
log "installed to $PREFIX"
