# Source this to set up the CUDA toolchain for the C->CUDA workflow.
#
# The toolchain already exists inside the 'faiss' conda env (CUDA 12.4 + nvcc),
# completed with the cudart dev/static headers+libs. Override any var below by
# exporting it before sourcing, e.g.  CUDA_ENV=mycuda  CUDA_ARCH=sm_89.

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CUDA_ENV="${CUDA_ENV:-faiss}"
export CUDA_ARCH="${CUDA_ARCH:-sm_86}"     # RTX 3070 Laptop = Ampere = sm_86

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CUDA_ENV"

export NVCC="${NVCC:-$(command -v nvcc)}"
export NVCC_FLAGS="${NVCC_FLAGS:--O3 -arch=$CUDA_ARCH -L$CONDA_PREFIX/lib}"
export CC="${CC:-cc}"
export CC_FLAGS="${CC_FLAGS:--O2 -std=c11 -Wall -Wextra}"
export LDLIBS="${LDLIBS:--lm}"

if [ -z "$NVCC" ]; then
    echo "env.sh: nvcc not found in conda env '$CUDA_ENV'" >&2
fi
