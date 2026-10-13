#!/bin/bash

set -euo pipefail
cd "$(dirname "$0")"

CUDA_ARCH="${CUDA_ARCH:-86}"
BUILD_DIR="${BUILD_DIR:-build}"
mkdir -p "$BUILD_DIR"

if [[ -n "${NVCC:-}" ]]; then
    nvcc="$NVCC"
elif command -v nvcc >/dev/null 2>&1; then
    nvcc="$(command -v nvcc)"
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    nvcc=/usr/local/cuda/bin/nvcc
else
    printf 'Error: nvcc not found; install the CUDA Toolkit or set NVCC.\n' >&2
    exit 127
fi

arch_flags=()
# A space-separated list builds native device code for multiple GPU architectures.
for cuda_arch in $CUDA_ARCH; do
    arch_flags+=(-gencode "arch=compute_${cuda_arch},code=sm_${cuda_arch}")
done

"$nvcc" -O3 -lineinfo \
    "${arch_flags[@]}" \
    benchmark.cu -o "$BUILD_DIR/coalesced-submatrix-load" -ldl
