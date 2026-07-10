#!/usr/bin/env bash
ARCH="${1:-${ARCH:-80}}"
JOBS="${JOBS:-8}"
root="$(cd "$(dirname "$0")" && pwd)"

if ! command -v cmake >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
        module load CMake >/dev/null 2>&1
    fi
fi
if ! command -v cmake >/dev/null 2>&1; then
    echo "setup_cuda.sh: cmake not found (install CMake or run: module load CMake)" >&2
    exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "setup_cuda.sh: nvcc not found (load the CUDA toolkit, e.g. source setup.sh)" >&2
    exit 1
fi

cmake -S "$root/cuda" -B "$root/cuda/build" -DCMAKE_CUDA_ARCHITECTURES="$ARCH" || exit 1
cmake --build "$root/cuda/build" -j"$JOBS" || exit 1

so="$root/cuda/build/libpeps_sampler.so"
echo "$so"
strings "$so" | grep cuQuantumNaturalfPEPS
