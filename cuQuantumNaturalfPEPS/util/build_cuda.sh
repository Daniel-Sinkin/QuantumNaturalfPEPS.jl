#!/usr/bin/env bash
ARCH="${1:-${ARCH:-80}}"
JOBS="${JOBS:-8}"
root="$(cd "$(dirname "$0")/.." && pwd)"
build_dir="${QNPEPS_CUDA_BUILD_DIR:-$root/cuda/build}"

if ! command -v cmake >/dev/null 2>&1 || ! command -v nvcc >/dev/null 2>&1; then
    if [ -r "$root/activate.sh" ]; then
        source "$root/activate.sh" || exit 1
    fi
fi
if ! command -v cmake >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
        module load CMake >/dev/null 2>&1
    fi
fi
if ! command -v cmake >/dev/null 2>&1; then
    echo "build_cuda.sh: cmake not found (install CMake or run: module load CMake)" >&2
    exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "build_cuda.sh: nvcc not found (install or load a CUDA toolkit)" >&2
    exit 1
fi

echo "CUDA build architectures: $ARCH"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
if command -v nvidia-smi >/dev/null 2>&1; then
    if gpu_info="$(nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv,noheader 2>&1)"; then
        echo "NVIDIA GPUs (index, name, compute capability, driver):"
        printf '%s\n' "$gpu_info"
    elif gpu_info="$(nvidia-smi -L 2>&1)"; then
        echo "NVIDIA GPUs:"
        printf '%s\n' "$gpu_info"
    else
        echo "NVIDIA GPUs: unavailable ($gpu_info)" >&2
    fi
else
    echo "NVIDIA GPUs: nvidia-smi not found"
fi

cmake -S "$root/cuda" -B "$build_dir" -DCMAKE_CUDA_ARCHITECTURES="$ARCH" || exit 1
cmake --build "$build_dir" -j"$JOBS" || exit 1

so="$build_dir/libpeps_sampler.so"
echo "$so"
if ! version="$(strings "$so" | grep -m1 -E '^cuQuantumNaturalfPEPS [0-9.]+ \([0-9]{4}-[0-9]{2}-[0-9]{2}\)$')"; then
    echo "build_cuda.sh: ABI version string not found in $so" >&2
    exit 1
fi
printf '%s\n' "$version"
