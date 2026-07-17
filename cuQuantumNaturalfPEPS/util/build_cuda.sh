#!/usr/bin/env bash
CUDA_ARCHITECTURES="75;80;90"
JOBS="${JOBS:-8}"
root="$(cd "$(dirname "$0")/.." && pwd)"
build_dir="${QNPEPS_CUDA_BUILD_DIR:-$root/cuda/build}"

if [ -d /p/home ] && [ "${QNPEPS_ACTIVE_ROOT:-}" != "$root" ]; then
    source "$root/activate.sh" || exit 1
elif ! command -v cmake >/dev/null 2>&1 || ! command -v nvcc >/dev/null 2>&1; then
    if [ -r "$root/activate.sh" ]; then
        source "$root/activate.sh" || exit 1
    fi
fi
if [ -d /p/home ] && [ "${QNPEPS_ACTIVE_ROOT:-}" != "$root" ]; then
    echo "[build_cuda.sh] cuQuantumNaturalfPEPS environment activation failed" >&2
    exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
        module load CMake >/dev/null 2>&1
    fi
fi
if ! command -v cmake >/dev/null 2>&1; then
    echo "[build_cuda.sh] cmake not found" >&2
    exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "[build_cuda.sh] nvcc not found (load CUDA toolkit before running this)" >&2
    exit 1
fi

printf 'CMake: %s\n' "$(cmake --version | head -n 1)"
printf 'CMake executable: %s\n' "$(readlink -f "$(command -v cmake)")"
echo "CUDA build architectures: $CUDA_ARCHITECTURES"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
if command -v nvidia-smi >/dev/null 2>&1; then
    if gpu_info="$(nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv,noheader 2>&1)"; then
        echo "NVIDIA GPUs"
        printf '%s\n%s\n' "index, name, compute capability, driver" "$gpu_info" \
            | column -t -s,
    elif gpu_info="$(nvidia-smi -L 2>&1)"; then
        echo "Nvidia GPUs:"
        printf '%s\n' "$gpu_info"
    else
        echo "NVIDIA GPUs: unavailable ($gpu_info)" >&2
    fi
else
    echo "NVIDIA GPUs: nvidia-smi not found"
fi

cmake -S "$root/cuda" -B "$build_dir" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" || exit 1
cmake --build "$build_dir" -j"$JOBS" || exit 1

so="$build_dir/qnpeps.so"
echo "$so"
if ! version="$(strings "$so" | grep -m1 -E '^cuQuantumNaturalfPEPS [0-9.]+ \([0-9]{4}-[0-9]{2}-[0-9]{2}\)$')"; then
    echo "[build_cuda.sh] Failed to find ABI version string in $so" >&2
    exit 1
fi
printf '%s\n' "$version"

if command -v cuobjdump >/dev/null 2>&1; then
    elf_arches="$(cuobjdump --list-elf "$so" 2>/dev/null \
        | sed -n 's/.*\.\(sm_[0-9][0-9]*\)\.cubin.*/\1/p' | sort -u | paste -sd, -)"
    ptx_arches="$(cuobjdump --list-ptx "$so" 2>/dev/null \
        | sed -n 's/.*\.\(sm_[0-9][0-9]*\)\.ptx.*/\1/p' | sort -u | paste -sd, -)"
    printf 'Native CUDA images: cubin=%s ptx=%s\n' \
        "${elf_arches:-<none>}" "${ptx_arches:-<none>}"
fi

echo "Resolved CUDA library dependencies:"
missing_cuda_library=0
while read -r name arrow path remainder; do
    case "$name" in
        libcudart.so.*|libcublas.so.*|libcublasLt.so.*|libcusolver.so.*|libcurand.so.*|libcusparse.so.*|libnvJitLink.so.*)
            if [ "$path" = "not" ]; then
                printf '  %s => not found\n' "$name" >&2
                missing_cuda_library=1
            else
                printf '  %s => %s\n' "$name" "$(readlink -f "$path")"
            fi
            ;;
    esac
done < <(ldd "$so")
if [ "$missing_cuda_library" -ne 0 ]; then
    echo "[build_cuda.sh] failed to resolve CUDA library dependencies" >&2
    exit 1
fi
