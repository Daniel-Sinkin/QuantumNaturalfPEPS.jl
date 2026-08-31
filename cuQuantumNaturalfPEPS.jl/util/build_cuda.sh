#!/usr/bin/env bash

show_paths=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --show-path)
            show_paths=1
            ;;
        *)
            echo "[build_cuda.sh] unknown option: $1" >&2
            exit 2
            ;;
    esac
    shift
done

JOBS="${JOBS:-8}"
root="$(cd "$(dirname "$0")/.." && pwd)"
build_dir="$root/build/cuda"

jsc_platform=""
systemname_file="${QNPEPS_SYSTEMNAME_FILE:-/etc/FZJ/systemname}"
if [ -r "$systemname_file" ]; then
    source "$root/util/platform.sh"
    if ! jsc_platform="$(_qnpeps_detect_platform)"; then
        unset -f _qnpeps_compute_kind
        unset -f _qnpeps_detect_platform
        unset -f _qnpeps_platform_label
        exit 1
    fi
    unset -f _qnpeps_compute_kind
    unset -f _qnpeps_detect_platform
    unset -f _qnpeps_platform_label
fi

environment_script="$root/util/environment.sh"
environment_args=()
[ "$show_paths" -eq 0 ] || environment_args+=(--show-path)
environment_required=0
if [ -n "$jsc_platform" ] && [ "${QNPEPS_ACTIVE_ROOT:-}" != "$root" ]; then
    environment_required=1
fi
if [ ! -r "$environment_script" ]; then
    echo "[build_cuda.sh] no $environment_script, using nvcc and cmake from PATH" >&2
elif [ "$environment_required" -eq 1 ]; then
    source "$environment_script" "${environment_args[@]}" || exit 1
    if [ "${QNPEPS_ACTIVE_ROOT:-}" != "$root" ]; then
        echo "[build_cuda.sh] cuQuantumNaturalfPEPS environment activation failed" >&2
        exit 1
    fi
elif ! command -v nvcc >/dev/null 2>&1; then
    source "$environment_script" "${environment_args[@]}" || true
fi
CUDA_ARCHITECTURES="${QNPEPS_CUDA_ARCHITECTURES:-75;80;90}"
if ! command -v cmake >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
        module load CMake >/dev/null 2>&1
    fi
fi
missing_tool=0
if ! command -v cmake >/dev/null 2>&1; then
    echo "[build_cuda.sh] cmake not found on PATH" >&2
    missing_tool=1
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "[build_cuda.sh] nvcc not found on PATH" >&2
    missing_tool=1
fi
if [ "$missing_tool" -ne 0 ]; then
    echo "[build_cuda.sh] put the CUDA toolkit and CMake on PATH before running this" >&2
    exit 1
fi

cmake_cache="$build_dir/CMakeCache.txt"
configured_cuda_compiler="${CUDACXX:-nvcc}"
configured_cxx_compiler="${CXX:-c++}"
configured_cuda_compiler="${configured_cuda_compiler%% *}"
configured_cxx_compiler="${configured_cxx_compiler%% *}"
current_cuda_compiler="$(command -v "$configured_cuda_compiler" 2>/dev/null || printf '%s' "$configured_cuda_compiler")"
current_cxx_compiler="$(command -v "$configured_cxx_compiler" 2>/dev/null || printf '%s' "$configured_cxx_compiler")"
current_cuda_compiler="$(readlink -f "$current_cuda_compiler" 2>/dev/null || printf '%s' "$current_cuda_compiler")"
current_cxx_compiler="$(readlink -f "$current_cxx_compiler" 2>/dev/null || printf '%s' "$current_cxx_compiler")"
refresh_cache=0
if [ -f "$cmake_cache" ]; then
    cached_cuda_compiler="$(sed -n 's/^CMAKE_CUDA_COMPILER:[^=]*=//p' "$cmake_cache" | head -n 1)"
    cached_cxx_compiler="$(sed -n 's/^CMAKE_CXX_COMPILER:[^=]*=//p' "$cmake_cache" | head -n 1)"
    cached_cuda_compiler="$(readlink -f "$cached_cuda_compiler" 2>/dev/null || printf '%s' "$cached_cuda_compiler")"
    cached_cxx_compiler="$(readlink -f "$cached_cxx_compiler" 2>/dev/null || printf '%s' "$cached_cxx_compiler")"
    if [ "$cached_cuda_compiler" != "$current_cuda_compiler" ] \
        || [ "$cached_cxx_compiler" != "$current_cxx_compiler" ]; then
        refresh_cache=1
    fi
fi

printf 'CMake %s\n' "$(cmake --version | head -n 1)"
if [ "$show_paths" -eq 1 ]; then
    printf 'Build PATH %s\n' "$PATH"
fi
echo "CUDA build architectures $CUDA_ARCHITECTURES"
echo "CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES:-<unset>}"
if command -v nvidia-smi >/dev/null 2>&1; then
    if gpu_info="$(nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv,noheader 2>&1)"; then
        echo "NVIDIA GPUs"
        printf '%s\n%s\n' "index, name, compute capability, driver" "$gpu_info" \
            | column -t -s,
    elif gpu_info="$(nvidia-smi -L 2>&1)"; then
        echo "NVIDIA GPUs"
        printf '%s\n' "$gpu_info"
    else
        echo "NVIDIA GPUs unavailable ($gpu_info)" >&2
    fi
else
    echo "NVIDIA GPUs nvidia-smi not found"
fi

cmake_fresh_args=()
if [ "$refresh_cache" -eq 1 ]; then
    echo "[build_cuda.sh] refreshing CMake cache after compiler change"
    if cmake --help 2>&1 | grep -q -- '--fresh'; then
        cmake_fresh_args+=(--fresh)
    else
        cmake -E remove -f "$cmake_cache" || exit 1
        cmake -E remove_directory "$build_dir/CMakeFiles" || exit 1
    fi
fi

cmake "${cmake_fresh_args[@]}" -S "$root/cuda" -B "$build_dir" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" || exit 1
cmake --build "$build_dir" -j"$JOBS" || exit 1

so="$build_dir/qnpeps.so"
echo "Native library build/cuda/qnpeps.so"
if ! version="$(strings "$so" | grep -m1 -E '^cuQuantumNaturalfPEPS [0-9]+\.[0-9]+\.[0-9]+$')"; then
    echo "[build_cuda.sh] failed to find version in $so" >&2
    exit 1
fi
expected_version="$(<"$root/c_api_version.txt")"
if [ "$version" != "cuQuantumNaturalfPEPS $expected_version" ]; then
    echo "[build_cuda.sh] version mismatch: $version, expected $expected_version" >&2
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

echo "Resolved CUDA library dependencies"
missing_cuda_library=0
while read -r name arrow path remainder; do
    case "$name" in
        libcudart.so.*|libcublas.so.*|libcublasLt.so.*|libcusolver.so.*|libcurand.so.*|libcusparse.so.*|libnvJitLink.so.*)
            if [ "$path" = "not" ]; then
                printf '  %s => not found\n' "$name" >&2
                missing_cuda_library=1
            else
                printf '  %s => %s\n' "$name" "$(basename "$(readlink -f "$path")")"
            fi
            ;;
    esac
done < <(ldd "$so")
if [ "$missing_cuda_library" -ne 0 ]; then
    echo "[build_cuda.sh] failed to resolve CUDA library dependencies" >&2
    exit 1
fi
