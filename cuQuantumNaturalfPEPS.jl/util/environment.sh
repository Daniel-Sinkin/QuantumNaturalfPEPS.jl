#!/usr/bin/env bash

_qnpeps_environment() {
    local show_paths=0
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --show-path)
                show_paths=1
                ;;
            --)
                shift
                break
                ;;
            *)
                echo "[environment.sh] unknown option $1" >&2
                return 2
                ;;
        esac
        shift
    done

    local package_dir nvcc_release nvcc_major platform_id platform_label
    local cuda_major cuda_architectures julia_depot
    package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 1

    source "$package_dir/util/platform.sh" || return 1
    if ! platform_id="$(_qnpeps_detect_platform)"; then
        unset -f _qnpeps_compute_kind
        unset -f _qnpeps_detect_platform
        unset -f _qnpeps_platform_label
        return 1
    fi
    if ! platform_label="$(_qnpeps_platform_label "$platform_id")"; then
        unset -f _qnpeps_compute_kind
        unset -f _qnpeps_detect_platform
        unset -f _qnpeps_platform_label
        return 1
    fi
    unset -f _qnpeps_compute_kind
    unset -f _qnpeps_detect_platform
    unset -f _qnpeps_platform_label

    if ! type module >/dev/null 2>&1; then
        if [ -n "${MODULESHOME:-}" ] && [ -r "$MODULESHOME/init/bash" ]; then
            source "$MODULESHOME/init/bash"
        elif [ -r /p/software/default/lmod/lmod/init/bash ]; then
            source /p/software/default/lmod/lmod/init/bash
        elif [ -r /e/software/default/lmod/lmod/init/bash ]; then
            source /e/software/default/lmod/lmod/init/bash
        fi
    fi
    if ! type module >/dev/null 2>&1; then
        echo "[environment.sh] module command unavailable on $platform_label" >&2
        return 1
    fi

    case "$platform_id" in
        jusuf-*)
            module load Stages/2025 || return 1
            module load GCC/13.3.0 ParaStationMPI/5.11.0-1 Julia/1.11.2 || return 1
            module load CUDA/12 cuTENSOR/2.0.2.5-CUDA-12 || return 1
            cuda_major=12
            cuda_architectures=70
            ;;
        jureca-*|jupiter-*)
            module load Stages/2026 || return 1
            module load GCC/14.3.0 ParaStationMPI/5.13.0-1 Julia/1.12.3 || return 1
            module load CUDA/13 cuTENSOR/2.3.1.0-CUDA-13 || return 1
            cuda_major=13
            cuda_architectures="75;80;90"
            ;;
        *)
            echo "[environment.sh] unsupported platform id $platform_id" >&2
            return 1
            ;;
    esac
    if [ -n "${QNPEPS_JULIA_DEPOT:-}" ]; then
        julia_depot="$QNPEPS_JULIA_DEPOT"
    elif [ -n "${PEPS_JULIA_DEPOT:-}" ]; then
        julia_depot="$PEPS_JULIA_DEPOT"
    elif [ -n "${SCRATCH:-}" ] && [ -n "${USER:-}" ]; then
        julia_depot="$SCRATCH/$USER/julia-peps-cuda"
    else
        echo "[setup.sh] set QNPEPS_JULIA_DEPOT because SCRATCH or USER is unavailable" >&2
        return 1
    fi

    if ! command -v nvcc >/dev/null 2>&1; then
        echo "[setup.sh] nvcc was not found for $platform_label" >&2
        return 1
    fi
    if ! command -v julia >/dev/null 2>&1; then
        echo "[setup.sh] julia was not found for $platform_label" >&2
        return 1
    fi

    nvcc_release="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
    nvcc_major="${nvcc_release%%.*}"
    if [ "$nvcc_major" != "$cuda_major" ]; then
        echo "[setup.sh] expected CUDA $cuda_major.x, nvcc reports ${nvcc_release:-<unknown>}" >&2
        return 1
    fi
    mkdir -p "$julia_depot" || return 1

    export QNPEPS_CUDA_VERSION="$nvcc_release"
    export QNPEPS_CUDA_ARCHITECTURES="$cuda_architectures"
    export QNPEPS_PLATFORM_ID="$platform_id"
    export QNPEPS_PLATFORM_LABEL="$platform_label"
    export QNPEPS_JULIA_DEPOT="$julia_depot"
    export JULIA_DEPOT_PATH="$julia_depot"
    export QNPEPS_ACTIVE_ROOT="$package_dir"

    echo "cuQuantumNaturalfPEPS $QNPEPS_PLATFORM_LABEL toolchain"
    printf '  Host %s\n' "$(hostname)"
    printf '  Julia %s\n' "$(julia --version)"
    printf '  CUDA compiler %s\n' "$(nvcc --version | tail -n 1)"
    printf '  CUDA architectures %s\n' "$QNPEPS_CUDA_ARCHITECTURES"
    if [ "$show_paths" -eq 1 ]; then
        printf '  Activation PATH %s\n' "$PATH"
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_info
        if gpu_info="$(nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv,noheader 2>&1)"; then
            echo "  Visible NVIDIA GPUs"
            while IFS= read -r gpu; do
                printf '    %s\n' "$gpu"
            done <<< "$gpu_info"
        else
            printf '  Visible NVIDIA GPUs unavailable (%s)\n' "$gpu_info"
        fi
    fi
}

_qnpeps_environment "$@"
_qnpeps_environment_status=$?
unset -f _qnpeps_environment

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    exit "$_qnpeps_environment_status"
else
    if [ "$_qnpeps_environment_status" -eq 0 ]; then
        unset _qnpeps_environment_status
        return 0
    else
        unset _qnpeps_environment_status
        return 1
    fi
fi
