#!/usr/bin/env bash

_qnpeps_environment() {
    local show_paths=0
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --show-path)
                show_paths=1
                ;;
            *)
                echo "environment.sh: unknown option: $1" >&2
                return 2
                ;;
        esac
        shift
    done

    local package_dir nvcc_release
    package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 1
    unset QNPEPS_ACTIVE_ROOT

    if ! type module >/dev/null 2>&1; then
        if [ -n "${MODULESHOME:-}" ] && [ -r "$MODULESHOME/init/bash" ]; then
            source "$MODULESHOME/init/bash"
        elif [ -r /p/software/default/lmod/lmod/init/bash ]; then
            source /p/software/default/lmod/lmod/init/bash
        fi
    fi
    if ! type module >/dev/null 2>&1; then
        echo "setup.sh: JURECA's module command is unavailable" >&2
        return 1
    fi

    module load Stages/2026 || return 1
    module load GCC/14.3.0 ParaStationMPI/5.13.0-1 Julia/1.12.3 || return 1
    module load CUDA/13 cuTENSOR/2.3.1.0-CUDA-13 || return 1

    export QNPEPS_CUDA_VERSION=13.0
    if [ -n "${QNPEPS_JULIA_DEPOT:-}" ]; then
        :
    elif [ -n "${PEPS_JULIA_DEPOT:-}" ]; then
        export QNPEPS_JULIA_DEPOT="$PEPS_JULIA_DEPOT"
    elif [ -n "${SCRATCH:-}" ] && [ -n "${USER:-}" ]; then
        export QNPEPS_JULIA_DEPOT="$SCRATCH/$USER/julia-peps-cuda"
    else
        echo "setup.sh: set QNPEPS_JULIA_DEPOT because SCRATCH or USER is unavailable" >&2
        return 1
    fi
    export JULIA_DEPOT_PATH="$QNPEPS_JULIA_DEPOT"
    mkdir -p "$JULIA_DEPOT_PATH" || return 1

    if ! command -v nvcc >/dev/null 2>&1; then
        echo "setup.sh: nvcc was not found after loading CUDA/13" >&2
        return 1
    fi
    if ! command -v julia >/dev/null 2>&1; then
        echo "setup.sh: julia was not found after loading Julia/1.12.3" >&2
        return 1
    fi

    nvcc_release="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
    if [ "$nvcc_release" != "$QNPEPS_CUDA_VERSION" ]; then
        echo "setup.sh: expected CUDA $QNPEPS_CUDA_VERSION, nvcc reports $nvcc_release" >&2
        return 1
    fi

    export QNPEPS_ACTIVE_ROOT="$package_dir"

    echo "cuQuantumNaturalfPEPS JURECA toolchain"
    printf '  Host: %s\n' "$(hostname)"
    printf '  Julia: %s\n' "$(julia --version)"
    printf '  CUDA compiler: %s\n' "$(nvcc --version | tail -n 1)"
    if [ "$show_paths" -eq 1 ]; then
        printf '  Activation PATH: %s\n' "$PATH"
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_info
        if gpu_info="$(nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv,noheader 2>&1)"; then
            echo "  Visible NVIDIA GPUs (index, name, compute capability, driver):"
            while IFS= read -r gpu; do
                printf '    %s\n' "$gpu"
            done <<< "$gpu_info"
        else
            printf '  Visible NVIDIA GPUs: unavailable (%s)\n' "$gpu_info"
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
