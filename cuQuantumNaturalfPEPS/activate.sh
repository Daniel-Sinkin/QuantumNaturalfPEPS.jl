#!/usr/bin/env bash

unset QNPEPS_ACTIVE_ROOT

_qnpeps_load_modules() {
    local package_dir nvcc_path julia_path nvcc_release toolkit_libdir
    package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || return 1

    if ! type module >/dev/null 2>&1; then
        if [ -n "${MODULESHOME:-}" ] && [ -r "$MODULESHOME/init/bash" ]; then
            source "$MODULESHOME/init/bash"
        elif [ -r /p/software/default/lmod/lmod/init/bash ]; then
            source /p/software/default/lmod/lmod/init/bash
        fi
    fi
    if ! type module >/dev/null 2>&1; then
        echo "activate.sh: JURECA's module command is unavailable" >&2
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
        echo "activate.sh: set QNPEPS_JULIA_DEPOT because SCRATCH or USER is unavailable" >&2
        return 1
    fi
    export JULIA_DEPOT_PATH="$QNPEPS_JULIA_DEPOT"
    mkdir -p "$JULIA_DEPOT_PATH" || return 1

    nvcc_path="$(readlink -f "$(command -v nvcc)")"
    julia_path="$(readlink -f "$(command -v julia)")"
    toolkit_libdir="$(readlink -f "$CUDA_HOME/targets/x86_64-linux/lib")"
    nvcc_release="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
    if [ "$nvcc_release" != "$QNPEPS_CUDA_VERSION" ]; then
        echo "activate.sh: expected CUDA $QNPEPS_CUDA_VERSION, nvcc reports $nvcc_release" >&2
        return 1
    fi

    export QNPEPS_ACTIVE_ROOT="$package_dir"

    echo "cuQuantumNaturalfPEPS JURECA toolchain"
    printf '  Host: %s\n' "$(hostname)"
    printf '  Julia: %s\n' "$(julia --version)"
    printf '  Julia executable: %s\n' "$julia_path"
    printf '  CUDA compiler: %s\n' "$(nvcc --version | tail -n 1)"
    printf '  CUDA compiler executable: %s\n' "$nvcc_path"
    printf '  CUDA_HOME: %s\n' "$CUDA_HOME"
    printf '  CUDA toolkit libraries: %s\n' "$toolkit_libdir"
    printf '  JULIA_DEPOT_PATH: %s\n' "$JULIA_DEPOT_PATH"
    printf '  QNPEPS_ACTIVE_ROOT: %s\n' "$QNPEPS_ACTIVE_ROOT"
    printf '  PATH: %s\n' "$PATH"
    printf '  LD_LIBRARY_PATH: %s\n' "${LD_LIBRARY_PATH:-<unset>}"
    printf '  LD_PRELOAD: %s\n' "${LD_PRELOAD:-<unset>}"
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

_qnpeps_load_modules
_qnpeps_load_modules_status=$?
unset -f _qnpeps_load_modules

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    if [ "$_qnpeps_load_modules_status" -eq 0 ]; then
        unset _qnpeps_load_modules_status
        exit 0
    else
        unset _qnpeps_load_modules_status
        exit 1
    fi
else
    if [ "$_qnpeps_load_modules_status" -eq 0 ]; then
        unset _qnpeps_load_modules_status
        return 0
    else
        unset _qnpeps_load_modules_status
        return 1
    fi
fi
