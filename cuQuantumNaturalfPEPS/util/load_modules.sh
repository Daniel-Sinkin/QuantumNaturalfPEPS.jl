#!/usr/bin/env bash

_qnpeps_load_modules() {
    if ! type module >/dev/null 2>&1; then
        if [ -n "${MODULESHOME:-}" ] && [ -r "$MODULESHOME/init/bash" ]; then
            source "$MODULESHOME/init/bash"
        elif [ -r /p/software/default/lmod/lmod/init/bash ]; then
            source /p/software/default/lmod/lmod/init/bash
        fi
    fi
    if ! type module >/dev/null 2>&1; then
        echo "load_modules.sh: JURECA's module command is unavailable" >&2
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
        echo "load_modules.sh: set QNPEPS_JULIA_DEPOT because SCRATCH or USER is unavailable" >&2
        return 1
    fi
    export JULIA_DEPOT_PATH="$QNPEPS_JULIA_DEPOT"
    mkdir -p "$JULIA_DEPOT_PATH" || return 1

    local nvcc_release
    nvcc_release="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
    if [ "$nvcc_release" != "$QNPEPS_CUDA_VERSION" ]; then
        echo "load_modules.sh: expected CUDA $QNPEPS_CUDA_VERSION, nvcc reports $nvcc_release" >&2
        return 1
    fi

    echo "cuQuantumNaturalfPEPS JURECA toolchain"
    printf '  Julia: %s\n' "$(julia --version)"
    printf '  CUDA compiler: %s\n' "$(nvcc --version | tail -n 1)"
    printf '  CUDA_HOME: %s\n' "$CUDA_HOME"
    printf '  JULIA_DEPOT_PATH: %s\n' "$JULIA_DEPOT_PATH"
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
