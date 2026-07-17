#!/usr/bin/env bash

_qnpeps_setup() {
    local package_dir
    package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || return 1

    source "$package_dir/activate.sh" || return 1

    echo "Setup Julia"
    julia --startup-file=no "$package_dir/util/configure_julia.jl" || return 1

    echo "Building qnpeps.so"
    "$package_dir/util/build_cuda.sh" || return 1

    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        echo "Check Julia and CUDA runtime"
        julia --startup-file=no --project="$package_dir" \
            "$package_dir/util/runtime_info.jl" || return 1
    fi

    echo "cuQuantumNaturalfPEPS setup complete"
}

_qnpeps_setup "$@"
_qnpeps_setup_status=$?
unset -f _qnpeps_setup

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    if [ "$_qnpeps_setup_status" -eq 0 ]; then
        unset _qnpeps_setup_status
        exit 0
    else
        unset _qnpeps_setup_status
        exit 1
    fi
else
    if [ "$_qnpeps_setup_status" -eq 0 ]; then
        unset _qnpeps_setup_status
        return 0
    else
        unset _qnpeps_setup_status
        return 1
    fi
fi
