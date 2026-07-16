#!/usr/bin/env bash

_qnpeps_setup() {
    local package_dir arch
    package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || return 1
    arch="${1:-${ARCH:-80}}"

    source "$package_dir/activate.sh" || return 1

    echo "Configuring and precompiling Julia environments"
    julia --startup-file=no "$package_dir/util/configure_julia.jl" || return 1

    module load CMake || return 1
    echo "Building libpeps_sampler.so"
    "$package_dir/util/build_cuda.sh" "$arch" || return 1

    echo "cuQuantumNaturalfPEPS setup complete"
    echo "For each new shell, run: source $package_dir/activate.sh"
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
