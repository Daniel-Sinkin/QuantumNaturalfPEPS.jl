#!/usr/bin/env bash

_qnpeps_bootstrap_jureca() {
    local package_dir arch
    package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 1
    arch="${1:-${ARCH:-80}}"

    source "$package_dir/util/load_modules.sh" || return 1

    echo "Configuring and precompiling Julia environments"
    julia --startup-file=no "$package_dir/setup.jl" || return 1

    module load CMake || return 1
    echo "Building libpeps_sampler.so"
    "$package_dir/setup_cuda.sh" "$arch" || return 1

    echo "cuQuantumNaturalfPEPS bootstrap complete"
    echo "For each new shell, run: source $package_dir/util/load_modules.sh"
}

_qnpeps_bootstrap_jureca "$@"
_qnpeps_bootstrap_status=$?
unset -f _qnpeps_bootstrap_jureca

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    if [ "$_qnpeps_bootstrap_status" -eq 0 ]; then
        unset _qnpeps_bootstrap_status
        exit 0
    else
        unset _qnpeps_bootstrap_status
        exit 1
    fi
else
    if [ "$_qnpeps_bootstrap_status" -eq 0 ]; then
        unset _qnpeps_bootstrap_status
        return 0
    else
        unset _qnpeps_bootstrap_status
        return 1
    fi
fi
