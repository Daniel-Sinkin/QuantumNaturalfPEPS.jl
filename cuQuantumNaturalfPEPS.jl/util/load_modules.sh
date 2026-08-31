#!/usr/bin/env bash

_qnpeps_load_modules() {
    local package_dir
    package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 1
    source "$package_dir/util/environment.sh" "$@"
}

_qnpeps_load_modules "$@"
_qnpeps_load_modules_status=$?
unset -f _qnpeps_load_modules

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    exit "$_qnpeps_load_modules_status"
else
    if [ "$_qnpeps_load_modules_status" -eq 0 ]; then
        unset _qnpeps_load_modules_status
        return 0
    else
        unset _qnpeps_load_modules_status
        return 1
    fi
fi
