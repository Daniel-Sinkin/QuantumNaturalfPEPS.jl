#!/usr/bin/env bash

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "setup.sh must be sourced: source ${BASH_SOURCE[0]}" >&2
    exit 1
fi

_qnpeps_setup() {
    local package_dir
    package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || return 1

    if ! command -v python3 >/dev/null 2>&1; then
        echo "setup.sh: python3 is required" >&2
        return 1
    fi

    local argument
    local needs_environment=1
    local show_paths=0
    for argument in "$@"; do
        case "$argument" in
            -c|--check|-cv|--check-version|--api-version|-h|--help)
                needs_environment=0
                ;;
            -f|--force)
                ;;
            --show-path)
                show_paths=1
                ;;
            *)
                needs_environment=0
                ;;
        esac
    done

    if [ "$needs_environment" -eq 1 ]; then
        local environment_args=()
        [ "$show_paths" -eq 0 ] || environment_args+=(--show-path)
        source "$package_dir/util/environment.sh" "${environment_args[@]}" || return 1
    fi

    python3 -B "$package_dir/util/setup.py" "$@"
}

_qnpeps_setup_cleanup() {
    local status="$1"
    unset -f _qnpeps_setup
    unset -f _qnpeps_setup_cleanup
    return "$status"
}

if _qnpeps_setup "$@"; then
    _qnpeps_setup_cleanup 0
else
    _qnpeps_setup_cleanup "$?"
fi
return $?
