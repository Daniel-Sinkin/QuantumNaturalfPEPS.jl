#!/usr/bin/env bash

_qnpeps_compute_kind() {
    local cluster="$1"
    local partition="$2"

    case "$cluster:$partition" in
        jureca:dc-gpu*)
            printf '%s\n' "jureca-dc-gpu"
            ;;
        jureca:dc-cpu*)
            printf '%s\n' "jureca-dc-cpu"
            ;;
        jureca:*)
            printf '%s\n' "jureca-dc"
            ;;
        jusuf:gpus|jusuf:develgpus)
            printf '%s\n' "jusuf-gpu"
            ;;
        jusuf:batch|jusuf:devel|jusuf:scraper)
            printf '%s\n' "jusuf-cpu"
            ;;
        jusuf:*)
            printf '%s\n' "jusuf-compute"
            ;;
        jupiter:booster|jupiter:largebooster|jupiter:*)
            printf '%s\n' "jupiter-booster"
            ;;
        *)
            return 1
            ;;
    esac
}

_qnpeps_detect_platform() {
    local systemname_file="${QNPEPS_SYSTEMNAME_FILE:-/etc/FZJ/systemname}"
    local systemname host partition cluster

    if [ ! -r "$systemname_file" ]; then
        echo "[platform.sh] cannot read $systemname_file" >&2
        return 1
    fi
    IFS= read -r systemname < "$systemname_file"
    systemname="${systemname,,}"
    host="${QNPEPS_HOSTNAME:-$(hostname -s)}"
    host="${host%%.*}"
    partition="${SLURM_JOB_PARTITION:-}"

    case "$systemname" in
        jureca|jurecadc)
            cluster="jureca"
            ;;
        jusuf)
            cluster="jusuf"
            ;;
        jupiter)
            cluster="jupiter"
            ;;
        *)
            echo "[platform.sh] unsupported JSC system ${systemname:-<empty>}" >&2
            return 1
            ;;
    esac

    case "$cluster:$host" in
        jureca:jrlogin*|jureca:jureca*)
            printf '%s\n' "jureca-login"
            return 0
            ;;
        jureca:jrc*)
            _qnpeps_compute_kind "$cluster" "$partition"
            return
            ;;
        jusuf:jsfl*|jusuf:jusuf*)
            printf '%s\n' "jusuf-login"
            return 0
            ;;
        jusuf:jsfc*)
            _qnpeps_compute_kind "$cluster" "$partition"
            return
            ;;
        jupiter:jpbl*)
            printf '%s\n' "jupiter-login"
            return 0
            ;;
        jupiter:jpbo*)
            printf '%s\n' "jupiter-booster"
            return 0
            ;;
    esac

    if [ -n "${SLURM_JOB_ID:-}" ]; then
        case "$cluster:$partition" in
            jureca:dc-*|jusuf:batch|jusuf:devel|jusuf:scraper|jusuf:gpus|jusuf:develgpus|jupiter:booster|jupiter:largebooster)
                _qnpeps_compute_kind "$cluster" "$partition"
                return
                ;;
        esac
    fi

    echo "[platform.sh] unrecognized $systemname host $host" >&2
    return 1
}

_qnpeps_platform_label() {
    case "$1" in
        jureca-login)
            printf '%s\n' "JURECA login"
            ;;
        jureca-dc-gpu)
            printf '%s\n' "JURECA DC GPU"
            ;;
        jureca-dc-cpu)
            printf '%s\n' "JURECA DC CPU"
            ;;
        jureca-dc)
            printf '%s\n' "JURECA DC compute"
            ;;
        jusuf-login)
            printf '%s\n' "JUSUF login"
            ;;
        jusuf-gpu)
            printf '%s\n' "JUSUF GPU"
            ;;
        jusuf-cpu)
            printf '%s\n' "JUSUF CPU"
            ;;
        jusuf-compute)
            printf '%s\n' "JUSUF compute"
            ;;
        jupiter-login)
            printf '%s\n' "JUPITER login"
            ;;
        jupiter-booster)
            printf '%s\n' "JUPITER Booster"
            ;;
        *)
            echo "[platform.sh] unsupported platform id $1" >&2
            return 1
            ;;
    esac
}
