#!/usr/bin/env bash

root="$(cd "$(dirname "$0")" && pwd)"
exec "$root/util/build_cuda.sh" "$@"
