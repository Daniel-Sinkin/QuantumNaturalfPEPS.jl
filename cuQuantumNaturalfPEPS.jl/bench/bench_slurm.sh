#!/bin/bash

PKG="${PKG_DIR:-$SLURM_SUBMIT_DIR}"

if [ ! -f "$PKG/cuda/capi/qnpeps.h" ]; then
    echo "[bench_slurm.sh] $PKG/cuda/capi/qnpeps.h not found (set PKG_DIR to the package root)" >&2
    exit 1
fi

source "$PKG/util/environment.sh" || exit 1

if [ ! -f "$PKG/build/cuda/qnpeps.so" ]; then
    JOBS="${SLURM_CPUS_PER_TASK:-8}" "$PKG/util/build_cuda.sh" || exit 1
fi

nvcc -O2 -std=c++20 -arch=sm_80 "$PKG/bench/bench_api.cu" -o "$PKG/bench/bench_api" \
    -I"$PKG/cuda" "$PKG/build/cuda/qnpeps.so" \
    -Xlinker -rpath -Xlinker "$PKG/build/cuda" || exit 1

export QNPEPS_LIB="$PKG/build/cuda/qnpeps.so"
julia --project="$PKG" -e 'using Pkg; Pkg.instantiate()' || { echo "julia instantiate failed"; exit 1; }

mkdir -p "$PKG/bench/out"
cd "$PKG" || exit 1

for c in 1 2; do
    bench/bench_api "$c" | tee "$PKG/bench/out/c_${c}.log"
    julia --project="$PKG" bench/bench_api.jl "$c" | tee "$PKG/bench/out/julia_${c}.log"
done

get() { grep -o "$2=[0-9.]*" "$1" | awk -F= '{print $2; exit}'; }
pct() { awk -v c="$1" -v j="$2" 'BEGIN{ if (j+0==0) print "NA"; else printf "%.1f", 100*c/j }'; }
cell() { if [ -n "$2" ]; then printf '%s ± %s' "$1" "$2"; else printf '%s' "$1"; fi; }

c0="$PKG/bench/out/c_1.log"; j0="$PKG/bench/out/julia_1.log"
c1="$PKG/bench/out/c_2.log"; j1="$PKG/bench/out/julia_2.log"

c0bm=$(get "$c0" build_ms); c0bs=$(get "$c0" build_sd); c0sm=$(get "$c0" sample_ms); c0ss=$(get "$c0" sample_sd)
j0bm=$(get "$j0" build_ms); j0bs=$(get "$j0" build_sd); j0sm=$(get "$j0" sample_ms); j0ss=$(get "$j0" sample_sd)
c1bm=$(get "$c1" build_ms); c1bs=$(get "$c1" build_sd); c1sm=$(get "$c1" sample_ms); c1ss=$(get "$c1" sample_sd)
j1bm=$(get "$j1" build_ms); j1bs=$(get "$j1" build_sd); j1sm=$(get "$j1" sample_ms); j1ss=$(get "$j1" sample_sd)

csv="$PKG/bench/out/bench_results_$(date +%Y-%m-%d)_j${SLURM_JOB_ID}.csv"
{
    echo "Params,Operation,CUDA (ms),Julia (ms),Perf (C/Julia)"
    echo "L=8 D=4 chi=4 n_samples=1024,build,$(cell "$c0bm" "$c0bs"),$(cell "$j0bm" "$j0bs"),$(pct "$c0bm" "$j0bm")%"
    echo "L=8 D=4 chi=4 n_samples=1024,sample,$(cell "$c0sm" "$c0ss"),$(cell "$j0sm" "$j0ss"),$(pct "$c0sm" "$j0sm")%"
    echo "L=16 D=7 chi=7 n_samples=512,build,$(cell "$c1bm" "$c1bs"),$(cell "$j1bm" "$j1bs"),$(pct "$c1bm" "$j1bm")%"
    echo "L=16 D=7 chi=7 n_samples=512,sample,$(cell "$c1sm" "$c1ss"),$(cell "$j1sm" "$j1ss"),$(pct "$c1sm" "$j1sm")%"
} > "$csv"

cat "$csv"
