#pragma once

#include "common.cuh"
#include "core/arena_cursor.cuh"
#include "core/complex.cuh"
#include "core/defer.cuh"
#include "core/session.cuh"
#include "density.cuh"
#include "dtensor.cuh"
#include "eloc_kernels.cuh"
#include "env_build.cuh"
#include "../linalg/eo.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <new>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "permutation.hpp"
#include "projection_kernels.hpp"
#include "arena.hpp"
#include "worker.hpp"
#include "environment_rows.hpp"
#include "tensor_network.hpp"
#include "active_lists.hpp"

namespace qn_eloc::env
{

#line 2492 "cuda/eo/env_build.cu"

__global__ auto cu_project_site_val(
    const cf* site,
    const u8* samples,
    i64 lane_stride,
    int site_idx,
    i64 slab,
    int value_mode,
    int fixed_val,
    int dim_phys,
    cf* out,
    i64 out_stride,
    int lanes
) -> void
{
    const auto total = i64{slab * lanes};
    const auto grid_stride = i64{static_cast<i64>(gridDim.x) * blockDim.x};
    for (auto tid = static_cast<i64>(blockIdx.x * blockDim.x + threadIdx.x); tid < total;
         tid += grid_stride)
    {
        const auto lane = i64{tid / slab};
        const auto elem = i64{tid % slab};
        const auto spin_raw = int{static_cast<int>(samples[lane * lane_stride + site_idx])};
        auto val = int{fixed_val};
        if (value_mode == -2)
            val = spin_raw;
        else if (value_mode < 0)
            val = dim_phys - 1 - spin_raw;
        out[lane * out_stride + elem] = site[static_cast<i64>(val) * slab + elem];
    }
}

__global__ auto cu_project_site_val_indexed(
    const cf* site,
    const u8* samples,
    i64 lane_stride,
    const int* active_indices,
    int site_idx,
    i64 slab,
    int value_mode,
    int fixed_val,
    int dim_phys,
    cf* out,
    i64 out_stride,
    int active
) -> void
{
    const auto total = i64{slab * active};
    const auto grid_stride = i64{static_cast<i64>(gridDim.x) * blockDim.x};
    for (auto tid = static_cast<i64>(blockIdx.x * blockDim.x + threadIdx.x); tid < total;
         tid += grid_stride)
    {
        const auto compact_lane = int{static_cast<int>(tid / slab)};
        const auto elem = i64{tid % slab};
        const auto lane = int{active_indices[compact_lane]};
        const auto spin_raw =
            int{static_cast<int>(samples[static_cast<i64>(lane) * lane_stride + site_idx])};
        auto val = int{fixed_val};
        if (value_mode == -2)
            val = spin_raw;
        else if (value_mode < 0)
            val = dim_phys - 1 - spin_raw;
        out[static_cast<i64>(compact_lane) * out_stride + elem] =
            site[static_cast<i64>(val) * slab + elem];
    }
}

__global__ auto cu_project_site_val_packed(
    const cf* site,
    const u8* samples,
    i64 lane_stride,
    int site_idx,
    i64 slab,
    int value_mode,
    int fixed_val,
    int dim_phys,
    const int* source_indices,
    cf* out,
    i64 out_stride,
    int lanes
) -> void
{
    const auto total = i64{slab * lanes};
    const auto grid_stride = i64{static_cast<i64>(gridDim.x) * blockDim.x};
    for (auto tid = static_cast<i64>(blockIdx.x * blockDim.x + threadIdx.x); tid < total;
         tid += grid_stride)
    {
        const auto lane = i64{tid / slab};
        const auto out_elem = int{static_cast<int>(tid % slab)};
        const auto source_elem = int{source_indices[out_elem]};
        const auto spin_raw = int{static_cast<int>(samples[lane * lane_stride + site_idx])};
        auto val = int{fixed_val};
        if (value_mode == -2)
            val = spin_raw;
        else if (value_mode < 0)
            val = dim_phys - 1 - spin_raw;
        out[lane * out_stride + out_elem] = site[static_cast<i64>(val) * slab + source_elem];
    }
}

__global__ auto cu_project_site_val_indexed_packed(
    const cf* site,
    const u8* samples,
    i64 lane_stride,
    const int* active_indices,
    int site_idx,
    i64 slab,
    int value_mode,
    int fixed_val,
    int dim_phys,
    const int* source_indices,
    cf* out,
    i64 out_stride,
    int active
) -> void
{
    const auto total = i64{slab * active};
    const auto grid_stride = i64{static_cast<i64>(gridDim.x) * blockDim.x};
    for (auto tid = static_cast<i64>(blockIdx.x * blockDim.x + threadIdx.x); tid < total;
         tid += grid_stride)
    {
        const auto compact_lane = int{static_cast<int>(tid / slab)};
        const auto out_elem = int{static_cast<int>(tid % slab)};
        const auto source_elem = int{source_indices[out_elem]};
        const auto lane = int{active_indices[compact_lane]};
        const auto spin_raw =
            int{static_cast<int>(samples[static_cast<i64>(lane) * lane_stride + site_idx])};
        auto val = int{fixed_val};
        if (value_mode == -2)
            val = spin_raw;
        else if (value_mode < 0)
            val = dim_phys - 1 - spin_raw;
        out[static_cast<i64>(compact_lane) * out_stride + out_elem] =
            site[static_cast<i64>(val) * slab + source_elem];
    }
}

__global__ auto cu_compact_lanes(
    cf* out,
    i64 out_stride,
    const cf* in,
    i64 in_stride,
    const int* active_indices,
    i64 elems,
    int active
) -> void
{
    const auto total = i64{elems * active};
    const auto grid_stride = i64{static_cast<i64>(gridDim.x) * blockDim.x};
    for (auto tid = static_cast<i64>(blockIdx.x * blockDim.x + threadIdx.x); tid < total;
         tid += grid_stride)
    {
        const auto compact_lane = int{static_cast<int>(tid / elems)};
        const auto elem = i64{tid % elems};
        const auto lane = int{active_indices[compact_lane]};
        out[static_cast<i64>(compact_lane) * out_stride + elem] =
            in[static_cast<i64>(lane) * in_stride + elem];
    }
}

__global__ auto cu_compact_lanes_packed(
    cf* out,
    i64 out_stride,
    const cf* in,
    i64 in_stride,
    const int* active_indices,
    const int* source_indices,
    i64 elems,
    int active
) -> void
{
    const auto total = i64{elems * active};
    const auto grid_stride = i64{static_cast<i64>(gridDim.x) * blockDim.x};
    for (auto tid = static_cast<i64>(blockIdx.x * blockDim.x + threadIdx.x); tid < total;
         tid += grid_stride)
    {
        const auto compact_lane = int{static_cast<int>(tid / elems)};
        const auto out_elem = int{static_cast<int>(tid % elems)};
        const auto lane = int{active_indices[compact_lane]};
        out[static_cast<i64>(compact_lane) * out_stride + out_elem] =
            in[static_cast<i64>(lane) * in_stride + source_indices[out_elem]];
    }
}

__global__ auto cu_diagonal(
    f64* e_loc,
    const u8* samples,
    i64 lane_stride,
    const int* site_a,
    const int* site_b,
    const f64* coeff,
    int n_diag,
    int lanes
) -> void
{
    const auto lane = int{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
    if (lane >= lanes) return;
    const auto base = i64{static_cast<i64>(lane) * lane_stride};
    auto e = f64{0.0};
    for (auto t = int{0}; t < n_diag; ++t)
    {
        const auto sa = int{1 - 2 * static_cast<int>(samples[base + site_a[t]])};
        const auto sb = int{1 - 2 * static_cast<int>(samples[base + site_b[t]])};
        e += coeff[t] * sa * sb;
    }
    e_loc[2 * lane] += e;
}

__global__ auto cu_diagonal_j2_half(
    f64* e_loc,
    const u8* samples,
    i64 lane_stride,
    const int* site_a,
    const int* site_b,
    const f64* coeff,
    int n_diag,
    int ly,
    int j2_mode,
    int j2_group,
    int lanes
) -> void
{
    const auto lane = int{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
    if (lane >= lanes) return;
    const auto base = i64{static_cast<i64>(lane) * lane_stride};
    auto e = f64{0.0};
    for (auto t = int{0}; t < n_diag; ++t)
    {
        const auto a = int{site_a[t]};
        const auto b = int{site_b[t]};
        const auto row_a = int{a / ly};
        const auto row_b = int{b / ly};
        const auto col_a = int{a % ly};
        const auto col_b = int{b % ly};
        const auto drow = int{row_a > row_b ? row_a - row_b : row_b - row_a};
        const auto dcol = int{col_a > col_b ? col_a - col_b : col_b - col_a};
        auto weight = f64{1.0};
        if (drow == 1 and dcol == 1)
        {
            const auto selector =
                int{j2_mode == QNPEPS_ELOC_J2_HALF_COLUMN_PAIRS ? (col_a < col_b ? col_a : col_b)
                                                                : (row_a < row_b ? row_a : row_b)};
            if (j2_group >= 0 and (selector & 1) != j2_group) continue;
            if (j2_group >= 0) weight = 2.0;
        }
        const auto sa = int{1 - 2 * static_cast<int>(samples[base + a])};
        const auto sb = int{1 - 2 * static_cast<int>(samples[base + b])};
        e += weight * coeff[t] * sa * sb;
    }
    e_loc[2 * lane] += e;
}

__global__ auto cu_reduce_term(
    f64* e_loc,
    const cf* value,
    const f64* f_top,
    const f64* f_down,
    const f64* f_rail_l,
    const f64* f_rail_r,
    const f64* f_chain,
    const f64* logpsi,
    const u8* samples,
    i64 lane_stride,
    int mask_a,
    int mask_b,
    f64 coeff_re,
    f64 coeff_im,
    int lanes
) -> void
{
    const auto lane = int{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
    if (lane >= lanes) return;
    if (mask_a >= 0)
    {
        const auto base = i64{static_cast<i64>(lane) * lane_stride};
        if (samples[base + mask_a] == samples[base + mask_b]) return;
    }
    const auto v = cf{value[lane]};
    const auto re = f64{static_cast<f64>(v.re)};
    const auto im = f64{static_cast<f64>(v.im)};
    const auto logmag = f64{0.5 * log(norm2(re, im))};
    const auto phase = f64{atan2(im, re)};
    const auto fsum =
        f64{(f_top ? f_top[lane] : 0.0) + (f_down ? f_down[lane] : 0.0)
            + (f_rail_l ? f_rail_l[lane] : 0.0) + (f_rail_r ? f_rail_r[lane] : 0.0)
            + (f_chain ? f_chain[lane] : 0.0)};
    const auto dre = f64{logmag + fsum - logpsi[2 * lane]};
    const auto dim = f64{phase - logpsi[2 * lane + 1]};
    const auto mag = f64{exp(dre)};
    const auto rr = f64{mag * cos(dim)};
    const auto ri = f64{mag * sin(dim)};
    e_loc[2 * lane] += coeff_re * rr - coeff_im * ri;
    e_loc[2 * lane + 1] += coeff_re * ri + coeff_im * rr;
}

__global__ auto cu_reduce_term_indexed(
    f64* e_loc,
    const cf* value,
    const f64* f_top,
    const f64* f_down,
    const f64* f_rail_l,
    const f64* f_rail_r,
    const f64* f_chain,
    const f64* logpsi,
    const int* active_indices,
    f64 coeff_re,
    f64 coeff_im,
    int active
) -> void
{
    const auto compact_lane = int{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
    if (compact_lane >= active) return;
    const auto lane = int{active_indices[compact_lane]};
    const auto v = cf{value[compact_lane]};
    const auto re = f64{static_cast<f64>(v.re)};
    const auto im = f64{static_cast<f64>(v.im)};
    const auto logmag = f64{0.5 * log(norm2(re, im))};
    const auto phase = f64{atan2(im, re)};
    const auto fsum =
        f64{(f_top ? f_top[lane] : 0.0) + (f_down ? f_down[lane] : 0.0)
            + (f_rail_l ? f_rail_l[lane] : 0.0) + (f_rail_r ? f_rail_r[lane] : 0.0)
            + f_chain[compact_lane]};
    const auto dre = f64{logmag + fsum - logpsi[2 * lane]};
    const auto dim = f64{phase - logpsi[2 * lane + 1]};
    const auto mag = f64{exp(dre)};
    const auto rr = f64{mag * cos(dim)};
    const auto ri = f64{mag * sin(dim)};
    e_loc[2 * lane] += coeff_re * rr - coeff_im * ri;
    e_loc[2 * lane + 1] += coeff_re * ri + coeff_im * rr;
}

inline auto top_bt(Worker& w, int i, int c) -> BT
{
    return (i > 0) ? BT{etf_site(w, i - 1, c), w.slot_env, top_dims(w, i, c)} : unit3(w);
}
inline auto bot_bt_h(Worker& w, int i, int c) -> BT
{
    return (i < w.sh.lx - 1) ? BT{eb_site(w, w.sh.lx - i - 2, c), w.slot_env, bot_dims_h(w, i, c)}
                             : unit3(w);
}
inline auto bot_bt_fb(Worker& w, int i, int c) -> BT
{
    return (i < w.sh.lx - 2) ? BT{eb_site(w, w.sh.lx - i - 3, c), w.slot_env, bot_dims_fb(w, i, c)}
                             : unit3(w);
}

inline auto hl_o_dims(Worker& w, int i, int c) -> std::vector<int>
{
    const auto tb = int{i > 0 ? w.ks_full[static_cast<usize>(i - 1)][static_cast<usize>(c)] : 1};
    const auto bb =
        int{i < w.sh.lx - 1 ? w.ks_down[static_cast<usize>(w.sh.lx - i - 2)][static_cast<usize>(c)]
                            : 1};
    return {tb, bd(w.sh.ly, c, w.sh.dim_bond), bb};
}

inline auto hr_o_dims(Worker& w, int i, int c) -> std::vector<int>
{
    const auto tb =
        int{i > 0 ? w.ks_full[static_cast<usize>(i - 1)][static_cast<usize>(c + 1)] : 1};
    const auto bb =
        int{i < w.sh.lx - 1
                ? w.ks_down[static_cast<usize>(w.sh.lx - i - 2)][static_cast<usize>(c + 1)]
                : 1};
    return {tb, bd(w.sh.ly, c + 1, w.sh.dim_bond), bb};
}

auto reduce_term(
    Worker& w,
    int mask_a,
    int mask_b,
    f64 coeff_re,
    f64 coeff_im,
    const cf* value,
    f64* f_top,
    f64* f_down,
    f64* f_rail_l,
    f64* f_rail_r,
    f64* e_loc,
    const f64* logpsi
) -> void
{
    const auto lanes = int{w.sh.lanes};
    const auto blocks = int{(lanes + 255) / 256};
    cu_reduce_term<<<blocks, 256, 0, w.la.stream()>>>(
        e_loc,
        value,
        f_top,
        f_down,
        f_rail_l,
        f_rail_r,
        w.f_chain,
        logpsi,
        w.samples,
        static_cast<i64>(w.sh.lx) * w.sh.ly,
        mask_a,
        mask_b,
        coeff_re,
        coeff_im,
        lanes
    );
    CUDA_CHECK(cudaGetLastError());
}

auto reduce_term_indexed(
    Worker& w,
    const FlipInst& term,
    const cf* value,
    const f64* f_chain,
    const int* active_indices,
    int active,
    f64* f_top,
    f64* f_down,
    f64* f_rail_l,
    f64* f_rail_r,
    f64* e_loc,
    const f64* logpsi
) -> void
{
    const auto blocks = int{(active + 255) / 256};
    cu_reduce_term_indexed<<<blocks, 256, 0, w.la.stream()>>>(
        e_loc,
        value,
        f_top,
        f_down,
        f_rail_l,
        f_rail_r,
        f_chain,
        logpsi,
        active_indices,
        term.coeff_re,
        term.coeff_im,
        active
    );
    CUDA_CHECK(cudaGetLastError());
}

auto launch_diagonal(
    Worker& w, f64* e_loc, const int* site_a, const int* site_b, const f64* coeff, int n_diag
) -> void
{
    if (n_diag < 1) return;
    const auto lanes = int{w.sh.lanes};
    const auto blocks = int{(lanes + 255) / 256};
    cu_diagonal<<<blocks, 256, 0, w.la.stream()>>>(
        e_loc, w.samples, static_cast<i64>(w.sh.lx) * w.sh.ly, site_a, site_b, coeff, n_diag, lanes
    );
    CUDA_CHECK(cudaGetLastError());
}

auto launch_diagonal_j2_half(
    Worker& w,
    f64* e_loc,
    const int* site_a,
    const int* site_b,
    const f64* coeff,
    int n_diag,
    int j2_mode,
    int j2_group
) -> void
{
    if (n_diag < 1) return;
    const auto lanes = int{w.sh.lanes};
    const auto blocks = int{(lanes + 255) / 256};
    cu_diagonal_j2_half<<<blocks, 256, 0, w.la.stream()>>>(
        e_loc,
        w.samples,
        static_cast<i64>(w.sh.lx) * w.sh.ly,
        site_a,
        site_b,
        coeff,
        n_diag,
        w.sh.ly,
        j2_mode,
        j2_group,
        lanes
    );
    CUDA_CHECK(cudaGetLastError());
}

__global__ auto cu_fourbody(
    int lanes, int mid, const cf* cl, i64 stride_l, const cf* cr, i64 stride_r, cf* out
) -> void
{
    extern __shared__ f32 smem[];
    auto partial{reinterpret_cast<cf*>(smem)};
    for (auto b = static_cast<int>(blockIdx.x); b < lanes; b += gridDim.x)
    {
        auto a{cl + static_cast<i64>(b) * stride_l};
        auto bb{cr + static_cast<i64>(b) * stride_r};
        auto local = cf{};
        for (auto k = static_cast<int>(threadIdx.x); k < mid; k += blockDim.x)
            fx::cf_acc(local, a[k], bb[k]);
        partial[threadIdx.x] = local;
        __syncthreads();
        for (auto h = static_cast<int>(blockDim.x / 2); h > 0; h >>= 1)
        {
            if (static_cast<int>(threadIdx.x) < h)
                partial[threadIdx.x] = qnpeps::to_cf(cuCaddf(
                    qnpeps::to_cu(partial[threadIdx.x]), qnpeps::to_cu(partial[threadIdx.x + h])
                ));
            __syncthreads();
        }
        if (threadIdx.x == 0) out[b] = partial[0];
    }
}

auto launch_dot(
    Worker& w,
    int mid,
    const cf* cl,
    i64 stride_l,
    const cf* cr,
    i64 stride_r,
    cf* out,
    int dim_batch = 0
) -> void
{
    const auto lanes = int{dim_batch > 0 ? dim_batch : w.sh.lanes};
    const auto block = int{128};
    const auto shmem = usize{static_cast<usize>(block) * sizeof(cf)};
    const auto grid = int{std::min(lanes, 65535)};
    cu_fourbody<<<grid, block, shmem, w.la.stream()>>>(lanes, mid, cl, stride_l, cr, stride_r, out);
    CUDA_CHECK(cudaGetLastError());
}

}
