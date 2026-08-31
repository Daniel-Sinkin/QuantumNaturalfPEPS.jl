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
#include "term_kernels.hpp"
#include "term_evaluation.hpp"

namespace qn_eloc::env
{

#line 3742 "cuda/eo/env_build.cu"

__global__ auto cu_gscale(
    cf* g,
    const f64* f_top,
    const f64* f_down,
    const f64* f_rail_l,
    const f64* f_rail_r,
    const f64* logpsi,
    int lanes
) -> void
{
    const auto lane = int{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
    if (lane >= lanes) return;
    const auto ft = f64{f_top ? f_top[lane] : 0.0};
    const auto fd = f64{f_down ? f_down[lane] : 0.0};
    const auto fl = f64{f_rail_l ? f_rail_l[lane] : 0.0};
    const auto fr = f64{f_rail_r ? f_rail_r[lane] : 0.0};
    const auto dre = f64{ft + fd + fl + fr - logpsi[2 * lane]};
    const auto dphase = f64{-logpsi[2 * lane + 1]};
    const auto mag = f64{exp(dre)};
    g[lane] = cf{static_cast<f32>(mag * cos(dphase)), static_cast<f32>(mag * sin(dphase))};
}

__global__ auto cu_write_o_block(
    cf* base,
    i64 compact_count,
    i64 compact_offset,
    const cf* src,
    i64 src_stride,
    int slice,
    const cf* g,
    int lanes
) -> void
{
    const auto total = i64{static_cast<i64>(slice) * lanes};
    const auto grid_stride = i64{static_cast<i64>(gridDim.x) * blockDim.x};
    for (auto tid = static_cast<i64>(blockIdx.x * blockDim.x + threadIdx.x); tid < total;
         tid += grid_stride)
    {
        const auto lane = i64{tid / slice};
        const auto k = i64{tid % slice};
        const auto v = cf{src[lane * src_stride + k]};
        base[lane * compact_count + compact_offset + k] =
            qnpeps::to_cf(cuCmulf(qnpeps::to_cu(g[lane]), qnpeps::to_cu(v)));
    }
}

__global__ auto cu_u8_to_i32(int* out, const u8* in, i64 total) -> void
{
    const auto grid_stride = i64{static_cast<i64>(gridDim.x) * blockDim.x};
    for (auto tid = static_cast<i64>(blockIdx.x * blockDim.x + threadIdx.x); tid < total;
         tid += grid_stride)
        out[tid] = static_cast<int>(in[tid]);
}

__global__ auto cu_add_lambda_diag(cf* t, int ntot, f64 lambda) -> void
{
    const auto s = int{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
    if (s >= ntot) return;
    t[static_cast<i64>(s) * ntot + s].re += static_cast<f32>(lambda);
}

auto ok_layout(const Shape& sh, std::vector<i64>& off, std::vector<int>& slice) -> i64
{
    const auto lx = int{sh.lx};
    const auto ly = int{sh.ly};
    int sites_int{};
    if (lx < 1 or ly < 1 or sh.dim_bond < 1
        or not arena_int(
            {static_cast<std::uint64_t>(lx), static_cast<std::uint64_t>(ly)}, sites_int
        ))
        return 0;
    const auto sites = usize{static_cast<usize>(sites_int)};
    off.assign(sites, 0);
    slice.assign(sites, 0);
    auto acc = i64{0};
    for (auto i = int{0}; i < lx; ++i)
    {
        for (auto c = int{0}; c < ly; ++c)
        {
            int s{};
            i64 next{};
            if (not arena_int(
                    {static_cast<std::uint64_t>(bd(ly, c, sh.dim_bond)),
                     static_cast<std::uint64_t>(bd(lx, i + 1, sh.dim_bond)),
                     static_cast<std::uint64_t>(bd(ly, c + 1, sh.dim_bond)),
                     static_cast<std::uint64_t>(bd(lx, i, sh.dim_bond))},
                    s
                )
                or not arena_i64_sum(acc, s, next))
                return 0;
            off[static_cast<usize>(i) * ly + c] = acc;
            slice[static_cast<usize>(i) * ly + c] = s;
            acc = next;
        }
    }
    return acc;
}

auto emit_o_rows(
    Worker& w,
    cf* wave_base,
    i64 compact_count,
    const std::vector<i64>& compact_off,
    const std::vector<int>& compact_slice,
    const f64* logpsi,
    cf* gscratch
) -> void
{
    const auto& sh = w.sh;
    const auto lanes = int{sh.lanes};
    const auto ly = int{sh.ly};
    const auto lx = int{sh.lx};
    const auto threads = int{256};
    for (auto i = int{0}; i < lx; ++i)
    {
        for (auto c = int{0}; c < ly; ++c)
        {
            if (qn::err_state() != QNPEPS_ELOC_OK) return;
            const auto site = usize{static_cast<usize>(i) * ly + c};
            const auto slice = int{compact_slice[site]};
            auto frame = Carver{w.scratch};
            auto envblk{frame.take_product<cf>(
                {static_cast<std::uint64_t>(slice), static_cast<std::uint64_t>(lanes)}
            )};
            if (qn::err_state() != QNPEPS_ELOC_OK) return;
            const auto hl =
                BT{c > 0 ? BT{hl_site(w, i, c - 1), w.slot_h, hl_o_dims(w, i, c)} : unit3(w)};
            const auto top = BT{top_bt(w, i, c)};
            const auto bot = BT{bot_bt_h(w, i, c)};
            const auto hr =
                BT{c < ly - 1 ? BT{hr_site(w, i, c), w.slot_h, hr_o_dims(w, i, c)} : unit3(w)};
            const auto t1 = BT{bcontract(w, frame, hl, {0}, top, {0}, nullptr, 0, 0)};
            const auto t2 = BT{bcontract(w, frame, t1, {1}, bot, {0}, nullptr, 0, 0)};
            bcontract(w, frame, t2, {2, 4}, hr, {0, 2}, envblk, static_cast<i64>(slice), 0);
            if (qn::err_state() != QNPEPS_ELOC_OK) return;

            auto f_top{i > 0 ? w.f_top + static_cast<i64>(i - 1) * lanes : nullptr};
            auto f_down{i < lx - 1 ? w.f_down + static_cast<i64>(lx - i - 2) * lanes : nullptr};
            auto f_rail_l{c > 0 ? f_hl_cut(w, i, c - 1) : nullptr};
            auto f_rail_r{c < ly - 1 ? f_hr_cut(w, i, c) : nullptr};
            const auto gblocks = int{(lanes + threads - 1) / threads};
            cu_gscale<<<gblocks, threads, 0, w.la.stream()>>>(
                gscratch, f_top, f_down, f_rail_l, f_rail_r, logpsi, lanes
            );
            CUDA_CHECK(cudaGetLastError());
            const auto nwrite = i64{static_cast<i64>(slice) * lanes};
            const auto wblocks =
                int{static_cast<int>(std::min<i64>(4096, (nwrite + threads - 1) / threads))};
            cu_write_o_block<<<wblocks, threads, 0, w.la.stream()>>>(
                wave_base,
                compact_count,
                compact_off[site],
                envblk,
                static_cast<i64>(slice),
                slice,
                gscratch,
                lanes
            );
            CUDA_CHECK(cudaGetLastError());
        }
    }
}

auto site_offsets(int lx, int ly, int dim_bond, int dim_phys) -> std::vector<i64>
{
    int sites_int{};
    usize slots{};
    if (lx < 1 or ly < 1 or dim_bond < 1 or dim_phys < 1
        or not arena_int(
            {static_cast<std::uint64_t>(lx), static_cast<std::uint64_t>(ly)}, sites_int
        )
        or not arena_sum(static_cast<usize>(sites_int), 1u, slots))
        return {};
    const auto sites = usize{static_cast<usize>(sites_int)};
    auto off = std::vector<i64>(slots, 0);
    auto o = i64{0};
    for (auto r = int{0}; r < lx; ++r)
    {
        for (auto c = int{0}; c < ly; ++c)
        {
            off[static_cast<usize>(r) * ly + c] = o;
            i64 site{};
            i64 next{};
            if (not arena_slot(
                    {static_cast<std::uint64_t>(bd(ly, c, dim_bond)),
                     static_cast<std::uint64_t>(bd(lx, r + 1, dim_bond)),
                     static_cast<std::uint64_t>(bd(ly, c + 1, dim_bond)),
                     static_cast<std::uint64_t>(bd(lx, r, dim_bond)),
                     static_cast<std::uint64_t>(dim_phys)},
                    site
                )
                or not arena_i64_sum(o, site, next))
                return {};
            o = next;
        }
    }
    off[sites] = o;
    return off;
}

__global__ auto cu_transpose_samples(const u8* samples, u8* out, int lx, int ly, i64 n) -> void
{
    const auto sites = i64{static_cast<i64>(lx) * ly};
    const auto total = i64{n * sites};
    const auto grid_stride = i64{static_cast<i64>(gridDim.x) * blockDim.x};
    for (auto tid = static_cast<i64>(blockIdx.x * blockDim.x + threadIdx.x); tid < total;
         tid += grid_stride)
    {
        const auto s = i64{tid / sites};
        const auto idx_t = i64{tid % sites};
        const auto r = int{static_cast<int>(idx_t / lx)};
        const auto c = int{static_cast<int>(idx_t % lx)};
        const auto orig = i64{static_cast<i64>(c) * ly + r};
        out[tid] = samples[s * sites + orig];
    }
}

auto build_transposed_peps(
    const QnpepsElocConfig& cfg, const cf* peps, cf* peps_t, cudaStream_t stream
) -> void
{
    const auto lx = int{cfg.lx};
    const auto ly = int{cfg.ly};
    const auto D = int{cfg.dim_bond};
    const auto dp = int{cfg.dim_phys};
    const auto off = std::vector<i64>{site_offsets(lx, ly, D, dp)};
    const auto offt = std::vector<i64>{site_offsets(ly, lx, D, dp)};
    if (qn::err_state() != QNPEPS_ELOC_OK or off.empty() or offt.empty()) return;
    dt::set_stream(stream);
    for (auto r = int{0}; r < ly; ++r)
    {
        for (auto c = int{0}; c < lx; ++c)
        {
            const auto src_dim = std::vector<int>{
                bd(ly, r, D), bd(lx, c + 1, D), bd(ly, r + 1, D), bd(lx, c, D), dp
            };
            const auto src = dt::DeviceTensor{dt::view(
                reinterpret_cast<cuFloatComplex*>(const_cast<cf*>(peps))
                    + off[static_cast<usize>(c) * ly + r],
                src_dim
            )};
            dt::permute_axes(
                src,
                {3, 2, 1, 0, 4},
                false,
                reinterpret_cast<cuFloatComplex*>(peps_t) + offt[static_cast<usize>(r) * lx + c]
            );
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

}
