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

namespace qn_eloc::env
{

#line 1124 "cuda/eo/env_build.cu"

auto project_row(Worker& w, int row) -> void
{
    const auto& sh = w.sh;
    for (auto col = int{0}; col < sh.ly; ++col)
    {
        i64 slab{};
        i64 total{};
        if (not arena_slot(
                {static_cast<std::uint64_t>(bd(sh.ly, col, sh.dim_bond)),
                 static_cast<std::uint64_t>(bd(sh.lx, row + 1, sh.dim_bond)),
                 static_cast<std::uint64_t>(bd(sh.ly, col + 1, sh.dim_bond)),
                 static_cast<std::uint64_t>(bd(sh.lx, row, sh.dim_bond))},
                slab
            )
            or not arena_slot(
                {static_cast<std::uint64_t>(slab), static_cast<std::uint64_t>(sh.lanes)}, total
            ))
            return;
        const auto threads = int{256};
        const auto blocks =
            int{static_cast<int>(std::min<i64>(4096, (total + threads - 1) / threads))};
        cu_project_slab<<<blocks, threads, 0, w.la.stream()>>>(
            w.peps + w.site_off[static_cast<usize>(row) * sh.ly + col],
            w.samples,
            static_cast<i64>(sh.lx) * sh.ly,
            row * sh.ly + col,
            slab,
            proj_site(w, col),
            w.slot_proj,
            sh.lanes
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

auto build_env_row(
    Worker& w,
    int row,
    bool contract_up,
    const std::vector<int>* ks_adj,
    const std::function<cf*(int)>& adj_site,
    const std::function<cf*(int)>& out_site,
    f64* f_lane,
    std::vector<int>& ks_out
) -> void
{
    const auto& sh = w.sh;
    if (sh.density)
    {
        if (sh.lanes != 1 or not w.density)
        {
            qn::set_err(QNPEPS_ELOC_ERR_INTERNAL);
            return;
        }
        project_row(w, row);
        usize rail_slots{};
        if (not arena_sum(static_cast<usize>(sh.ly), 1u, rail_slots)) return;
        ks_out.assign(rail_slots, 1);
        auto shapes = std::vector<qn_eloc::density::SiteShape>(static_cast<usize>(sh.ly));
        auto projected = std::vector<const cf*>(static_cast<usize>(sh.ly));
        auto adjacent = std::vector<const cf*>(static_cast<usize>(sh.ly));
        auto output = std::vector<cf*>(static_cast<usize>(sh.ly));
        for (auto col = int{0}; col < sh.ly; ++col)
        {
            shapes[static_cast<usize>(col)] = qn_eloc::density::SiteShape{
                bd(sh.ly, col, sh.dim_bond),
                bd(sh.lx, row + 1, sh.dim_bond),
                bd(sh.ly, col + 1, sh.dim_bond),
                bd(sh.lx, row, sh.dim_bond)
            };
            projected[static_cast<usize>(col)] = proj_site(w, col);
            adjacent[static_cast<usize>(col)] = ks_adj ? adj_site(col) : nullptr;
            output[static_cast<usize>(col)] = out_site(col);
        }
        const auto args = qn_eloc::density::RowArgs{
            .sites = sh.ly,
            .max_bond = sh.chi,
            .cutoff = sh.density_cutoff,
            .contract_up = contract_up,
            .shapes = shapes.data(),
            .projected = projected.data(),
            .adjacent = adjacent.data(),
            .output = output.data(),
            .adjacent_ranks = ks_adj ? ks_adj->data() : nullptr,
            .output_ranks = ks_out.data(),
            .device_gauge = f_lane
        };
        const auto status =
            int{ks_adj ? qn_eloc::density::contract(*w.density, args)
                       : qn_eloc::density::boundary(*w.density, args)};
        if (status != 0) qn::set_err(status == 5 ? QNPEPS_ELOC_ERR_OOM : QNPEPS_ELOC_ERR_INTERNAL);
        return;
    }
    auto exact_factorization = bool{true};
    auto exact_rank = int{1};
    for (auto axis = int{1}; axis < std::max(sh.lx, sh.ly); ++axis)
    {
        if (exact_rank > sh.chi / sh.dim_bond)
        {
            exact_factorization = false;
            break;
        }
        exact_rank *= sh.dim_bond;
    }
    exact_factorization = exact_factorization and exact_rank <= sh.chi;
    if (rangefinder_force_nonexact_from_env()) exact_factorization = false;
    const auto lanes = int{sh.lanes};
    project_row(w, row);

    usize rail_slots{};
    if (not arena_sum(static_cast<usize>(sh.ly), 1u, rail_slots)) return;
    ks_out.assign(rail_slots, 1);
    const auto threads = int{256};
    const auto launch = [&](i64 n) -> int
    {
        i64 total{};
        if (n < 0
            or not arena_slot(
                {static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(lanes)}, total
            ))
            return 0;
        return static_cast<int>(std::min<i64>(4096, (total + threads - 1) / threads));
    };

    cu_fill_first_one<<<launch(1), threads, 0, w.la.stream()>>>(
        w.rroll.p, w.rroll.stride, 1, lanes
    );
    CUDA_CHECK(cudaGetLastError());

    auto previous_rank = int{1};
    for (auto col = int{0}; col < sh.ly; ++col)
    {
        if (qn::err_state() != QNPEPS_ELOC_OK) return;
        const auto wdim = int{bd(sh.ly, col, sh.dim_bond)};
        const auto sdim = int{bd(sh.lx, row + 1, sh.dim_bond)};
        const auto edim = int{bd(sh.ly, col + 1, sh.dim_bond)};
        const auto ndim = int{bd(sh.lx, row, sh.dim_bond)};
        const auto vdim = int{contract_up ? sdim : ndim};
        const auto cdim = int{contract_up ? ndim : sdim};
        const auto lp = int{ks_adj ? (*ks_adj)[static_cast<usize>(col)] : 1};
        const auto rp = int{ks_adj ? (*ks_adj)[static_cast<usize>(col) + 1] : 1};

        cf* combined{};
        int a_dim{};
        int b_dim{};
        int projected_rows{};
        int adjacent_cols{};
        int contraction_cols{};
        if (not arena_int({static_cast<std::uint64_t>(wdim), static_cast<std::uint64_t>(lp)}, a_dim)
            or not arena_int(
                {static_cast<std::uint64_t>(edim), static_cast<std::uint64_t>(rp)}, b_dim
            )
            or not arena_int(
                {static_cast<std::uint64_t>(wdim),
                 static_cast<std::uint64_t>(edim),
                 static_cast<std::uint64_t>(vdim)},
                projected_rows
            )
            or not arena_int(
                {static_cast<std::uint64_t>(lp), static_cast<std::uint64_t>(rp)}, adjacent_cols
            )
            or not arena_int(
                {static_cast<std::uint64_t>(vdim), static_cast<std::uint64_t>(b_dim)},
                contraction_cols
            ))
            return;
        if (ks_adj)
        {
            device_permute(
                w.la,
                w.permutation_index_maps,
                {.dst = w.tmp_a,
                 .src = {proj_site(w, col), w.slot_proj},
                 .dims_in = {wdim, sdim, edim, ndim},
                 .perm = contract_up ? std::vector<int>{0, 2, 1, 3} : std::vector<int>{0, 2, 3, 1},
                 .batch = lanes}
            );
            device_permute(
                w.la,
                w.permutation_index_maps,
                {.dst = w.tmp_b,
                 .src = {adj_site(col), w.slot_env},
                 .dims_in = {lp, cdim, rp},
                 .perm = {1, 0, 2},
                 .batch = lanes}
            );
            w.la.matmul_batched(
                CuMatrixBatchedCF32Const{
                    reinterpret_cast<const cuFloatComplex*>(w.tmp_a.p),
                    w.tmp_a.stride,
                    projected_rows,
                    cdim
                },
                CuMatrixBatchedCF32Const{
                    reinterpret_cast<const cuFloatComplex*>(w.tmp_b.p),
                    w.tmp_b.stride,
                    cdim,
                    adjacent_cols
                },
                CuMatrixBatchedCF32{
                    reinterpret_cast<cuFloatComplex*>(w.tmp_c.p),
                    w.tmp_c.stride,
                    projected_rows,
                    adjacent_cols
                },
                lanes
            );
            device_permute(
                w.la,
                w.permutation_index_maps,
                {.dst = w.tmp_a,
                 .src = w.tmp_c,
                 .dims_in = {wdim, edim, vdim, lp, rp},
                 .perm = {0, 3, 2, 1, 4},
                 .batch = lanes}
            );
            combined = w.tmp_a.p;
        }
        else
        {
            if (contract_up)
            {
                combined = proj_site(w, col);
            }
            else
            {
                device_permute(
                    w.la,
                    w.permutation_index_maps,
                    {.dst = w.tmp_a,
                     .src = {proj_site(w, col), w.slot_proj},
                     .dims_in = {wdim, edim, ndim},
                     .perm = {0, 2, 1},
                     .batch = lanes}
                );
                combined = w.tmp_a.p;
            }
        }
        const auto comb_stride = i64{(not ks_adj and contract_up) ? w.slot_proj : w.tmp_a.stride};

        w.la.matmul_batched(
            CuMatrixBatchedCF32Const{
                reinterpret_cast<const cuFloatComplex*>(w.rroll.p),
                w.rroll.stride,
                previous_rank,
                a_dim
            },
            CuMatrixBatchedCF32Const{
                reinterpret_cast<const cuFloatComplex*>(combined),
                comb_stride,
                a_dim,
                contraction_cols
            },
            CuMatrixBatchedCF32{
                reinterpret_cast<cuFloatComplex*>(w.tmp_b.p),
                w.tmp_b.stride,
                previous_rank,
                contraction_cols
            },
            lanes
        );

        int rangefinder_rows{};
        if (not arena_int(
                {static_cast<std::uint64_t>(previous_rank), static_cast<std::uint64_t>(vdim)},
                rangefinder_rows
            ))
            return;
        const auto next_rank = int{std::max(1, std::min({sh.chi, rangefinder_rows, b_dim}))};
        const auto call_stride = usize{2 * static_cast<usize>(lanes)};
        const auto rangefinder_call = int{w.rangefinder_call++};
        const auto log_offset = usize{static_cast<usize>(rangefinder_call) * call_stride};
        if (log_offset + call_stride > w.failure_log_count)
        {
            qn::set_err(QNPEPS_ELOC_ERR_INTERNAL);
            return;
        }
        auto robust_scratch{
            w.qr_scratch ? static_cast<void*>(w.qr_scratch) : static_cast<void*>(w.tmp_a.p)
        };
        usize fallback_scratch_bytes{};
        if (not w.qr_scratch
            and not arena_product(
                {static_cast<std::uint64_t>(w.tmp_a.stride),
                 static_cast<std::uint64_t>(lanes),
                 sizeof(cf)},
                fallback_scratch_bytes
            ))
            return;
        const auto robust_scratch_bytes =
            usize{w.qr_scratch ? w.qr_scratch_bytes : fallback_scratch_bytes};
        const auto sketch_width = int{rangefinder_sketch_width(rangefinder_rows, b_dim, next_rank)};
        batched_rangefinder(
            w.la,
            {w.tmp_b.p, w.tmp_b.stride, rangefinder_rows, b_dim},
            next_rank,
            exact_factorization,
            omega_for(w, b_dim, sketch_width),
            {out_site(col), w.slot_env, rangefinder_rows, next_rank},
            {w.rnext.p, w.rnext.stride, next_rank, b_dim},
            lanes,
            w.sketch,
            w.proj_rf,
            w.gram,
            w.gram_ptrs,
            w.sketch_ptrs,
            w.info,
            w.fail_flag,
            w.failure_log + log_offset,
            w.fallback_info ? w.fallback_info + log_offset : nullptr,
            robust_scratch,
            robust_scratch_bytes
        );
        int normalized_elements{};
        if (not arena_int(
                {static_cast<std::uint64_t>(next_rank), static_cast<std::uint64_t>(b_dim)},
                normalized_elements
            ))
            return;
        cu_normalize_log<<<lanes, threads, 0, w.la.stream()>>>(
            w.rnext.p, normalized_elements, w.rnext.stride, f_lane, lanes, nullptr
        );
        CUDA_CHECK(cudaGetLastError());
        std::swap(w.rroll.p, w.rnext.p);
        std::swap(w.rroll.stride, w.rnext.stride);
        ks_out[static_cast<usize>(col) + 1] = next_rank;
        previous_rank = next_rank;
    }

    const auto last = int{sh.ly - 1};
    int n_last_int{};
    if (not arena_int(
            {static_cast<std::uint64_t>(ks_out[static_cast<usize>(last)]),
             static_cast<std::uint64_t>(
                 contract_up ? bd(sh.lx, row + 1, sh.dim_bond) : bd(sh.lx, row, sh.dim_bond)
             ),
             static_cast<std::uint64_t>(ks_out[static_cast<usize>(last) + 1])},
            n_last_int
        ))
        return;
    const auto n_last = i64{n_last_int};
    cu_scale_last<<<launch(n_last), threads, 0, w.la.stream()>>>(
        out_site(last), w.slot_env, n_last, w.rroll.p, w.rroll.stride, lanes
    );
    CUDA_CHECK(cudaGetLastError());
}

auto compute_logpsi(Worker& w, f64* logpsi_out, bool full_top) -> void
{
    const auto& sh = w.sh;
    const auto lanes = int{sh.lanes};
    const auto lanes_u = usize{static_cast<usize>(lanes)};
    const auto threads = int{256};
    usize vertical_count{};
    usize vertical_bytes{};
    usize overlap_bytes{};
    if (not arena_product(
            {static_cast<std::uint64_t>(lanes), static_cast<std::uint64_t>(sh.lx - 1)},
            vertical_count
        )
        or not arena_product({sizeof(f64), vertical_count}, vertical_bytes)
        or not arena_product({sizeof(f64), lanes_u}, overlap_bytes))
        return;

    CUDA_CHECK(cudaMemsetAsync(w.f_down, 0, vertical_bytes, w.la.stream()));
    CUDA_CHECK(cudaMemsetAsync(w.f_top, 0, vertical_bytes, w.la.stream()));
    CUDA_CHECK(cudaMemsetAsync(w.f_ov, 0, overlap_bytes, w.la.stream()));

    for (auto k = int{0}; k < sh.lx - 1; ++k)
    {
        if (qn::err_state() != QNPEPS_ELOC_OK) return;
        auto f_lane{w.f_down + static_cast<i64>(k) * lanes};
        if (k > 0)
        {
            CUDA_CHECK(cudaMemcpyAsync(
                f_lane,
                w.f_down + static_cast<i64>(k - 1) * lanes,
                overlap_bytes,
                cudaMemcpyDeviceToDevice,
                w.la.stream()
            ));
        }
        const auto row = int{sh.lx - 1 - k};
        auto adj{k > 0 ? &w.ks_down[static_cast<usize>(k - 1)] : nullptr};
        build_env_row(
            w,
            row,
            false,
            adj,
            [&](int col) { return k > 0 ? eb_site(w, k - 1, col) : nullptr; },
            [&](int col) { return eb_site(w, k, col); },
            f_lane,
            w.ks_down[static_cast<usize>(k)]
        );
    }

    auto pos = int{(sh.lx - 1) / 2};
    if (pos < 1) pos = 1;
    const auto ti = int{pos - 1};
    const auto di = int{sh.lx - pos - 1};
    const auto top_rows = int{full_top ? sh.lx - 1 : ti + 1};

    auto cur = int{0};
    for (auto k = int{0}; k < top_rows; ++k)
    {
        if (qn::err_state() != QNPEPS_ELOC_OK) return;
        auto f_lane{w.f_top + static_cast<i64>(k) * lanes};
        if (k > 0)
        {
            CUDA_CHECK(cudaMemcpyAsync(
                f_lane,
                w.f_top + static_cast<i64>(k - 1) * lanes,
                overlap_bytes,
                cudaMemcpyDeviceToDevice,
                w.la.stream()
            ));
        }
        const auto nxt = int{1 - cur};
        auto adj{k > 0 ? &w.ks_top[static_cast<usize>(k - 1)] : nullptr};
        const auto kk = int{k};
        const auto adj_site = [&w, full_top, kk, cur](int col) -> cf*
        {
            if (kk == 0) return nullptr;
            return full_top ? etf_site(w, kk - 1, col) : et_site(w, cur, col);
        };
        const auto out_site = [&w, full_top, kk, nxt](int col) -> cf*
        { return full_top ? etf_site(w, kk, col) : et_site(w, nxt, col); };
        build_env_row(w, k, true, adj, adj_site, out_site, f_lane, w.ks_top[static_cast<usize>(k)]);
        cur = nxt;
    }

    const auto& kt{w.ks_top[static_cast<usize>(ti)]};
    const auto& kd{w.ks_down[static_cast<usize>(di)]};
    const auto ov_site = [&w, full_top, ti](int col) -> cf*
    { return full_top ? etf_site(w, ti, col) : et_site(w, static_cast<int>(ti % 2 == 0), col); };
    cu_fill_first_one<<<1, threads, 0, w.la.stream()>>>(w.rroll.p, w.rroll.stride, 1, lanes);
    CUDA_CHECK(cudaGetLastError());
    for (auto col = int{0}; col < sh.ly; ++col)
    {
        if (qn::err_state() != QNPEPS_ELOC_OK) return;
        const auto ktl = int{kt[static_cast<usize>(col)]};
        const auto ktr = int{kt[static_cast<usize>(col) + 1]};
        const auto kdl = int{kd[static_cast<usize>(col)]};
        const auto kdr = int{kd[static_cast<usize>(col) + 1]};
        const auto vdim = int{bd(sh.lx, pos, sh.dim_bond)};
        int top_width{};
        int bottom_width{};
        int overlap_elements{};
        if (not arena_int(
                {static_cast<std::uint64_t>(vdim), static_cast<std::uint64_t>(ktr)}, top_width
            )
            or not arena_int(
                {static_cast<std::uint64_t>(kdl), static_cast<std::uint64_t>(vdim)}, bottom_width
            )
            or not arena_int(
                {static_cast<std::uint64_t>(kdr), static_cast<std::uint64_t>(ktr)}, overlap_elements
            ))
            return;
        w.la.matmul_batched(
            CuMatrixBatchedCF32Const{
                reinterpret_cast<const cuFloatComplex*>(w.rroll.p), w.rroll.stride, kdl, ktl
            },
            CuMatrixBatchedCF32Const{
                reinterpret_cast<const cuFloatComplex*>(ov_site(col)), w.slot_env, ktl, top_width
            },
            CuMatrixBatchedCF32{
                reinterpret_cast<cuFloatComplex*>(w.tmp_a.p), w.tmp_a.stride, kdl, top_width
            },
            lanes
        );
        w.la.matmul_batched(
            CuMatrixBatchedCF32Const{
                reinterpret_cast<const cuFloatComplex*>(eb_site(w, di, col)),
                w.slot_env,
                bottom_width,
                kdr
            },
            CuMatrixBatchedCF32Const{
                reinterpret_cast<const cuFloatComplex*>(w.tmp_a.p),
                w.tmp_a.stride,
                bottom_width,
                ktr
            },
            CuMatrixBatchedCF32{
                reinterpret_cast<cuFloatComplex*>(w.rnext.p), w.rnext.stride, kdr, ktr
            },
            lanes,
            {.op_a = BlasOp::trans, .op_b = BlasOp::none}
        );
        cu_normalize_log<<<lanes, threads, 0, w.la.stream()>>>(
            w.rnext.p, overlap_elements, w.rnext.stride, w.f_ov, lanes, nullptr
        );
        CUDA_CHECK(cudaGetLastError());
        std::swap(w.rroll.p, w.rnext.p);
        std::swap(w.rroll.stride, w.rnext.stride);
    }

    const auto blocks = int{(lanes + threads - 1) / threads};
    cu_log_finish<<<blocks, threads, 0, w.la.stream()>>>(
        w.rroll.p,
        w.rroll.stride,
        w.f_top + static_cast<i64>(ti) * lanes,
        w.f_down + static_cast<i64>(di) * lanes,
        w.f_ov,
        logpsi_out,
        lanes
    );
    CUDA_CHECK(cudaGetLastError());
}

}
