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

namespace qn_eloc::env
{

#line 1593 "cuda/eo/env_build.cu"

struct BT
{
    cf* p{};
    i64 stride{};
    std::vector<int> dim{};
};

inline auto bt_elems(const std::vector<int>& dim) -> i64
{
    auto n = i64{1};
    for (const int d : dim)
    {
        if (d < 0
            or not arena_slot({static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(d)}, n))
            return 0;
    }
    return n;
}

auto bcontract(
    Worker& w,
    Carver& frame,
    const BT& A,
    const std::vector<int>& ca,
    const BT& B,
    const std::vector<int>& cb,
    cf* out_p,
    i64 out_stride,
    int dim_batch = 0
) -> BT
{
    const auto lanes = int{dim_batch > 0 ? dim_batch : w.sh.lanes};
    const auto ra = int{static_cast<int>(A.dim.size())};
    const auto rb = int{static_cast<int>(B.dim.size())};
    auto is_ca = std::vector<char>(static_cast<usize>(ra), 0);
    auto is_cb = std::vector<char>(static_cast<usize>(rb), 0);
    for (const int ax : ca)
        is_ca[static_cast<usize>(ax)] = 1;
    for (const int ax : cb)
        is_cb[static_cast<usize>(ax)] = 1;
    std::vector<int> free_a{};
    std::vector<int> free_b{};
    for (auto ax = int{0}; ax < ra; ++ax)
        if (not is_ca[static_cast<usize>(ax)]) free_a.push_back(ax);
    for (auto ax = int{0}; ax < rb; ++ax)
        if (not is_cb[static_cast<usize>(ax)]) free_b.push_back(ax);
    auto perm_a = std::vector<int>{free_a};
    perm_a.insert(perm_a.end(), ca.begin(), ca.end());
    auto perm_b = std::vector<int>{cb};
    perm_b.insert(perm_b.end(), free_b.begin(), free_b.end());
    auto M = i64{1};
    auto K = i64{1};
    auto N = i64{1};
    for (const int ax : free_a)
    {
        if (not arena_slot(
                {static_cast<std::uint64_t>(M),
                 static_cast<std::uint64_t>(A.dim[static_cast<usize>(ax)])},
                M
            ))
            return {};
    }
    for (const int ax : ca)
    {
        if (not arena_slot(
                {static_cast<std::uint64_t>(K),
                 static_cast<std::uint64_t>(A.dim[static_cast<usize>(ax)])},
                K
            ))
            return {};
    }
    for (const int ax : free_b)
    {
        if (not arena_slot(
                {static_cast<std::uint64_t>(N),
                 static_cast<std::uint64_t>(B.dim[static_cast<usize>(ax)])},
                N
            ))
            return {};
    }
    i64 mk{};
    i64 kn{};
    i64 mn{};
    if (not arena_slot({static_cast<std::uint64_t>(M), static_cast<std::uint64_t>(K)}, mk)
        or not arena_slot({static_cast<std::uint64_t>(K), static_cast<std::uint64_t>(N)}, kn)
        or not arena_slot({static_cast<std::uint64_t>(M), static_cast<std::uint64_t>(N)}, mn))
        return {};
    int m_int{};
    int contraction_extent{};
    int n_int{};
    if (not arena_int({static_cast<std::uint64_t>(M)}, m_int)
        or not arena_int({static_cast<std::uint64_t>(K)}, contraction_extent)
        or not arena_int({static_cast<std::uint64_t>(N)}, n_int))
        return {};

    std::vector<int> res{};
    for (const int ax : free_a)
        res.push_back(A.dim[static_cast<usize>(ax)]);
    for (const int ax : free_b)
        res.push_back(B.dim[static_cast<usize>(ax)]);
    if (res.empty()) res.push_back(1);

    auto out{out_p};
    auto stride = i64{out_stride};
    if (out == nullptr)
    {
        stride = mn;
        out = frame.take_product<cf>(
            {static_cast<std::uint64_t>(stride), static_cast<std::uint64_t>(lanes)}
        );
    }
    if (qn::err_state() != QNPEPS_ELOC_OK) return BT{out, stride, res};

    const auto is_identity = [](const std::vector<int>& perm) -> bool
    {
        for (auto ax = usize{0}; ax < perm.size(); ++ax)
            if (perm[ax] != static_cast<int>(ax)) return false;
        return true;
    };
    auto a_leading_block = bool{not ca.empty()};
    for (auto i = usize{0}; i < ca.size(); ++i)
        if (ca[i] != static_cast<int>(i)) a_leading_block = false;
    const auto a_read_trans = bool{a_leading_block and not is_identity(perm_a)};
    auto b_trailing_block = bool{not cb.empty()};
    const auto b_trailing_begin = int{rb - static_cast<int>(cb.size())};
    for (auto i = usize{0}; i < cb.size(); ++i)
        if (cb[i] != b_trailing_begin + static_cast<int>(i)) b_trailing_block = false;
    const auto b_read_trans = bool{b_trailing_block and not is_identity(perm_b)};

    auto sub = Carver{frame};
    auto pa{A.p};
    auto pa_stride = i64{A.stride};
    if (not a_read_trans and not is_identity(perm_a))
    {
        auto dst{sub.take_product<cf>(
            {static_cast<std::uint64_t>(mk), static_cast<std::uint64_t>(lanes)}
        )};
        if (qn::err_state() != QNPEPS_ELOC_OK) return BT{out, stride, res};
        device_permute(
            w.la,
            w.permutation_index_maps,
            {.dst = {dst, mk},
             .src = {A.p, A.stride},
             .dims_in = A.dim,
             .perm = perm_a,
             .batch = lanes,
             .kind = "perm_a",
             .m = m_int,
             .n = n_int,
             .k = contraction_extent}
        );
        pa = dst;
        pa_stride = mk;
    }
    auto pb{B.p};
    auto pb_stride = i64{B.stride};
    if (not b_read_trans and not is_identity(perm_b))
    {
        auto dst{sub.take_product<cf>(
            {static_cast<std::uint64_t>(kn), static_cast<std::uint64_t>(lanes)}
        )};
        if (qn::err_state() != QNPEPS_ELOC_OK) return BT{out, stride, res};
        device_permute(
            w.la,
            w.permutation_index_maps,
            {.dst = {dst, kn},
             .src = {B.p, B.stride},
             .dims_in = B.dim,
             .perm = perm_b,
             .batch = lanes,
             .kind = "perm_b",
             .m = m_int,
             .n = n_int,
             .k = contraction_extent}
        );
        pb = dst;
        pb_stride = kn;
    }
    if (a_read_trans and b_read_trans)
    {
        w.la.matmul_batched(
            CuMatrixBatchedCF32Const{
                reinterpret_cast<const cuFloatComplex*>(pa), pa_stride, contraction_extent, m_int
            },
            CuMatrixBatchedCF32Const{
                reinterpret_cast<const cuFloatComplex*>(pb), pb_stride, n_int, contraction_extent
            },
            CuMatrixBatchedCF32{reinterpret_cast<cuFloatComplex*>(out), stride, m_int, n_int},
            lanes,
            {.op_a = BlasOp::trans, .op_b = BlasOp::trans}
        );
    }
    else if (a_read_trans)
    {
        w.la.matmul_batched(
            CuMatrixBatchedCF32Const{
                reinterpret_cast<const cuFloatComplex*>(pa), pa_stride, contraction_extent, m_int
            },
            CuMatrixBatchedCF32Const{
                reinterpret_cast<const cuFloatComplex*>(pb), pb_stride, contraction_extent, n_int
            },
            CuMatrixBatchedCF32{reinterpret_cast<cuFloatComplex*>(out), stride, m_int, n_int},
            lanes,
            {.op_a = BlasOp::trans, .op_b = BlasOp::none}
        );
    }
    else if (b_read_trans)
    {
        w.la.matmul_batched(
            CuMatrixBatchedCF32Const{
                reinterpret_cast<const cuFloatComplex*>(pa), pa_stride, m_int, contraction_extent
            },
            CuMatrixBatchedCF32Const{
                reinterpret_cast<const cuFloatComplex*>(pb), pb_stride, n_int, contraction_extent
            },
            CuMatrixBatchedCF32{reinterpret_cast<cuFloatComplex*>(out), stride, m_int, n_int},
            lanes,
            {.op_a = BlasOp::none, .op_b = BlasOp::trans}
        );
    }
    else
        w.la.matmul_batched(
            CuMatrixBatchedCF32Const{
                reinterpret_cast<const cuFloatComplex*>(pa), pa_stride, m_int, contraction_extent
            },
            CuMatrixBatchedCF32Const{
                reinterpret_cast<const cuFloatComplex*>(pb), pb_stride, contraction_extent, n_int
            },
            CuMatrixBatchedCF32{reinterpret_cast<cuFloatComplex*>(out), stride, m_int, n_int},
            lanes
        );
    return BT{out, stride, res};
}

auto fold_left(
    Worker& w,
    Carver frame,
    const BT& h,
    const BT& top,
    const BT& pr,
    const BT& bot,
    cf* out,
    i64 out_stride,
    int dim_batch = 0,
    bool projected_packed = false
) -> std::vector<int>
{
    const auto projected_axes =
        std::vector<int>{projected_packed ? std::vector<int>{0, 1} : std::vector<int>{0, 3}};
    const auto t1 = BT{bcontract(w, frame, h, {0}, top, {0}, nullptr, 0, dim_batch)};
    const auto t2 = BT{bcontract(w, frame, t1, {0, 2}, pr, projected_axes, nullptr, 0, dim_batch)};
    return bcontract(w, frame, t2, {0, 2}, bot, {0, 1}, out, out_stride, dim_batch).dim;
}

auto fold_right(
    Worker& w,
    Carver frame,
    const BT& h,
    const BT& top,
    const BT& pr,
    const BT& bot,
    cf* out,
    i64 out_stride
) -> std::vector<int>
{
    const auto t1 = BT{bcontract(w, frame, h, {0}, top, {2}, nullptr, 0, 0)};
    const auto t2 = BT{bcontract(w, frame, t1, {0, 3}, pr, {2, 3}, nullptr, 0, 0)};
    return bcontract(w, frame, t2, {0, 3}, bot, {2, 1}, out, out_stride, 0).dim;
}

auto fold_left4(
    Worker& w,
    Carver frame,
    const BT& h,
    const BT& top,
    const BT& pri,
    const BT& prj,
    const BT& bot,
    cf* out,
    i64 out_stride,
    int dim_batch = 0,
    bool projected_packed = false
) -> std::vector<int>
{
    const auto projected_axes =
        std::vector<int>{projected_packed ? std::vector<int>{0, 1} : std::vector<int>{0, 3}};
    const auto t1 = BT{bcontract(w, frame, h, {0}, top, {0}, nullptr, 0, dim_batch)};
    const auto t2 = BT{bcontract(w, frame, t1, {0, 3}, pri, projected_axes, nullptr, 0, dim_batch)};
    const auto t3 = BT{bcontract(w, frame, t2, {0, 3}, prj, projected_axes, nullptr, 0, dim_batch)};
    return bcontract(w, frame, t3, {0, 3}, bot, {0, 1}, out, out_stride, dim_batch).dim;
}

auto fold_right4(
    Worker& w,
    Carver frame,
    const BT& h,
    const BT& top,
    const BT& pri,
    const BT& prj,
    const BT& bot,
    cf* out,
    i64 out_stride,
    int dim_batch = 0,
    bool bottom_packed = false
) -> std::vector<int>
{
    const auto bottom_axes =
        std::vector<int>{bottom_packed ? std::vector<int>{0, 1} : std::vector<int>{2, 1}};
    const auto t1 = BT{bcontract(w, frame, h, {0}, top, {2}, nullptr, 0, dim_batch)};
    const auto t2 = BT{bcontract(w, frame, t1, {0, 4}, pri, {2, 3}, nullptr, 0, dim_batch)};
    const auto t3 = BT{bcontract(w, frame, t2, {0, 4}, prj, {2, 3}, nullptr, 0, dim_batch)};
    return bcontract(w, frame, t3, {0, 4}, bot, bottom_axes, out, out_stride, dim_batch).dim;
}

auto project_column_into(Worker& w, int row, int col, cf* out, i64 out_stride) -> void
{
    const auto& sh = w.sh;
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
    const auto blocks = int{static_cast<int>(std::min<i64>(4096, (total + threads - 1) / threads))};
    cu_project_slab<<<blocks, threads, 0, w.la.stream()>>>(
        w.peps + w.site_off[static_cast<usize>(row) * sh.ly + col],
        w.samples,
        static_cast<i64>(sh.lx) * sh.ly,
        row * sh.ly + col,
        slab,
        out,
        out_stride,
        sh.lanes
    );
    CUDA_CHECK(cudaGetLastError());
}

auto project_column_dual_packed(
    Worker& w,
    int row,
    int col,
    cf* native_out,
    i64 native_stride,
    cf* packed_out,
    i64 packed_stride
) -> BT
{
    const auto& sh = w.sh;
    const auto dims = std::vector<int>{
        bd(sh.ly, col, sh.dim_bond),
        bd(sh.lx, row + 1, sh.dim_bond),
        bd(sh.ly, col + 1, sh.dim_bond),
        bd(sh.lx, row, sh.dim_bond)
    };
    const auto perm = std::vector<int>{0, 3, 1, 2};
    const auto slab = i64{bt_elems(dims)};
    auto destination_indices{w.permutation_index_maps.get_inverse(dims, perm)};
    i64 total{};
    if (qn::err_state() != QNPEPS_ELOC_OK or slab < 1 or not destination_indices
        or not arena_slot(
            {static_cast<std::uint64_t>(slab), static_cast<std::uint64_t>(sh.lanes)}, total
        ))
        return {};
    const auto threads = int{256};
    const auto blocks = int{static_cast<int>(std::min<i64>(4096, (total + threads - 1) / threads))};
    cu_project_slab_dual_packed<<<blocks, threads, 0, w.la.stream()>>>(
        w.peps + w.site_off[static_cast<usize>(row) * sh.ly + col],
        w.samples,
        static_cast<i64>(sh.lx) * sh.ly,
        row * sh.ly + col,
        slab,
        destination_indices,
        native_out,
        native_stride,
        packed_out,
        packed_stride,
        sh.lanes
    );
    CUDA_CHECK(cudaGetLastError());
    return BT{packed_out, packed_stride, permuted_dims(dims, perm)};
}

auto project_row_into(Worker& w, int row, cf* base, i64 slot) -> void
{
    for (auto col = int{0}; col < w.sh.ly; ++col)
        project_column_into(w, row, col, base + static_cast<i64>(col) * slot * w.sh.lanes, slot);
}

inline auto hl_site(Worker& w, int row, int k) -> cf*
{
    return w.hl.p + (static_cast<i64>(row) * (w.sh.ly - 1) + k) * w.slot_h * w.sh.lanes;
}
inline auto hr_site(Worker& w, int row, int k) -> cf*
{
    return w.hr.p + (static_cast<i64>(row) * (w.sh.ly - 1) + k) * w.slot_h * w.sh.lanes;
}
inline auto fbl_site(Worker& w, int row, int k) -> cf*
{
    return w.fbl.p + (static_cast<i64>(row) * (w.sh.ly - 1) + k) * w.slot_fb * w.sh.lanes;
}
inline auto fbr_site(Worker& w, int row, int k) -> cf*
{
    return w.fbr.p + (static_cast<i64>(row) * (w.sh.ly - 1) + k) * w.slot_fb * w.sh.lanes;
}

inline auto top_dims(Worker& w, int i, int c) -> std::vector<int>
{
    return {
        w.ks_full[static_cast<usize>(i - 1)][static_cast<usize>(c)],
        bd(w.sh.lx, i, w.sh.dim_bond),
        w.ks_full[static_cast<usize>(i - 1)][static_cast<usize>(c) + 1]
    };
}
inline auto bot_dims_h(Worker& w, int i, int c) -> std::vector<int>
{
    const auto k = int{w.sh.lx - i - 2};
    return {
        w.ks_down[static_cast<usize>(k)][static_cast<usize>(c)],
        bd(w.sh.lx, i + 1, w.sh.dim_bond),
        w.ks_down[static_cast<usize>(k)][static_cast<usize>(c) + 1]
    };
}
inline auto bot_dims_fb(Worker& w, int i, int c) -> std::vector<int>
{
    const auto k = int{w.sh.lx - i - 3};
    return {
        w.ks_down[static_cast<usize>(k)][static_cast<usize>(c)],
        bd(w.sh.lx, i + 2, w.sh.dim_bond),
        w.ks_down[static_cast<usize>(k)][static_cast<usize>(c) + 1]
    };
}
inline auto proj_dims(Worker& w, int i, int c) -> std::vector<int>
{
    return {
        bd(w.sh.ly, c, w.sh.dim_bond),
        bd(w.sh.lx, i + 1, w.sh.dim_bond),
        bd(w.sh.ly, c + 1, w.sh.dim_bond),
        bd(w.sh.lx, i, w.sh.dim_bond)
    };
}
inline auto unit3(Worker& w) -> BT
{
    return {w.unit, 1, {1, 1, 1}};
}
inline auto unit4(Worker& w) -> BT
{
    return {w.unit, 1, {1, 1, 1, 1}};
}

auto rail_normalize(Worker& w, cf* rail, i64 n, i64 slot, f64* f_cut, const f64* f_prev) -> void
{
    const auto lanes = int{w.sh.lanes};
    int elements{};
    usize lane_bytes{};
    if (n < 1 or not arena_int({static_cast<std::uint64_t>(n)}, elements)
        or not arena_product({sizeof(f64), static_cast<std::uint64_t>(lanes)}, lane_bytes))
        return;
    if (f_prev)
    {
        CUDA_CHECK(
            cudaMemcpyAsync(f_cut, f_prev, lane_bytes, cudaMemcpyDeviceToDevice, w.la.stream())
        );
    }
    else
        CUDA_CHECK(cudaMemsetAsync(f_cut, 0, lane_bytes, w.la.stream()));
    cu_normalize_log<<<lanes, 256, 0, w.la.stream()>>>(rail, elements, slot, f_cut, lanes, nullptr);
    CUDA_CHECK(cudaGetLastError());
}

inline auto f_hl_cut(Worker& w, int row, int cut) -> f64*
{
    return w.f_hl + (static_cast<i64>(row) * (w.sh.ly - 1) + cut) * w.sh.lanes;
}
inline auto f_hr_cut(Worker& w, int row, int cut) -> f64*
{
    return w.f_hr + (static_cast<i64>(row) * (w.sh.ly - 1) + cut) * w.sh.lanes;
}
inline auto f_fbl_cut(Worker& w, int row, int cut) -> f64*
{
    return w.f_fbl + (static_cast<i64>(row) * (w.sh.ly - 1) + cut) * w.sh.lanes;
}
inline auto f_fbr_cut(Worker& w, int row, int cut) -> f64*
{
    return w.f_fbr + (static_cast<i64>(row) * (w.sh.ly - 1) + cut) * w.sh.lanes;
}

auto build_h_rails(Worker& w, int i, bool dual_projected = false) -> void
{
    const auto& sh = w.sh;
    const auto ly = int{sh.ly};
    const auto top = [&](int c) -> BT
    { return (i > 0) ? BT{etf_site(w, i - 1, c), w.slot_env, top_dims(w, i, c)} : unit3(w); };
    const auto bot = [&](int c) -> BT
    {
        return (i < sh.lx - 1) ? BT{eb_site(w, sh.lx - i - 2, c), w.slot_env, bot_dims_h(w, i, c)}
                               : unit3(w);
    };
    const auto pr = [&](int c) -> BT { return {proj_site(w, c), w.slot_proj, proj_dims(w, i, c)}; };

    auto h = BT{unit3(w)};
    for (auto c = int{0}; c <= ly - 2; ++c)
    {
        if (qn::err_state() != QNPEPS_ELOC_OK) return;
        auto frame = Carver{w.scratch};
        auto projected = BT{pr(c)};
        if (dual_projected)
        {
            auto packed{frame.take_product<cf>(
                {static_cast<std::uint64_t>(w.slot_proj), static_cast<std::uint64_t>(sh.lanes)}
            )};
            if (qn::err_state() != QNPEPS_ELOC_OK) return;
            projected = project_column_dual_packed(
                w, i, c, proj_site(w, c), w.slot_proj, packed, w.slot_proj
            );
        }
        const auto d = std::vector<int>{fold_left(
            w, frame, h, top(c), projected, bot(c), hl_site(w, i, c), w.slot_h, 0, dual_projected
        )};
        h = BT{hl_site(w, i, c), w.slot_h, d};
        rail_normalize(
            w,
            h.p,
            bt_elems(d),
            w.slot_h,
            f_hl_cut(w, i, c),
            c > 0 ? f_hl_cut(w, i, c - 1) : nullptr
        );
    }
    if (dual_projected) project_column_into(w, i, ly - 1, proj_site(w, ly - 1), w.slot_proj);
    auto r = BT{unit3(w)};
    for (auto c = int{ly - 1}; c >= 1; --c)
    {
        if (qn::err_state() != QNPEPS_ELOC_OK) return;
        const auto d = std::vector<int>{
            fold_right(w, w.scratch, r, top(c), pr(c), bot(c), hr_site(w, i, c - 1), w.slot_h)
        };
        r = BT{hr_site(w, i, c - 1), w.slot_h, d};
        rail_normalize(
            w,
            r.p,
            bt_elems(d),
            w.slot_h,
            f_hr_cut(w, i, c - 1),
            c < ly - 1 ? f_hr_cut(w, i, c) : nullptr
        );
    }
}

auto build_fb_rails(Worker& w, int i, bool dual_projected = false) -> void
{
    const auto& sh = w.sh;
    const auto ly = int{sh.ly};
    const auto top = [&](int c) -> BT
    { return (i > 0) ? BT{etf_site(w, i - 1, c), w.slot_env, top_dims(w, i, c)} : unit3(w); };
    const auto bot = [&](int c) -> BT
    {
        return (i < sh.lx - 2) ? BT{eb_site(w, sh.lx - i - 3, c), w.slot_env, bot_dims_fb(w, i, c)}
                               : unit3(w);
    };
    const auto pri = [&](int c) -> BT
    { return {proj_site(w, c), w.slot_proj, proj_dims(w, i, c)}; };
    const auto prj = [&](int c) -> BT
    {
        return {
            w.proj2.p + static_cast<i64>(c) * w.slot_proj * sh.lanes,
            w.slot_proj,
            proj_dims(w, i + 1, c)
        };
    };

    auto h = BT{unit4(w)};
    for (auto c = int{0}; c <= ly - 2; ++c)
    {
        if (qn::err_state() != QNPEPS_ELOC_OK) return;
        auto frame = Carver{w.scratch};
        auto projected_i = BT{pri(c)};
        auto projected_j = BT{prj(c)};
        if (dual_projected)
        {
            auto packed_i{frame.take_product<cf>(
                {static_cast<std::uint64_t>(w.slot_proj), static_cast<std::uint64_t>(sh.lanes)}
            )};
            auto packed_j{frame.take_product<cf>(
                {static_cast<std::uint64_t>(w.slot_proj), static_cast<std::uint64_t>(sh.lanes)}
            )};
            if (qn::err_state() != QNPEPS_ELOC_OK) return;
            projected_i = project_column_dual_packed(
                w, i, c, proj_site(w, c), w.slot_proj, packed_i, w.slot_proj
            );
            projected_j = project_column_dual_packed(
                w,
                i + 1,
                c,
                w.proj2.p + static_cast<i64>(c) * w.slot_proj * sh.lanes,
                w.slot_proj,
                packed_j,
                w.slot_proj
            );
        }
        const auto d = std::vector<int>{fold_left4(
            w,
            frame,
            h,
            top(c),
            projected_i,
            projected_j,
            bot(c),
            fbl_site(w, i, c),
            w.slot_fb,
            0,
            dual_projected
        )};
        h = BT{fbl_site(w, i, c), w.slot_fb, d};
        rail_normalize(
            w,
            h.p,
            bt_elems(d),
            w.slot_fb,
            f_fbl_cut(w, i, c),
            c > 0 ? f_fbl_cut(w, i, c - 1) : nullptr
        );
    }
    if (dual_projected)
    {
        project_column_into(w, i, ly - 1, proj_site(w, ly - 1), w.slot_proj);
        project_column_into(
            w,
            i + 1,
            ly - 1,
            w.proj2.p + static_cast<i64>(ly - 1) * w.slot_proj * sh.lanes,
            w.slot_proj
        );
    }
    auto r = BT{unit4(w)};
    for (auto c = int{ly - 1}; c >= 1; --c)
    {
        if (qn::err_state() != QNPEPS_ELOC_OK) return;
        const auto d = std::vector<int>{fold_right4(
            w, w.scratch, r, top(c), pri(c), prj(c), bot(c), fbr_site(w, i, c - 1), w.slot_fb
        )};
        r = BT{fbr_site(w, i, c - 1), w.slot_fb, d};
        rail_normalize(
            w,
            r.p,
            bt_elems(d),
            w.slot_fb,
            f_fbr_cut(w, i, c - 1),
            c < ly - 1 ? f_fbr_cut(w, i, c) : nullptr
        );
    }
}

auto emit_stage_c(Worker& w, bool want_h, bool want_fb) -> void
{
    const auto& sh = w.sh;
    const auto dual_projected = bool{(packed_producers_from_env() & 4u) != 0u};
    if (not want_h and not want_fb) return;
    w.ks_full = w.ks_top;
    if (want_h)
    {
        for (auto i = int{0}; i < sh.lx; ++i)
        {
            if (qn::err_state() != QNPEPS_ELOC_OK) return;
            if (not dual_projected) project_row_into(w, i, w.proj.p, w.slot_proj);
            build_h_rails(w, i, dual_projected);
        }
    }
    if (want_fb)
    {
        for (auto i = int{0}; i < sh.lx - 1; ++i)
        {
            if (qn::err_state() != QNPEPS_ELOC_OK) return;
            if (not dual_projected)
            {
                project_row_into(w, i, w.proj.p, w.slot_proj);
                project_row_into(w, i + 1, w.proj2.p, w.slot_proj);
            }
            build_fb_rails(w, i, dual_projected);
        }
    }
}

auto emit_stage_c_j2_group(Worker& w, bool want_h, bool want_fb, int j2_group) -> void
{
    const auto& sh = w.sh;
    const auto dual_projected = bool{(packed_producers_from_env() & 4u) != 0u};
    if (not want_h and not want_fb) return;
    w.ks_full = w.ks_top;
    if (want_h)
    {
        for (auto i = int{0}; i < sh.lx; ++i)
        {
            if (qn::err_state() != QNPEPS_ELOC_OK) return;
            if (not dual_projected) project_row_into(w, i, w.proj.p, w.slot_proj);
            build_h_rails(w, i, dual_projected);
        }
    }
    if (want_fb)
    {
        for (auto i = int{0}; i < sh.lx - 1; ++i)
        {
            if (j2_group < 0 or (i & 1) == j2_group)
            {
                if (qn::err_state() != QNPEPS_ELOC_OK) return;
                if (not dual_projected)
                {
                    project_row_into(w, i, w.proj.p, w.slot_proj);
                    project_row_into(w, i + 1, w.proj2.p, w.slot_proj);
                }
                build_fb_rails(w, i, dual_projected);
            }
        }
    }
}

auto emit_stage_c_j2_column_group(Worker& w, bool want_h, bool want_fb, int j2_group) -> void
{
    static_cast<void>(j2_group);
    emit_stage_c(w, want_h, want_fb);
}

}
