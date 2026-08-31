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

namespace qn_eloc::env
{

#line 3017 "cuda/eo/env_build.cu"

auto launch_fourbody(Worker& w, int mid, const cf* cl, const cf* cr, cf* out, int dim_batch = 0)
    -> void
{
    launch_dot(w, mid, cl, mid, cr, mid, out, dim_batch);
}

inline auto fb_left_dims(Worker& w, int upper, int k) -> std::vector<int>
{
    const auto t = BT{top_bt(w, upper, k)};
    const auto b = BT{bot_bt_fb(w, upper, k)};
    const auto pi = std::vector<int>{proj_dims(w, upper, k)};
    const auto pj = std::vector<int>{proj_dims(w, upper + 1, k)};
    return {t.dim[2], pi[2], pj[2], b.dim[2]};
}
inline auto fb_right_dims(Worker& w, int upper, int k) -> std::vector<int>
{
    const auto c = int{k + 1};
    const auto t = BT{top_bt(w, upper, c)};
    const auto b = BT{bot_bt_fb(w, upper, c)};
    const auto pi = std::vector<int>{proj_dims(w, upper, c)};
    const auto pj = std::vector<int>{proj_dims(w, upper + 1, c)};
    return {t.dim[0], pi[0], pj[0], b.dim[0]};
}

auto corner_mode(const FlipInst& t, int r, int c, int ly) -> int
{
    const auto idx = int{r * ly + c};
    for (auto k = int{0}; k < t.n_flips; ++k)
        if (t.site[static_cast<usize>(k)] == idx) return t.value[static_cast<usize>(k)];
    return -2;
}

auto project_corner(Worker& w, int r, int c, int mode, cf* out, i64 out_stride, bool packed = false)
    -> BT
{
    const auto& sh = w.sh;
    const auto pd = std::vector<int>{proj_dims(w, r, c)};
    const auto slab = i64{bt_elems(pd)};
    i64 total{};
    if (qn::err_state() != QNPEPS_ELOC_OK or slab < 1
        or not arena_slot(
            {static_cast<std::uint64_t>(slab), static_cast<std::uint64_t>(sh.lanes)}, total
        ))
        return {};
    const auto threads = int{256};
    const auto blocks = int{static_cast<int>(std::min<i64>(4096, (total + threads - 1) / threads))};
    auto out_dims = std::vector<int>{pd};
    if (packed)
    {
        const auto perm = std::vector<int>{0, 3, 1, 2};
        auto source_indices{w.permutation_index_maps.get(pd, perm)};
        if (qn::err_state() != QNPEPS_ELOC_OK or not source_indices) return {};
        cu_project_site_val_packed<<<blocks, threads, 0, w.la.stream()>>>(
            w.peps + w.site_off[static_cast<usize>(r) * sh.ly + c],
            w.samples,
            static_cast<i64>(sh.lx) * sh.ly,
            r * sh.ly + c,
            slab,
            mode,
            mode,
            sh.dim_phys,
            source_indices,
            out,
            out_stride,
            sh.lanes
        );
        out_dims = permuted_dims(pd, perm);
    }
    else
        cu_project_site_val<<<blocks, threads, 0, w.la.stream()>>>(
            w.peps + w.site_off[static_cast<usize>(r) * sh.ly + c],
            w.samples,
            static_cast<i64>(sh.lx) * sh.ly,
            r * sh.ly + c,
            slab,
            mode,
            mode,
            sh.dim_phys,
            out,
            out_stride,
            sh.lanes
        );
    CUDA_CHECK(cudaGetLastError());
    return BT{out, out_stride, std::move(out_dims)};
}

auto project_corner_indexed(
    Worker& w,
    int r,
    int c,
    int mode,
    const int* active_indices,
    int active,
    cf* out,
    i64 out_stride,
    bool packed = false
) -> BT
{
    const auto& sh = w.sh;
    const auto pd = std::vector<int>{proj_dims(w, r, c)};
    const auto slab = i64{bt_elems(pd)};
    i64 total{};
    if (qn::err_state() != QNPEPS_ELOC_OK or slab < 1 or active < 1
        or not arena_slot(
            {static_cast<std::uint64_t>(slab), static_cast<std::uint64_t>(active)}, total
        ))
        return {};
    const auto threads = int{256};
    const auto blocks = int{static_cast<int>(std::min<i64>(4096, (total + threads - 1) / threads))};
    auto out_dims = std::vector<int>{pd};
    if (packed)
    {
        const auto perm = std::vector<int>{0, 3, 1, 2};
        auto source_indices{w.permutation_index_maps.get(pd, perm)};
        if (qn::err_state() != QNPEPS_ELOC_OK or not source_indices) return {};
        cu_project_site_val_indexed_packed<<<blocks, threads, 0, w.la.stream()>>>(
            w.peps + w.site_off[static_cast<usize>(r) * sh.ly + c],
            w.samples,
            static_cast<i64>(sh.lx) * sh.ly,
            active_indices,
            r * sh.ly + c,
            slab,
            mode,
            mode,
            sh.dim_phys,
            source_indices,
            out,
            out_stride,
            active
        );
        out_dims = permuted_dims(pd, perm);
    }
    else
        cu_project_site_val_indexed<<<blocks, threads, 0, w.la.stream()>>>(
            w.peps + w.site_off[static_cast<usize>(r) * sh.ly + c],
            w.samples,
            static_cast<i64>(sh.lx) * sh.ly,
            active_indices,
            r * sh.ly + c,
            slab,
            mode,
            mode,
            sh.dim_phys,
            out,
            out_stride,
            active
        );
    CUDA_CHECK(cudaGetLastError());
    return BT{out, out_stride, std::move(out_dims)};
}

auto compact_tensor(Worker& w, const BT& src, const int* active_indices, int active, cf* out) -> BT
{
    const auto elems = i64{bt_elems(src.dim)};
    i64 total{};
    if (qn::err_state() != QNPEPS_ELOC_OK or elems < 1 or active < 1
        or not arena_slot(
            {static_cast<std::uint64_t>(elems), static_cast<std::uint64_t>(active)}, total
        ))
        return {};
    const auto threads = int{256};
    const auto blocks = int{static_cast<int>(std::min<i64>(4096, (total + threads - 1) / threads))};
    cu_compact_lanes<<<blocks, threads, 0, w.la.stream()>>>(
        out, elems, src.p, src.stride, active_indices, elems, active
    );
    CUDA_CHECK(cudaGetLastError());
    return BT{out, elems, src.dim};
}

auto compact_tensor_packed(
    Worker& w,
    const BT& src,
    const int* active_indices,
    int active,
    cf* out,
    const std::vector<int>& perm
) -> BT
{
    const auto elems = i64{bt_elems(src.dim)};
    i64 total{};
    if (qn::err_state() != QNPEPS_ELOC_OK or elems < 1 or active < 1
        or not arena_slot(
            {static_cast<std::uint64_t>(elems), static_cast<std::uint64_t>(active)}, total
        ))
        return {};
    const auto threads = int{256};
    const auto blocks = int{static_cast<int>(std::min<i64>(4096, (total + threads - 1) / threads))};
    auto source_indices{w.permutation_index_maps.get(src.dim, perm)};
    if (qn::err_state() != QNPEPS_ELOC_OK or not source_indices) return {};
    cu_compact_lanes_packed<<<blocks, threads, 0, w.la.stream()>>>(
        out, elems, src.p, src.stride, active_indices, source_indices, elems, active
    );
    CUDA_CHECK(cudaGetLastError());
    return BT{out, elems, permuted_dims(src.dim, perm)};
}

auto chain_normalize(Worker& w, cf* x, i64 n, i64 stride) -> void
{
    const auto lanes = int{w.sh.lanes};
    int elements{};
    if (n < 1 or not arena_int({static_cast<std::uint64_t>(n)}, elements)) return;
    cu_normalize_log<<<lanes, 256, 0, w.la.stream()>>>(
        x, elements, stride, w.f_chain, lanes, nullptr
    );
    CUDA_CHECK(cudaGetLastError());
}

auto chain_normalize_batch(Worker& w, cf* x, i64 n, i64 stride, f64* f_chain, int dim_batch) -> void
{
    int elements{};
    if (n < 1 or not arena_int({static_cast<std::uint64_t>(n)}, elements)) return;
    cu_normalize_log<<<dim_batch, 256, 0, w.la.stream()>>>(
        x, elements, stride, f_chain, dim_batch, nullptr
    );
    CUDA_CHECK(cudaGetLastError());
}

auto eval_horizontal(Worker& w, const FlipInst& t, cf* value_out) -> void
{
    const auto& sh = w.sh;
    const auto ly = int{sh.ly};
    const auto lanes = int{sh.lanes};
    const auto packed_projected = bool{(packed_producers_from_env() & 1u) != 0u};
    const auto i = int{t.site[0] / ly};
    auto c0 = int{ly};
    auto cend = int{-1};
    for (auto k = int{0}; k < t.n_flips; ++k)
    {
        const auto c = int{t.site[static_cast<usize>(k)] % ly};
        c0 = std::min(c0, c);
        cend = std::max(cend, c);
    }
    const auto val_at = [&](int c) -> int
    {
        for (auto k = int{0}; k < t.n_flips; ++k)
            if (t.site[static_cast<usize>(k)] % ly == c) return t.value[static_cast<usize>(k)];
        return -2;
    };

    auto frame = Carver{w.scratch};
    auto v = BT{c0 > 0 ? BT{hl_site(w, i, c0 - 1), w.slot_h, hl_o_dims(w, i, c0)} : unit3(w)};
    for (auto c = int{c0}; c <= cend; ++c)
    {
        auto flip{frame.take_product<cf>(
            {static_cast<std::uint64_t>(w.slot_proj), static_cast<std::uint64_t>(lanes)}
        )};
        auto nxt{frame.take_product<cf>(
            {static_cast<std::uint64_t>(w.slot_h), static_cast<std::uint64_t>(lanes)}
        )};
        if (qn::err_state() != QNPEPS_ELOC_OK) return;
        const auto mode = int{val_at(c)};
        const auto projected =
            BT{project_corner(w, i, c, mode, flip, w.slot_proj, packed_projected)};
        const auto d = std::vector<int>{fold_left(
            w,
            frame,
            v,
            top_bt(w, i, c),
            projected,
            bot_bt_h(w, i, c),
            nxt,
            w.slot_h,
            0,
            packed_projected
        )};
        v = BT{nxt, w.slot_h, d};
        chain_normalize(w, v.p, bt_elems(d), w.slot_h);
    }

    auto vend_p{cend < ly - 1 ? hr_site(w, i, cend) : w.unit};
    const auto vend_stride = i64{cend < ly - 1 ? w.slot_h : 1};
    launch_dot(w, static_cast<int>(bt_elems(v.dim)), v.p, w.slot_h, vend_p, vend_stride, value_out);
}

auto eval_horizontal_compact(
    Worker& w, const FlipInst& t, const int* active_indices, int active, cf* value_out
) -> f64*
{
    const auto& sh = w.sh;
    const auto ly = int{sh.ly};
    const auto packed_projected = bool{(packed_producers_from_env() & 1u) != 0u};
    const auto i = int{t.site[0] / ly};
    auto c0 = int{ly};
    auto cend = int{-1};
    for (auto k = int{0}; k < t.n_flips; ++k)
    {
        const auto c = int{t.site[static_cast<usize>(k)] % ly};
        c0 = std::min(c0, c);
        cend = std::max(cend, c);
    }
    const auto val_at = [&](int c) -> int
    {
        for (auto k = int{0}; k < t.n_flips; ++k)
            if (t.site[static_cast<usize>(k)] % ly == c) return t.value[static_cast<usize>(k)];
        return -2;
    };

    const auto left_src =
        BT{c0 > 0 ? BT{hl_site(w, i, c0 - 1), w.slot_h, hl_o_dims(w, i, c0)} : unit3(w)};
    const auto right_src =
        BT{cend < ly - 1 ? BT{hr_site(w, i, cend), w.slot_h, hr_o_dims(w, i, cend)} : unit3(w)};

    auto frame = Carver{w.scratch};
    auto chain_a{frame.take_product<cf>(
        {static_cast<std::uint64_t>(w.slot_h), static_cast<std::uint64_t>(active)}
    )};
    auto chain_b{frame.take_product<cf>(
        {static_cast<std::uint64_t>(w.slot_h), static_cast<std::uint64_t>(active)}
    )};
    auto right{frame.take_product<cf>(
        {static_cast<std::uint64_t>(w.slot_h), static_cast<std::uint64_t>(active)}
    )};
    auto flip{frame.take_product<cf>(
        {static_cast<std::uint64_t>(w.slot_proj), static_cast<std::uint64_t>(active)}
    )};
    auto compact_top{frame.take_product<cf>(
        {static_cast<std::uint64_t>(w.slot_env), static_cast<std::uint64_t>(active)}
    )};
    auto compact_bot{frame.take_product<cf>(
        {static_cast<std::uint64_t>(w.slot_env), static_cast<std::uint64_t>(active)}
    )};
    auto f_chain{frame.take<f64>(static_cast<usize>(active))};
    if (qn::err_state() != QNPEPS_ELOC_OK) return nullptr;
    usize f_chain_bytes{};
    if (not arena_product({sizeof(f64), static_cast<std::uint64_t>(active)}, f_chain_bytes))
        return nullptr;

    CUDA_CHECK(cudaMemsetAsync(f_chain, 0, f_chain_bytes, w.la.stream()));
    auto v = BT{compact_tensor(w, left_src, active_indices, active, chain_a)};
    auto next{chain_b};
    for (auto c = int{c0}; c <= cend; ++c)
    {
        if (qn::err_state() != QNPEPS_ELOC_OK) return nullptr;
        const auto proj_elems = i64{bt_elems(proj_dims(w, i, c))};
        const auto projected = BT{project_corner_indexed(
            w, i, c, val_at(c), active_indices, active, flip, proj_elems, packed_projected
        )};
        const auto top =
            BT{compact_tensor(w, top_bt(w, i, c), active_indices, active, compact_top)};
        const auto bot =
            BT{compact_tensor(w, bot_bt_h(w, i, c), active_indices, active, compact_bot)};
        const auto d = std::vector<int>{
            fold_left(w, frame, v, top, projected, bot, next, w.slot_h, active, packed_projected)
        };
        v = BT{next, w.slot_h, d};
        chain_normalize_batch(w, v.p, bt_elems(d), w.slot_h, f_chain, active);
        next = next == chain_a ? chain_b : chain_a;
    }

    const auto vend = BT{compact_tensor(w, right_src, active_indices, active, right)};
    launch_dot(
        w, static_cast<int>(bt_elems(v.dim)), v.p, v.stride, vend.p, vend.stride, value_out, active
    );
    return f_chain;
}

auto eval_fourbody(Worker& w, const FlipInst& t, cf* value_out) -> void
{
    const auto& sh = w.sh;
    const auto ly = int{sh.ly};
    const auto lanes = int{sh.lanes};
    const auto packed_projected = bool{(packed_producers_from_env() & 1u) != 0u};
    const auto r0 = int{t.site[0] / ly};
    const auto c0 = int{t.site[0] % ly};
    const auto r1 = int{t.site[1] / ly};
    const auto c1 = int{t.site[1] % ly};
    const auto minr = int{std::min(r0, r1)};
    const auto maxr = int{std::max(r0, r1)};
    const auto miny = int{std::min(c0, c1)};
    const auto maxy = int{std::max(c0, c1)};
    const auto upper = int{minr};

    auto frame = Carver{w.scratch};
    auto piL{frame.take_product<cf>(
        {static_cast<std::uint64_t>(w.slot_proj), static_cast<std::uint64_t>(lanes)}
    )};
    auto pjL{frame.take_product<cf>(
        {static_cast<std::uint64_t>(w.slot_proj), static_cast<std::uint64_t>(lanes)}
    )};
    auto piR{frame.take_product<cf>(
        {static_cast<std::uint64_t>(w.slot_proj), static_cast<std::uint64_t>(lanes)}
    )};
    auto pjR{frame.take_product<cf>(
        {static_cast<std::uint64_t>(w.slot_proj), static_cast<std::uint64_t>(lanes)}
    )};
    if (qn::err_state() != QNPEPS_ELOC_OK) return;
    const auto projected_pi_l = BT{project_corner(
        w, minr, miny, corner_mode(t, minr, miny, ly), piL, w.slot_proj, packed_projected
    )};
    const auto projected_pj_l = BT{project_corner(
        w, maxr, miny, corner_mode(t, maxr, miny, ly), pjL, w.slot_proj, packed_projected
    )};
    const auto projected_pi_r =
        BT{project_corner(w, minr, maxy, corner_mode(t, minr, maxy, ly), piR, w.slot_proj)};
    const auto projected_pj_r =
        BT{project_corner(w, maxr, maxy, corner_mode(t, maxr, maxy, ly), pjR, w.slot_proj)};

    const auto topL = BT{top_bt(w, upper, miny)};
    const auto botL = BT{bot_bt_fb(w, upper, miny)};
    const auto topR = BT{top_bt(w, upper, maxy)};
    const auto botR = BT{bot_bt_fb(w, upper, maxy)};
    const auto piLd = std::vector<int>{proj_dims(w, minr, miny)};
    const auto pjLd = std::vector<int>{proj_dims(w, maxr, miny)};
    const auto piRd = std::vector<int>{proj_dims(w, minr, maxy)};
    const auto pjRd = std::vector<int>{proj_dims(w, maxr, maxy)};
    int mid{};
    if (not arena_int(
            {static_cast<std::uint64_t>(topL.dim[2]),
             static_cast<std::uint64_t>(piLd[2]),
             static_cast<std::uint64_t>(pjLd[2]),
             static_cast<std::uint64_t>(botL.dim[2])},
            mid
        ))
        return;

    auto cl{
        frame.take_product<cf>({static_cast<std::uint64_t>(mid), static_cast<std::uint64_t>(lanes)})
    };
    auto cr{
        frame.take_product<cf>({static_cast<std::uint64_t>(mid), static_cast<std::uint64_t>(lanes)})
    };
    if (qn::err_state() != QNPEPS_ELOC_OK) return;

    const auto fbL =
        BT{miny > 0 ? BT{fbl_site(w, upper, miny - 1), w.slot_fb, fb_left_dims(w, upper, miny - 1)}
                    : unit4(w)};
    const auto fbR =
        BT{maxy < ly - 1 ? BT{fbr_site(w, upper, maxy), w.slot_fb, fb_right_dims(w, upper, maxy)}
                         : unit4(w)};
    fold_left4(
        w,
        frame,
        fbL,
        topL,
        projected_pi_l,
        projected_pj_l,
        botL,
        cl,
        static_cast<i64>(mid),
        0,
        packed_projected
    );
    fold_right4(
        w, frame, fbR, topR, projected_pi_r, projected_pj_r, botR, cr, static_cast<i64>(mid)
    );
    chain_normalize(w, cl, mid, mid);
    chain_normalize(w, cr, mid, mid);
    launch_fourbody(w, mid, cl, cr, value_out);
}

auto eval_fourbody_compact(
    Worker& w, const FlipInst& t, const int* active_indices, int active, cf* value_out
) -> f64*
{
    const auto& sh = w.sh;
    const auto ly = int{sh.ly};
    const auto packed_mask = unsigned{packed_producers_from_env()};
    const auto packed_projected = bool{(packed_mask & 1u) != 0u};
    const auto packed_bottom = bool{(packed_mask & 2u) != 0u};
    const auto r0 = int{t.site[0] / ly};
    const auto c0 = int{t.site[0] % ly};
    const auto r1 = int{t.site[1] / ly};
    const auto c1 = int{t.site[1] % ly};
    const auto minr = int{std::min(r0, r1)};
    const auto maxr = int{std::max(r0, r1)};
    const auto miny = int{std::min(c0, c1)};
    const auto maxy = int{std::max(c0, c1)};
    const auto upper = int{minr};

    const auto piLd = std::vector<int>{proj_dims(w, minr, miny)};
    const auto pjLd = std::vector<int>{proj_dims(w, maxr, miny)};
    const auto piRd = std::vector<int>{proj_dims(w, minr, maxy)};
    const auto pjRd = std::vector<int>{proj_dims(w, maxr, maxy)};
    const auto piL_elems = i64{bt_elems(piLd)};
    const auto pjL_elems = i64{bt_elems(pjLd)};
    const auto piR_elems = i64{bt_elems(piRd)};
    const auto pjR_elems = i64{bt_elems(pjRd)};

    const auto topL = BT{top_bt(w, upper, miny)};
    const auto botL = BT{bot_bt_fb(w, upper, miny)};
    const auto topR = BT{top_bt(w, upper, maxy)};
    const auto botR = BT{bot_bt_fb(w, upper, maxy)};
    const auto fbL =
        BT{miny > 0 ? BT{fbl_site(w, upper, miny - 1), w.slot_fb, fb_left_dims(w, upper, miny - 1)}
                    : unit4(w)};
    const auto fbR =
        BT{maxy < ly - 1 ? BT{fbr_site(w, upper, maxy), w.slot_fb, fb_right_dims(w, upper, maxy)}
                         : unit4(w)};
    int mid{};
    if (not arena_int(
            {static_cast<std::uint64_t>(topL.dim[2]),
             static_cast<std::uint64_t>(piLd[2]),
             static_cast<std::uint64_t>(pjLd[2]),
             static_cast<std::uint64_t>(botL.dim[2])},
            mid
        ))
        return nullptr;

    auto frame = Carver{w.scratch};
    auto piL{frame.take_product<cf>(
        {static_cast<std::uint64_t>(piL_elems), static_cast<std::uint64_t>(active)}
    )};
    auto pjL{frame.take_product<cf>(
        {static_cast<std::uint64_t>(pjL_elems), static_cast<std::uint64_t>(active)}
    )};
    auto piR{frame.take_product<cf>(
        {static_cast<std::uint64_t>(piR_elems), static_cast<std::uint64_t>(active)}
    )};
    auto pjR{frame.take_product<cf>(
        {static_cast<std::uint64_t>(pjR_elems), static_cast<std::uint64_t>(active)}
    )};
    auto cl{frame.take_product<cf>(
        {static_cast<std::uint64_t>(mid), static_cast<std::uint64_t>(active)}
    )};
    auto cr{frame.take_product<cf>(
        {static_cast<std::uint64_t>(mid), static_cast<std::uint64_t>(active)}
    )};
    auto f_chain{frame.take<f64>(static_cast<usize>(active))};
    const auto fb_elems = i64{std::max(bt_elems(fbL.dim), bt_elems(fbR.dim))};
    const auto top_elems = i64{std::max(bt_elems(topL.dim), bt_elems(topR.dim))};
    const auto bot_elems = i64{std::max(bt_elems(botL.dim), bt_elems(botR.dim))};
    auto compact_fb{frame.take_product<cf>(
        {static_cast<std::uint64_t>(fb_elems), static_cast<std::uint64_t>(active)}
    )};
    auto compact_top{frame.take_product<cf>(
        {static_cast<std::uint64_t>(top_elems), static_cast<std::uint64_t>(active)}
    )};
    auto compact_bot{frame.take_product<cf>(
        {static_cast<std::uint64_t>(bot_elems), static_cast<std::uint64_t>(active)}
    )};
    if (qn::err_state() != QNPEPS_ELOC_OK) return nullptr;

    const auto projected_pi_l = BT{project_corner_indexed(
        w,
        minr,
        miny,
        corner_mode(t, minr, miny, ly),
        active_indices,
        active,
        piL,
        piL_elems,
        packed_projected
    )};
    const auto projected_pj_l = BT{project_corner_indexed(
        w,
        maxr,
        miny,
        corner_mode(t, maxr, miny, ly),
        active_indices,
        active,
        pjL,
        pjL_elems,
        packed_projected
    )};
    const auto projected_pi_r = BT{project_corner_indexed(
        w, minr, maxy, corner_mode(t, minr, maxy, ly), active_indices, active, piR, piR_elems
    )};
    const auto projected_pj_r = BT{project_corner_indexed(
        w, maxr, maxy, corner_mode(t, maxr, maxy, ly), active_indices, active, pjR, pjR_elems
    )};
    usize f_chain_bytes{};
    if (not arena_product({sizeof(f64), static_cast<std::uint64_t>(active)}, f_chain_bytes))
        return nullptr;
    CUDA_CHECK(cudaMemsetAsync(f_chain, 0, f_chain_bytes, w.la.stream()));

    const auto compact_fb_l = BT{compact_tensor(w, fbL, active_indices, active, compact_fb)};
    const auto compact_top_l = BT{compact_tensor(w, topL, active_indices, active, compact_top)};
    const auto compact_bot_l = BT{compact_tensor(w, botL, active_indices, active, compact_bot)};
    fold_left4(
        w,
        frame,
        compact_fb_l,
        compact_top_l,
        projected_pi_l,
        projected_pj_l,
        compact_bot_l,
        cl,
        static_cast<i64>(mid),
        active,
        packed_projected
    );

    const auto compact_fb_r = BT{compact_tensor(w, fbR, active_indices, active, compact_fb)};
    const auto compact_top_r = BT{compact_tensor(w, topR, active_indices, active, compact_top)};
    const auto compact_bot_r =
        BT{packed_bottom
               ? compact_tensor_packed(w, botR, active_indices, active, compact_bot, {2, 1, 0})
               : compact_tensor(w, botR, active_indices, active, compact_bot)};
    fold_right4(
        w,
        frame,
        compact_fb_r,
        compact_top_r,
        projected_pi_r,
        projected_pj_r,
        compact_bot_r,
        cr,
        static_cast<i64>(mid),
        active,
        packed_bottom
    );
    chain_normalize_batch(w, cl, mid, mid, f_chain, active);
    chain_normalize_batch(w, cr, mid, mid, f_chain, active);
    launch_fourbody(w, mid, cl, cr, value_out, active);
    return f_chain;
}

auto eval_flip_inst(
    Worker& w,
    const FlipInst& t,
    cf* value_buf,
    f64* e_loc,
    const f64* logpsi,
    const int* active_indices = nullptr,
    int active = -1,
    int compact_min = 2
) -> void
{
    const auto& sh = w.sh;
    const auto ly = int{sh.ly};
    const auto lanes = int{sh.lanes};
    const auto lx = int{sh.lx};
    if (active == 0) return;
    const auto use_compact =
        bool{active_indices != nullptr and active >= compact_min and active < lanes};
    usize f_chain_bytes{};
    if (not arena_product({sizeof(f64), static_cast<std::uint64_t>(lanes)}, f_chain_bytes)) return;
    if (not use_compact) CUDA_CHECK(cudaMemsetAsync(w.f_chain, 0, f_chain_bytes, w.la.stream()));
    f64* compact_f_chain{};
    f64* f_top{};
    f64* f_down{};
    f64* f_rail_l{};
    f64* f_rail_r{};
    if (t.bucket == Bucket::fourbody)
    {
        if (use_compact)
            compact_f_chain = eval_fourbody_compact(w, t, active_indices, active, value_buf);
        else
            eval_fourbody(w, t, value_buf);
        if (qn::err_state() != QNPEPS_ELOC_OK or (use_compact and compact_f_chain == nullptr))
            return;
        const auto r0 = int{t.site[0] / ly};
        const auto r1 = int{t.site[1] / ly};
        const auto minr = int{std::min(r0, r1)};
        const auto maxr = int{std::max(r0, r1)};
        const auto c0 = int{t.site[0] % ly};
        const auto c1 = int{t.site[1] % ly};
        const auto miny = int{std::min(c0, c1)};
        const auto maxy = int{std::max(c0, c1)};
        f_top = minr > 0 ? w.f_top + static_cast<i64>(minr - 1) * lanes : nullptr;
        f_down = maxr < lx - 1 ? w.f_down + static_cast<i64>(lx - maxr - 2) * lanes : nullptr;
        f_rail_l = miny > 0 ? f_fbl_cut(w, minr, miny - 1) : nullptr;
        f_rail_r = maxy < ly - 1 ? f_fbr_cut(w, minr, maxy) : nullptr;
    }
    else
    {
        if (use_compact)
            compact_f_chain = eval_horizontal_compact(w, t, active_indices, active, value_buf);
        else
            eval_horizontal(w, t, value_buf);
        if (qn::err_state() != QNPEPS_ELOC_OK or (use_compact and compact_f_chain == nullptr))
            return;
        const auto i = int{t.site[0] / ly};
        auto c0 = int{ly};
        auto cend = int{-1};
        for (auto k = int{0}; k < t.n_flips; ++k)
        {
            const auto c = int{t.site[static_cast<usize>(k)] % ly};
            c0 = std::min(c0, c);
            cend = std::max(cend, c);
        }
        f_top = i > 0 ? w.f_top + static_cast<i64>(i - 1) * lanes : nullptr;
        f_down = i < lx - 1 ? w.f_down + static_cast<i64>(lx - i - 2) * lanes : nullptr;
        f_rail_l = c0 > 0 ? f_hl_cut(w, i, c0 - 1) : nullptr;
        f_rail_r = cend < ly - 1 ? f_hr_cut(w, i, cend) : nullptr;
    }
    if (use_compact)
    {
        reduce_term_indexed(
            w,
            t,
            value_buf,
            compact_f_chain,
            active_indices,
            active,
            f_top,
            f_down,
            f_rail_l,
            f_rail_r,
            e_loc,
            logpsi
        );
    }
    else
        reduce_term(
            w,
            t.mask_a,
            t.mask_b,
            t.coeff_re,
            t.coeff_im,
            value_buf,
            f_top,
            f_down,
            f_rail_l,
            f_rail_r,
            e_loc,
            logpsi
        );
}

}
