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

namespace qn_eloc::env
{

#line 458 "cuda/eo/env_build.cu"

struct Worker
{
    explicit Worker(Linalg& linalg) : la(linalg) {}

    Shape sh{};
    bool want_h{};
    bool want_fb{};
    Linalg& la;
    PermutationCache permutation_index_maps{};
    ContextArena* arena{};
    Carver scratch{};
    std::map<i64, cf*> omegas{};
    qn_eloc::density::Context* density{};

    const cf* peps{};
    const u8* samples{};
    std::vector<i64> site_off{};

    EoDeviceBuffer eb{};
    qnpeps::CuArray<EoDeviceBuffer, 2> et{};
    EoDeviceBuffer proj{};
    EoDeviceBuffer tmp_a{};
    EoDeviceBuffer tmp_b{};
    EoDeviceBuffer tmp_c{};
    EoDeviceBuffer rroll{};
    EoDeviceBuffer rnext{};
    EoDeviceBuffer sketch{};
    EoDeviceBuffer proj_rf{};
    EoDeviceBuffer gram{};
    cf** gram_ptrs{};
    cf** sketch_ptrs{};
    int* info{};
    int* fail_flag{};
    int* failure_log{};
    usize failure_log_count{};
    char* qr_scratch{};
    usize qr_scratch_bytes{};
    int rangefinder_call{};
    const int* fallback_info{};
    f64* f_down{};
    f64* f_top{};
    f64* f_ov{};

    i64 slot_env{};
    i64 slot_proj{};
    i64 slot_tmp{};
    i64 slot_r{};
    i64 slot_rf{};

    std::vector<std::vector<int>> ks_down{};
    std::vector<std::vector<int>> ks_top{};

    EoDeviceBuffer et_full{};
    EoDeviceBuffer hl{};
    EoDeviceBuffer hr{};
    EoDeviceBuffer fbl{};
    EoDeviceBuffer fbr{};
    EoDeviceBuffer proj2{};
    cf* unit{};
    f64* f_hl{};
    f64* f_hr{};
    f64* f_fbl{};
    f64* f_fbr{};
    f64* f_chain{};
    i64 slot_h{};
    i64 slot_fb{};
    std::vector<std::vector<int>> ks_full{};
};

inline auto bd(int axis_len, int pos, int dim_bond) -> int
{
    return bond_dim(axis_len, pos, dim_bond);
}

inline auto omega_host(int n, int k) -> std::vector<cf>
{
    auto seed = u64{0x777ull ^ (static_cast<u64>(n) << 20) ^ static_cast<u64>(k)};
    if (auto value{std::getenv("QNPEPS_E0194_OMEGA_SEED")}; value and value[0] != '\0')
    {
        char* end{};
        const auto parsed{std::strtoull(value, &end, 0)};
        if (not end or end == value or end[0] != '\0')
        {
            qn::set_err(QNPEPS_ELOC_ERR_BAD_CONFIG);
            return {};
        }
        seed ^= parsed;
    }
    auto rng = std::mt19937_64(seed);
    auto gauss = std::normal_distribution<f32>(0.0f, 1.0f);
    usize count{};
    if (n < 1 or k < 1
        or not arena_product({static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(k)}, count))
        return {};
    auto host = std::vector<cf>(count);
    for (auto& z : host)
        z = cf{gauss(rng), gauss(rng)};
    return host;
}

inline auto omega_for(Worker& w, int n, int k) -> cf*
{
    const auto key = i64{static_cast<i64>(n) * 1000000 + k};
    auto it = w.omegas.find(key);
    if (it != w.omegas.end()) return it->second;
    const auto host = std::vector<cf>{omega_host(n, k)};
    if (host.empty()) return nullptr;
    if (not w.arena) return nullptr;
    const auto bytes = host.size() * sizeof(cf);
    auto* d = w.arena->take<cf>(host.size());
    CUDA_CHECK(cudaMemcpy(d, host.data(), bytes, cudaMemcpyHostToDevice));
    w.omegas[key] = d;
    return d;
}

inline auto eb_site(Worker& w, int stack_k, int col) -> cf*
{
    return w.eb.p + (static_cast<i64>(stack_k) * w.sh.ly + col) * w.slot_env * w.sh.lanes;
}

inline auto et_site(Worker& w, int buf, int col) -> cf*
{
    return w.et[static_cast<usize>(buf)].p + static_cast<i64>(col) * w.slot_env * w.sh.lanes;
}

inline auto proj_site(Worker& w, int col) -> cf*
{
    return w.proj.p + static_cast<i64>(col) * w.slot_proj * w.sh.lanes;
}

inline auto etf_site(Worker& w, int k, int col) -> cf*
{
    return w.et_full.p + (static_cast<i64>(k) * w.sh.ly + col) * w.slot_env * w.sh.lanes;
}

auto carve_worker(Worker& w, const Shape& sh, bool want_h, bool want_fb, Carver& cv) -> usize
{
    if (sh.lx < 1 or sh.ly < 1 or sh.dim_phys < 1 or sh.dim_bond < 1 or sh.chi < 1 or sh.lanes < 1)
    {
        qn::set_err(QNPEPS_ELOC_ERR_BAD_CONFIG);
        return 0;
    }
    const auto lanes = usize{static_cast<usize>(sh.lanes)};
    const auto C = std::uint64_t{static_cast<std::uint64_t>(sh.chi)};
    const auto D = std::uint64_t{static_cast<std::uint64_t>(sh.dim_bond)};
    i64 gram_slot{};
    i64 fold4_slot{};
    int chi_bond{};
    int rangefinder_calls{};
    if (not arena_slot({C, D, C}, w.slot_env) or not arena_slot({D, D, D, D}, w.slot_proj)
        or not arena_slot({C, C, D, D, D}, w.slot_tmp) or not arena_slot({C, C, D}, w.slot_r)
        or not arena_slot({C, C, D}, w.slot_rf) or not arena_slot({C, D, C}, w.slot_h)
        or not arena_slot({C, D, D, C}, w.slot_fb) or not arena_slot({C, C}, gram_slot)
        or not arena_slot({C, D, D, C, D}, fold4_slot) or not arena_int({C, D}, chi_bond)
        or not arena_int(
            {2u, static_cast<std::uint64_t>(sh.lx - 1), static_cast<std::uint64_t>(sh.ly)},
            rangefinder_calls
        ))
        return 0;
    i64 env_cuts{};
    i64 h_cuts_i64{};
    i64 fb_cuts_i64{};
    if (not arena_slot(
            {static_cast<std::uint64_t>(sh.lx - 1), static_cast<std::uint64_t>(sh.ly)}, env_cuts
        )
        or not arena_slot(
            {static_cast<std::uint64_t>(sh.lx), static_cast<std::uint64_t>(sh.ly - 1)}, h_cuts_i64
        )
        or not arena_slot(
            {static_cast<std::uint64_t>(sh.lx - 1), static_cast<std::uint64_t>(sh.ly - 1)},
            fb_cuts_i64
        ))
        return 0;
    const auto grab = [&](i64 slot, i64 count) -> EoDeviceBuffer
    {
        EoDeviceBuffer b{};
        b.stride = slot;
        i64 elements{};
        if (slot < 0 or count < 0
            or not arena_slot(
                {static_cast<std::uint64_t>(slot),
                 static_cast<std::uint64_t>(count),
                 static_cast<std::uint64_t>(lanes)},
                elements
            ))
            return b;
        b.p = cv.take<cf>(static_cast<usize>(elements));
        return b;
    };
    w.eb = grab(w.slot_env, env_cuts);
    w.et[0] = grab(w.slot_env, sh.ly);
    w.et[1] = grab(w.slot_env, sh.ly);
    w.proj = grab(w.slot_proj, sh.ly);
    w.tmp_a = grab(w.slot_tmp, 1);
    w.tmp_b = grab(w.slot_tmp, 1);
    w.tmp_c = grab(w.slot_tmp, 1);
    w.rroll = grab(w.slot_r, 1);
    w.rnext = grab(w.slot_r, 1);
    w.sketch = grab(w.slot_rf, 1);
    w.proj_rf = grab(w.slot_rf, 1);
    w.gram = grab(gram_slot, 1);
    w.gram_ptrs = cv.take<cf*>(lanes);
    w.sketch_ptrs = cv.take<cf*>(lanes);
    w.info = cv.take<int>(lanes);
    w.fail_flag = cv.take<int>(1);
    if (not arena_product(
            {2u,
             static_cast<std::uint64_t>(rangefinder_calls),
             static_cast<std::uint64_t>(sh.lanes)},
            w.failure_log_count
        ))
        return 0;
    w.failure_log = cv.take<int>(w.failure_log_count);
    if (rangefinder_route_uses_householder_scratch())
    {
        w.qr_scratch_bytes = qnpeps::qr_scratch_bytes(w.la, chi_bond, sh.chi);
        w.qr_scratch = cv.take<char>(w.qr_scratch_bytes);
    }
    else if (rangefinder_route_uses_experimental_scratch())
    {
        rangefinder_experimental_carve(
            w.la, cv, chi_bond, sh.chi, sh.lanes, w.qr_scratch, w.qr_scratch_bytes
        );
    }
    else
    {
        w.qr_scratch = nullptr;
        w.qr_scratch_bytes = 0;
    }
    usize vertical_scales{};
    if (not arena_product(
            {static_cast<std::uint64_t>(lanes), static_cast<std::uint64_t>(sh.lx - 1)},
            vertical_scales
        ))
        return 0;
    w.f_down = cv.take<f64>(vertical_scales);
    w.f_top = cv.take<f64>(vertical_scales);
    w.f_ov = cv.take<f64>(lanes);
    if (want_h or want_fb)
    {
        w.et_full = grab(w.slot_env, env_cuts);
        w.unit = cv.take<cf>(lanes);
        w.f_chain = cv.take<f64>(lanes);
    }
    if (want_h)
    {
        w.hl = grab(w.slot_h, h_cuts_i64);
        w.hr = grab(w.slot_h, h_cuts_i64);
        usize h_scales{};
        if (not arena_product(
                {static_cast<std::uint64_t>(lanes), static_cast<std::uint64_t>(h_cuts_i64)},
                h_scales
            ))
            return 0;
        w.f_hl = cv.take<f64>(h_scales);
        w.f_hr = cv.take<f64>(h_scales);
    }
    if (want_fb)
    {
        w.fbl = grab(w.slot_fb, fb_cuts_i64);
        w.fbr = grab(w.slot_fb, fb_cuts_i64);
        w.proj2 = grab(w.slot_proj, sh.ly);
        usize fb_scales{};
        if (not arena_product(
                {static_cast<std::uint64_t>(lanes), static_cast<std::uint64_t>(fb_cuts_i64)},
                fb_scales
            ))
            return 0;
        w.f_fbl = cv.take<f64>(fb_scales);
        w.f_fbr = cv.take<f64>(fb_scales);
    }
    if (want_h or want_fb)
    {
        const auto slot_bytes = [&](i64 elems) -> usize
        {
            usize bytes{};
            usize aligned{};
            if (elems < 0
                or not arena_product(
                    {sizeof(cf),
                     static_cast<std::uint64_t>(elems),
                     static_cast<std::uint64_t>(lanes)},
                    bytes
                )
                or not arena_align(bytes, aligned))
                return 0;
            return aligned;
        };
        const auto h_e = i64{w.slot_h};
        const auto fb_e = i64{w.slot_fb};
        const auto proj_e = i64{w.slot_proj};
        const auto fold4_e = i64{fold4_slot};
        const auto ok_e = i64{w.slot_tmp};
        const auto proj_bytes = usize{slot_bytes(proj_e)};
        const auto h_bytes = usize{slot_bytes(h_e)};
        const auto fb_bytes = usize{slot_bytes(fb_e)};
        const auto fold4_bytes = usize{slot_bytes(fold4_e)};
        const auto ok_bytes = usize{slot_bytes(ok_e)};
        if (qn::err_state() != QNPEPS_ELOC_OK) return 0;
        usize reserve{};
        const auto reserve_slots = [&](std::uint64_t count, usize bytes) -> bool
        {
            usize term{};
            return arena_product({count, static_cast<std::uint64_t>(bytes)}, term)
                   and arena_accumulate(reserve, term);
        };
        usize row_pair{};
        usize row_slots{};
        usize tail{};
        if (not reserve_slots(6u, proj_bytes) or not reserve_slots(4u, h_bytes)
            or (want_fb and not reserve_slots(4u, fb_bytes)) or not reserve_slots(12u, fold4_bytes)
            or not reserve_slots(8u, ok_bytes) or not arena_sum(proj_bytes, h_bytes, row_pair)
            or not arena_product(
                {static_cast<std::uint64_t>(sh.ly), static_cast<std::uint64_t>(row_pair)}, row_slots
            )
            or not arena_accumulate(reserve, row_slots) or not arena_product({64u, 256u}, tail)
            or not arena_accumulate(reserve, tail))
            return 0;
        auto* const scratch_base = cv.take<char>(reserve);
        if (qn::err_state() != QNPEPS_ELOC_OK) return 0;
        w.scratch = Carver::carve(scratch_base, reserve);
    }
    return cv.total();
}

auto worker_setup(
    Worker& w,
    const Shape& sh,
    bool want_h,
    bool want_fb,
    const cf* peps,
    const u8* samples,
    cudaStream_t stream,
    ContextArena& context_arena
) -> void
{
    if (stream != w.la.stream())
    {
        qn::set_err(QNPEPS_ELOC_ERR_BAD_CONFIG);
        return;
    }
    w.sh = sh;
    w.want_h = want_h;
    w.want_fb = want_fb;
    w.peps = peps;
    w.samples = samples;
    w.arena = &context_arena;
    carve_worker(w, sh, want_h, want_fb, context_arena);
    if (qn::err_state() != QNPEPS_ELOC_OK) return;
    if (sh.density)
    {
        const auto status = int{
            qn_eloc::density::create(w.la, context_arena, sh.ly, sh.dim_bond, sh.chi, &w.density)
        };
        if (status != 0 or not w.density)
        {
            qn::set_err(status == 5 ? QNPEPS_ELOC_ERR_OOM : QNPEPS_ELOC_ERR_INTERNAL);
            return;
        }
    }
    const auto lanes = usize{static_cast<usize>(sh.lanes)};
    auto hp = std::vector<cf*>(lanes);
    for (auto lane = usize{0}; lane < lanes; ++lane)
        hp[lane] = w.gram.p + static_cast<i64>(lane) * w.gram.stride;
    usize pointer_bytes{};
    usize failure_bytes{};
    if (not arena_product({sizeof(cf*), lanes}, pointer_bytes)
        or not arena_product({sizeof(int), w.failure_log_count}, failure_bytes))
        return;
    CUDA_CHECK(cudaMemcpy(w.gram_ptrs, hp.data(), pointer_bytes, cudaMemcpyHostToDevice));
    for (auto lane = usize{0}; lane < lanes; ++lane)
        hp[lane] = w.sketch.p + static_cast<i64>(lane) * w.sketch.stride;
    CUDA_CHECK(cudaMemcpy(w.sketch_ptrs, hp.data(), pointer_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemsetAsync(w.fail_flag, 0, sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(w.failure_log, 0, failure_bytes, stream));

    int site_count_int{};
    usize site_slots{};
    if (not arena_int(
            {static_cast<std::uint64_t>(sh.lx), static_cast<std::uint64_t>(sh.ly)}, site_count_int
        )
        or not arena_sum(static_cast<usize>(site_count_int), 1u, site_slots))
        return;
    const auto site_count = usize{static_cast<usize>(site_count_int)};
    w.site_off.assign(site_slots, 0);
    i64 off{};
    for (auto row = int{0}; row < sh.lx; ++row)
    {
        for (auto col = int{0}; col < sh.ly; ++col)
        {
            w.site_off[static_cast<usize>(row) * sh.ly + col] = off;
            i64 n_site{};
            i64 next{};
            if (not arena_slot(
                    {static_cast<std::uint64_t>(bd(sh.ly, col, sh.dim_bond)),
                     static_cast<std::uint64_t>(bd(sh.lx, row + 1, sh.dim_bond)),
                     static_cast<std::uint64_t>(bd(sh.ly, col + 1, sh.dim_bond)),
                     static_cast<std::uint64_t>(bd(sh.lx, row, sh.dim_bond)),
                     static_cast<std::uint64_t>(sh.dim_phys)},
                    n_site
                )
                or not arena_i64_sum(off, n_site, next))
                return;
            off = next;
        }
    }
    w.site_off[site_count] = off;
    w.ks_down.assign(static_cast<usize>(sh.lx - 1), {});
    w.ks_top.assign(static_cast<usize>(sh.lx - 1), {});
    if (want_h or want_fb)
    {
        cu_fill_first_one<<<1, 256, 0, w.la.stream()>>>(w.unit, 1, 1, sh.lanes);
        CUDA_CHECK(cudaGetLastError());
    }
}

auto worker_teardown(Worker& w) -> void
{
    if (w.la.stream()) CUDA_NOCHECK(cudaStreamSynchronize(w.la.stream()));
    qn_eloc::density::destroy(w.density);
    w.density = nullptr;
    w.omegas.clear();
    w.permutation_index_maps.release();
    w.arena = nullptr;
}

auto worker_rearm_constants(Worker& w, cudaStream_t stream) -> void
{
    if (not w.fail_flag)
    {
        qn::set_err(QNPEPS_ELOC_ERR_INTERNAL);
        return;
    }
    const auto lanes = usize{static_cast<usize>(w.sh.lanes)};
    auto pointers = std::vector<cf*>(lanes);
    usize pointer_bytes{};
    usize failure_bytes{};
    if (not arena_product({sizeof(cf*), lanes}, pointer_bytes)
        or not arena_product({sizeof(int), w.failure_log_count}, failure_bytes))
        return;
    for (auto lane = usize{0}; lane < lanes; ++lane)
        pointers[lane] = w.gram.p + static_cast<i64>(lane) * w.gram.stride;
    CUDA_CHECK(
        cudaMemcpyAsync(w.gram_ptrs, pointers.data(), pointer_bytes, cudaMemcpyHostToDevice, stream)
    );
    for (auto lane = usize{0}; lane < lanes; ++lane)
        pointers[lane] = w.sketch.p + static_cast<i64>(lane) * w.sketch.stride;
    CUDA_CHECK(cudaMemcpyAsync(
        w.sketch_ptrs, pointers.data(), pointer_bytes, cudaMemcpyHostToDevice, stream
    ));
    CUDA_CHECK(cudaMemsetAsync(w.fail_flag, 0, sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(w.failure_log, 0, failure_bytes, stream));
    if (w.want_h or w.want_fb)
    {
        cu_fill_first_one<<<1, 256, 0, stream>>>(w.unit, 1, 1, w.sh.lanes);
        CUDA_CHECK(cudaGetLastError());
    }
}

auto worker_begin_rangefinders(Worker& w, const std::vector<int>* fallback) -> void
{
    if (fallback and fallback->size() != w.failure_log_count)
    {
        qn::set_err(QNPEPS_ELOC_ERR_INTERNAL);
        return;
    }
    if (fallback)
    {
        for (const int info : *fallback)
        {
            if (info < 0)
            {
                qn::set_err(QNPEPS_ELOC_ERR_INTERNAL);
                return;
            }
        }
    }
    w.rangefinder_call = 0;
    w.fallback_info = fallback ? fallback->data() : nullptr;
    CUDA_CHECK(cudaMemsetAsync(w.fail_flag, 0, sizeof(int), w.la.stream()));
    usize failure_bytes{};
    if (not arena_product({sizeof(int), w.failure_log_count}, failure_bytes)) return;
    if (fallback)
    {
        CUDA_CHECK(cudaMemcpyAsync(
            w.failure_log, fallback->data(), failure_bytes, cudaMemcpyHostToDevice, w.la.stream()
        ));
    }
    else
        CUDA_CHECK(cudaMemsetAsync(w.failure_log, 0, failure_bytes, w.la.stream()));
}

auto worker_rangefinder_failed(Worker& w) -> int
{
    int fail{};
    CUDA_CHECK(cudaStreamSynchronize(w.la.stream()));
    CUDA_CHECK(cudaMemcpy(&fail, w.fail_flag, sizeof(int), cudaMemcpyDeviceToHost));
    return qn::err_state() == QNPEPS_ELOC_OK ? fail : -1;
}

auto worker_failure_log(Worker& w) -> std::vector<int>
{
    auto log = std::vector<int>(w.failure_log_count);
    usize failure_bytes{};
    if (not arena_product({sizeof(int), w.failure_log_count}, failure_bytes)) return {};
    CUDA_CHECK(cudaMemcpy(log.data(), w.failure_log, failure_bytes, cudaMemcpyDeviceToHost));
    return log;
}

auto expand_rangefinder_fallback(Worker& w, std::vector<int>& fallback) -> bool
{
    const auto expanded = std::vector<int>{worker_failure_log(w)};
    if (qn::err_state() != QNPEPS_ELOC_OK or expanded.size() != fallback.size()) return false;
    bool changed{};
    for (auto index = usize{0}; index < expanded.size(); ++index)
    {
        if (fallback[index] == 0 and expanded[index] != 0)
        {
            changed = true;
        }
    }
    if (not changed) return false;
    fallback = expanded;
    return true;
}

auto worker_ensure(
    Worker& w,
    const Shape& sh,
    bool want_h,
    bool want_fb,
    const cf* peps,
    const u8* samples,
    cudaStream_t stream,
    ContextArena& context_arena
) -> void
{
    if (w.arena
        and (w.sh.lanes != sh.lanes or w.sh.density != sh.density or w.sh.density_cutoff != sh.density_cutoff or w.want_h != want_h or w.want_fb != want_fb))
    {
        auto& linalg = w.la;
        worker_teardown(w);
        w.~Worker();
        new (&w) Worker{linalg};
    }
    if (not w.arena)
    {
        worker_setup(w, sh, want_h, want_fb, peps, samples, stream, context_arena);
        return;
    }
    w.peps = peps;
    w.samples = samples;
    CUDA_CHECK(cudaMemsetAsync(w.fail_flag, 0, sizeof(int), stream));
}

}
