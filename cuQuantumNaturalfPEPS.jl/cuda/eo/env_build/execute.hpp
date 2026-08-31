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
#include "outputs.hpp"
#include "planning.hpp"
#include "graph_cache.hpp"

#line 4909 "cuda/eo/env_build.cu"

auto qn_eloc_envs_logpsi(
    const QnpepsElocConfig& cfg,
    const cf* peps,
    const u8* samples,
    i64 n_samples,
    f64* logpsi_out,
    Linalg& linalg
) -> void
{
    using namespace qn_eloc::env;
    auto transient_arena = linalg.transient_arena();
    auto& context_arena = transient_arena.cursor();
    Worker w{linalg};
    DEFER([&] { worker_teardown(w); });
    const auto stream = linalg.stream();
    int sites_int{};
    i64 sample_sites{};
    i64 output_elements{};
    if (not arena_int(
            {static_cast<std::uint64_t>(cfg.lx), static_cast<std::uint64_t>(cfg.ly)}, sites_int
        )
        or not arena_slot(
            {static_cast<std::uint64_t>(n_samples), static_cast<std::uint64_t>(sites_int)},
            sample_sites
        )
        or not arena_slot({2u, static_cast<std::uint64_t>(n_samples)}, output_elements))
        return;
    const auto sites = i64{sites_int};
    (void) sample_sites;
    auto done = i64{0};
    while (done < n_samples)
    {
        const auto density = bool{cfg.truncation_route == 6};
        const auto lanes =
            int{density ? 1 : static_cast<int>(std::min<i64>(cfg.meo, n_samples - done))};
        i64 output_offset{};
        if (not arena_slot({2u, static_cast<std::uint64_t>(done)}, output_offset)) return;
        auto wave_samples{samples + done * sites};
        auto sh = Shape{
            cfg.lx,
            cfg.ly,
            cfg.dim_phys,
            cfg.dim_bond,
            cfg.chi_eo,
            lanes,
            density,
            cfg.density_cutoff
        };
        worker_ensure(w, sh, false, false, peps, wave_samples, stream, context_arena);
        if (qn::err_state() != QNPEPS_ELOC_OK) return;
        worker_begin_rangefinders(w, nullptr);
        compute_logpsi(w, logpsi_out + output_offset, false);
        auto fail = int{worker_rangefinder_failed(w)};
        if (fail > 0)
        {
            if ((fail & 2) != 0)
            {
                qn::set_backend_err(
                    QNPEPS_ELOC_ERR_CUDA, "rangefinder_householder", fail, __FILE__, __LINE__
                );
                return;
            }
            auto fallback = std::vector<int>{worker_failure_log(w)};
            for (auto attempt = int{0}; attempt < k_rangefinder_recovery_limit; ++attempt)
            {
                worker_begin_rangefinders(w, &fallback);
                if (qn::err_state() != QNPEPS_ELOC_OK) return;
                compute_logpsi(w, logpsi_out + output_offset, false);
                fail = worker_rangefinder_failed(w);
                if (fail <= 0 or (fail & 2) != 0) break;
                if (not expand_rangefinder_fallback(w, fallback)) break;
            }
        }
        if (fail != 0)
        {
            qn::set_backend_err(
                QNPEPS_ELOC_ERR_CUDA, "rangefinder_recovery", fail, __FILE__, __LINE__
            );
        }
        if (qn::err_state() != QNPEPS_ELOC_OK) return;
        done += lanes;
    }
    (void) output_elements;
}

auto run_cache_execute(
    qn_eloc::env::RunCache& state,
    const cf* peps,
    const u8* samples,
    f64* logpsi_out,
    f64* e_loc_out,
    cf* o_rows_dev,
    cf* o_rows_host,
    cf* T_dev,
    f64 lambda,
    const qn_eloc::env::J2Selection& j2_selection
) -> void
{
    using namespace qn_eloc::env;
    const auto& cfg{state.cfg};
    const auto n_samples = i64{state.n_samples};
    auto tt{&state.term_table};
    const auto stream = cudaStream_t{state.stream};
    int sites_int{};
    i64 sample_sites{};
    if (not arena_int(
            {static_cast<std::uint64_t>(cfg.lx), static_cast<std::uint64_t>(cfg.ly)}, sites_int
        )
        or not arena_slot(
            {static_cast<std::uint64_t>(n_samples), static_cast<std::uint64_t>(sites_int)},
            sample_sites
        ))
        return;
    const auto sites = i64{sites_int};
    const auto want_o = bool{(state.flags & QNPEPS_ELOC_CTX_O_ROWS) != 0};
    const auto want_gram = bool{(state.flags & QNPEPS_ELOC_CTX_GRAM) != 0};
    const auto compact_count = i64{state.compact_count};
    i64 output_elements{};
    i64 o_elements{};
    int gram_samples{};
    int gram_compact{};
    usize output_bytes{};
    usize o_bytes{};
    usize gram_bytes{};
    if (not arena_slot({2u, static_cast<std::uint64_t>(n_samples)}, output_elements)
        or not arena_product(
            {sizeof(f64), static_cast<std::uint64_t>(output_elements)}, output_bytes
        ))
        return;
    if (want_o
        and (not arena_slot({static_cast<std::uint64_t>(n_samples), static_cast<std::uint64_t>(compact_count)}, o_elements) or not arena_product({sizeof(cf), static_cast<std::uint64_t>(o_elements)}, o_bytes)))
        return;
    if (want_gram and (not arena_int({static_cast<std::uint64_t>(n_samples)}, gram_samples) or not arena_int({static_cast<std::uint64_t>(compact_count)}, gram_compact) or not arena_product({sizeof(cf), static_cast<std::uint64_t>(n_samples), static_cast<std::uint64_t>(n_samples)}, gram_bytes)))
        return;
    (void) output_bytes;
    (void) o_bytes;
    (void) gram_bytes;
    const auto& ok_off{state.ok_off};
    const auto& ok_slice{state.ok_slice};
    const auto& orig_terms{state.orig_terms};
    const auto& tr_terms{state.tr_terms};
    const auto active_compact_min = int{state.active_compact_min};
    auto peps_t{state.peps_t};
    auto samples_t{state.samples_t};
    auto d_a{state.d_a};
    auto d_b{state.d_b};
    auto d_j{state.d_j};
    auto value_buf{state.value_buf};
    auto logpsi_tr{state.logpsi_tr};
    auto gscratch{state.gscratch};
    auto d_boff{state.d_boff};
    auto d_bslice{state.d_bslice};
    auto spins_i32{state.spins_i32};
    auto& orig_active{state.orig_active};
    auto& tr_active{state.tr_active};
    auto graph_binding{
        state.graph_enabled
            ? find_graph_binding(state, peps, samples, logpsi_out, e_loc_out, o_rows_dev)
            : nullptr
    };

    if (not tr_terms.empty())
    {
        const auto preprocess = [&]
        {
            build_transposed_peps(cfg, peps, peps_t, stream);
            if (qn::err_state() != QNPEPS_ELOC_OK) return;
            const auto threads = int{256};
            const auto total = i64{sample_sites};
            const auto blocks =
                int{static_cast<int>(std::min<i64>(4096, (total + threads - 1) / threads))};
            cu_transpose_samples<<<blocks, threads, 0, stream>>>(
                samples, samples_t, cfg.lx, cfg.ly, n_samples
            );
            CUDA_CHECK(cudaGetLastError());
        };
        run_graph_region(state, graph_binding ? &graph_binding->preprocess : nullptr, preprocess);
    }
    if (want_gram)
    {
        const auto total = i64{sample_sites};
        const auto threads = int{256};
        const auto blocks =
            int{static_cast<int>(std::min<i64>(4096, (total + threads - 1) / threads))};
        cu_u8_to_i32<<<blocks, threads, 0, stream>>>(spins_i32, samples, total);
        CUDA_CHECK(cudaGetLastError());
    }
    if (qn::err_state() != QNPEPS_ELOC_OK) return;

    const auto orig_fb = bool{state.orig_fb};
    const auto tr_fb = bool{state.tr_fb};
    const auto orig_h = bool{state.orig_h};
    const std::vector<std::pair<i64, int>>& wave_ranges{state.wave_ranges};
    auto& w{state.w};
    auto& wt{state.wt};
    auto done = i64{0};
    usize wave_index{};
    while (done < n_samples)
    {
        const auto density = bool{cfg.truncation_route == 6};
        const auto lanes =
            int{density ? 1 : static_cast<int>(std::min<i64>(cfg.meo, n_samples - done))};
        const auto so = i64{done * sites};
        i64 output_offset{};
        usize energy_bytes{};
        usize o_wave_bytes{};
        if (not arena_slot({2u, static_cast<std::uint64_t>(done)}, output_offset)
            or not arena_product({sizeof(f64), 2u, static_cast<std::uint64_t>(lanes)}, energy_bytes)
            or not arena_product(
                {sizeof(cf),
                 static_cast<std::uint64_t>(lanes),
                 static_cast<std::uint64_t>(compact_count)},
                o_wave_bytes
            ))
            return;
        const auto j2_group = int{j2_group_for_wave(j2_selection, wave_index)};
        const auto j2_exact = bool{j2_selection.mode == QNPEPS_ELOC_J2_EXACT};
        auto wave_graphs{graph_binding ? &graph_binding->waves[wave_index] : nullptr};
        auto sh = Shape{
            cfg.lx,
            cfg.ly,
            cfg.dim_phys,
            cfg.dim_bond,
            cfg.chi_eo,
            lanes,
            density,
            cfg.density_cutoff
        };
        worker_ensure(w, sh, orig_h, orig_fb, peps, samples + so, stream, *state.context_arena);
        if (qn::err_state() != QNPEPS_ELOC_OK) return;
        prepare_active_lists(orig_active, w.samples, sites, lanes, stream);
        if (qn::err_state() != QNPEPS_ELOC_OK) return;
        const auto run_orig = [&](const std::vector<int>* fallback) -> int
        {
            worker_begin_rangefinders(w, fallback);
            if (qn::err_state() != QNPEPS_ELOC_OK) return -1;
            run_graph_region(
                state,
                wave_graphs ? &wave_graphs->orig_logpsi : nullptr,
                [&] { compute_logpsi(w, logpsi_out + output_offset, orig_h or orig_fb); },
                &w.rangefinder_call
            );
            if (qn::err_state() != QNPEPS_ELOC_OK) return -1;
            {
                run_graph_region(
                    state,
                    wave_graphs ? &wave_graphs->orig_stage[j2_graph_slot(j2_selection, j2_group)]
                                : nullptr,
                    [&]
                    {
                        if (j2_exact)
                            emit_stage_c(w, orig_h, orig_fb);
                        else if (j2_selection.mode == QNPEPS_ELOC_J2_HALF_COLUMN_PAIRS)
                            emit_stage_c_j2_column_group(w, orig_h, orig_fb, j2_group);
                        else
                            emit_stage_c_j2_group(w, orig_h, orig_fb, j2_group);
                        CUDA_CHECK(
                            cudaMemsetAsync(e_loc_out + output_offset, 0, energy_bytes, stream)
                        );
                        if (j2_exact)
                        {
                            launch_diagonal(
                                w, e_loc_out + output_offset, d_a, d_b, d_j, tt->n_diag
                            );
                        }
                        else
                            launch_diagonal_j2_half(
                                w,
                                e_loc_out + output_offset,
                                d_a,
                                d_b,
                                d_j,
                                tt->n_diag,
                                j2_selection.mode,
                                j2_group
                            );
                    },
                    &w.rangefinder_call
                );
            }
            if (qn::err_state() != QNPEPS_ELOC_OK) return -1;
            {
                for (const FlipInst& t : orig_terms)
                {
                    if (not j2_exact and j2_group >= 0 and t.bucket == Bucket::fourbody
                        and j2_flip_group(t, j2_selection.mode) != j2_group)
                        continue;
                    const auto active =
                        int{t.active_slot >= 0
                                ? orig_active.host_counts[static_cast<usize>(t.active_slot)]
                                : -1};
                    auto active_indices{
                        t.active_slot >= 0
                            ? orig_active.indices
                                  + static_cast<i64>(t.active_slot) * orig_active.max_lanes
                            : nullptr
                    };
                    auto selected = FlipInst{t};
                    if (not j2_exact and j2_group >= 0 and selected.bucket == Bucket::fourbody)
                    {
                        selected.coeff_re *= 2.0;
                        selected.coeff_im *= 2.0;
                    }
                    eval_flip_inst(
                        w,
                        selected,
                        value_buf,
                        e_loc_out + output_offset,
                        logpsi_out + output_offset,
                        active_indices,
                        active,
                        active_compact_min
                    );
                }
            }
            if (qn::err_state() != QNPEPS_ELOC_OK) return -1;
            if (want_o)
            {
                auto wave_base{o_rows_dev + done * compact_count};
                run_graph_region(
                    state,
                    wave_graphs ? &wave_graphs->orig_o : nullptr,
                    [&]
                    {
                        emit_o_rows(
                            w,
                            wave_base,
                            compact_count,
                            ok_off,
                            ok_slice,
                            logpsi_out + output_offset,
                            gscratch
                        );
                    },
                    &w.rangefinder_call
                );
                if (o_rows_host)
                {
                    CUDA_CHECK(cudaMemcpyAsync(
                        o_rows_host + done * compact_count,
                        wave_base,
                        o_wave_bytes,
                        cudaMemcpyDeviceToHost,
                        stream
                    ));
                }
            }
            return worker_rangefinder_failed(w);
        };
        const auto run_orig_recovered = [&]() -> int
        {
            std::vector<int> transient_fallback{};
            auto& fallback{wave_graphs ? wave_graphs->orig_fallback : transient_fallback};
            const auto active_fallback = [&]() -> const std::vector<int>*
            { return fallback.empty() ? nullptr : &fallback; };
            auto orig_fail = int{run_orig(active_fallback())};
            if (orig_fail <= 0) return orig_fail;
            if ((orig_fail & 2) != 0)
            {
                return orig_fail;
            }
            bool changed{};
            if (fallback.empty())
            {
                fallback = worker_failure_log(w);
                changed = qn::err_state() == QNPEPS_ELOC_OK;
            }
            else
                changed = expand_rangefinder_fallback(w, fallback);
            if (not changed) return orig_fail;
            if (wave_graphs) destroy_orig_graphs(*wave_graphs);
            for (auto attempt = int{0}; attempt < k_rangefinder_recovery_limit; ++attempt)
            {
                orig_fail = run_orig(&fallback);
                if (orig_fail <= 0 or (orig_fail & 2) != 0) break;
                if (not expand_rangefinder_fallback(w, fallback)) break;
                if (wave_graphs) destroy_orig_graphs(*wave_graphs);
            }
            return orig_fail;
        };
        auto fail = int{run_orig_recovered()};
        if (fail != 0)
        {
            qn::set_backend_err(
                QNPEPS_ELOC_ERR_CUDA, "rangefinder_recovery", fail, __FILE__, __LINE__
            );
        }
        if (qn::err_state() != QNPEPS_ELOC_OK) return;
        if (not tr_terms.empty())
        {
            auto sht = Shape{
                cfg.ly,
                cfg.lx,
                cfg.dim_phys,
                cfg.dim_bond,
                cfg.chi_eo,
                lanes,
                density,
                cfg.density_cutoff
            };
            worker_ensure(
                wt, sht, true, tr_fb, peps_t, samples_t + so, stream, *state.context_arena
            );
            if (qn::err_state() != QNPEPS_ELOC_OK) return;
            prepare_active_lists(tr_active, wt.samples, sites, lanes, stream);
            if (qn::err_state() != QNPEPS_ELOC_OK) return;
            const auto run_transposed = [&](const std::vector<int>* fallback) -> int
            {
                worker_begin_rangefinders(wt, fallback);
                if (qn::err_state() != QNPEPS_ELOC_OK) return -1;
                run_graph_region(
                    state,
                    wave_graphs ? &wave_graphs->tr_logpsi : nullptr,
                    [&] { compute_logpsi(wt, logpsi_tr, true); },
                    &wt.rangefinder_call
                );
                if (qn::err_state() != QNPEPS_ELOC_OK) return -1;
                {
                    run_graph_region(
                        state,
                        wave_graphs ? &wave_graphs->tr_stage : nullptr,
                        [&] { emit_stage_c(wt, true, tr_fb); },
                        &wt.rangefinder_call
                    );
                }
                if (qn::err_state() != QNPEPS_ELOC_OK) return -1;
                {
                    for (const FlipInst& t : tr_terms)
                    {
                        const auto active =
                            int{t.active_slot >= 0
                                    ? tr_active.host_counts[static_cast<usize>(t.active_slot)]
                                    : -1};
                        auto active_indices{
                            t.active_slot >= 0
                                ? tr_active.indices
                                      + static_cast<i64>(t.active_slot) * tr_active.max_lanes
                                : nullptr
                        };
                        eval_flip_inst(
                            wt,
                            t,
                            value_buf,
                            e_loc_out + output_offset,
                            logpsi_tr,
                            active_indices,
                            active,
                            active_compact_min
                        );
                    }
                }
                return worker_rangefinder_failed(wt);
            };
            std::vector<int> transient_fallback{};
            auto& fallback{wave_graphs ? wave_graphs->tr_fallback : transient_fallback};
            auto fail = int{run_transposed(fallback.empty() ? nullptr : &fallback)};
            if (fail > 0)
            {
                if ((fail & 2) != 0)
                {
                    qn::set_backend_err(
                        QNPEPS_ELOC_ERR_CUDA, "rangefinder_householder", fail, __FILE__, __LINE__
                    );
                    return;
                }
                bool changed{};
                if (fallback.empty())
                {
                    fallback = worker_failure_log(wt);
                    changed = qn::err_state() == QNPEPS_ELOC_OK;
                }
                else
                    changed = expand_rangefinder_fallback(wt, fallback);
                if (not changed)
                {
                    qn::set_backend_err(
                        QNPEPS_ELOC_ERR_CUDA, "rangefinder_recovery", fail, __FILE__, __LINE__
                    );
                    return;
                }
                if (wave_graphs) destroy_transposed_graphs(*wave_graphs);
                for (auto attempt = int{0}; attempt < k_rangefinder_recovery_limit; ++attempt)
                {
                    fail = run_orig_recovered();
                    if (fail != 0) break;
                    fail = run_transposed(&fallback);
                    if (fail <= 0 or (fail & 2) != 0) break;
                    if (not expand_rangefinder_fallback(wt, fallback)) break;
                    if (wave_graphs) destroy_transposed_graphs(*wave_graphs);
                }
            }
            if (fail != 0)
            {
                qn::set_backend_err(
                    QNPEPS_ELOC_ERR_CUDA, "rangefinder_recovery", fail, __FILE__, __LINE__
                );
            }
        }
        if (qn::err_state() != QNPEPS_ELOC_OK) return;
        done += lanes;
        ++wave_index;
    }

    if (want_gram)
    {
        const auto ntot = int{gram_samples};
        const auto n_blocks = int{static_cast<int>(sites)};
        const auto tile = [&](const std::pair<i64, int>& sr, const std::pair<i64, int>& ur) -> void
        {
            qn_eloc::launch_gram_tile(
                T_dev,
                ntot,
                gram_compact,
                n_blocks,
                o_rows_dev,
                o_rows_dev,
                spins_i32,
                spins_i32,
                d_boff,
                d_bslice,
                static_cast<int>(sr.first),
                sr.second,
                static_cast<int>(ur.first),
                ur.second,
                stream
            );
            CUDA_CHECK(cudaGetLastError());
        };
        for (auto j = usize{0}; j < wave_ranges.size(); ++j)
        {
            for (auto i = usize{0}; i <= j; ++i)
            {
                tile(wave_ranges[i], wave_ranges[j]);
                if (i != j) tile(wave_ranges[j], wave_ranges[i]);
            }
        }
        const auto threads = int{256};
        cu_add_lambda_diag<<<(ntot + threads - 1) / threads, threads, 0, stream>>>(
            T_dev, ntot, lambda
        );
        CUDA_CHECK(cudaGetLastError());
    }
}
