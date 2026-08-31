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

namespace qn_eloc::env
{

#line 4149 "cuda/eo/env_build.cu"

struct GraphSlot
{
    cudaGraphExec_t exec{};
    int rangefinder_calls{};
    bool warmed{};
    bool failed{};
};

struct WaveGraphs
{
    GraphSlot orig_logpsi{};
    std::array<GraphSlot, 7> orig_stage{};
    GraphSlot orig_o{};
    GraphSlot tr_logpsi{};
    GraphSlot tr_stage{};
    std::vector<int> orig_fallback{};
    std::vector<int> tr_fallback{};
};

struct GraphBinding
{
    const cf* peps{};
    const u8* samples{};
    f64* logpsi_out{};
    f64* e_loc_out{};
    cf* o_rows_dev{};
    GraphSlot preprocess{};
    std::vector<WaveGraphs> waves{};
};

struct J2Selection
{
    uint32_t mode{QNPEPS_ELOC_J2_EXACT};
    uint32_t draw{QNPEPS_ELOC_J2_BALANCED};
    u64 seed{};
    u64 epoch{};
};

struct RunCache
{
    explicit RunCache(Linalg& linalg) : w(linalg), wt(linalg) {}

    QnpepsElocConfig cfg{};
    i64 n_samples{};
    uint32_t flags{};
    cudaStream_t stream{};
    std::vector<QnpepsElocDiagBond> diag_terms{};
    std::vector<QnpepsElocFlipTerm> flip_terms{};
    QnpepsElocTermTable term_table{};
    std::vector<i64> ok_off{};
    std::vector<int> ok_slice{};
    i64 compact_count{};
    std::vector<FlipInst> orig_terms{};
    std::vector<FlipInst> tr_terms{};
    bool active_compact{};
    int active_compact_min{};
    ActiveLists orig_active{};
    ActiveLists tr_active{};
    cf* peps_t{};
    u8* samples_t{};
    int* d_a{};
    int* d_b{};
    f64* d_j{};
    cf* value_buf{};
    f64* logpsi_tr{};
    cf* gscratch{};
    int* d_boff{};
    int* d_bslice{};
    int* spins_i32{};
    bool orig_fb{};
    bool orig_hterms{};
    bool tr_fb{};
    bool orig_h{};
    Worker w;
    Worker wt;
    std::vector<std::pair<i64, int>> wave_ranges{};
    std::vector<GraphBinding> graph_bindings{};
    bool graph_enabled{};
    ContextArena* context_arena{};
    bool ready{};
};

template <typename T>
auto cache_alloc(RunCache& state, T*& pointer, usize count) -> void
{
    pointer = state.context_arena->take<T>(count);
}

auto graph_requested_from_env() -> bool
{
    auto value{std::getenv("QNPEPS_ELOC_GRAPH")};
    return value == nullptr or value[0] == '\0' or value[0] != '0';
}

auto splitmix64(u64 value) -> u64
{
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

auto j2_group_for_wave(const J2Selection& selection, usize wave_index) -> int
{
    if (selection.mode == QNPEPS_ELOC_J2_EXACT) return -1;
    if (selection.draw == QNPEPS_ELOC_J2_FORCE_ALL) return -2;
    if (selection.draw == QNPEPS_ELOC_J2_FORCE_GROUP_0) return 0;
    if (selection.draw == QNPEPS_ELOC_J2_FORCE_GROUP_1) return 1;
    const auto first =
        int{static_cast<int>(splitmix64(selection.seed ^ splitmix64(selection.epoch)) & 1ull)};
    return first ^ static_cast<int>(wave_index & 1u);
}

auto j2_graph_slot(const J2Selection& selection, int group) -> usize
{
    if (selection.mode == QNPEPS_ELOC_J2_EXACT) return 0;
    const auto orientation = usize{selection.mode == QNPEPS_ELOC_J2_HALF_COLUMN_PAIRS ? 1u : 0u};
    if (group < 0) return 3u + 3u * orientation;
    return 1u + 3u * orientation + static_cast<usize>(group);
}

auto j2_flip_group(const FlipInst& term, uint32_t mode) -> int
{
    return mode == QNPEPS_ELOC_J2_HALF_COLUMN_PAIRS ? term.j2_column_group : term.j2_group;
}

auto destroy_graph_slot(GraphSlot& slot) -> void
{
    if (slot.exec) CUDA_NOCHECK(cudaGraphExecDestroy(slot.exec));
    slot = GraphSlot{};
}

auto destroy_orig_graphs(WaveGraphs& wave) -> void
{
    destroy_graph_slot(wave.orig_logpsi);
    for (GraphSlot& slot : wave.orig_stage)
        destroy_graph_slot(slot);
    destroy_graph_slot(wave.orig_o);
}

auto destroy_transposed_graphs(WaveGraphs& wave) -> void
{
    destroy_graph_slot(wave.tr_logpsi);
    destroy_graph_slot(wave.tr_stage);
}

auto destroy_graph_binding(GraphBinding& binding) -> void
{
    destroy_graph_slot(binding.preprocess);
    for (WaveGraphs& wave : binding.waves)
    {
        destroy_orig_graphs(wave);
        destroy_transposed_graphs(wave);
    }
}

auto find_graph_binding(
    RunCache& state,
    const cf* peps,
    const u8* samples,
    f64* logpsi_out,
    f64* e_loc_out,
    cf* o_rows_dev
) -> GraphBinding*
{
    for (GraphBinding& binding : state.graph_bindings)
    {
        if (binding.peps == peps and binding.samples == samples and binding.logpsi_out == logpsi_out
            and binding.e_loc_out == e_loc_out and binding.o_rows_dev == o_rows_dev)
            return &binding;
    }
    GraphBinding binding{};
    binding.peps = peps;
    binding.samples = samples;
    binding.logpsi_out = logpsi_out;
    binding.e_loc_out = e_loc_out;
    binding.o_rows_dev = o_rows_dev;
    binding.waves.resize(state.wave_ranges.size());
    state.graph_bindings.push_back(std::move(binding));
    return &state.graph_bindings.back();
}

template <typename Enqueue>
auto run_graph_region(
    RunCache& state, GraphSlot* slot, Enqueue&& enqueue, int* rangefinder_call = nullptr
) -> void
{
    if (not slot or not state.graph_enabled)
    {
        enqueue();
        return;
    }
    if (slot->exec)
    {
        CUDA_CHECK(cudaGraphLaunch(slot->exec, state.stream));
        if (rangefinder_call) *rangefinder_call += slot->rangefinder_calls;
        return;
    }
    if (slot->failed)
    {
        qn::set_err(QNPEPS_ELOC_ERR_CUDA);
        return;
    }
    if (not slot->warmed)
    {
        const auto call_before = int{rangefinder_call ? *rangefinder_call : 0};
        enqueue();
        if (rangefinder_call) slot->rangefinder_calls = *rangefinder_call - call_before;
        slot->warmed = qn::err_state() == QNPEPS_ELOC_OK;
        return;
    }

    cudaGraph_t graph{};
    const auto begin_status =
        cudaError_t{cudaStreamBeginCapture(state.stream, cudaStreamCaptureModeThreadLocal)};
    if (begin_status != cudaSuccess)
    {
        cudaGetLastError();
        slot->failed = true;
        qn::set_err(QNPEPS_ELOC_ERR_CUDA);
        return;
    }
    const auto call_before = int{rangefinder_call ? *rangefinder_call : 0};
    enqueue();
    if (rangefinder_call and *rangefinder_call - call_before != slot->rangefinder_calls)
        qn::set_err(QNPEPS_ELOC_ERR_INTERNAL);
    const auto end_status = cudaError_t{cudaStreamEndCapture(state.stream, &graph)};
    const auto enqueued = bool{qn::err_state() == QNPEPS_ELOC_OK};
    cudaGraphExec_t exec{};
    const auto instantiate_status = cudaError_t{
        end_status == cudaSuccess and enqueued
            ? cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0)
            : cudaErrorUnknown
    };
    if (end_status == cudaSuccess and enqueued and instantiate_status == cudaSuccess)
    {
        CUDA_NOCHECK(cudaGraphDestroy(graph));
        slot->exec = exec;
        CUDA_CHECK(cudaGraphLaunch(slot->exec, state.stream));
        return;
    }

    if (exec) CUDA_NOCHECK(cudaGraphExecDestroy(exec));
    if (graph) CUDA_NOCHECK(cudaGraphDestroy(graph));
    cudaGetLastError();
    slot->failed = true;
    qn::set_err(QNPEPS_ELOC_ERR_CUDA);
}

auto run_cache_teardown(RunCache& state) -> void
{
    if (state.stream) CUDA_NOCHECK(cudaStreamSynchronize(state.stream));
    for (GraphBinding& binding : state.graph_bindings)
        destroy_graph_binding(binding);
    state.graph_bindings.clear();
    worker_teardown(state.w);
    worker_teardown(state.wt);
    active_lists_teardown(state.orig_active);
    active_lists_teardown(state.tr_active);
    state.peps_t = nullptr;
    state.samples_t = nullptr;
    state.d_a = nullptr;
    state.d_b = nullptr;
    state.d_j = nullptr;
    state.value_buf = nullptr;
    state.logpsi_tr = nullptr;
    state.gscratch = nullptr;
    state.d_boff = nullptr;
    state.d_bslice = nullptr;
    state.spins_i32 = nullptr;
    state.context_arena = nullptr;
    state.ready = false;
}

auto run_cache_setup(
    RunCache& state,
    const QnpepsElocConfig& cfg,
    i64 n_samples,
    const QnpepsElocTermTable& tt,
    uint32_t flags,
    ContextArena& context_arena
) -> void
{
    state.cfg = cfg;
    state.n_samples = n_samples;
    state.flags = flags;
    state.stream = state.w.la.stream();
    state.context_arena = &context_arena;
    if (tt.n_diag > 0) state.diag_terms.assign(tt.diag, tt.diag + tt.n_diag);
    if (tt.n_flip > 0) state.flip_terms.assign(tt.flip, tt.flip + tt.n_flip);
    state.term_table.n_diag = static_cast<int32_t>(state.diag_terms.size());
    state.term_table.diag = state.diag_terms.empty() ? nullptr : state.diag_terms.data();
    state.term_table.n_flip = static_cast<int32_t>(state.flip_terms.size());
    state.term_table.flip = state.flip_terms.empty() ? nullptr : state.flip_terms.data();

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
    const auto osh = Shape{cfg.lx, cfg.ly, cfg.dim_phys, cfg.dim_bond, cfg.chi_eo, 1};
    state.compact_count = ok_layout(osh, state.ok_off, state.ok_slice);
    if (qn::err_state() != QNPEPS_ELOC_OK or state.compact_count < 1) return;
    for (const QnpepsElocFlipTerm& term : state.flip_terms)
    {
        FlipInst fi{};
        int pass{};
        if (not classify_route(cfg, term, fi, pass))
        {
            qn::set_err(QNPEPS_ELOC_ERR_BAD_CONFIG);
            return;
        }
        (pass == 0 ? state.orig_terms : state.tr_terms).push_back(fi);
    }
    state.active_compact = active_compact_requested_from_env();
    state.active_compact_min = active_compact_min_from_env();
    const auto orig_active_slots = int{assign_active_slots(state.orig_terms, state.active_compact)};
    const auto tr_active_slots = int{assign_active_slots(state.tr_terms, state.active_compact)};
    active_lists_setup(
        state.orig_active, state.orig_terms, orig_active_slots, cfg.meo, context_arena
    );
    active_lists_setup(state.tr_active, state.tr_terms, tr_active_slots, cfg.meo, context_arena);
    if (qn::err_state() != QNPEPS_ELOC_OK) return;

    if (not state.tr_terms.empty())
    {
        const auto offt =
            std::vector<i64>{site_offsets(cfg.ly, cfg.lx, cfg.dim_bond, cfg.dim_phys)};
        if (qn::err_state() != QNPEPS_ELOC_OK or offt.size() <= static_cast<usize>(sites)) return;
        cache_alloc(state, state.peps_t, static_cast<usize>(offt[static_cast<usize>(sites)]));
        cache_alloc(state, state.samples_t, static_cast<usize>(sample_sites));
    }

    if (not state.diag_terms.empty())
    {
        auto ha = std::vector<int>(state.diag_terms.size());
        auto hb = std::vector<int>(state.diag_terms.size());
        auto hj = std::vector<f64>(state.diag_terms.size());
        for (auto i = usize{0}; i < state.diag_terms.size(); ++i)
        {
            ha[i] = state.diag_terms[i].site_a;
            hb[i] = state.diag_terms[i].site_b;
            hj[i] = state.diag_terms[i].coeff;
        }
        cache_alloc(state, state.d_a, ha.size());
        cache_alloc(state, state.d_b, hb.size());
        cache_alloc(state, state.d_j, hj.size());
        if (qn::err_state() != QNPEPS_ELOC_OK) return;
        usize index_bytes{};
        usize coefficient_bytes{};
        if (not arena_product({sizeof(int), ha.size()}, index_bytes)
            or not arena_product({sizeof(f64), hj.size()}, coefficient_bytes))
            return;
        CUDA_CHECK(cudaMemcpy(state.d_a, ha.data(), index_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state.d_b, hb.data(), index_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state.d_j, hj.data(), coefficient_bytes, cudaMemcpyHostToDevice));
    }

    cache_alloc(state, state.value_buf, static_cast<usize>(cfg.meo));
    usize logpsi_count{};
    if (not arena_product({2u, static_cast<std::uint64_t>(cfg.meo)}, logpsi_count)) return;
    cache_alloc(state, state.logpsi_tr, logpsi_count);
    const auto want_o = bool{(flags & QNPEPS_ELOC_CTX_O_ROWS) != 0};
    const auto want_gram = bool{(flags & QNPEPS_ELOC_CTX_GRAM) != 0};
    if (want_o) cache_alloc(state, state.gscratch, static_cast<usize>(cfg.meo));
    if (want_gram)
    {
        const auto n_blocks = int{static_cast<int>(sites)};
        cache_alloc(state, state.d_boff, static_cast<usize>(n_blocks));
        cache_alloc(state, state.d_bslice, static_cast<usize>(n_blocks));
        cache_alloc(state, state.spins_i32, static_cast<usize>(sample_sites));
        if (qn::err_state() != QNPEPS_ELOC_OK) return;
        usize block_bytes{};
        if (not arena_product({sizeof(int), static_cast<std::uint64_t>(n_blocks)}, block_bytes))
            return;
        auto hoff = std::vector<int>(static_cast<usize>(n_blocks));
        for (auto b = int{0}; b < n_blocks; ++b)
            hoff[static_cast<usize>(b)] = static_cast<int>(state.ok_off[static_cast<usize>(b)]);
        CUDA_CHECK(cudaMemcpy(state.d_boff, hoff.data(), block_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(state.d_bslice, state.ok_slice.data(), block_bytes, cudaMemcpyHostToDevice)
        );
    }
    if (qn::err_state() != QNPEPS_ELOC_OK) return;

    for (const FlipInst& term : state.orig_terms)
        (term.bucket == Bucket::fourbody ? state.orig_fb : state.orig_hterms) = true;
    for (const FlipInst& term : state.tr_terms)
        if (term.bucket == Bucket::fourbody) state.tr_fb = true;
    state.orig_h = want_o or state.orig_hterms;
    const auto density = bool{cfg.truncation_route == 6};
    const auto lanes = int{density ? 1 : static_cast<int>(std::min<i64>(cfg.meo, n_samples))};
    const auto sh = Shape{
        cfg.lx, cfg.ly, cfg.dim_phys, cfg.dim_bond, cfg.chi_eo, lanes, density, cfg.density_cutoff
    };
    worker_setup(
        state.w, sh, state.orig_h, state.orig_fb, nullptr, nullptr, state.stream, context_arena
    );
    if (not state.tr_terms.empty())
    {
        const auto sht = Shape{
            cfg.ly,
            cfg.lx,
            cfg.dim_phys,
            cfg.dim_bond,
            cfg.chi_eo,
            lanes,
            density,
            cfg.density_cutoff
        };
        worker_setup(
            state.wt, sht, true, state.tr_fb, nullptr, nullptr, state.stream, context_arena
        );
    }
    if (qn::err_state() != QNPEPS_ELOC_OK) return;

    const auto wave_width = int{density ? 1 : cfg.meo};
    for (auto done = i64{0}; done < n_samples; done += wave_width)
    {
        const auto wave_lanes = int{static_cast<int>(std::min<i64>(wave_width, n_samples - done))};
        state.wave_ranges.push_back({done, wave_lanes});
    }
    state.ready = true;
}
}
