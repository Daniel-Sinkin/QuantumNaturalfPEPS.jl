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
#include "execute.hpp"

#line 5497 "cuda/eo/env_build.cu"

struct qnpeps_eloc_ctx
{
    explicit qnpeps_eloc_ctx(std::unique_ptr<qnpeps::Session> incoming)
        : session(std::move(incoming)), state(session->linalg())
    {
    }

    std::unique_ptr<qnpeps::Session> session{};
    qn_eloc::env::RunCache state;
};

auto qn_eloc_run_impl(
    const QnpepsElocConfig& cfg,
    const cf* peps,
    const u8* samples,
    i64 n_samples,
    const QnpepsElocTermTable* tt,
    f64* logpsi_out,
    f64* e_loc_out,
    cf* o_rows_dev,
    cf* o_rows_host,
    cf* T_dev,
    f64 lambda,
    Linalg& linalg
) -> void
{
    using namespace qn_eloc::env;
    auto transient_arena = linalg.transient_arena();
    auto& context_arena = transient_arena.cursor();
    RunCache state{linalg};
    DEFER([&] { run_cache_teardown(state); });
    auto flags = uint32_t{QNPEPS_ELOC_CTX_ENERGY};
    if (o_rows_dev) flags |= QNPEPS_ELOC_CTX_O_ROWS;
    if (T_dev) flags |= QNPEPS_ELOC_CTX_GRAM;
    run_cache_setup(state, cfg, n_samples, *tt, flags, context_arena);
    if (qn::err_state() != QNPEPS_ELOC_OK) return;
    run_cache_execute(
        state,
        peps,
        samples,
        logpsi_out,
        e_loc_out,
        o_rows_dev,
        o_rows_host,
        T_dev,
        lambda,
        J2Selection{}
    );
}

auto qn_eloc_ctx_create_impl(
    const QnpepsElocConfig& cfg,
    i64 n_samples,
    const QnpepsElocTermTable& terms,
    uint32_t flags,
    std::unique_ptr<qnpeps::Session> session,
    qnpeps_eloc_ctx** out
) -> void
{
    using namespace qn_eloc::env;
    auto* ctx{new (std::nothrow) qnpeps_eloc_ctx{std::move(session)}};
    if (not ctx)
    {
        qn::set_err(QNPEPS_ELOC_ERR_OOM);
        return;
    }
    auto& context_arena = ctx->session->linalg().persistent_arena();
    run_cache_setup(ctx->state, cfg, n_samples, terms, flags, context_arena);
    if (qn::err_state() != QNPEPS_ELOC_OK)
    {
        run_cache_teardown(ctx->state);
        delete ctx;
        return;
    }
    ctx->state.graph_enabled =
        cfg.truncation_route != 6 and graph_requested_from_env() and n_samples % cfg.meo == 0;
    *out = ctx;
}

auto qn_eloc_ctx_destroy_impl(qnpeps_eloc_ctx* ctx) -> void
{
    if (not ctx) return;
    qn_eloc::env::run_cache_teardown(ctx->state);
    delete ctx;
}

auto qn_eloc_ctx_run_async_impl(
    qnpeps_eloc_ctx& ctx,
    const cf* peps,
    const u8* samples,
    f64* logpsi_out,
    f64* e_loc_out,
    cf* o_rows_dev,
    cf* o_rows_host,
    cf* T_dev,
    f64 lambda,
    uint32_t j2_mode,
    uint32_t j2_draw,
    u64 j2_seed,
    u64 j2_epoch
) -> void
{
    using namespace qn_eloc::env;
    auto& state{ctx.state};
    if (not state.ready)
    {
        qn::set_err(QNPEPS_ELOC_ERR_INTERNAL);
        return;
    }
    const auto want_o = bool{(state.flags & QNPEPS_ELOC_CTX_O_ROWS) != 0};
    const auto want_gram = bool{(state.flags & QNPEPS_ELOC_CTX_GRAM) != 0};
    if (want_o != (o_rows_dev != nullptr) or want_gram != (T_dev != nullptr)
        or (not want_o and o_rows_host))
    {
        qn::set_err(QNPEPS_ELOC_ERR_BAD_CONFIG);
        return;
    }
    run_cache_execute(
        state,
        peps,
        samples,
        logpsi_out,
        e_loc_out,
        o_rows_dev,
        o_rows_host,
        T_dev,
        lambda,
        J2Selection{j2_mode, j2_draw, j2_seed, j2_epoch}
    );
}

auto qn_eloc_ctx_run_impl(
    qnpeps_eloc_ctx& ctx,
    const cf* peps,
    const u8* samples,
    f64* logpsi_out,
    f64* e_loc_out,
    cf* o_rows_dev,
    cf* o_rows_host,
    cf* T_dev,
    f64 lambda,
    uint32_t j2_mode,
    uint32_t j2_draw,
    u64 j2_seed,
    u64 j2_epoch
) -> void
{
    qn_eloc_ctx_run_async_impl(
        ctx,
        peps,
        samples,
        logpsi_out,
        e_loc_out,
        o_rows_dev,
        o_rows_host,
        T_dev,
        lambda,
        j2_mode,
        j2_draw,
        j2_seed,
        j2_epoch
    );
    if (qn::err_state() != QNPEPS_ELOC_OK) return;
    auto& state{ctx.state};
    CUDA_CHECK(cudaStreamSynchronize(state.stream));
}

auto qn_eloc_ctx_stats_impl(const qnpeps_eloc_ctx&, QnpepsElocCtxStats& out) -> void
{
    const auto struct_size = uint32_t{out.struct_size};
    out = {};
    out.struct_size = struct_size;
}

auto qn_eloc_gram_tile_impl(
    const QnpepsElocConfig& cfg,
    const cf* rows_a,
    const u8* samples_a,
    i64 ns_a,
    const cf* rows_b,
    const u8* samples_b,
    i64 ns_b,
    cf* tile_out,
    cudaStream_t stream
) -> void
{
    using namespace qn_eloc::env;
    int n_blocks{};
    int ns_a_int{};
    int ns_b_int{};
    i64 spins_a_count{};
    i64 spins_b_count{};
    if (not arena_int(
            {static_cast<std::uint64_t>(cfg.lx), static_cast<std::uint64_t>(cfg.ly)}, n_blocks
        )
        or not arena_int({static_cast<std::uint64_t>(ns_a)}, ns_a_int)
        or not arena_int({static_cast<std::uint64_t>(ns_b)}, ns_b_int)
        or not arena_slot(
            {static_cast<std::uint64_t>(ns_a), static_cast<std::uint64_t>(n_blocks)}, spins_a_count
        )
        or not arena_slot(
            {static_cast<std::uint64_t>(ns_b), static_cast<std::uint64_t>(n_blocks)}, spins_b_count
        ))
        return;
    const auto sites = i64{n_blocks};

    const auto osh = Shape{cfg.lx, cfg.ly, cfg.dim_phys, cfg.dim_bond, cfg.chi_eo, 1};
    std::vector<i64> ok_off{};
    std::vector<int> ok_slice{};
    const auto compact_count = i64{ok_layout(osh, ok_off, ok_slice)};
    int compact_count_int{};
    if (qn::err_state() != QNPEPS_ELOC_OK or compact_count < 1
        or not arena_int({static_cast<std::uint64_t>(compact_count)}, compact_count_int))
        return;

    int* d_boff{};
    int* d_bslice{};
    int* spins_a{};
    int* spins_b{};
    DEFER(
        [&]
        {
            maybe_free_device(d_boff);
            maybe_free_device(d_bslice);
            maybe_free_device(spins_a);
            maybe_free_device(spins_b);
        }
    );

    usize block_bytes{};
    usize spins_a_bytes{};
    usize spins_b_bytes{};
    if (not arena_product({sizeof(int), static_cast<std::uint64_t>(n_blocks)}, block_bytes)
        or not arena_product(
            {sizeof(int), static_cast<std::uint64_t>(spins_a_count)}, spins_a_bytes
        )
        or not arena_product(
            {sizeof(int), static_cast<std::uint64_t>(spins_b_count)}, spins_b_bytes
        ))
        return;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_boff), block_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bslice), block_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&spins_a), spins_a_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&spins_b), spins_b_bytes));
    if (qn::err_state() != QNPEPS_ELOC_OK) return;

    auto hoff = std::vector<int>(static_cast<usize>(n_blocks));
    for (auto b = int{0}; b < n_blocks; ++b)
        hoff[static_cast<usize>(b)] = static_cast<int>(ok_off[static_cast<usize>(b)]);
    CUDA_CHECK(cudaMemcpy(d_boff, hoff.data(), block_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bslice, ok_slice.data(), block_bytes, cudaMemcpyHostToDevice));
    if (qn::err_state() != QNPEPS_ELOC_OK) return;

    const auto threads = int{256};
    const auto convert = [&](int* dst, const u8* src, i64 count) -> void
    {
        i64 total{};
        if (not arena_slot(
                {static_cast<std::uint64_t>(count), static_cast<std::uint64_t>(sites)}, total
            ))
            return;
        const auto blocks =
            int{static_cast<int>(std::min<i64>(4096, (total + threads - 1) / threads))};
        cu_u8_to_i32<<<blocks, threads, 0, stream>>>(dst, src, total);
        CUDA_CHECK(cudaGetLastError());
    };
    convert(spins_a, samples_a, ns_a);
    convert(spins_b, samples_b, ns_b);
    if (qn::err_state() != QNPEPS_ELOC_OK) return;

    qn_eloc::launch_gram_tile(
        tile_out,
        ns_b_int,
        compact_count_int,
        n_blocks,
        rows_a,
        rows_b,
        spins_a,
        spins_b,
        d_boff,
        d_bslice,
        0,
        ns_a_int,
        0,
        ns_b_int,
        stream
    );
    CUDA_CHECK(cudaGetLastError());
}

auto qn_eloc_run_scratch(const QnpepsElocConfig&, i64, const QnpepsElocTermTable*) -> u64
{
    return static_cast<u64>(qnpeps::arena_reservation_bytes());
}

auto qn_eloc_compact_count(const QnpepsElocConfig& cfg) -> i64
{
    const auto sh = qn_eloc::env::Shape{cfg.lx, cfg.ly, cfg.dim_phys, cfg.dim_bond, cfg.chi_eo, 1};
    std::vector<i64> off{};
    std::vector<int> slice{};
    return qn_eloc::env::ok_layout(sh, off, slice);
}
