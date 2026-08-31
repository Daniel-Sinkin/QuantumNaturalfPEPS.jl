#ifndef QNPEPS_ELOC_ENV_BUILD_CUH
#define QNPEPS_ELOC_ENV_BUILD_CUH

#include "common.cuh"
#include "core/session.cuh"

#include <memory>

struct qnpeps_eloc_ctx;

auto qn_eloc_ctx_create_impl(
    const QnpepsElocConfig& cfg,
    i64 n_samples,
    const QnpepsElocTermTable& terms,
    uint32_t flags,
    std::unique_ptr<qnpeps::Session> session,
    qnpeps_eloc_ctx** out
) -> void;

auto qn_eloc_ctx_destroy_impl(qnpeps_eloc_ctx* ctx) -> void;

auto qn_eloc_ctx_run_impl(
    qnpeps_eloc_ctx& ctx,
    const cf* peps,
    const uint8_t* samples,
    f64* logpsi_out,
    f64* e_loc_out,
    cf* o_rows_dev,
    cf* o_rows_host,
    cf* T_dev,
    f64 lambda,
    uint32_t j2_mode,
    uint32_t j2_draw,
    std::uint64_t j2_seed,
    std::uint64_t j2_epoch
) -> void;

auto qn_eloc_ctx_run_async_impl(
    qnpeps_eloc_ctx& ctx,
    const cf* peps,
    const uint8_t* samples,
    f64* logpsi_out,
    f64* e_loc_out,
    cf* o_rows_dev,
    cf* o_rows_host,
    cf* T_dev,
    f64 lambda,
    uint32_t j2_mode,
    uint32_t j2_draw,
    std::uint64_t j2_seed,
    std::uint64_t j2_epoch
) -> void;

auto qn_eloc_ctx_stats_impl(const qnpeps_eloc_ctx& ctx, QnpepsElocCtxStats& out) -> void;

auto qn_eloc_run_impl(
    const QnpepsElocConfig& cfg,
    const cf* peps,
    const std::uint8_t* samples,
    std::int64_t n_samples,
    const QnpepsElocTermTable* terms,
    f64* logpsi_out,
    f64* e_loc_out,
    cf* o_rows_dev,
    cf* o_rows_host,
    cf* gram,
    f64 lambda,
    qnpeps::Linalg& linalg
) -> void;

#endif
