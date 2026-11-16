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
    f64 lambda
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
    f64 lambda
) -> void;

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

auto qn_eloc_gram_tile_impl(
    const QnpepsElocConfig& cfg,
    const cf* rows_a,
    const std::uint8_t* samples_a,
    std::int64_t ns_a,
    const cf* rows_b,
    const std::uint8_t* samples_b,
    std::int64_t ns_b,
    cf* tile_out,
    cudaStream_t stream
) -> void;

#endif
