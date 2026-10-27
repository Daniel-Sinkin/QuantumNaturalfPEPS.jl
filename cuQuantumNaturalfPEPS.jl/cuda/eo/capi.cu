#include "common.cuh"
#include "capi/qnpeps.h"
#include "dans_qnpeps_eloc.h"
#include "eloc_kernels.cuh"
#include "env_build.cuh"
#include "term_validation.cuh"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#ifndef QNPEPS_PACKAGE_VERSION
#    error "QNPEPS_PACKAGE_VERSION must be provided by CMake"
#endif

namespace
{
auto check_cfg(const QnpepsElocConfig* c) -> qnpeps_eloc_status
{
    if (not c) return QNPEPS_ELOC_ERR_NULL_ARG;
    if (c->struct_size != sizeof(QnpepsElocConfig)) return QNPEPS_ELOC_ERR_BAD_VERSION;
    if (c->lx < 2 or c->ly < 2 or c->dim_phys < 1 or c->dim_bond < 1 or c->chi_eo < 1 or c->meo < 1)
        return QNPEPS_ELOC_ERR_BAD_CONFIG;
    const auto known_route = c->truncation_route == QNPEPS_TRUNCATION_DEFAULT
                             or c->truncation_route == QNPEPS_TRUNCATION_SVD
                             or c->truncation_route == QNPEPS_TRUNCATION_HOUSEHOLDER
                             or c->truncation_route == QNPEPS_TRUNCATION_DENSITY;
    if (not known_route or not std::isfinite(c->density_cutoff) or c->density_cutoff < 0.0
        or not std::isfinite(c->svd_cutoff) or c->svd_cutoff < 0.0 or c->svd_cutoff > 1.0)
        return QNPEPS_ELOC_ERR_BAD_CONFIG;
    return QNPEPS_ELOC_OK;
}
auto check_terms(const QnpepsElocConfig& config, const QnpepsElocTermTable* terms)
    -> qnpeps_eloc_status
{
    switch (qnpeps::eo::validate_terms(config.lx, config.ly, terms))
    {
        case qnpeps::eo::TermValidation::ok:
            return QNPEPS_ELOC_OK;
        case qnpeps::eo::TermValidation::null_argument:
            return QNPEPS_ELOC_ERR_NULL_ARG;
        case qnpeps::eo::TermValidation::bad_config:
            return QNPEPS_ELOC_ERR_BAD_CONFIG;
    }
    return QNPEPS_ELOC_ERR_INTERNAL;
}
auto as_cf(const qnpeps_eloc_cbuf* p) -> const cf*
{
    return reinterpret_cast<const cf*>(p);
}
auto as_cf(qnpeps_eloc_cbuf* p) -> cf*
{
    return reinterpret_cast<cf*>(p);
}

}

auto qn_eloc_envs_logpsi(
    const QnpepsElocConfig& cfg,
    const cf* peps,
    const std::uint8_t* samples,
    std::int64_t n_samples,
    f64* logpsi_out,
    qnpeps::Linalg& linalg
) -> void;

auto qn_eloc_run_impl(
    const QnpepsElocConfig& cfg,
    const cf* peps,
    const std::uint8_t* samples,
    std::int64_t n_samples,
    const QnpepsElocTermTable* tt,
    f64* logpsi_out,
    f64* e_loc_out,
    cf* o_rows_dev,
    cf* o_rows_host,
    cf* T_dev,
    f64 lambda,
    qnpeps::Linalg& linalg
) -> void;

auto qn_eloc_compact_count(const QnpepsElocConfig& cfg) -> std::int64_t;

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

extern "C" qnpeps_eloc_status qnpeps_eloc_logpsi(
    const QnpepsElocConfig* cfg,
    const qnpeps_eloc_cbuf* device_peps,
    const uint8_t* device_samples,
    int64_t n_samples,
    double* logpsi_out,
    void* stream
)
{
    const auto v = check_cfg(cfg);
    if (v != QNPEPS_ELOC_OK) return v;
    if (not device_peps or not device_samples or not logpsi_out) return QNPEPS_ELOC_ERR_NULL_ARG;
    if (n_samples <= 0) return QNPEPS_ELOC_OK;
    qn::clear_err();
    auto session = qnpeps::make_session(static_cast<cudaStream_t>(stream));
    if (not session) return qn::err_state();
    qn_eloc_envs_logpsi(
        *cfg, as_cf(device_peps), device_samples, n_samples, logpsi_out, session->linalg()
    );
    CUDA_CHECK(cudaStreamSynchronize(session->stream()));
    return qn::err_state();
}

extern "C" qnpeps_eloc_status qnpeps_eloc_run(
    const QnpepsElocConfig* cfg,
    const qnpeps_eloc_cbuf* device_peps,
    const uint8_t* device_samples,
    int64_t n_samples,
    const QnpepsElocTermTable* terms,
    double* logpsi_out,
    double* e_loc_out,
    qnpeps_eloc_cbuf* o_rows_dev,
    qnpeps_eloc_cbuf* o_rows_host,
    qnpeps_eloc_cbuf* T_dev,
    double lambda,
    void* stream
)
{
    const auto v = check_cfg(cfg);
    if (v != QNPEPS_ELOC_OK) return v;
    if (not device_peps or not device_samples or not terms or not logpsi_out or not e_loc_out)
        return QNPEPS_ELOC_ERR_NULL_ARG;
    if (terms->n_flip > 0 and not terms->flip) return QNPEPS_ELOC_ERR_NULL_ARG;
    if (const auto status = check_terms(*cfg, terms); status != QNPEPS_ELOC_OK) return status;
    if ((o_rows_host or T_dev) and not o_rows_dev) return QNPEPS_ELOC_ERR_NULL_ARG;
    if (n_samples <= 0) return QNPEPS_ELOC_OK;
    qn::clear_err();
    auto session = qnpeps::make_session(static_cast<cudaStream_t>(stream));
    if (not session) return qn::err_state();
    qn_eloc_run_impl(
        *cfg,
        as_cf(device_peps),
        device_samples,
        n_samples,
        terms,
        logpsi_out,
        e_loc_out,
        as_cf(o_rows_dev),
        as_cf(o_rows_host),
        as_cf(T_dev),
        lambda,
        session->linalg()
    );
    CUDA_CHECK(cudaStreamSynchronize(session->stream()));
    return qn::err_state();
}

extern "C" qnpeps_eloc_status qnpeps_eloc_ctx_create(
    const QnpepsElocConfig* cfg,
    int64_t n_samples,
    const QnpepsElocTermTable* terms,
    uint32_t flags,
    void* stream,
    qnpeps_eloc_ctx** out
)
{
    const auto v = check_cfg(cfg);
    if (v != QNPEPS_ELOC_OK) return v;
    if (not terms or not out) return QNPEPS_ELOC_ERR_NULL_ARG;
    *out = nullptr;
    if (terms->n_flip > 0 and not terms->flip) return QNPEPS_ELOC_ERR_NULL_ARG;
    if (const auto status = check_terms(*cfg, terms); status != QNPEPS_ELOC_OK) return status;
    if (n_samples <= 0) return QNPEPS_ELOC_ERR_BAD_CONFIG;
    const auto known = uint32_t{QNPEPS_ELOC_CTX_O_ROWS | QNPEPS_ELOC_CTX_GRAM};
    if ((flags & ~known) != 0
        or ((flags & QNPEPS_ELOC_CTX_GRAM) != 0 and (flags & QNPEPS_ELOC_CTX_O_ROWS) == 0))
        return QNPEPS_ELOC_ERR_BAD_CONFIG;

    qn::clear_err();
    auto session = qnpeps::make_session(static_cast<cudaStream_t>(stream));
    if (session) qn_eloc_ctx_create_impl(*cfg, n_samples, *terms, flags, std::move(session), out);
    return qn::err_state();
}

extern "C" void qnpeps_eloc_ctx_destroy(qnpeps_eloc_ctx* ctx)
{
    qn_eloc_ctx_destroy_impl(ctx);
}

extern "C" qnpeps_eloc_status qnpeps_eloc_ctx_run(
    qnpeps_eloc_ctx* ctx, const QnpepsElocCtxRunArgs* args
)
{
    if (not ctx or not args) return QNPEPS_ELOC_ERR_NULL_ARG;
    if (args->struct_size != sizeof(QnpepsElocCtxRunArgs)) return QNPEPS_ELOC_ERR_BAD_VERSION;
    if (not args->device_peps or not args->device_samples or not args->logpsi_out
        or not args->e_loc_out)
        return QNPEPS_ELOC_ERR_NULL_ARG;
    if ((args->o_rows_host or args->T_dev) and not args->o_rows_dev)
        return QNPEPS_ELOC_ERR_NULL_ARG;
    qn::clear_err();
    qn_eloc_ctx_run_impl(
        *ctx,
        as_cf(args->device_peps),
        args->device_samples,
        args->logpsi_out,
        args->e_loc_out,
        as_cf(args->o_rows_dev),
        as_cf(args->o_rows_host),
        as_cf(args->T_dev),
        args->lambda
    );
    return qn::err_state();
}

extern "C" qnpeps_eloc_status qnpeps_eloc_gram_tile(
    const QnpepsElocConfig* cfg,
    const qnpeps_eloc_cbuf* rows_a,
    const uint8_t* samples_a,
    int64_t ns_a,
    const qnpeps_eloc_cbuf* rows_b,
    const uint8_t* samples_b,
    int64_t ns_b,
    qnpeps_eloc_cbuf* tile_out,
    void* stream
)
{
    const auto v = check_cfg(cfg);
    if (v != QNPEPS_ELOC_OK) return v;
    if (not rows_a or not samples_a or not rows_b or not samples_b or not tile_out)
        return QNPEPS_ELOC_ERR_NULL_ARG;
    if (ns_a <= 0 or ns_b <= 0) return QNPEPS_ELOC_OK;
    qn::clear_err();
    const auto s = static_cast<cudaStream_t>(stream);
    qn_eloc_gram_tile_impl(
        *cfg, as_cf(rows_a), samples_a, ns_a, as_cf(rows_b), samples_b, ns_b, as_cf(tile_out), s
    );
    CUDA_CHECK(cudaStreamSynchronize(s));
    return qn::err_state();
}

extern "C" qnpeps_eloc_status qnpeps_eloc_compact_count(
    const QnpepsElocConfig* cfg, int64_t* out_count
)
{
    const auto v = check_cfg(cfg);
    if (v != QNPEPS_ELOC_OK) return v;
    if (not out_count) return QNPEPS_ELOC_ERR_NULL_ARG;
    *out_count = qn_eloc_compact_count(*cfg);
    return QNPEPS_ELOC_OK;
}

extern "C" const char* qnpeps_eloc_strerror(qnpeps_eloc_status status)
{
    switch (status)
    {
        case QNPEPS_ELOC_OK:
            return "ok";
        case QNPEPS_ELOC_ERR_NULL_ARG:
            return "a required pointer was NULL";
        case QNPEPS_ELOC_ERR_BAD_CONFIG:
            return "descriptor failed validation";
        case QNPEPS_ELOC_ERR_BAD_VERSION:
            return "descriptor struct_size not recognized";
        case QNPEPS_ELOC_ERR_CUDA:
            {
                auto backend{qnpeps::err_backend()};
                if (not backend or std::strcmp(backend, "none") == 0)
                    return "an underlying CUDA call failed";
                static thread_local qnpeps::CuArray<char, 512> message{};
                std::snprintf(
                    message.data(),
                    sizeof(message),
                    "an underlying CUDA call failed (%s code %d at %s:%d)",
                    backend,
                    qnpeps::err_backend_code(),
                    qnpeps::err_file(),
                    qnpeps::err_line()
                );
                return message.data();
            }
        case QNPEPS_ELOC_ERR_OOM:
            return "device allocation failed";
        case QNPEPS_ELOC_ERR_INTERNAL:
            return "internal invariant violated (please report)";
    }
    return "unknown status";
}

extern "C" const char* qnpeps_eloc_version(void)
{
    return "cuQuantumNaturalfPEPS eo " QNPEPS_PACKAGE_VERSION;
}

extern "C" uint64_t qnpeps_eloc_capabilities(void)
{
    return QNPEPS_ELOC_CAP_ONE_SITE_Z;
}
