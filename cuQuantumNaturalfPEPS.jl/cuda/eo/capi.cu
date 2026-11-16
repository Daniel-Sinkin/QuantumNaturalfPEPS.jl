#include "capi/internal/format.cuh"
#include "common.cuh"
#include "capi/qnpeps.h"
#include "dans_qnpeps_eloc.h"
#include "eloc_kernels.cuh"
#include "env_build.cuh"
#include "eo/shards.cuh"
#include "term_validation.cuh"

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#ifndef QNPEPS_PACKAGE_VERSION
#    error "QNPEPS_PACKAGE_VERSION must be provided by CMake"
#endif

namespace
{
auto validate_config(const QnpepsElocConfig* config) -> void
{
    require(config != nullptr, Status::null_argument, "local energy config is null");
    require_struct_size(*config, "local energy config");
    const auto positive = config->lx >= 2 and config->ly >= 2 and config->dim_phys >= 1
                          and config->dim_bond >= 1 and config->chi_eo >= 1 and config->meo >= 1;
    require(positive, Status::bad_config, "local energy config rejects {}", *config);
    const auto known_route = qn_eloc::is_local_energy_route(config->truncation_route);
    const auto valid_density_cutoff =
        std::isfinite(config->density_cutoff) and config->density_cutoff >= 0.0;
    const auto valid_svd_cutoff = std::isfinite(config->svd_cutoff) and config->svd_cutoff >= 0.0
                                  and config->svd_cutoff <= 1.0;
    require(
        known_route and valid_density_cutoff and valid_svd_cutoff,
        Status::bad_config,
        "local energy config rejects the truncation of {}",
        *config
    );
}

auto validate_terms(const QnpepsElocConfig& config, const QnpepsElocTermTable* terms) -> void
{
    switch (qnpeps::eo::validate_terms(config.lx, config.ly, config.dim_phys, terms))
    {
        case qnpeps::eo::TermValidation::ok:
            require(
                qnpeps::eo::is_routable(config.lx, config.ly, *terms),
                Status::bad_config,
                "local energy term table is not routable on a {}x{} lattice",
                config.lx,
                config.ly
            );
            return;
        case qnpeps::eo::TermValidation::null_argument:
            fail(Status::null_argument, "local energy term table misses an array");
        case qnpeps::eo::TermValidation::bad_version:
            fail(Status::bad_version, "local energy term table struct_size is not recognized");
        case qnpeps::eo::TermValidation::bad_config:
            fail(Status::bad_config, "local energy term table rejects its entries");
    }
    fail("local energy term validation returned an unknown verdict");
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

extern "C" qnpeps_status qnpeps_eloc_logpsi(
    const QnpepsElocConfig* cfg,
    const qnpeps_eloc_cbuf* device_peps,
    const uint8_t* device_samples,
    int64_t n_samples,
    double* logpsi_out,
    void* stream
)
{
    return guarded(
        [&]
        {
            validate_config(cfg);
            const auto has_buffers = device_peps and device_samples and logpsi_out;
            require(has_buffers, Status::null_argument, "local energy logpsi misses a buffer");
            if (n_samples <= 0) return;
            auto session = qnpeps::make_session(static_cast<cudaStream_t>(stream));
            qn_eloc_envs_logpsi(
                *cfg, as_cf(device_peps), device_samples, n_samples, logpsi_out, session->linalg()
            );
            check(cudaStreamSynchronize(session->stream()));
        }
    );
}

extern "C" qnpeps_status qnpeps_eloc_run(
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
    return guarded(
        [&]
        {
            validate_config(cfg);
            const auto has_buffers =
                device_peps and device_samples and terms and logpsi_out and e_loc_out;
            require(has_buffers, Status::null_argument, "local energy run misses a buffer");
            validate_terms(*cfg, terms);
            const auto rows_requested = o_rows_host or T_dev;
            require(
                not rows_requested or o_rows_dev,
                Status::null_argument,
                "local energy run asks for host rows or a Gram tile without device rows"
            );
            if (n_samples <= 0) return;
            auto session = qnpeps::make_session(static_cast<cudaStream_t>(stream));
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
            check(cudaStreamSynchronize(session->stream()));
        }
    );
}

extern "C" qnpeps_status qnpeps_eloc_ctx_create(
    const QnpepsElocConfig* cfg,
    int64_t n_samples,
    const QnpepsElocTermTable* terms,
    uint32_t flags,
    void* stream,
    qnpeps_eloc_ctx** out
)
{
    return guarded(
        [&]
        {
            validate_config(cfg);
            require(
                terms != nullptr and out != nullptr,
                Status::null_argument,
                "local energy context needs a term table and an output"
            );
            *out = nullptr;
            validate_terms(*cfg, terms);
            require(
                n_samples > 0,
                Status::bad_config,
                "local energy context sample count {} is not positive",
                n_samples
            );
            const auto known = uint32_t{QNPEPS_ELOC_CTX_O_ROWS | QNPEPS_ELOC_CTX_GRAM};
            const auto unknown_flags = (flags & ~known) != 0;
            const auto gram_without_rows =
                (flags & QNPEPS_ELOC_CTX_GRAM) != 0 and (flags & QNPEPS_ELOC_CTX_O_ROWS) == 0;
            require(
                not unknown_flags and not gram_without_rows,
                Status::bad_config,
                "local energy context flags {} are not a valid combination",
                flags
            );
            auto session = qnpeps::make_session(static_cast<cudaStream_t>(stream));
            qn_eloc_ctx_create_impl(*cfg, n_samples, *terms, flags, std::move(session), out);
        }
    );
}

extern "C" void qnpeps_eloc_ctx_destroy(qnpeps_eloc_ctx* ctx)
{
    static_cast<void>(guarded_keeping_message([&] { qn_eloc_ctx_destroy_impl(ctx); }));
}

extern "C" qnpeps_status qnpeps_eloc_ctx_run(qnpeps_eloc_ctx* ctx, const QnpepsElocCtxRunArgs* args)
{
    return guarded(
        [&]
        {
            require(
                ctx != nullptr and args != nullptr,
                Status::null_argument,
                "local energy context run needs a context and arguments"
            );
            require_struct_size(*args, "local energy run arguments");
            const auto has_buffers =
                args->device_peps and args->device_samples and args->logpsi_out and args->e_loc_out;
            require(has_buffers, Status::null_argument, "local energy context run misses a buffer");
            const auto rows_requested = args->o_rows_host or args->T_dev;
            require(
                not rows_requested or args->o_rows_dev,
                Status::null_argument,
                "local energy context run asks for host rows or a Gram tile without device rows"
            );
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
        }
    );
}

extern "C" qnpeps_status qnpeps_eloc_gram_tile(
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
    return guarded(
        [&]
        {
            validate_config(cfg);
            const auto has_buffers = rows_a and samples_a and rows_b and samples_b and tile_out;
            require(has_buffers, Status::null_argument, "local energy Gram tile misses a buffer");
            qnpeps::eo::gram_tile({
                .config = cfg,
                .rows_a = as_cf(rows_a),
                .samples_a = samples_a,
                .count_a = ns_a,
                .rows_b = as_cf(rows_b),
                .samples_b = samples_b,
                .count_b = ns_b,
                .tile = as_cf(tile_out),
                .stream = static_cast<cudaStream_t>(stream),
            });
        }
    );
}

extern "C" qnpeps_status qnpeps_eloc_compact_count(const QnpepsElocConfig* cfg, int64_t* out_count)
{
    return guarded(
        [&]
        {
            validate_config(cfg);
            require(
                out_count != nullptr,
                Status::null_argument,
                "local energy compact count output is null"
            );
            *out_count = qn_eloc_compact_count(*cfg);
        }
    );
}

extern "C" const char* qnpeps_eloc_version(void)
{
    return "cuQuantumNaturalfPEPS eo " QNPEPS_PACKAGE_VERSION;
}

extern "C" uint64_t qnpeps_eloc_capabilities(void)
{
    return QNPEPS_ELOC_CAP_LOCAL_OPERATORS;
}
