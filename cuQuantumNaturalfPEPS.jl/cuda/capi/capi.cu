#include "capi/qnpeps.h"
#include "core/arena_cursor.cuh"
#include "core/cuda_utils.cuh"
#include "core/qnpeps_ctx.cuh"
#include "dlenv/build.cuh"
#include "gram/gram.cuh"
#include "linalg/linalg.cuh"
#include "linalg/rangefinder_rng.cuh"
#include "linalg/transfer.cuh"
#include "minsr/minsr.cuh"
#include "peps/init.cuh"
#include "peps/peps.cuh"
#include "sampler/draw.cuh"
#include "zipup/zipup_mpo_mps.cuh"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <utility>
#include <vector>

#ifndef QNPEPS_C_API_VERSION
#    error "QNPEPS_C_API_VERSION must be provided by CMake"
#endif

using namespace qnpeps;

namespace qnpeps::dlenv
{
auto dl_free(qnpeps_ctx& ctx) -> void;
}

namespace
{
auto check_cfg(const QnpepsConfig* config) -> qnpeps_status
{
    if (not config) return QNPEPS_ERR_NULL_ARG;
    if (config->struct_size != sizeof(QnpepsConfig)) return QNPEPS_ERR_BAD_VERSION;
    const auto invalid_dimensions = config->dim_phys < 1 or config->lx < 2 or config->ly < 2
                                    or config->dim_bond < 1 or config->chi_s < 1
                                    or config->chi_dl < 1;
    if (invalid_dimensions) return QNPEPS_ERR_BAD_CONFIG;
    const auto invalid_sampling_mode = config->sampling_mode != QNPEPS_SAMPLING_FAST
                                       and config->sampling_mode != QNPEPS_SAMPLING_FULL;
    if (invalid_sampling_mode) return QNPEPS_ERR_BAD_CONFIG;
    if (config->sampling_mode == QNPEPS_SAMPLING_FULL and config->chi_c < 1)
        return QNPEPS_ERR_BAD_CONFIG;
    const auto invalid_dlenv_route =
        config->dlenv_truncation_route != QNPEPS_TRUNCATION_DEFAULT
        and config->dlenv_truncation_route != QNPEPS_TRUNCATION_DENSITY;
    const auto invalid_sampler_route =
        config->sampler_truncation_route != QNPEPS_TRUNCATION_DEFAULT
        and config->sampler_truncation_route != QNPEPS_TRUNCATION_DENSITY;
    const auto invalid_dlenv_cutoff =
        not std::isfinite(config->dlenv_density_cutoff) or config->dlenv_density_cutoff < 0.0;
    const auto invalid_sampler_cutoff =
        not std::isfinite(config->sampler_density_cutoff) or config->sampler_density_cutoff < 0.0;
    const auto invalid_projected_cutoff = not std::isfinite(config->projected_density_cutoff)
                                          or config->projected_density_cutoff < 0.0;
    const auto invalid_truncation = invalid_dlenv_route or invalid_sampler_route
                                    or invalid_dlenv_cutoff or invalid_sampler_cutoff
                                    or invalid_projected_cutoff;
    if (invalid_truncation) return QNPEPS_ERR_BAD_CONFIG;
    return QNPEPS_OK;
}

auto dispatch_sample(const QnpepsConfig& config, int gpus, const qnpeps::sampler::SampleArgs& args)
    -> qnpeps_status
{
    if (gpus < 1) return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
    if (gpus > 1)
    {
        if (args.scratch) return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
        qnpeps::sampler::sample_multigpu(
            config,
            {
                .device_peps = args.device_peps,
                .device_dlenv = args.device_dlenv,
                .gpus = gpus,
                .output = args.output,
                .logpc_out = args.logpc_out,
                .lognorm_out = args.lognorm_out,
                .n_samples = args.n_samples,
                .batch_base = args.batch_base,
                .dim_batch = args.dim_batch,
                .output_location = args.output_location,
            }
        );
        return qnpeps::err_state();
    }
    qnpeps::sampler::sample(config, args);
    return qnpeps::err_state();
}

auto check_sample_shape(
    const QnpepsConfig& config, uint64_t n_samples, uint64_t dim_batch, uint64_t batch_base
) -> qnpeps_status
{
    if (dim_batch < 1 or dim_batch > static_cast<uint64_t>(k_max_batch_size))
        return QNPEPS_ERR_BAD_CONFIG;
    if (n_samples > static_cast<uint64_t>(std::numeric_limits<i64>::max()))
        return QNPEPS_ERR_BAD_CONFIG;
    if (n_samples == 0) return QNPEPS_OK;

    const auto batches = n_samples / dim_batch + static_cast<uint64_t>(n_samples % dim_batch != 0);
    if (batches > static_cast<uint64_t>(INT_MAX)) return QNPEPS_ERR_BAD_CONFIG;
    if (batch_base > std::numeric_limits<uint64_t>::max() - (batches - 1))
        return QNPEPS_ERR_BAD_CONFIG;

    const auto sites = static_cast<uint64_t>(config.lx) * static_cast<uint64_t>(config.ly);
    const auto size_limit = static_cast<uint64_t>(std::numeric_limits<usize>::max());
    if (sites == 0 or n_samples > size_limit / sites) return QNPEPS_ERR_BAD_CONFIG;
    if (n_samples > size_limit / sizeof(f64)) return QNPEPS_ERR_BAD_CONFIG;
    return QNPEPS_OK;
}

auto ctx_sample_impl(
    qnpeps_ctx* ctx,
    const QnpepsCtxSampleArgs* args,
    qnpeps::sampler::SampleOutputLocation output_location
) -> qnpeps_status
{
    qnpeps::reset_err();
    if (not ctx or not args) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    if (args->struct_size != sizeof(QnpepsCtxSampleArgs))
        return qnpeps::set_err(QNPEPS_ERR_BAD_VERSION);
    if (args->n_samples > 0 and not args->samples_out) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    const auto shape_status =
        check_sample_shape(ctx->cfg, args->n_samples, args->dim_batch, args->batch_base);
    if (shape_status != QNPEPS_OK) return qnpeps::set_err(shape_status);
    return qnpeps::sampler::ctx_sample(
        *ctx,
        {
            .output = args->samples_out,
            .logpc_out = args->log_prob_config,
            .lognorm_out = args->log_gauge,
            .n_samples = args->n_samples,
            .batch_base = args->batch_base,
            .dim_batch = args->dim_batch,
            .output_location = output_location,
        }
    );
}

auto sampler_host_sampling_bytes(const qnpeps_ctx& ctx, const int32_t* dims, uint64_t count)
    -> uint64_t
{
    const auto expected =
        static_cast<uint64_t>(ctx.cfg.lx - 1) * static_cast<uint64_t>(ctx.cfg.ly) * 4_u64;
    if (not dims or count != expected) return 0;
    uint64_t elements{};
    for (auto site = 0_u64; site < expected / 4_u64; ++site)
    {
        uint64_t site_elements{1};
        for (auto axis = 0_u64; axis < 4_u64; ++axis)
        {
            const auto dim = dims[site * 4_u64 + axis];
            if (dim < 1) return 0;
            const auto dim_u = static_cast<uint64_t>(dim);
            if (site_elements > std::numeric_limits<uint64_t>::max() / dim_u) return 0;
            site_elements *= dim_u;
        }
        if (elements > std::numeric_limits<uint64_t>::max() - site_elements) return 0;
        elements += site_elements;
    }
    if (elements > std::numeric_limits<uint64_t>::max() / (2_u64 * sizeof(cuFloatComplex)))
        return 0;
    return elements * 2_u64 * sizeof(cuFloatComplex);
}

auto sampler_host_upload_pointers(
    qnpeps_ctx& ctx, const void* const* pointers, uint64_t pointer_count
) -> qnpeps_status
{
    if (not ctx.sampler.ready() or not ctx.sampler.execution.host_managed)
        return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
    if (not pointers) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    const auto expected = static_cast<uint64_t>(qnpeps::dlenv::k_sampling_layout_count)
                          * static_cast<uint64_t>(ctx.cfg.lx - 1)
                          * static_cast<uint64_t>(ctx.cfg.ly)
                          * static_cast<uint64_t>(ctx.sampler.allocation.dim_batch_capacity);
    if (pointer_count != expected) return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
    auto& samp = ctx.sampler.samp;
    upload_async(
        ctx.linalg(), samp.dlenv_env_ptrs()[0][0], pointers, static_cast<usize>(pointer_count)
    );
    if (qnpeps::err_state() == QNPEPS_OK) ctx.sampler.execution.host_pointers = true;
    return qnpeps::err_state();
}
}

extern "C" qnpeps_status qnpeps_ctx_create(
    const QnpepsConfig* config, void* stream, qnpeps_ctx** out
)
{
    qnpeps::reset_err();
    if (not out) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    *out = nullptr;
    const auto config_status = check_cfg(config);
    if (config_status != QNPEPS_OK) return qnpeps::set_err(config_status);

    auto session = make_session(static_cast<cudaStream_t>(stream));
    if (not session) return qnpeps::err_state();
    auto* ctx = new (std::nothrow) qnpeps_ctx{*config, std::move(session)};
    if (not ctx)
    {
        return qnpeps::set_err(QNPEPS_ERR_OOM);
    }
    *out = ctx;
    return QNPEPS_OK;
}

extern "C" void qnpeps_ctx_destroy(qnpeps_ctx* ctx)
{
    if (not ctx) return;
    CUDA_NOCHECK(cudaStreamSynchronize(ctx->stream()));
    qnpeps::sampler::ctx_sampler_free(*ctx);
    qnpeps::dlenv::dl_free(*ctx);
    delete ctx;
}

extern "C" qnpeps_status qnpeps_ctx_build_dlenv(
    qnpeps_ctx* ctx, const qnpeps_device_peps* peps, double* cumulative_row_logs
)
{
    qnpeps::reset_err();
    if (not ctx or not peps) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    qnpeps::dlenv::build_dlenv(*ctx, peps, cumulative_row_logs);
    return qnpeps::err_state();
}

extern "C" qnpeps_status qnpeps_ctx_copy_dlenv_host(
    const qnpeps_ctx* ctx, void* output, uint64_t output_bytes
)
{
    qnpeps::reset_err();
    if (not ctx or not output) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    const auto& active_lane = ctx->dlenv.lanes[ctx->dlenv.active_lane];
    if (not active_lane.valid or not active_lane.packed)
        return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
    const auto required = qnpeps_dlenv_bytes(&ctx->cfg);
    if (required < 0 or output_bytes < static_cast<uint64_t>(required))
        return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
    download(
        static_cast<u8*>(output),
        reinterpret_cast<const u8*>(active_lane.packed),
        static_cast<usize>(required)
    );
    return qnpeps::err_state();
}

extern "C" qnpeps_status qnpeps_ctx_sample(qnpeps_ctx* ctx, const QnpepsCtxSampleArgs* args)
{
    return ctx_sample_impl(ctx, args, qnpeps::sampler::SampleOutputLocation::device);
}

extern "C" qnpeps_status qnpeps_ctx_sample_host(qnpeps_ctx* ctx, const QnpepsCtxSampleArgs* args)
{
    return ctx_sample_impl(ctx, args, qnpeps::sampler::SampleOutputLocation::host);
}

extern "C" qnpeps_status qnpeps_sampler_host_upload_pointers(
    qnpeps_ctx* ctx, const void* const* pointers, uint64_t pointer_count
)
{
    qnpeps::reset_err();
    if (not ctx) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    return sampler_host_upload_pointers(*ctx, pointers, pointer_count);
}

extern "C" qnpeps_status qnpeps_sampler_host_refresh(
    qnpeps_ctx* ctx, const QnpepsSamplerHostRefreshArgs* args
)
{
    qnpeps::reset_err();
    if (not ctx or not args) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    if (args->struct_size != sizeof(QnpepsSamplerHostRefreshArgs))
        return qnpeps::set_err(QNPEPS_ERR_BAD_VERSION);
    if (not args->peps or not args->dlenv_values or not args->sampling)
        return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    if (not ctx->sampler.ready() or not ctx->sampler.execution.host_managed)
        return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
    const auto required = sampler_host_sampling_bytes(
        *ctx, ctx->dlenv.dims.data(), static_cast<uint64_t>(ctx->dlenv.dims.size())
    );
    if (required == 0 or args->sampling_bytes < required)
        return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
    auto& lane = ctx->dlenv.lanes[ctx->dlenv.active_lane];
    lane.sampling = static_cast<cuFloatComplex*>(args->sampling);
    lane.sampling_owned = false;
    qnpeps::dlenv::materialize_sampling_buffer(
        *ctx, reinterpret_cast<const cuFloatComplex*>(args->dlenv_values), lane.sampling
    );
    const auto layout =
        args->peps_layout == 0 ? qnpeps::PepsLayout::canonical : qnpeps::PepsLayout::reverse_packed;
    qnpeps::sampler::ctx_sample_refresh(*ctx, args->peps, layout);
    return qnpeps::err_state();
}

extern "C" qnpeps_status qnpeps_sampler_host_batch(
    qnpeps_ctx* ctx, const QnpepsSamplerHostBatchArgs* args
)
{
    qnpeps::reset_err();
    if (not ctx or not args) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    if (args->struct_size != sizeof(QnpepsSamplerHostBatchArgs))
        return qnpeps::set_err(QNPEPS_ERR_BAD_VERSION);
    const auto missing_input = not args->peps or not args->dlenv_dims or not args->dlenv_values
                               or not args->sampling or not args->dlenv_pointers
                               or not args->samples_out or not args->log_prob_config
                               or not args->log_gauge;
    if (missing_input)
    {
        return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    }
    const auto invalid_batch = args->dim_batch < 1
                               or args->dim_batch > static_cast<uint64_t>(k_max_batch_size)
                               or args->batch_id > static_cast<uint64_t>(INT_MAX);
    if (invalid_batch)
    {
        return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
    }
    const auto sampling_bytes =
        sampler_host_sampling_bytes(*ctx, args->dlenv_dims, args->dlenv_dims_count);
    if (sampling_bytes == 0 or args->sampling_bytes < sampling_bytes)
    {
        return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
    }
    if (not ctx->sampler.allocation.allocated)
    {
        ctx->sampler.execution.host_managed = true;
        ctx->sampler.execution.dim_batch = static_cast<int>(args->dim_batch);
        ctx->sampler.allocation.dim_batch_capacity = static_cast<int>(args->dim_batch);
        ctx->dlenv.dims.assign(
            args->dlenv_dims, args->dlenv_dims + static_cast<usize>(args->dlenv_dims_count)
        );
        auto& lane = ctx->dlenv.lanes.front();
        lane.sampling = static_cast<cuFloatComplex*>(args->sampling);
        lane.sampling_owned = false;
        lane.valid = true;
        ctx->dlenv.active_lane = 0;
        const qnpeps::DlEnvView view{
            .dims = ctx->dlenv.dims.data(),
            .values = reinterpret_cast<const cuFloatComplex*>(args->dlenv_values),
        };
        qnpeps::sampler::ctx_sampler_setup(*ctx, &view, nullptr, 0);
        if (qnpeps::err_state() != QNPEPS_OK) return qnpeps::err_state();
        sampler_host_upload_pointers(*ctx, args->dlenv_pointers, args->dlenv_pointer_count);
        if (qnpeps::err_state() != QNPEPS_OK) return qnpeps::err_state();
        const auto layout = args->peps_layout == 0 ? qnpeps::PepsLayout::canonical
                                                   : qnpeps::PepsLayout::reverse_packed;
        qnpeps::sampler::ctx_sample_refresh(*ctx, args->peps, layout);
        if (qnpeps::err_state() != QNPEPS_OK) return qnpeps::err_state();
    }
    const auto invalid_host_state =
        not ctx->sampler.execution.host_managed
        or ctx->sampler.execution.dim_batch != static_cast<int>(args->dim_batch);
    if (invalid_host_state)
    {
        return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
    }
    constexpr uint64_t seed_multiplier{1000003};
    ctx->sampler.samp.cfg().batch_base =
        args->batch_seed - ctx->sampler.samp.cfg().seed * seed_multiplier - args->batch_id;
    const auto batch_id = static_cast<int>(args->batch_id);
    const std::array<int, 1> batch_ids{batch_id};
    auto& staging = ctx->sampler.staging;
    auto* const old_samples = staging.h_samples;
    auto* const old_logpc = staging.h_logpc;
    auto* const old_lognorm = staging.h_lognorm;
    staging.h_samples = args->samples_out;
    staging.h_logpc = args->log_prob_config;
    staging.h_lognorm = args->log_gauge;
    qnpeps::sampler::HostSampleOutput output{
        .samples = args->samples_out,
        .logpc = args->log_prob_config,
        .lognorm = args->log_gauge,
        .n_samples = args->dim_batch,
        .batch_origin = args->batch_id,
    };
    qnpeps::sampler::ctx_sample_run(*ctx, batch_ids, &output);
    staging.h_samples = old_samples;
    staging.h_logpc = old_logpc;
    staging.h_lognorm = old_lognorm;
    return qnpeps::err_state();
}

namespace qnpeps::dlenv
{
auto build_dlenv_packed(
    const QnpepsConfig& config,
    const void* device_peps,
    void* output,
    f64* cumulative_row_logs,
    Linalg& linalg
) -> qnpeps_status
{
    qnpeps_ctx ctx{config, linalg};
    build_dlenv(ctx, device_peps, cumulative_row_logs);
    if (err_state() == QNPEPS_OK)
    {
        const auto bytes = qnpeps_dlenv_bytes(&config);
        const auto& active_lane = ctx.dlenv.lanes[ctx.dlenv.active_lane];
        copy_device(
            static_cast<u8*>(output),
            reinterpret_cast<const u8*>(active_lane.packed),
            static_cast<usize>(bytes)
        );
    }
    const auto status = err_state();
    CUDA_NOCHECK(cudaStreamSynchronize(linalg.stream()));
    dl_free(ctx);
    return status;
}
}

extern "C" qnpeps_status qnpeps_build_dlenv(
    const QnpepsConfig* config,
    const qnpeps_device_peps* device_peps,
    qnpeps_device_dlenv* dlenv_out,
    double* cumulative_row_logs,
    void* stream
)
{
    qnpeps::reset_err();
    const auto config_status = check_cfg(config);
    if (config_status != QNPEPS_OK) return qnpeps::set_err(config_status);
    if (not device_peps or not dlenv_out) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);

    auto session = make_session(static_cast<cudaStream_t>(stream));
    if (not session) return qnpeps::err_state();
    return qnpeps::dlenv::build_dlenv_packed(
        *config, device_peps, dlenv_out, cumulative_row_logs, session->linalg()
    );
}

extern "C" qnpeps_status qnpeps_sample(const QnpepsConfig* config, const QnpepsSampleArgs* args)
{
    qnpeps::reset_err();
    if (not config or not args) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    if (args->struct_size != sizeof(QnpepsSampleArgs))
        return qnpeps::set_err(QNPEPS_ERR_BAD_VERSION);
    if (not args->peps or not args->dlenv) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    if (args->n_samples > 0 and not args->samples_out) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);

    const auto config_status = check_cfg(config);
    if (config_status != QNPEPS_OK) return qnpeps::set_err(config_status);
    const auto shape_status =
        check_sample_shape(*config, args->n_samples, args->dim_batch, args->batch_base);
    if (shape_status != QNPEPS_OK) return qnpeps::set_err(shape_status);

    if (args->n_samples == 0) return QNPEPS_OK;
    return dispatch_sample(
        *config,
        args->gpus,
        {
            .device_peps = args->peps,
            .device_dlenv = args->dlenv,
            .scratch = nullptr,
            .scratch_bytes = 0,
            .output = args->samples_out,
            .logpc_out = args->log_prob_config,
            .lognorm_out = args->log_gauge,
            .n_samples = args->n_samples,
            .batch_base = args->batch_base,
            .dim_batch = args->dim_batch,
            .stream = args->stream,
            .output_location = qnpeps::sampler::SampleOutputLocation::device,
        }
    );
}

extern "C" qnpeps_status qnpeps_sample_host(
    const QnpepsConfig* config, const QnpepsSampleHostArgs* args
)
{
    qnpeps::reset_err();
    if (not config or not args) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    if (args->struct_size != sizeof(QnpepsSampleHostArgs))
        return qnpeps::set_err(QNPEPS_ERR_BAD_VERSION);
    if (not args->peps or not args->dlenv) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    if (args->n_samples > 0 and not args->samples_out) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);

    const auto config_status = check_cfg(config);
    if (config_status != QNPEPS_OK) return qnpeps::set_err(config_status);
    const auto shape_status =
        check_sample_shape(*config, args->n_samples, args->dim_batch, args->batch_base);
    if (shape_status != QNPEPS_OK) return qnpeps::set_err(shape_status);
    if (args->n_samples == 0) return QNPEPS_OK;
    return dispatch_sample(
        *config,
        args->gpus,
        {
            .device_peps = args->peps,
            .device_dlenv = args->dlenv,
            .scratch = nullptr,
            .scratch_bytes = 0,
            .output = args->samples_out,
            .logpc_out = args->log_prob_config,
            .lognorm_out = args->log_gauge,
            .n_samples = args->n_samples,
            .batch_base = args->batch_base,
            .dim_batch = args->dim_batch,
            .stream = args->stream,
            .output_location = qnpeps::sampler::SampleOutputLocation::host,
        }
    );
}

extern "C" qnpeps_status qnpeps_random_unitary_peps(
    const QnpepsConfig* config,
    qnpeps_device_peps* peps_out,
    uint64_t peps_bytes,
    uint64_t seed,
    double alpha,
    void* stream
)
{
    qnpeps::reset_err();
    const auto config_status = check_cfg(config);
    if (config_status != QNPEPS_OK) return qnpeps::set_err(config_status);
    if (not peps_out) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    const auto required_bytes = qnpeps_peps_bytes(config);
    if (required_bytes < 0 or peps_bytes < static_cast<uint64_t>(required_bytes))
        return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
    auto session = make_session(static_cast<cudaStream_t>(stream));
    if (not session) return qnpeps::err_state();
    qnpeps::peps::random_unitary(
        session->linalg(),
        {
            .config = *config,
            .output = reinterpret_cast<cuFloatComplex*>(peps_out),
            .output_bytes = static_cast<usize>(peps_bytes),
            .seed = seed,
            .alpha = alpha,
        }
    );
    return qnpeps::err_state();
}

extern "C" int64_t qnpeps_zipup_peps_row_bytes(const QnpepsConfig* config, int maxdim)
{
    if (check_cfg(config) != QNPEPS_OK) return -1;
    return qnpeps::dlenv::zipup_peps_row_bytes(*config, maxdim);
}

extern "C" qnpeps_status qnpeps_zipup_ctx_create(
    const QnpepsConfig* config, int maxdim, void* stream, qnpeps_zipup_ctx** out
)
{
    qnpeps::reset_err();
    if (not out) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    *out = nullptr;
    const auto config_status = check_cfg(config);
    if (config_status != QNPEPS_OK) return qnpeps::set_err(config_status);
    if (maxdim < 1) return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);

    auto* context =
        qnpeps::dlenv::create_zipup_context(*config, maxdim, static_cast<cudaStream_t>(stream));
    if (not context) return qnpeps::err_state();
    *out = context;
    return QNPEPS_OK;
}

extern "C" void qnpeps_zipup_ctx_destroy(qnpeps_zipup_ctx* context)
{
    qnpeps::dlenv::destroy_zipup_context(context);
}

extern "C" qnpeps_status qnpeps_zipup_ctx_begin(qnpeps_zipup_ctx* context)
{
    qnpeps::reset_err();
    if (not context) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    return qnpeps::dlenv::begin_zipup_context(*context);
}

extern "C" qnpeps_status qnpeps_zipup_ctx_enqueue_peps_row(
    qnpeps_zipup_ctx* context, const QnpepsZipupPepsRowArgs* args
)
{
    qnpeps::reset_err();
    if (not context or not args) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    if (args->struct_size != sizeof(QnpepsZipupPepsRowArgs))
        return qnpeps::set_err(QNPEPS_ERR_BAD_VERSION);
    if (not args->peps_row or not args->output_dims or not args->output_values)
        return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    return qnpeps::dlenv::enqueue_peps_row(*context, *args);
}

extern "C" qnpeps_status qnpeps_zipup_ctx_finish(
    qnpeps_zipup_ctx* context, double* scales, uint64_t count
)
{
    qnpeps::reset_err();
    if (not context or not scales) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    return qnpeps::dlenv::finish_zipup_context(*context, scales, static_cast<usize>(count));
}

extern "C" int64_t qnpeps_zipup_mpo_mps_bytes(const QnpepsZipupMpoMpsDesc* descriptor)
{
    return qnpeps::zipup::output_bytes(descriptor);
}

extern "C" qnpeps_status qnpeps_zipup_mpo_mps(
    const QnpepsZipupMpoMpsDesc* descriptor, const QnpepsZipupMpoMpsArgs* args
)
{
    qnpeps::reset_err();
    return qnpeps::zipup::execute(descriptor, args);
}

extern "C" int64_t qnpeps_minsr_dense_count(const QnpepsMinsrDesc* descriptor)
{
    qnpeps::reset_err();
    return qnpeps::minsr::descriptor_dense_count(descriptor);
}

extern "C" int64_t qnpeps_minsr_compact_count(const QnpepsMinsrDesc* descriptor)
{
    qnpeps::reset_err();
    return qnpeps::minsr::descriptor_compact_count(descriptor);
}

extern "C" int64_t qnpeps_minsr_scratch_bytes(const QnpepsMinsrDesc* descriptor)
{
    qnpeps::reset_err();
    return qnpeps::minsr::descriptor_scratch_bytes(descriptor);
}

extern "C" qnpeps_status qnpeps_minsr(
    const QnpepsMinsrDesc* descriptor, const QnpepsMinsrArgs* args
)
{
    qnpeps::reset_err();
    return qnpeps::minsr::execute(descriptor, args);
}

extern "C" qnpeps_status qnpeps_minsr_ctx_create(
    const QnpepsMinsrDesc* descriptor, void* stream, qnpeps_minsr_ctx** out
)
{
    qnpeps::reset_err();
    return qnpeps::minsr::ctx_create(descriptor, stream, out);
}

extern "C" qnpeps_status qnpeps_minsr_ctx_run(qnpeps_minsr_ctx* ctx, const QnpepsMinsrArgs* args)
{
    qnpeps::reset_err();
    return qnpeps::minsr::ctx_run(ctx, args);
}

extern "C" void qnpeps_minsr_ctx_destroy(qnpeps_minsr_ctx* ctx)
{
    qnpeps::minsr::ctx_destroy(ctx);
}

extern "C" qnpeps_status qnpeps_gram_ctx_create(
    const QnpepsGramDesc* descriptor, void* stream, qnpeps_gram_ctx** out
)
{
    qnpeps::reset_err();
    return qnpeps::gram::ctx_create(descriptor, stream, out);
}

extern "C" qnpeps_status qnpeps_gram_ctx_run(qnpeps_gram_ctx* ctx, const QnpepsGramArgs* args)
{
    qnpeps::reset_err();
    return qnpeps::gram::ctx_run(ctx, args);
}

extern "C" qnpeps_status qnpeps_gram_ctx_footprint(
    const qnpeps_gram_ctx* ctx, QnpepsGramFootprint* out
)
{
    qnpeps::reset_err();
    return qnpeps::gram::ctx_footprint(ctx, out);
}

extern "C" void qnpeps_gram_ctx_destroy(qnpeps_gram_ctx* ctx)
{
    qnpeps::gram::ctx_destroy(ctx);
}

extern "C" int64_t qnpeps_batched_rangefinder_scratch_bytes(int rows, int cols, int rank, int batch)
{
    if (rows < 1 or cols < 1 or rank < 1 or batch < 1) return -1;
    if (rank > rows or rank > cols) return -1;
    const auto bytes = arena_reservation_bytes();
    return err_state() == QNPEPS_OK ? static_cast<int64_t>(bytes) : -1;
}

extern "C" qnpeps_status qnpeps_batched_rangefinder(
    const void* input,
    int rows,
    int cols,
    int rank,
    int batch,
    int64_t input_stride,
    uint64_t seed,
    void* q_out,
    int64_t q_stride,
    void* r_out,
    int64_t r_stride,
    void* scratch,
    uint64_t scratch_bytes,
    void* stream
)
{
    qnpeps::reset_err();
    if (not input or not q_out or not r_out or not scratch)
        return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    if (rows < 1 or cols < 1 or rank < 1 or batch < 1)
        return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
    if (rank > rows or rank > cols) return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);

    const auto rows_i64 = static_cast<i64>(rows);
    const auto cols_i64 = static_cast<i64>(cols);
    const auto rank_i64 = static_cast<i64>(rank);
    const auto batch_i64 = static_cast<i64>(batch);
    const auto batch_size = static_cast<usize>(batch_i64);
    const auto omega_size = static_cast<usize>(cols_i64 * rank_i64);
    const auto invalid_stride = input_stride < rows_i64 * cols_i64 or q_stride < rows_i64 * rank_i64
                                or r_stride < rank_i64 * cols_i64;
    if (invalid_stride)
    {
        return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
    }

    auto cursor = ArenaCursor::carve(scratch, static_cast<usize>(scratch_bytes));
    auto* device_sketch =
        cursor.take<cuFloatComplex>(static_cast<usize>(rows_i64 * rank_i64 * batch_i64));
    auto* device_projection =
        cursor.take<cuFloatComplex>(static_cast<usize>(cols_i64 * rank_i64 * batch_i64));
    auto* device_gram =
        cursor.take<cuFloatComplex>(static_cast<usize>(rank_i64 * rank_i64 * batch_i64));
    auto* device_omega = cursor.take<cuFloatComplex>(omega_size);
    auto* device_gram_pointers = cursor.take<cuFloatComplex*>(batch_size);
    auto* device_sketch_pointers = cursor.take<cuFloatComplex*>(batch_size);
    auto* device_info = cursor.take<int>(batch_size);
    auto* device_fail_flag = cursor.take<int>(1);
    if (qnpeps::err_state() != QNPEPS_OK) return qnpeps::err_state();
    auto session = make_session(static_cast<cudaStream_t>(stream));
    if (not session) return qnpeps::err_state();
    auto& linalg = session->linalg();
    const auto cuda_stream = linalg.stream();

    auto rng = RangefinderRng::from_seed_and_width(seed, cols);
    std::vector<cuFloatComplex> host_omega{};
    host_omega.resize(omega_size);
    rng.fill_complex_normal(std::span{host_omega});
    upload_async(linalg, device_omega, host_omega.data(), host_omega.size());

    std::vector<cuFloatComplex*> host_gram_pointers{};
    host_gram_pointers.resize(batch_size);
    std::vector<cuFloatComplex*> host_sketch_pointers{};
    host_sketch_pointers.resize(batch_size);
    for (auto lane = 0_i64; lane < batch_i64; ++lane)
    {
        const auto lane_index = static_cast<usize>(lane);
        host_gram_pointers[lane_index] = device_gram + lane * rank_i64 * rank_i64;
        host_sketch_pointers[lane_index] = device_sketch + lane * rows_i64 * rank_i64;
    }
    upload_async(
        linalg, device_gram_pointers, host_gram_pointers.data(), host_gram_pointers.size()
    );
    upload_async(
        linalg, device_sketch_pointers, host_sketch_pointers.data(), host_sketch_pointers.size()
    );
    if (qnpeps::err_state() != QNPEPS_OK)
    {
        CUDA_NOCHECK(cudaStreamSynchronize(cuda_stream));
        return qnpeps::err_state();
    }
    zero_async(linalg, device_fail_flag, 1);
    batched_rangefinder(
        linalg,
        {
            .input =
                CuMatrixBatchedCF32Const{
                    static_cast<const cuFloatComplex*>(input), input_stride, rows, cols
                },
            .rank = rank,
            .omega = device_omega,
            .q_out = CuMatrixBatchedCF32{static_cast<cuFloatComplex*>(q_out), q_stride, rows, rank},
            .r_out = CuMatrixBatchedCF32{static_cast<cuFloatComplex*>(r_out), r_stride, rank, cols},
            .dim_batch = batch,
            .sketch = CuSpanCF32{device_sketch, rows_i64 * rank_i64},
            .projection = CuSpanCF32{device_projection, cols_i64 * rank_i64},
            .gram = CuSpanCF32{device_gram, rank_i64 * rank_i64},
            .gram_ptrs = device_gram_pointers,
            .sketch_ptrs = device_sketch_pointers,
            .info = device_info,
            .fail_flag = device_fail_flag,
        }
    );
    CUDA_CHECK(cudaStreamSynchronize(linalg.stream()));
    int host_fail_flag{};
    download(&host_fail_flag, device_fail_flag, 1);
    if (host_fail_flag != 0) qnpeps::set_err(QNPEPS_ERR_CUDA);
    return qnpeps::err_state();
}

extern "C" int64_t qnpeps_peps_bytes(const QnpepsConfig* config)
{
    if (check_cfg(config) != QNPEPS_OK) return -1;
    const PepsDims dims{config->lx, config->ly, config->dim_phys, config->dim_bond};
    return peps_elems(dims) * static_cast<i64>(sizeof(cuFloatComplex));
}

extern "C" int64_t qnpeps_sample_bytes(const QnpepsConfig* config, uint64_t count)
{
    if (check_cfg(config) != QNPEPS_OK) return -1;
    const auto sites = static_cast<uint64_t>(config->lx) * static_cast<uint64_t>(config->ly);
    const auto limit = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    if (sites == 0 or count > limit / sites) return -1;
    return static_cast<int64_t>(count * sites);
}

extern "C" int64_t qnpeps_dlenv_bytes(const QnpepsConfig* config)
{
    if (check_cfg(config) != QNPEPS_OK) return -1;
    const i64 lx{config->lx};
    const i64 ly{config->ly};
    const i64 dim_bond{config->dim_bond};
    const i64 chi_dl{config->chi_dl};
    const auto int32_bytes = static_cast<i64>(sizeof(int32_t));
    const auto f32_bytes = static_cast<i64>(sizeof(f32));
    const auto rows_below = lx - 1;
    const auto num_sites = rows_below * ly;
    const auto bond_pair = dim_bond * dim_bond;
    const auto chi_c = std::min(chi_dl, bond_pair);
    const auto header = num_sites * 4 * int32_bytes;
    const auto values = num_sites * chi_c * chi_c * bond_pair;
    return header + values * 2 * f32_bytes;
}

extern "C" int64_t qnpeps_sample_footprint_bytes(
    const QnpepsConfig* config, uint64_t count, uint64_t dim_batch
)
{
    const auto invalid = check_cfg(config) != QNPEPS_OK or dim_batch < 1
                         or dim_batch > static_cast<uint64_t>(k_max_batch_size);
    if (invalid)
    {
        return -1;
    }
    return qnpeps_sample_bytes(config, count);
}

extern "C" int64_t qnpeps_sample_scratch_bytes(const QnpepsConfig* config, uint64_t dim_batch)
{
    const auto invalid = check_cfg(config) != QNPEPS_OK or dim_batch < 1
                         or dim_batch > static_cast<uint64_t>(k_max_batch_size);
    if (invalid)
    {
        return -1;
    }
    return 0;
}

extern "C" void qnpeps_sampler_pool_release(void) {}

extern "C" const char* qnpeps_last_error_file(void)
{
    return qnpeps::err_file();
}

extern "C" int32_t qnpeps_last_error_line(void)
{
    return qnpeps::err_line();
}

extern "C" const char* qnpeps_last_error_message(void)
{
    return qnpeps::err_message();
}

extern "C" const char* qnpeps_strerror(qnpeps_status status)
{
    switch (status)
    {
        case QNPEPS_OK:
            return "ok";
        case QNPEPS_ERR_NULL_ARG:
            return "a required pointer was NULL";
        case QNPEPS_ERR_BAD_CONFIG:
            return "descriptor or batch dimensions failed validation, or dlenv header is "
                   "inconsistent with the config";
        case QNPEPS_ERR_BAD_VERSION:
            return "descriptor struct_size not recognized";
        case QNPEPS_ERR_CUDA:
            return "an underlying CUDA/cuBLAS/cuSOLVER call failed";
        case QNPEPS_ERR_OOM:
            return "device allocation failed";
        case QNPEPS_ERR_INTERNAL:
            return "internal invariant violated";
    }
    return "unknown status";
}

extern "C" const char* qnpeps_capi_version(void)
{
    return "cuQuantumNaturalfPEPS " QNPEPS_C_API_VERSION;
}
