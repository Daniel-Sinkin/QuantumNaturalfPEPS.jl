#include "capi/internal/arena_reading.cuh"
#include "capi/internal/validation.cuh"
#include "capi/qnpeps.h"
#include "core/arena_cursor.cuh"
#include "core/cuda_utils.cuh"
#include "core/defer.cuh"
#include "core/qnpeps_ctx.cuh"
#include "dlenv/build.cuh"
#include "linalg/linalg.cuh"
#include "linalg/transfer.cuh"
#include "peps/init.cuh"
#include "peps/peps.cuh"
#include "sampler/draw.cuh"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#ifndef QNPEPS_C_API_VERSION
#    error "QNPEPS_C_API_VERSION must be provided by CMake"
#endif

using namespace qnpeps;
using qnpeps::capi::validate_config;

namespace qnpeps::dlenv
{
auto dl_free(qnpeps_ctx& ctx) -> void;
}

namespace
{
auto validate_sample_shape(
    const QnpepsConfig& config, uint64_t n_samples, uint64_t dim_batch, uint64_t batch_base
) -> void
{
    require(
        dim_batch >= 1 and dim_batch <= static_cast<uint64_t>(k_max_batch_size),
        Status::bad_config,
        "sample batch {} outside 1 to {}",
        dim_batch,
        k_max_batch_size
    );
    require(
        n_samples <= static_cast<uint64_t>(std::numeric_limits<i64>::max()),
        Status::bad_config,
        "sample count {} exceeds the i64 range",
        n_samples
    );
    if (n_samples == 0) return;

    const auto batches = n_samples / dim_batch + static_cast<uint64_t>(n_samples % dim_batch != 0);
    require(
        batches <= static_cast<uint64_t>(INT_MAX),
        Status::bad_config,
        "{} sample batches exceed the int range",
        batches
    );
    require(
        batch_base <= std::numeric_limits<uint64_t>::max() - (batches - 1),
        Status::bad_config,
        "batch base {} overflows with {} batches",
        batch_base,
        batches
    );

    const auto sites = static_cast<uint64_t>(config.lx) * static_cast<uint64_t>(config.ly);
    const auto size_limit = static_cast<uint64_t>(std::numeric_limits<usize>::max());
    const auto fits =
        sites != 0 and n_samples <= size_limit / sites and n_samples <= size_limit / sizeof(f64);
    require(
        fits, Status::bad_config, "{} samples over {} sites exceed the size range", n_samples, sites
    );
}

auto dispatch_sample(const QnpepsConfig& config, int gpus, const qnpeps::sampler::SampleArgs& args)
    -> void
{
    require(gpus >= 1, Status::bad_config, "sampling over {} GPUs", gpus);
    if (gpus > 1)
    {
        require(
            not args.scratch, Status::bad_config, "multi-GPU sampling does not take caller scratch"
        );
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
        return;
    }
    qnpeps::sampler::sample(config, args);
}

auto ctx_sample(
    qnpeps_ctx* ctx,
    const QnpepsCtxSampleArgs* args,
    qnpeps::sampler::SampleOutputLocation output_location
) -> void
{
    require(
        ctx != nullptr and args != nullptr,
        Status::null_argument,
        "context sampling needs a context and arguments"
    );
    require_struct_size(*args, "context sample arguments");
    require(
        args->n_samples == 0 or args->samples_out != nullptr,
        Status::null_argument,
        "context sampling of {} samples has no sample output",
        args->n_samples
    );
    validate_sample_shape(ctx->cfg, args->n_samples, args->dim_batch, args->batch_base);
    qnpeps::sampler::ctx_sample(
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
) -> void
{
    require(
        ctx.sampler.ready() and ctx.sampler.execution.host_managed,
        Status::bad_config,
        "host sampler pointers before a host-managed sampler is ready"
    );
    require(pointers != nullptr, Status::null_argument, "host sampler pointer table is null");
    const auto expected = static_cast<uint64_t>(qnpeps::dlenv::k_sampling_layout_count)
                          * static_cast<uint64_t>(ctx.cfg.lx - 1)
                          * static_cast<uint64_t>(ctx.cfg.ly)
                          * static_cast<uint64_t>(ctx.sampler.allocation.dim_batch_capacity);
    require(
        pointer_count == expected,
        Status::bad_config,
        "host sampler pointer table holds {} entries, expected {}",
        pointer_count,
        expected
    );
    auto& samp = ctx.sampler.samp;
    upload_async(
        ctx.linalg(), samp.dlenv_env_ptrs()[0][0], pointers, static_cast<usize>(pointer_count)
    );
    ctx.sampler.execution.host_pointers = true;
}

auto sampler_host_refresh(qnpeps_ctx* ctx, const QnpepsSamplerHostRefreshArgs* args) -> void
{
    require(
        ctx != nullptr and args != nullptr,
        Status::null_argument,
        "host refresh needs a context and arguments"
    );
    require_struct_size(*args, "host refresh arguments");
    const auto has_buffers = args->peps and args->dlenv_values and args->sampling;
    require(has_buffers, Status::null_argument, "host refresh misses a buffer");
    require(
        ctx->sampler.ready() and ctx->sampler.execution.host_managed,
        Status::bad_config,
        "host refresh before a host-managed sampler is ready"
    );
    const auto required = sampler_host_sampling_bytes(
        *ctx, ctx->dlenv.dims.data(), static_cast<uint64_t>(ctx->dlenv.dims.size())
    );
    require(
        required != 0 and args->sampling_bytes >= required,
        Status::bad_config,
        "host refresh sampling buffer holds {} bytes, needs {}",
        args->sampling_bytes,
        required
    );
    auto& lane = ctx->dlenv.lanes[ctx->dlenv.active_lane];
    lane.sampling = static_cast<cuFloatComplex*>(args->sampling);
    lane.sampling_owned = false;
    qnpeps::dlenv::materialize_sampling_buffer(
        *ctx, reinterpret_cast<const cuFloatComplex*>(args->dlenv_values), lane.sampling
    );
    const auto layout =
        args->peps_layout == 0 ? qnpeps::PepsLayout::canonical : qnpeps::PepsLayout::reverse_packed;
    qnpeps::sampler::ctx_sample_refresh(*ctx, args->peps, layout);
}

auto sampler_host_batch(qnpeps_ctx* ctx, const QnpepsSamplerHostBatchArgs* args) -> void
{
    require(
        ctx != nullptr and args != nullptr,
        Status::null_argument,
        "host batch needs a context and arguments"
    );
    require_struct_size(*args, "host batch arguments");
    const auto missing_input = not args->peps or not args->dlenv_dims or not args->dlenv_values
                               or not args->sampling or not args->dlenv_pointers
                               or not args->samples_out or not args->log_prob_config
                               or not args->log_gauge;
    require(not missing_input, Status::null_argument, "host batch misses a buffer");
    const auto invalid_batch = args->dim_batch < 1
                               or args->dim_batch > static_cast<uint64_t>(k_max_batch_size)
                               or args->batch_id > static_cast<uint64_t>(INT_MAX);
    require(
        not invalid_batch,
        Status::bad_config,
        "host batch {} with batch size {} is out of range",
        args->batch_id,
        args->dim_batch
    );
    const auto sampling_bytes =
        sampler_host_sampling_bytes(*ctx, args->dlenv_dims, args->dlenv_dims_count);
    require(
        sampling_bytes != 0 and args->sampling_bytes >= sampling_bytes,
        Status::bad_config,
        "host batch sampling buffer holds {} bytes, needs {}",
        args->sampling_bytes,
        sampling_bytes
    );
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
        sampler_host_upload_pointers(*ctx, args->dlenv_pointers, args->dlenv_pointer_count);
        const auto layout = args->peps_layout == 0 ? qnpeps::PepsLayout::canonical
                                                   : qnpeps::PepsLayout::reverse_packed;
        qnpeps::sampler::ctx_sample_refresh(*ctx, args->peps, layout);
    }
    const auto invalid_host_state =
        not ctx->sampler.execution.host_managed
        or ctx->sampler.execution.dim_batch != static_cast<int>(args->dim_batch);
    require(
        not invalid_host_state,
        Status::bad_config,
        "host batch size {} does not match the host-managed sampler batch {}",
        args->dim_batch,
        ctx->sampler.execution.dim_batch
    );
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
    const Defer restore_staging{[&]
                                {
                                    staging.h_samples = old_samples;
                                    staging.h_logpc = old_logpc;
                                    staging.h_lognorm = old_lognorm;
                                }};
    qnpeps::sampler::HostSampleOutput output{
        .samples = args->samples_out,
        .logpc = args->log_prob_config,
        .lognorm = args->log_gauge,
        .n_samples = args->dim_batch,
        .batch_origin = args->batch_id,
    };
    qnpeps::sampler::ctx_sample_run(*ctx, batch_ids, &output);
}

auto sample_from_config(
    const QnpepsConfig* config,
    uint32_t struct_size,
    usize expected_size,
    const qnpeps::sampler::SampleArgs& args,
    int gpus
) -> void
{
    require(
        struct_size == expected_size,
        Status::bad_version,
        "sample arguments struct_size {}, expected {}",
        struct_size,
        expected_size
    );
    const auto has_inputs = args.device_peps and args.device_dlenv;
    require(has_inputs, Status::null_argument, "sampling needs a PEPS and an environment");
    require(
        args.n_samples == 0 or args.output != nullptr,
        Status::null_argument,
        "sampling of {} samples has no sample output",
        args.n_samples
    );
    validate_config(config);
    validate_sample_shape(*config, args.n_samples, args.dim_batch, args.batch_base);
    if (args.n_samples == 0) return;
    dispatch_sample(*config, gpus, args);
}
}

namespace qnpeps::dlenv
{
auto build_dlenv_packed(
    const QnpepsConfig& config,
    const void* device_peps,
    void* output,
    f64* cumulative_row_logs,
    Linalg& linalg
) -> void
{
    qnpeps_ctx ctx{config, linalg};
    const Defer release_context{[&]
                                {
                                    unchecked(cudaStreamSynchronize(linalg.stream()));
                                    dl_free(ctx);
                                }};
    build_dlenv(ctx, device_peps, cumulative_row_logs);
    const auto bytes = qnpeps_dlenv_bytes(&config);
    const auto& active_lane = ctx.dlenv.lanes[ctx.dlenv.active_lane];
    copy_device(
        static_cast<u8*>(output),
        reinterpret_cast<const u8*>(active_lane.packed),
        static_cast<usize>(bytes)
    );
}
}

extern "C" qnpeps_status qnpeps_ctx_create(
    const QnpepsConfig* config, void* stream, qnpeps_ctx** out
)
{
    return guarded(
        [&]
        {
            require(out != nullptr, Status::null_argument, "context output is null");
            *out = nullptr;
            validate_config(config);
            auto session = make_session(static_cast<cudaStream_t>(stream));
            *out = new qnpeps_ctx{*config, std::move(session)};
        }
    );
}

extern "C" void qnpeps_ctx_destroy(qnpeps_ctx* ctx)
{
    static_cast<void>(guarded_keeping_message(
        [&]
        {
            if (not ctx) return;
            unchecked(cudaStreamSynchronize(ctx->stream()));
            qnpeps::sampler::ctx_sampler_free(*ctx);
            qnpeps::dlenv::dl_free(*ctx);
            delete ctx;
        }
    ));
}

extern "C" qnpeps_status qnpeps_ctx_build_dlenv(
    qnpeps_ctx* ctx, const qnpeps_device_peps* peps, double* cumulative_row_logs
)
{
    return guarded(
        [&]
        {
            require(
                ctx != nullptr and peps != nullptr,
                Status::null_argument,
                "environment build needs a context and a PEPS"
            );
            qnpeps::dlenv::build_dlenv(*ctx, peps, cumulative_row_logs);
        }
    );
}

extern "C" qnpeps_status qnpeps_ctx_copy_dlenv_host(
    const qnpeps_ctx* ctx, void* output, uint64_t output_bytes
)
{
    return guarded(
        [&]
        {
            require(
                ctx != nullptr and output != nullptr,
                Status::null_argument,
                "environment copy needs a context and an output"
            );
            const auto& active_lane = ctx->dlenv.lanes[ctx->dlenv.active_lane];
            require(
                active_lane.valid and active_lane.packed,
                Status::bad_config,
                "environment copy before an environment was built"
            );
            const auto required = qnpeps_dlenv_bytes(&ctx->cfg);
            require(
                required >= 0 and output_bytes >= static_cast<uint64_t>(required),
                Status::bad_config,
                "environment copy output holds {} bytes, needs {}",
                output_bytes,
                required
            );
            download(
                static_cast<u8*>(output),
                reinterpret_cast<const u8*>(active_lane.packed),
                static_cast<usize>(required)
            );
        }
    );
}

extern "C" qnpeps_status qnpeps_ctx_sample(qnpeps_ctx* ctx, const QnpepsCtxSampleArgs* args)
{
    return guarded([&] { ctx_sample(ctx, args, qnpeps::sampler::SampleOutputLocation::device); });
}

extern "C" qnpeps_status qnpeps_ctx_sample_host(qnpeps_ctx* ctx, const QnpepsCtxSampleArgs* args)
{
    return guarded([&] { ctx_sample(ctx, args, qnpeps::sampler::SampleOutputLocation::host); });
}

extern "C" qnpeps_status qnpeps_sampler_host_refresh(
    qnpeps_ctx* ctx, const QnpepsSamplerHostRefreshArgs* args
)
{
    return guarded([&] { sampler_host_refresh(ctx, args); });
}

extern "C" qnpeps_status qnpeps_sampler_host_batch(
    qnpeps_ctx* ctx, const QnpepsSamplerHostBatchArgs* args
)
{
    return guarded([&] { sampler_host_batch(ctx, args); });
}

extern "C" qnpeps_status qnpeps_build_dlenv(
    const QnpepsConfig* config,
    const qnpeps_device_peps* device_peps,
    qnpeps_device_dlenv* dlenv_out,
    double* cumulative_row_logs,
    void* stream
)
{
    return guarded(
        [&]
        {
            validate_config(config);
            require(
                device_peps != nullptr and dlenv_out != nullptr,
                Status::null_argument,
                "environment build needs a PEPS and an output"
            );
            auto session = make_session(static_cast<cudaStream_t>(stream));
            qnpeps::dlenv::build_dlenv_packed(
                *config, device_peps, dlenv_out, cumulative_row_logs, session->linalg()
            );
        }
    );
}

extern "C" qnpeps_status qnpeps_sample(const QnpepsConfig* config, const QnpepsSampleArgs* args)
{
    return guarded(
        [&]
        {
            require(
                config != nullptr and args != nullptr,
                Status::null_argument,
                "sampling needs a config and arguments"
            );
            sample_from_config(
                config,
                args->struct_size,
                sizeof(QnpepsSampleArgs),
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
                },
                args->gpus
            );
        }
    );
}

extern "C" qnpeps_status qnpeps_sample_host(
    const QnpepsConfig* config, const QnpepsSampleHostArgs* args
)
{
    return guarded(
        [&]
        {
            require(
                config != nullptr and args != nullptr,
                Status::null_argument,
                "sampling needs a config and arguments"
            );
            sample_from_config(
                config,
                args->struct_size,
                sizeof(QnpepsSampleHostArgs),
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
                },
                args->gpus
            );
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
    return guarded(
        [&]
        {
            validate_config(config);
            require(peps_out != nullptr, Status::null_argument, "random PEPS output is null");
            const auto required_bytes = qnpeps_peps_bytes(config);
            require(
                required_bytes >= 0 and peps_bytes >= static_cast<uint64_t>(required_bytes),
                Status::bad_config,
                "random PEPS output holds {} bytes, needs {}",
                peps_bytes,
                required_bytes
            );
            auto session = make_session(static_cast<cudaStream_t>(stream));
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
        }
    );
}

namespace
{
[[nodiscard]] auto sample_bytes(const QnpepsConfig* config, uint64_t count) -> int64_t
{
    validate_config(config);
    const auto sites = static_cast<uint64_t>(config->lx) * static_cast<uint64_t>(config->ly);
    const auto limit = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    require(
        sites != 0 and count <= limit / sites,
        Status::bad_config,
        "{} samples over {} sites exceed the i64 byte range",
        count,
        sites
    );
    return static_cast<int64_t>(count * sites);
}

auto validate_sample_batch(const QnpepsConfig* config, uint64_t dim_batch) -> void
{
    validate_config(config);
    require(
        dim_batch >= 1 and dim_batch <= static_cast<uint64_t>(k_max_batch_size),
        Status::bad_config,
        "sample batch {} outside 1 to {}",
        dim_batch,
        k_max_batch_size
    );
}
}

extern "C" int64_t qnpeps_peps_bytes(const QnpepsConfig* config)
{
    return guarded_or(
        int64_t{-1},
        [&]
        {
            validate_config(config);
            const PepsDims dims{config->lx, config->ly, config->dim_phys, config->dim_bond};
            return peps_elems(dims) * static_cast<i64>(sizeof(cuFloatComplex));
        }
    );
}

extern "C" int64_t qnpeps_sample_bytes(const QnpepsConfig* config, uint64_t count)
{
    return guarded_or(int64_t{-1}, [&] { return sample_bytes(config, count); });
}

extern "C" int64_t qnpeps_dlenv_bytes(const QnpepsConfig* config)
{
    return guarded_or(
        int64_t{-1},
        [&]
        {
            validate_config(config);
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
    );
}

extern "C" int64_t qnpeps_sample_footprint_bytes(
    const QnpepsConfig* config, uint64_t count, uint64_t dim_batch
)
{
    return guarded_or(
        int64_t{-1},
        [&]
        {
            validate_sample_batch(config, dim_batch);
            return sample_bytes(config, count);
        }
    );
}

extern "C" int64_t qnpeps_sample_scratch_bytes(const QnpepsConfig* config, uint64_t dim_batch)
{
    return guarded_or(
        int64_t{-1},
        [&]
        {
            validate_sample_batch(config, dim_batch);
            return int64_t{0};
        }
    );
}

extern "C" void qnpeps_sampler_pool_release(void) {}

extern "C" qnpeps_status qnpeps_device_arena_reading(int32_t device, QnpepsArenaReading* out)
{
    return guarded([&] { qnpeps::capi::read_device_arena(device, out); });
}

extern "C" const char* qnpeps_capi_version(void)
{
    return "cuQuantumNaturalfPEPS " QNPEPS_C_API_VERSION;
}
