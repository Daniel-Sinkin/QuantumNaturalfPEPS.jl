#include "capi/internal/format.cuh"
#include "../minsr/solve.cuh"
#include "common.cuh"
#include "core/defer.cuh"
#include "dans_qnpeps_e2e.h"
#include "layout.cuh"

#include <cstdint>

#ifndef QNPEPS_PACKAGE_VERSION
#    error "QNPEPS_PACKAGE_VERSION must be provided by CMake"
#endif

namespace
{
auto validate_config(const QnpepsE2eConfig* config) -> void
{
    require(config != nullptr, Status::null_argument, "end-to-end config is null");
    require_struct_size(*config, "end-to-end config");
    const auto valid_geometry =
        config->lx >= 2 and config->ly >= 2 and config->dim_phys == 2 and config->dim_bond >= 1;
    require(valid_geometry, Status::bad_config, "end-to-end config rejects {}", *config);
}

auto as_cf(const qnpeps_e2e_cbuf* p) -> const qn_e2e::cf*
{
    return reinterpret_cast<const qn_e2e::cf*>(p);
}

auto as_cf(qnpeps_e2e_cbuf* p) -> qn_e2e::cf*
{
    return reinterpret_cast<qn_e2e::cf*>(p);
}

struct MinsrInputs
{
    const uint8_t* device_samples;
    const double* device_log_amplitudes;
    const double* device_local_energies;
    const double* device_log_proposals;
    const qnpeps_e2e_cbuf* device_gram;
    const qnpeps_e2e_cbuf* device_rows;
    const qnpeps_e2e_cbuf* host_rows;
    double relative_cut;
    double absolute_cut;
    qnpeps_e2e_cbuf* theta_output;
    double* energy_mean_output;
    double* energy_variance_output;
    double* ess_output;
};

auto validate_inputs(const MinsrInputs& inputs) -> void
{
    const auto has_buffers =
        inputs.device_samples and inputs.device_log_amplitudes and inputs.device_local_energies
        and inputs.device_log_proposals and inputs.device_gram and inputs.theta_output
        and inputs.energy_mean_output and inputs.energy_variance_output and inputs.ess_output;
    require(
        has_buffers, Status::null_argument, "end-to-end MinSR misses an input or output buffer"
    );
    require(
        static_cast<bool>(inputs.device_rows) != static_cast<bool>(inputs.host_rows),
        Status::null_argument,
        "end-to-end MinSR needs exactly one of device rows and host rows"
    );
}

auto minsr_context_create(
    const QnpepsE2eConfig* config,
    int64_t n_samples,
    int64_t host_tile_bytes,
    void* stream,
    qnpeps_e2e_minsr_ctx** out
) -> void
{
    require(out != nullptr, Status::null_argument, "end-to-end MinSR context output is null");
    *out = nullptr;
    validate_config(config);
    require(
        n_samples >= 2,
        Status::bad_config,
        "end-to-end MinSR needs at least 2 samples, got {}",
        n_samples
    );
    qnpeps::minsr::minsr_context_create(
        *config, n_samples, host_tile_bytes, static_cast<cudaStream_t>(stream), out
    );
}

auto minsr_context_run(qnpeps_e2e_minsr_ctx& context, const MinsrInputs& inputs) -> void
{
    qnpeps::minsr::minsr_context_run(
        context,
        inputs.device_samples,
        inputs.device_log_amplitudes,
        inputs.device_local_energies,
        inputs.device_log_proposals,
        as_cf(inputs.device_gram),
        as_cf(inputs.device_rows),
        as_cf(inputs.host_rows),
        inputs.relative_cut,
        inputs.absolute_cut,
        as_cf(inputs.theta_output),
        inputs.energy_mean_output,
        inputs.energy_variance_output,
        inputs.ess_output
    );
}

auto validate_count_query(const QnpepsE2eConfig* config, const void* output) -> void
{
    validate_config(config);
    require(output != nullptr, Status::null_argument, "end-to-end count query output is null");
}
}

extern "C" qnpeps_status qnpeps_e2e_minsr_ctx_create(
    const QnpepsE2eConfig* cfg,
    int64_t n_samples,
    int64_t host_tile_bytes,
    void* stream,
    qnpeps_e2e_minsr_ctx** out
)
{
    return guarded([&] { minsr_context_create(cfg, n_samples, host_tile_bytes, stream, out); });
}

extern "C" qnpeps_status qnpeps_e2e_minsr_ctx_run(
    qnpeps_e2e_minsr_ctx* ctx,
    const uint8_t* device_samples,
    const double* device_log_amplitudes,
    const double* device_local_energies,
    const double* device_log_proposals,
    const qnpeps_e2e_cbuf* device_gram,
    const qnpeps_e2e_cbuf* device_rows,
    const qnpeps_e2e_cbuf* host_rows,
    double relative_cut,
    double absolute_cut,
    qnpeps_e2e_cbuf* theta_output,
    double* energy_mean_output,
    double* energy_variance_output,
    double* ess_output
)
{
    return guarded(
        [&]
        {
            require(
                ctx != nullptr, Status::null_argument, "end-to-end MinSR run without a context"
            );
            const MinsrInputs inputs{
                device_samples,
                device_log_amplitudes,
                device_local_energies,
                device_log_proposals,
                device_gram,
                device_rows,
                host_rows,
                relative_cut,
                absolute_cut,
                theta_output,
                energy_mean_output,
                energy_variance_output,
                ess_output
            };
            validate_inputs(inputs);
            minsr_context_run(*ctx, inputs);
        }
    );
}

extern "C" void qnpeps_e2e_minsr_ctx_destroy(qnpeps_e2e_minsr_ctx* ctx)
{
    static_cast<void>(guarded_keeping_message([&] { qnpeps::minsr::minsr_context_destroy(ctx); }));
}

extern "C" qnpeps_status qnpeps_e2e_minsr(
    const QnpepsE2eConfig* cfg,
    int64_t n_samples,
    const uint8_t* device_samples,
    const double* device_log_amplitudes,
    const double* device_local_energies,
    const double* device_log_proposals,
    const qnpeps_e2e_cbuf* device_gram,
    const qnpeps_e2e_cbuf* device_rows,
    const qnpeps_e2e_cbuf* host_rows,
    int64_t host_tile_bytes,
    double relative_cut,
    double absolute_cut,
    qnpeps_e2e_cbuf* theta_output,
    double* energy_mean_output,
    double* energy_variance_output,
    double* ess_output,
    void* stream
)
{
    return guarded(
        [&]
        {
            validate_config(cfg);
            const MinsrInputs inputs{
                device_samples,
                device_log_amplitudes,
                device_local_energies,
                device_log_proposals,
                device_gram,
                device_rows,
                host_rows,
                relative_cut,
                absolute_cut,
                theta_output,
                energy_mean_output,
                energy_variance_output,
                ess_output
            };
            validate_inputs(inputs);
            require(
                n_samples >= 2,
                Status::bad_config,
                "end-to-end MinSR needs at least 2 samples, got {}",
                n_samples
            );
            qnpeps_e2e_minsr_ctx* context{};
            minsr_context_create(cfg, n_samples, host_tile_bytes, stream, &context);
            const qnpeps::Defer destroy_context{[&]
                                                { qnpeps::minsr::minsr_context_destroy(context); }};
            minsr_context_run(*context, inputs);
        }
    );
}

extern "C" qnpeps_status qnpeps_e2e_dense_count(const QnpepsE2eConfig* cfg, int64_t* out_count)
{
    return guarded(
        [&]
        {
            validate_count_query(cfg, out_count);
            *out_count = qn_e2e::dense_count(*cfg);
        }
    );
}

extern "C" qnpeps_status qnpeps_e2e_compact_count(const QnpepsE2eConfig* cfg, int64_t* out_count)
{
    return guarded(
        [&]
        {
            validate_count_query(cfg, out_count);
            *out_count = qn_e2e::compact_count(*cfg);
        }
    );
}

extern "C" qnpeps_status qnpeps_e2e_minsr_scratch_bytes(
    const QnpepsE2eConfig* cfg, int64_t n_samples, int64_t host_tile_bytes, uint64_t* out_bytes
)
{
    return guarded(
        [&]
        {
            validate_count_query(cfg, out_bytes);
            *out_bytes = qnpeps::minsr::qn_e2e_minsr_scratch(
                *cfg, n_samples < 1 ? 1 : n_samples, host_tile_bytes
            );
        }
    );
}

extern "C" const char* qnpeps_e2e_version(void)
{
    return "cuQuantumNaturalfPEPS e2e " QNPEPS_PACKAGE_VERSION;
}
