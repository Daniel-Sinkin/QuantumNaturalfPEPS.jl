#include "../minsr/solve.cuh"
#include "common.cuh"
#include "dans_qnpeps_e2e.h"
#include "layout.cuh"

#include <cstdint>

#ifndef QNPEPS_PACKAGE_VERSION
#    error "QNPEPS_PACKAGE_VERSION must be provided by CMake"
#endif

namespace
{
auto check_cfg(const QnpepsE2eConfig* c) -> qnpeps_e2e_status
{
    if (not c) return QNPEPS_E2E_ERR_NULL_ARG;
    if (c->struct_size != sizeof(QnpepsE2eConfig)) return QNPEPS_E2E_ERR_BAD_VERSION;
    if (c->lx < 2 or c->ly < 2 or c->dim_phys != 2 or c->dim_bond < 1)
        return QNPEPS_E2E_ERR_BAD_CONFIG;
    return QNPEPS_E2E_OK;
}
auto as_cf(const qnpeps_e2e_cbuf* p) -> const qn_e2e::cf*
{
    return reinterpret_cast<const qn_e2e::cf*>(p);
}
auto as_cf(qnpeps_e2e_cbuf* p) -> qn_e2e::cf*
{
    return reinterpret_cast<qn_e2e::cf*>(p);
}
}

extern "C" qnpeps_e2e_status qnpeps_e2e_minsr_ctx_create(
    const QnpepsE2eConfig* cfg,
    int64_t n_samples,
    int64_t host_tile_bytes,
    void* stream,
    qnpeps_e2e_minsr_ctx** out
)
{
    qn_e2e::clear_err();
    if (not out) return QNPEPS_E2E_ERR_NULL_ARG;
    *out = nullptr;
    const qnpeps_e2e_status v{check_cfg(cfg)};
    if (v != QNPEPS_E2E_OK) return v;
    if (n_samples < 2) return QNPEPS_E2E_ERR_BAD_CONFIG;

    qnpeps::minsr::minsr_context_create(
        *cfg, n_samples, host_tile_bytes, static_cast<cudaStream_t>(stream), out
    );
    return qn_e2e::err_state();
}

extern "C" qnpeps_e2e_status qnpeps_e2e_minsr_ctx_run(
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
    qn_e2e::clear_err();
    const auto missing_input = not ctx or not device_samples or not device_log_amplitudes
                               or not device_local_energies or not device_log_proposals
                               or not device_gram or not theta_output or not energy_mean_output
                               or not energy_variance_output or not ess_output;
    if (missing_input) return QNPEPS_E2E_ERR_NULL_ARG;
    if (static_cast<bool>(device_rows) == static_cast<bool>(host_rows))
        return QNPEPS_E2E_ERR_NULL_ARG;

    qnpeps::minsr::minsr_context_run(
        *ctx,
        device_samples,
        device_log_amplitudes,
        device_local_energies,
        device_log_proposals,
        as_cf(device_gram),
        as_cf(device_rows),
        as_cf(host_rows),
        relative_cut,
        absolute_cut,
        as_cf(theta_output),
        energy_mean_output,
        energy_variance_output,
        ess_output
    );
    return qn_e2e::err_state();
}

extern "C" void qnpeps_e2e_minsr_ctx_destroy(qnpeps_e2e_minsr_ctx* ctx)
{
    qnpeps::minsr::minsr_context_destroy(ctx);
}

extern "C" qnpeps_e2e_status qnpeps_e2e_minsr(
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
    const qnpeps_e2e_status v{check_cfg(cfg)};
    if (v != QNPEPS_E2E_OK) return v;
    const auto missing_input = not device_samples or not device_log_amplitudes
                               or not device_local_energies or not device_log_proposals
                               or not device_gram or not theta_output or not energy_mean_output
                               or not energy_variance_output or not ess_output;
    if (missing_input) return QNPEPS_E2E_ERR_NULL_ARG;
    if (static_cast<bool>(device_rows) == static_cast<bool>(host_rows))
        return QNPEPS_E2E_ERR_NULL_ARG;
    if (n_samples < 2) return QNPEPS_E2E_ERR_BAD_CONFIG;

    qnpeps_e2e_minsr_ctx* ctx{};
    qnpeps_e2e_status status{
        qnpeps_e2e_minsr_ctx_create(cfg, n_samples, host_tile_bytes, stream, &ctx)
    };
    if (status != QNPEPS_E2E_OK) return status;
    status = qnpeps_e2e_minsr_ctx_run(
        ctx,
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
    );
    qnpeps_e2e_minsr_ctx_destroy(ctx);
    return status;
}

extern "C" qnpeps_e2e_status qnpeps_e2e_dense_count(const QnpepsE2eConfig* cfg, int64_t* out_count)
{
    const qnpeps_e2e_status v{check_cfg(cfg)};
    if (v != QNPEPS_E2E_OK) return v;
    if (not out_count) return QNPEPS_E2E_ERR_NULL_ARG;
    *out_count = qn_e2e::dense_count(*cfg);
    return QNPEPS_E2E_OK;
}

extern "C" qnpeps_e2e_status qnpeps_e2e_compact_count(
    const QnpepsE2eConfig* cfg, int64_t* out_count
)
{
    const qnpeps_e2e_status v{check_cfg(cfg)};
    if (v != QNPEPS_E2E_OK) return v;
    if (not out_count) return QNPEPS_E2E_ERR_NULL_ARG;
    *out_count = qn_e2e::compact_count(*cfg);
    return QNPEPS_E2E_OK;
}

extern "C" qnpeps_e2e_status qnpeps_e2e_minsr_scratch_bytes(
    const QnpepsE2eConfig* cfg, int64_t n_samples, int64_t host_tile_bytes, uint64_t* out_bytes
)
{
    const qnpeps_e2e_status v{check_cfg(cfg)};
    if (v != QNPEPS_E2E_OK) return v;
    if (not out_bytes) return QNPEPS_E2E_ERR_NULL_ARG;
    *out_bytes =
        qnpeps::minsr::qn_e2e_minsr_scratch(*cfg, n_samples < 1 ? 1 : n_samples, host_tile_bytes);
    return QNPEPS_E2E_OK;
}

extern "C" const char* qnpeps_e2e_strerror(qnpeps_e2e_status status)
{
    switch (status)
    {
        case QNPEPS_E2E_OK:
            return "ok";
        case QNPEPS_E2E_ERR_NULL_ARG:
            return "a required pointer was NULL (or both/neither row source supplied)";
        case QNPEPS_E2E_ERR_BAD_CONFIG:
            return "descriptor or sample count failed validation";
        case QNPEPS_E2E_ERR_BAD_VERSION:
            return "descriptor struct_size not recognized";
        case QNPEPS_E2E_ERR_CUDA:
            return "an underlying CUDA / cuBLAS / cuSOLVER call failed";
        case QNPEPS_E2E_ERR_OOM:
            return "device allocation failed";
        case QNPEPS_E2E_ERR_INTERNAL:
            return "internal invariant violated (please report)";
    }
    return "unknown status";
}

extern "C" const char* qnpeps_e2e_version(void)
{
    return "cuQuantumNaturalfPEPS e2e " QNPEPS_PACKAGE_VERSION;
}
