#include "e2e/node/compat.cuh"

#include "e2e/common.cuh"
#include "e2e/mg_common.cuh"

#include <algorithm>
#include <cstring>

using qn_e2e::err_state;

namespace
{
auto composed_step(
    const QnpepsE2eConfig* cfg,
    const void* device_peps,
    i64 n_samples,
    const void* terms,
    int gpus,
    i64 peer_tile_bytes,
    i64 host_tile_bytes,
    f64 relative_cut,
    f64 absolute_cut,
    qnpeps_e2e_cbuf* theta_dot_out,
    f64* e_mean_out,
    f64* e_var_out,
    f64* ess_out,
    u8* samples_out,
    f64* logq_out,
    f64* log_gauge_out,
    f64* logpsi_out,
    f64* e_loc_out,
    qnpeps_e2e_cbuf* o_rows_host
) -> qnpeps_e2e_status
{
    const qnpeps_e2e_status config_status{qn_e2e::mg::_config_check(cfg)};
    if (config_status != QNPEPS_E2E_OK) return config_status;
    const auto missing_output = not device_peps or not terms or not theta_dot_out or not e_mean_out
                                or not e_var_out or not ess_out;
    if (missing_output) return QNPEPS_E2E_ERR_NULL_ARG;
    if (n_samples < 2 or gpus < 1) return QNPEPS_E2E_ERR_BAD_CONFIG;

    const i64 dim_batch{
        cfg->sample_batch > 0 ? cfg->sample_batch
                              : std::min<i64>(n_samples, qnpeps::k_max_batch_size)
    };
    const i64 capacity{((n_samples + dim_batch - 1) / dim_batch) * dim_batch};
    qnpeps_e2e_node* node{};
    qnpeps_e2e_status status{
        qnpeps_e2e_node_create(cfg, gpus, capacity, 0, dim_batch, host_tile_bytes, terms, &node)
    };
    if (status != QNPEPS_E2E_OK) return status;

    node->state_.peer_tile_bytes = peer_tile_bytes;
    status = qnpeps_e2e_node_submit_theta(node, device_peps);
    if (status == QNPEPS_E2E_OK)
    {
        status = qnpeps_e2e_node_step(
            node,
            n_samples,
            relative_cut,
            absolute_cut,
            theta_dot_out,
            e_mean_out,
            e_var_out,
            ess_out,
            samples_out,
            logq_out,
            log_gauge_out,
            logpsi_out,
            e_loc_out,
            o_rows_host,
            nullptr
        );
    }
    const auto destroy_status = qnpeps_e2e_node_destroy(node);
    return status == QNPEPS_E2E_OK ? destroy_status : status;
}
}

extern "C" qnpeps_e2e_status qnpeps_e2e_step_multigpu(
    const QnpepsE2eConfig* cfg,
    const void* device_peps,
    int64_t n_samples,
    const void* terms,
    int gpus,
    int64_t host_tile_bytes,
    double relative_cut,
    double absolute_cut,
    qnpeps_e2e_cbuf* theta_dot_out,
    double* e_mean_out,
    double* e_var_out,
    double* ess_out,
    uint8_t* samples_out,
    double* logq_out,
    double* log_gauge_out,
    double* logpsi_out,
    double* e_loc_out,
    qnpeps_e2e_cbuf* o_rows_host
)
{
    return composed_step(
        cfg,
        device_peps,
        n_samples,
        terms,
        gpus,
        0,
        host_tile_bytes,
        relative_cut,
        absolute_cut,
        theta_dot_out,
        e_mean_out,
        e_var_out,
        ess_out,
        samples_out,
        logq_out,
        log_gauge_out,
        logpsi_out,
        e_loc_out,
        o_rows_host
    );
}

extern "C" qnpeps_e2e_status qnpeps_e2e_step_multigpu_dist(
    const QnpepsE2eConfig* cfg,
    const void* device_peps,
    int64_t n_samples,
    const void* terms,
    int gpus,
    int64_t peer_tile_bytes,
    int64_t host_tile_bytes,
    double relative_cut,
    double absolute_cut,
    qnpeps_e2e_cbuf* theta_dot_out,
    double* e_mean_out,
    double* e_var_out,
    double* ess_out,
    uint8_t* samples_out,
    double* logq_out,
    double* log_gauge_out,
    double* logpsi_out,
    double* e_loc_out,
    qnpeps_e2e_cbuf* o_rows_host,
    QnpepsE2eDistTimings* timings_out
)
{
    if (timings_out)
    {
        if (timings_out->struct_size != sizeof(QnpepsE2eDistTimings))
            return QNPEPS_E2E_ERR_BAD_VERSION;
        const u32 struct_size{timings_out->struct_size};
        std::memset(timings_out, 0, sizeof(QnpepsE2eDistTimings));
        timings_out->struct_size = struct_size;
    }
    return composed_step(
        cfg,
        device_peps,
        n_samples,
        terms,
        gpus,
        peer_tile_bytes,
        host_tile_bytes,
        relative_cut,
        absolute_cut,
        theta_dot_out,
        e_mean_out,
        e_var_out,
        ess_out,
        samples_out,
        logq_out,
        log_gauge_out,
        logpsi_out,
        e_loc_out,
        o_rows_host
    );
}

extern "C" qnpeps_e2e_status qnpeps_e2e_step_multigpu_scratch_bytes(
    const QnpepsE2eConfig* cfg,
    int64_t n_samples,
    const void* terms,
    int gpus,
    int64_t host_tile_bytes,
    uint64_t* out_bytes
)
{
    const qnpeps_e2e_status status{qn_e2e::mg::_config_check(cfg)};
    if (status != QNPEPS_E2E_OK) return status;
    if (not out_bytes) return QNPEPS_E2E_ERR_NULL_ARG;
    if (gpus < 1) return QNPEPS_E2E_ERR_BAD_CONFIG;
    (void) n_samples;
    (void) terms;
    (void) host_tile_bytes;
    const auto reservation = qnpeps::arena_reservation_bytes();
    if (err_state() != QNPEPS_E2E_OK) return err_state();
    *out_bytes = static_cast<u64>(reservation);
    return QNPEPS_E2E_OK;
}

extern "C" qnpeps_e2e_status qnpeps_e2e_step_multigpu_dist_scratch_bytes(
    const QnpepsE2eConfig* cfg,
    int64_t n_samples,
    const void* terms,
    int gpus,
    int64_t peer_tile_bytes,
    uint64_t* out_bytes
)
{
    const qnpeps_e2e_status status{qn_e2e::mg::_config_check(cfg)};
    if (status != QNPEPS_E2E_OK) return status;
    if (not out_bytes) return QNPEPS_E2E_ERR_NULL_ARG;
    if (gpus < 1) return QNPEPS_E2E_ERR_BAD_CONFIG;
    (void) n_samples;
    (void) terms;
    (void) peer_tile_bytes;
    const auto reservation = qnpeps::arena_reservation_bytes();
    if (err_state() != QNPEPS_E2E_OK) return err_state();
    *out_bytes = static_cast<u64>(reservation);
    return QNPEPS_E2E_OK;
}
