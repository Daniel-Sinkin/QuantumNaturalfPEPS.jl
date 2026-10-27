#ifndef QNPEPS_E2E_H
#define QNPEPS_E2E_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#    pragma GCC visibility push(default)
#endif

    typedef enum qnpeps_e2e_status
    {
        QNPEPS_E2E_OK = 0,
        QNPEPS_E2E_ERR_NULL_ARG = 1,
        QNPEPS_E2E_ERR_BAD_CONFIG = 2,
        QNPEPS_E2E_ERR_BAD_VERSION = 3,
        QNPEPS_E2E_ERR_CUDA = 4,
        QNPEPS_E2E_ERR_OOM = 5,
        QNPEPS_E2E_ERR_INTERNAL = 6
    } qnpeps_e2e_status;

    typedef struct QnpepsE2eConfig
    {
        uint32_t struct_size;
        int32_t lx;
        int32_t ly;
        int32_t dim_phys;
        int32_t dim_bond;
        int32_t chi_s;
        int32_t chi_dl;
        int32_t chi_eo;
        int32_t meo;
        uint64_t seed;
        int32_t sampling_mode;
        int32_t contract_dim;
        int32_t sample_batch;
        int32_t truncation_route;
        double dlenv_density_cutoff;
        double sampler_density_cutoff;
        double projected_density_cutoff;
        double local_energy_density_cutoff;
        double local_energy_svd_cutoff;
    } QnpepsE2eConfig;

    typedef struct qnpeps_e2e_cbuf qnpeps_e2e_cbuf;
    typedef struct qnpeps_e2e_minsr_ctx qnpeps_e2e_minsr_ctx;
    typedef struct qnpeps_e2e_update_ctx qnpeps_e2e_update_ctx;

    qnpeps_e2e_status qnpeps_e2e_minsr_ctx_create(
        const QnpepsE2eConfig* cfg,
        int64_t n_samples,
        int64_t host_tile_bytes,
        void* stream,
        qnpeps_e2e_minsr_ctx** out
    );

    qnpeps_e2e_status qnpeps_e2e_minsr_ctx_run(
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
    );

    void qnpeps_e2e_minsr_ctx_destroy(qnpeps_e2e_minsr_ctx* ctx);

    qnpeps_e2e_status qnpeps_e2e_minsr(
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
    );

    qnpeps_e2e_status qnpeps_e2e_step(
        const QnpepsE2eConfig* cfg,
        const void* device_peps,
        int64_t n_samples,
        const void* terms,
        int64_t host_tile_bytes,
        double relative_cut,
        double absolute_cut,
        qnpeps_e2e_cbuf* theta_output,
        double* energy_mean_output,
        double* energy_variance_output,
        double* ess_output,
        uint8_t* samples_out,
        double* logq_out,
        double* log_gauge_out,
        double* logpsi_out,
        double* e_loc_out,
        qnpeps_e2e_cbuf* host_rows,
        void* stream
    );

    qnpeps_e2e_status qnpeps_e2e_step_scratch_bytes(
        const QnpepsE2eConfig* cfg,
        int64_t n_samples,
        const void* terms,
        int64_t host_tile_bytes,
        uint64_t* out_bytes
    );

    qnpeps_e2e_status qnpeps_e2e_step_multigpu(
        const QnpepsE2eConfig* cfg,
        const void* device_peps,
        int64_t n_samples,
        const void* terms,
        int gpus,
        int64_t host_tile_bytes,
        double relative_cut,
        double absolute_cut,
        qnpeps_e2e_cbuf* theta_output,
        double* energy_mean_output,
        double* energy_variance_output,
        double* ess_output,
        uint8_t* samples_out,
        double* logq_out,
        double* log_gauge_out,
        double* logpsi_out,
        double* e_loc_out,
        qnpeps_e2e_cbuf* host_rows
    );

    qnpeps_e2e_status qnpeps_e2e_step_multigpu_scratch_bytes(
        const QnpepsE2eConfig* cfg,
        int64_t n_samples,
        const void* terms,
        int gpus,
        int64_t host_tile_bytes,
        uint64_t* out_bytes
    );

    typedef struct QnpepsE2eDistTimings
    {
        uint32_t struct_size;
        double sample_s;
        double eo_s;
        double gram_s;
        double gram_rows_peer_sum_s;
        double gram_rows_peer_max_s;
        double gram_gather_s;
        double solve_s;
        double scatter_s;
        double scatter_acc_peer_s;
        uint64_t gram_rows_peer_bytes;
        uint64_t scatter_acc_peer_bytes;
    } QnpepsE2eDistTimings;

    qnpeps_e2e_status qnpeps_e2e_step_multigpu_dist(
        const QnpepsE2eConfig* cfg,
        const void* device_peps,
        int64_t n_samples,
        const void* terms,
        int gpus,
        int64_t peer_tile_bytes,
        int64_t host_tile_bytes,
        double relative_cut,
        double absolute_cut,
        qnpeps_e2e_cbuf* theta_output,
        double* energy_mean_output,
        double* energy_variance_output,
        double* ess_output,
        uint8_t* samples_out,
        double* logq_out,
        double* log_gauge_out,
        double* logpsi_out,
        double* e_loc_out,
        qnpeps_e2e_cbuf* host_rows,
        QnpepsE2eDistTimings* timings_out
    );

    qnpeps_e2e_status qnpeps_e2e_step_multigpu_dist_scratch_bytes(
        const QnpepsE2eConfig* cfg,
        int64_t n_samples,
        const void* terms,
        int gpus,
        int64_t peer_tile_bytes,
        uint64_t* out_bytes
    );

    typedef struct qnpeps_e2e_node qnpeps_e2e_node;

    typedef enum qnpeps_e2e_update_precision
    {
        QNPEPS_E2E_UPDATE_F32 = 0,
        QNPEPS_E2E_UPDATE_F64_MASTER = 1
    } qnpeps_e2e_update_precision;

    typedef struct QnpepsE2eEulerStepArgs
    {
        uint32_t struct_size;
        int32_t precision;
        int64_t n_samples;
        double relative_cut;
        double absolute_cut;
        double learning_rate;
        void* state_f64_io;
        uint64_t state_f64_bytes;
        qnpeps_e2e_cbuf* peps_f32_io;
        uint64_t peps_f32_bytes;
        qnpeps_e2e_cbuf* theta_output;
        uint64_t theta_dot_bytes;
        double* energy_mean_output;
        double* energy_variance_output;
        double* ess_output;
        uint8_t* samples_out;
        double* logq_out;
        double* log_gauge_out;
        double* logpsi_out;
        double* e_loc_out;
        qnpeps_e2e_cbuf* host_rows;
        int64_t* epoch_out;
    } QnpepsE2eEulerStepArgs;

    typedef struct QnpepsE2eUpdateArgs
    {
        uint32_t struct_size;
        uint32_t reserved;
        void* state_f64_io;
        uint64_t state_f64_bytes;
        const qnpeps_e2e_cbuf* theta_dot;
        uint64_t theta_dot_bytes;
        qnpeps_e2e_cbuf* peps_f32_out;
        uint64_t peps_f32_bytes;
        double learning_rate;
        void* stream;
    } QnpepsE2eUpdateArgs;

    qnpeps_e2e_status qnpeps_e2e_update_ctx_create(
        const QnpepsE2eConfig* cfg, qnpeps_e2e_update_ctx** out
    );

    qnpeps_e2e_status qnpeps_e2e_update_ctx_run(
        qnpeps_e2e_update_ctx* ctx, const QnpepsE2eUpdateArgs* args
    );

    qnpeps_e2e_status qnpeps_e2e_update_ctx_destroy(qnpeps_e2e_update_ctx* ctx);

    qnpeps_e2e_status qnpeps_e2e_node_create(
        const QnpepsE2eConfig* cfg,
        int gpus,
        int64_t ns_capacity,
        int64_t ns_ahead,
        int64_t dim_batch,
        int64_t host_tile_bytes,
        const void* terms,
        qnpeps_e2e_node** node_out
    );

    qnpeps_e2e_status qnpeps_e2e_node_submit_theta(qnpeps_e2e_node* node, const void* device_peps);

    qnpeps_e2e_status qnpeps_e2e_node_step(
        qnpeps_e2e_node* node,
        int64_t ns,
        double relative_cut,
        double absolute_cut,
        qnpeps_e2e_cbuf* theta_output,
        double* energy_mean_output,
        double* energy_variance_output,
        double* ess_output,
        uint8_t* samples_out,
        double* logq_out,
        double* log_gauge_out,
        double* logpsi_out,
        double* e_loc_out,
        qnpeps_e2e_cbuf* host_rows,
        int64_t* epoch_out
    );

    qnpeps_e2e_status qnpeps_e2e_node_step_euler(
        qnpeps_e2e_node* node, const QnpepsE2eEulerStepArgs* args
    );

    qnpeps_e2e_status qnpeps_e2e_node_destroy(qnpeps_e2e_node* node);

    qnpeps_e2e_status qnpeps_e2e_node_footprint_bytes(
        const QnpepsE2eConfig* cfg,
        int gpus,
        int64_t ns_capacity,
        int64_t ns_ahead,
        int64_t dim_batch,
        const void* terms,
        uint64_t* out_bytes
    );

    qnpeps_e2e_status qnpeps_e2e_dense_count(const QnpepsE2eConfig* cfg, int64_t* out_count);

    qnpeps_e2e_status qnpeps_e2e_compact_count(const QnpepsE2eConfig* cfg, int64_t* out_count);

    qnpeps_e2e_status qnpeps_e2e_minsr_scratch_bytes(
        const QnpepsE2eConfig* cfg, int64_t n_samples, int64_t host_tile_bytes, uint64_t* out_bytes
    );

    const char* qnpeps_e2e_strerror(qnpeps_e2e_status status);
    const char* qnpeps_e2e_version(void);

#ifdef __cplusplus
#    pragma GCC visibility pop
}
#endif

#endif
