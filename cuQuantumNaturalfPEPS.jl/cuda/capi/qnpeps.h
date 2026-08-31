#ifndef QNPEPS_H
#define QNPEPS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#    pragma GCC visibility push(default)
#endif

    typedef enum qnpeps_status
    {
        QNPEPS_OK = 0,
        QNPEPS_ERR_NULL_ARG = 1,
        QNPEPS_ERR_BAD_CONFIG = 2,
        QNPEPS_ERR_BAD_VERSION = 3,
        QNPEPS_ERR_CUDA = 4,
        QNPEPS_ERR_OOM = 5,
        QNPEPS_ERR_INTERNAL = 6
    } qnpeps_status;

    typedef enum qnpeps_sampling_mode
    {
        QNPEPS_SAMPLING_FAST = 0,
        QNPEPS_SAMPLING_FULL = 1
    } qnpeps_sampling_mode;

    typedef enum qnpeps_truncation_route
    {
        QNPEPS_TRUNCATION_DEFAULT = 0,
        QNPEPS_TRUNCATION_DENSITY = 6
    } qnpeps_truncation_route;

    typedef struct QnpepsDensitySettings
    {
        uint32_t struct_size;
        uint32_t precision;
        uint32_t mindim;
        uint32_t reserved;
        double relative_cutoff;
    } QnpepsDensitySettings;

    typedef struct QnpepsDensityRankRecord
    {
        uint32_t struct_size;
        uint32_t flags;
        int64_t matrix_order;
        int64_t natural_cap;
        int64_t applied_cap;
        int64_t rank_after_cap;
        int64_t active_rank;
        int64_t hard_discarded;
        int64_t cutoff_discarded;
        double truncation_error;
        double docut;
        double retained_sum;
        double discarded_weight;
    } QnpepsDensityRankRecord;

    typedef struct QnpepsDeviceBuffer
    {
        uint32_t struct_size;
        uint32_t reserved;
        uint64_t values;
        uint64_t bytes;
    } QnpepsDeviceBuffer;

    typedef struct QnpepsDensitySitePlan
    {
        uint32_t struct_size;
        uint32_t site;
        uint32_t num_sites;
        uint32_t reserved;
        int64_t state_left;
        int64_t physical_input;
        int64_t state_right;
        int64_t operator_left;
        int64_t operator_input;
        int64_t physical_output;
        int64_t operator_right;
        uint64_t state_offset;
        uint64_t operator_offset;
        uint64_t output_offset;
    } QnpepsDensitySitePlan;

    typedef struct QnpepsDensityWorkspace
    {
        uint32_t struct_size;
        uint32_t reserved;
        QnpepsDeviceBuffer left_environments;
        QnpepsDeviceBuffer right_blocks;
        QnpepsDeviceBuffer temporaries;
        QnpepsDeviceBuffer density;
        QnpepsDeviceBuffer eigenvalues;
        QnpepsDeviceBuffer sorted_eigenvalues;
        QnpepsDeviceBuffer sort_indices;
        QnpepsDeviceBuffer active_ranks;
        QnpepsDeviceBuffer truncation_records;
        QnpepsDeviceBuffer solver_workspace;
        QnpepsDeviceBuffer solver_information;
    } QnpepsDensityWorkspace;

    typedef struct QnpepsDensityTrace
    {
        uint32_t struct_size;
        uint32_t enabled;
        QnpepsDeviceBuffer left_environments;
        QnpepsDeviceBuffer right_blocks;
        QnpepsDeviceBuffer conjugate_right_blocks;
        QnpepsDeviceBuffer density_matrices;
        QnpepsDeviceBuffer solver_spectra;
        QnpepsDeviceBuffer sorted_spectra;
        QnpepsDeviceBuffer retained_spectra;
        QnpepsDeviceBuffer right_bases;
        QnpepsDeviceBuffer left_bases;
        QnpepsDeviceBuffer projectors;
        QnpepsDeviceBuffer accumulated_gauges;
        QnpepsDeviceBuffer matrix_orders;
        QnpepsDeviceBuffer combined_inputs;
    } QnpepsDensityTrace;

    typedef struct QnpepsDensityApplyArgs
    {
        uint32_t struct_size;
        uint32_t normalize;
        QnpepsDensitySettings settings;
        int64_t upper_bond;
        uint64_t num_sites;
        uint64_t sites;
        uint64_t sites_bytes;
        double input_gauge;
        QnpepsDensityWorkspace workspace;
        QnpepsDensityTrace trace;
        QnpepsDeviceBuffer state_values;
        QnpepsDeviceBuffer operator_values;
        QnpepsDeviceBuffer result_dimensions;
        QnpepsDeviceBuffer result_values;
        QnpepsDeviceBuffer normalization_log;
        QnpepsDeviceBuffer output_gauge;
    } QnpepsDensityApplyArgs;

    typedef struct QnpepsDensityFilterArgs
    {
        uint32_t struct_size;
        uint32_t reserved;
        QnpepsDensitySettings settings;
        int64_t order;
        int64_t natural_cap;
        int64_t applied_cap;
        QnpepsDeviceBuffer solver_values;
        QnpepsDeviceBuffer sorted_values;
        QnpepsDeviceBuffer retained_values;
        QnpepsDeviceBuffer sorted_indices;
        QnpepsDeviceBuffer active_rank;
        QnpepsDeviceBuffer record;
    } QnpepsDensityFilterArgs;

    typedef struct QnpepsDensityWorkspaceSizes
    {
        uint32_t struct_size;
        uint32_t reserved;
        uint64_t combined_input;
        uint64_t capped_output;
        uint64_t left_environment_bytes;
        uint64_t right_block_bytes;
        uint64_t temporary_bytes;
        uint64_t density_bytes;
        uint64_t eigenvalues_bytes;
        uint64_t sort_keys_bytes;
        uint64_t sort_indices_bytes;
        uint64_t active_ranks_bytes;
        uint64_t truncation_records_bytes;
        uint64_t solver_workspace_bytes;
        uint64_t solver_information_bytes;
        uint64_t arena_bytes;
    } QnpepsDensityWorkspaceSizes;

    typedef struct QnpepsDensityWorkspaceQuery
    {
        uint32_t struct_size;
        uint32_t reserved;
        int64_t num_sites;
        int64_t input_bond;
        int64_t operator_bond;
        int64_t output_dimension;
        int64_t upper_bond;
        int64_t lanes;
        uint64_t sizes_out;
    } QnpepsDensityWorkspaceQuery;

    typedef struct QnpepsConfig
    {
        uint32_t struct_size;
        int32_t lx;
        int32_t ly;
        int32_t dim_phys;
        int32_t dim_bond;
        int32_t chi_s;
        int32_t chi_dl;
        uint64_t seed;
        int32_t sampling_mode;
        int32_t chi_c;
        int32_t dlenv_truncation_route;
        int32_t sampler_truncation_route;
        double dlenv_density_cutoff;
        double sampler_density_cutoff;
        double projected_density_cutoff;
    } QnpepsConfig;

    typedef struct qnpeps_device_peps qnpeps_device_peps;
    typedef struct qnpeps_device_dlenv qnpeps_device_dlenv;

    typedef struct qnpeps_ctx qnpeps_ctx;
    typedef struct qnpeps_zipup_ctx qnpeps_zipup_ctx;
    typedef struct qnpeps_gram_ctx qnpeps_gram_ctx;
    typedef struct qnpeps_minsr_ctx qnpeps_minsr_ctx;

    typedef enum qnpeps_gram_consumer
    {
        QNPEPS_GRAM_CONSUMER_SLAB = 0,
        QNPEPS_GRAM_CONSUMER_CUSTOM = 1,
        QNPEPS_GRAM_CONSUMER_DENSE = 2
    } qnpeps_gram_consumer;

    typedef struct QnpepsSampleArgs
    {
        uint32_t struct_size;
        const qnpeps_device_peps* peps;
        const qnpeps_device_dlenv* dlenv;
        int32_t gpus;
        void* scratch;
        uint64_t scratch_bytes;
        uint8_t* samples_out;
        double* log_prob_config;
        double* log_gauge;
        uint64_t n_samples;
        uint64_t batch_base;
        uint64_t dim_batch;
        void* stream;
    } QnpepsSampleArgs;

    typedef struct QnpepsCtxSampleArgs
    {
        uint32_t struct_size;
        uint8_t* samples_out;
        double* log_prob_config;
        double* log_gauge;
        uint64_t n_samples;
        uint64_t batch_base;
        uint64_t dim_batch;
    } QnpepsCtxSampleArgs;

    typedef struct QnpepsSamplerHostBatchArgs
    {
        uint32_t struct_size;
        uint32_t reserved;
        const qnpeps_device_peps* peps;
        const int32_t* dlenv_dims;
        uint64_t dlenv_dims_count;
        const qnpeps_device_dlenv* dlenv_values;
        void* scratch;
        uint64_t scratch_bytes;
        void* sampling;
        uint64_t sampling_bytes;
        const void* const* dlenv_pointers;
        uint64_t dlenv_pointer_count;
        uint8_t* samples_out;
        double* log_prob_config;
        double* log_gauge;
        uint64_t batch_seed;
        uint64_t batch_id;
        uint64_t dim_batch;
        int32_t peps_layout;
        int32_t reserved2;
    } QnpepsSamplerHostBatchArgs;

    typedef struct QnpepsSamplerHostRefreshArgs
    {
        uint32_t struct_size;
        int32_t peps_layout;
        const qnpeps_device_peps* peps;
        const qnpeps_device_dlenv* dlenv_values;
        void* sampling;
        uint64_t sampling_bytes;
    } QnpepsSamplerHostRefreshArgs;

    typedef struct QnpepsSampleHostArgs
    {
        uint32_t struct_size;
        int32_t gpus;
        const qnpeps_device_peps* peps;
        const qnpeps_device_dlenv* dlenv;
        uint8_t* samples_out;
        double* log_prob_config;
        double* log_gauge;
        uint64_t n_samples;
        uint64_t batch_base;
        uint64_t dim_batch;
        void* stream;
    } QnpepsSampleHostArgs;

    typedef struct QnpepsZipupMpoMpsDesc
    {
        uint32_t struct_size;
        int32_t num_sites;
        int32_t maxdim;
        int32_t reserved;
        const int32_t* mpo_dims;
        const int32_t* mps_dims;
    } QnpepsZipupMpoMpsDesc;

    typedef struct QnpepsZipupMpoMpsArgs
    {
        uint32_t struct_size;
        uint32_t reserved;
        const void* mpo;
        uint64_t mpo_bytes;
        const void* mps;
        uint64_t mps_bytes;
        void* output;
        uint64_t output_bytes;
        double* log_gauge;
        void* stream;
    } QnpepsZipupMpoMpsArgs;

    typedef struct QnpepsZipupPepsRowArgs
    {
        uint32_t struct_size;
        int32_t row;
        const qnpeps_device_peps* peps_row;
        uint64_t peps_row_bytes;
        const int32_t* mps_dims;
        const qnpeps_device_dlenv* mps_values;
        uint64_t mps_bytes;
        int32_t* output_dims;
        qnpeps_device_dlenv* output_values;
        uint64_t output_bytes;
    } QnpepsZipupPepsRowArgs;

    typedef struct QnpepsGramDesc
    {
        uint32_t struct_size;
        int32_t lx;
        int32_t ly;
        int32_t dim_phys;
        int32_t dim_bond;
        int32_t consumer;
        int32_t reserved;
        int64_t n_samples;
    } QnpepsGramDesc;

    typedef struct QnpepsGramArgs
    {
        uint32_t struct_size;
        uint32_t reserved;
        const void* samples;
        uint64_t samples_bytes;
        const void* o_rows;
        uint64_t o_rows_bytes;
        void* gram_out;
        uint64_t gram_out_bytes;
        void* stream;
    } QnpepsGramArgs;

    typedef struct QnpepsGramFootprint
    {
        uint32_t struct_size;
        uint32_t reserved;
        uint64_t context_device_bytes;
        uint64_t geometry_device_bytes;
        uint64_t dense_a_device_bytes;
        uint64_t dense_b_device_bytes;
        uint64_t caller_samples_bytes;
        uint64_t caller_rows_bytes;
        uint64_t caller_gram_bytes;
    } QnpepsGramFootprint;

    typedef struct QnpepsMinsrDesc
    {
        uint32_t struct_size;
        int32_t lx;
        int32_t ly;
        int32_t dim_phys;
        int32_t dim_bond;
        int32_t diagnostics;
        int32_t reserved;
        int64_t n_samples;
        int64_t host_tile_bytes;
    } QnpepsMinsrDesc;

    typedef struct QnpepsMinsrArgs
    {
        uint32_t struct_size;
        uint32_t reserved;
        const void* samples;
        uint64_t samples_bytes;
        const double* logpsi;
        uint64_t logpsi_bytes;
        const double* e_loc;
        uint64_t e_loc_bytes;
        const double* logq;
        uint64_t logq_bytes;
        const void* gram;
        uint64_t gram_bytes;
        const void* o_rows_device;
        const void* o_rows_host;
        uint64_t o_rows_bytes;
        void* theta_dot_out;
        uint64_t theta_dot_out_bytes;
        double relative_cut;
        double absolute_cut;
        double* e_mean_out;
        double* e_var_out;
        double* ess_out;
        void* stream;
    } QnpepsMinsrArgs;

    qnpeps_status qnpeps_ctx_create(const QnpepsConfig* config, void* stream, qnpeps_ctx** out);
    void qnpeps_ctx_destroy(qnpeps_ctx* ctx);
    qnpeps_status qnpeps_ctx_build_dlenv(
        qnpeps_ctx* ctx, const qnpeps_device_peps* peps, double* cumulative_row_logs
    );
    qnpeps_status qnpeps_ctx_copy_dlenv_host(
        const qnpeps_ctx* ctx, void* output, uint64_t output_bytes
    );
    qnpeps_status qnpeps_ctx_sample(qnpeps_ctx* ctx, const QnpepsCtxSampleArgs* args);
    qnpeps_status qnpeps_ctx_sample_host(qnpeps_ctx* ctx, const QnpepsCtxSampleArgs* args);
    qnpeps_status qnpeps_sampler_host_batch(
        qnpeps_ctx* ctx, const QnpepsSamplerHostBatchArgs* args
    );
    qnpeps_status qnpeps_sampler_host_refresh(
        qnpeps_ctx* ctx, const QnpepsSamplerHostRefreshArgs* args
    );
    qnpeps_status qnpeps_sampler_host_upload_pointers(
        qnpeps_ctx* ctx, const void* const* pointers, uint64_t pointer_count
    );

    qnpeps_status qnpeps_build_dlenv(
        const QnpepsConfig* config,
        const qnpeps_device_peps* device_peps,
        qnpeps_device_dlenv* dlenv_out,
        double* cumulative_row_logs,
        void* stream
    );

    qnpeps_status qnpeps_sample(const QnpepsConfig* config, const QnpepsSampleArgs* args);
    qnpeps_status qnpeps_sample_host(const QnpepsConfig* config, const QnpepsSampleHostArgs* args);

    qnpeps_status qnpeps_random_unitary_peps(
        const QnpepsConfig* config,
        qnpeps_device_peps* peps_out,
        uint64_t peps_bytes,
        uint64_t seed,
        double alpha,
        void* stream
    );

    qnpeps_status qnpeps_zipup_ctx_create(
        const QnpepsConfig* config, int maxdim, void* stream, qnpeps_zipup_ctx** out
    );
    void qnpeps_zipup_ctx_destroy(qnpeps_zipup_ctx* context);
    qnpeps_status qnpeps_zipup_ctx_begin(qnpeps_zipup_ctx* context);
    qnpeps_status qnpeps_zipup_ctx_enqueue_peps_row(
        qnpeps_zipup_ctx* context, const QnpepsZipupPepsRowArgs* args
    );
    qnpeps_status qnpeps_zipup_ctx_finish(
        qnpeps_zipup_ctx* context, double* scales, uint64_t count
    );
    int64_t qnpeps_zipup_peps_row_bytes(const QnpepsConfig* config, int maxdim);

    int64_t qnpeps_zipup_mpo_mps_bytes(const QnpepsZipupMpoMpsDesc* descriptor);
    qnpeps_status qnpeps_zipup_mpo_mps(
        const QnpepsZipupMpoMpsDesc* descriptor, const QnpepsZipupMpoMpsArgs* args
    );
    qnpeps_status qnpeps_density_workspace_sizes(
        qnpeps_ctx* ctx, const QnpepsDensityWorkspaceQuery* query
    );
    qnpeps_status qnpeps_densitymatrix_apply(
        qnpeps_ctx* ctx, const QnpepsDensityApplyArgs* args, void* stream
    );
    qnpeps_status qnpeps_density_filter_apply(
        qnpeps_ctx* ctx, const QnpepsDensityFilterArgs* args, void* stream
    );

    int64_t qnpeps_minsr_dense_count(const QnpepsMinsrDesc* descriptor);
    int64_t qnpeps_minsr_compact_count(const QnpepsMinsrDesc* descriptor);
    int64_t qnpeps_minsr_scratch_bytes(const QnpepsMinsrDesc* descriptor);
    qnpeps_status qnpeps_minsr(const QnpepsMinsrDesc* descriptor, const QnpepsMinsrArgs* args);
    qnpeps_status qnpeps_minsr_ctx_create(
        const QnpepsMinsrDesc* descriptor, void* stream, qnpeps_minsr_ctx** out
    );
    qnpeps_status qnpeps_minsr_ctx_run(qnpeps_minsr_ctx* ctx, const QnpepsMinsrArgs* args);
    void qnpeps_minsr_ctx_destroy(qnpeps_minsr_ctx* ctx);

    qnpeps_status qnpeps_gram_ctx_create(
        const QnpepsGramDesc* descriptor, void* stream, qnpeps_gram_ctx** out
    );
    qnpeps_status qnpeps_gram_ctx_run(qnpeps_gram_ctx* ctx, const QnpepsGramArgs* args);
    qnpeps_status qnpeps_gram_ctx_footprint(const qnpeps_gram_ctx* ctx, QnpepsGramFootprint* out);
    void qnpeps_gram_ctx_destroy(qnpeps_gram_ctx* ctx);

    qnpeps_status qnpeps_batched_rangefinder(
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
    );
    int64_t qnpeps_batched_rangefinder_scratch_bytes(int rows, int cols, int rank, int batch);

    int64_t qnpeps_peps_bytes(const QnpepsConfig* config);
    int64_t qnpeps_dlenv_bytes(const QnpepsConfig* config);
    int64_t qnpeps_sample_bytes(const QnpepsConfig* config, uint64_t count);
    int64_t qnpeps_sample_footprint_bytes(
        const QnpepsConfig* config, uint64_t count, uint64_t dim_batch
    );

    int64_t qnpeps_sample_scratch_bytes(const QnpepsConfig* config, uint64_t dim_batch);

    void qnpeps_sampler_pool_release(void);
    const char* qnpeps_strerror(qnpeps_status status);
    const char* qnpeps_last_error_file(void);
    int32_t qnpeps_last_error_line(void);
    const char* qnpeps_last_error_message(void);
    const char* qnpeps_capi_version(void);

#ifdef __cplusplus
#    pragma GCC visibility pop
}
#endif

#endif
