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
    } QnpepsConfig;

    typedef struct qnpeps_device_peps qnpeps_device_peps;
    typedef struct qnpeps_device_dlenv qnpeps_device_dlenv;

    typedef struct qnpeps_ctx qnpeps_ctx;
    typedef struct qnpeps_zipup_ctx qnpeps_zipup_ctx;

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

    qnpeps_status qnpeps_build_dlenv(
        const QnpepsConfig* config,
        const qnpeps_device_peps* device_peps,
        qnpeps_device_dlenv* dlenv_out,
        double* cumulative_row_logs,
        void* stream
    );

    qnpeps_status qnpeps_sample(const QnpepsConfig* config, const QnpepsSampleArgs* args);
    qnpeps_status qnpeps_sample_host(
        const QnpepsConfig* config, const QnpepsSampleHostArgs* args
    );

    qnpeps_status qnpeps_zipup_ctx_create(
        const QnpepsConfig* config,
        int maxdim,
        void* stream,
        qnpeps_zipup_ctx** out
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
