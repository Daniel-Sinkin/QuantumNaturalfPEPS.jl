#ifndef QNPEPS_ELOC_H
#define QNPEPS_ELOC_H

#include <stdint.h>

#define QNPEPS_ELOC_DIAG_ONE_SITE_Z (-1)
#define QNPEPS_ELOC_CAP_ONE_SITE_Z UINT64_C(1)

#ifdef __cplusplus
extern "C"
{
#    pragma GCC visibility push(default)
#endif

    typedef enum qnpeps_eloc_status
    {
        QNPEPS_ELOC_OK = 0,
        QNPEPS_ELOC_ERR_NULL_ARG = 1,
        QNPEPS_ELOC_ERR_BAD_CONFIG = 2,
        QNPEPS_ELOC_ERR_BAD_VERSION = 3,
        QNPEPS_ELOC_ERR_CUDA = 4,
        QNPEPS_ELOC_ERR_OOM = 5,
        QNPEPS_ELOC_ERR_INTERNAL = 6
    } qnpeps_eloc_status;

    typedef struct QnpepsElocConfig
    {
        uint32_t struct_size;
        int32_t lx;
        int32_t ly;
        int32_t dim_phys;
        int32_t dim_bond;
        int32_t chi_eo;
        int32_t meo;
        int32_t truncation_route;
        double density_cutoff;
        double svd_cutoff;
    } QnpepsElocConfig;

    typedef struct qnpeps_eloc_cbuf qnpeps_eloc_cbuf;
    typedef struct qnpeps_eloc_ctx qnpeps_eloc_ctx;

    typedef enum qnpeps_eloc_ctx_flags
    {
        QNPEPS_ELOC_CTX_ENERGY = 0,
        QNPEPS_ELOC_CTX_O_ROWS = 1,
        QNPEPS_ELOC_CTX_GRAM = 2
    } qnpeps_eloc_ctx_flags;

    typedef struct QnpepsElocCtxRunArgs
    {
        uint32_t struct_size;
        const qnpeps_eloc_cbuf* device_peps;
        const uint8_t* device_samples;
        double* logpsi_out;
        double* e_loc_out;
        qnpeps_eloc_cbuf* o_rows_dev;
        qnpeps_eloc_cbuf* o_rows_host;
        qnpeps_eloc_cbuf* T_dev;
        double lambda;
    } QnpepsElocCtxRunArgs;

    typedef struct QnpepsElocDiagBond
    {
        int32_t site_a;
        int32_t site_b;
        double coeff;
    } QnpepsElocDiagBond;

    typedef struct QnpepsElocFlipTerm
    {
        int32_t n_flips;
        int32_t flip_site[4];
        int32_t flip_value[4];
        int32_t mask_a;
        int32_t mask_b;
        double coeff_re;
        double coeff_im;
    } QnpepsElocFlipTerm;

    typedef struct QnpepsElocTermTable
    {
        int32_t n_diag;
        const QnpepsElocDiagBond* diag;
        int32_t n_flip;
        const QnpepsElocFlipTerm* flip;
    } QnpepsElocTermTable;

    qnpeps_eloc_status qnpeps_eloc_logpsi(
        const QnpepsElocConfig* cfg,
        const qnpeps_eloc_cbuf* device_peps,
        const uint8_t* device_samples,
        int64_t n_samples,
        double* logpsi_out,
        void* stream
    );

    qnpeps_eloc_status qnpeps_eloc_run(
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
    );

    qnpeps_eloc_status qnpeps_eloc_ctx_create(
        const QnpepsElocConfig* cfg,
        int64_t n_samples,
        const QnpepsElocTermTable* terms,
        uint32_t flags,
        void* stream,
        qnpeps_eloc_ctx** out
    );

    void qnpeps_eloc_ctx_destroy(qnpeps_eloc_ctx* ctx);

    qnpeps_eloc_status qnpeps_eloc_ctx_run(qnpeps_eloc_ctx* ctx, const QnpepsElocCtxRunArgs* args);

    qnpeps_eloc_status qnpeps_eloc_compact_count(const QnpepsElocConfig* cfg, int64_t* out_count);

    qnpeps_eloc_status qnpeps_eloc_gram_tile(
        const QnpepsElocConfig* cfg,
        const qnpeps_eloc_cbuf* rows_a,
        const uint8_t* samples_a,
        int64_t ns_a,
        const qnpeps_eloc_cbuf* rows_b,
        const uint8_t* samples_b,
        int64_t ns_b,
        qnpeps_eloc_cbuf* tile_out,
        void* stream
    );

    const char* qnpeps_eloc_strerror(qnpeps_eloc_status status);
    const char* qnpeps_eloc_version(void);
    uint64_t qnpeps_eloc_capabilities(void);

#ifdef __cplusplus
#    pragma GCC visibility pop
}
#endif

#endif
