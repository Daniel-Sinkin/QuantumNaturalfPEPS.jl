#ifndef QNPEPS_ELOC_H
#define QNPEPS_ELOC_H

#include "qnpeps_status.h"

#include <stdint.h>

#define QNPEPS_ELOC_TERM_MAX_SITES 4
#define QNPEPS_ELOC_CAP_LOCAL_OPERATORS UINT64_C(2)
#define QNPEPS_ELOC_CAP_PLAQUETTE_OPERATORS UINT64_C(4)
#define QNPEPS_ELOC_CAP_SLAB_OPERATORS UINT64_C(8)

#ifdef __cplusplus
extern "C"
{
#    pragma GCC visibility push(default)
#endif

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

    typedef struct QnpepsElocComplex
    {
        double re;
        double im;
    } QnpepsElocComplex;

    typedef struct QnpepsElocDiagonalTerm
    {
        int64_t value_offset;
        int32_t site_count;
        int32_t state_count;
        int32_t site[QNPEPS_ELOC_TERM_MAX_SITES];
    } QnpepsElocDiagonalTerm;

    typedef struct QnpepsElocOffDiagonalPiece
    {
        int64_t state_offset;
        int64_t weight_offset;
        int32_t site_count;
        int32_t state_count;
        int32_t site[QNPEPS_ELOC_TERM_MAX_SITES];
    } QnpepsElocOffDiagonalPiece;

    typedef struct QnpepsElocTermTable
    {
        uint32_t struct_size;
        int32_t dim_phys;
        int32_t diagonal_count;
        int32_t piece_count;
        const QnpepsElocDiagonalTerm* diagonal;
        const QnpepsElocOffDiagonalPiece* piece;
        int64_t diagonal_value_count;
        const QnpepsElocComplex* diagonal_values;
        int64_t piece_state_count;
        const uint8_t* piece_active;
        const QnpepsElocComplex* piece_coefficients;
        int64_t piece_weight_count;
        const QnpepsElocComplex* piece_weights;
    } QnpepsElocTermTable;

    qnpeps_status qnpeps_eloc_logpsi(
        const QnpepsElocConfig* cfg,
        const qnpeps_eloc_cbuf* device_peps,
        const uint8_t* device_samples,
        int64_t n_samples,
        double* logpsi_out,
        void* stream
    );

    qnpeps_status qnpeps_eloc_run(
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

    qnpeps_status qnpeps_eloc_ctx_create(
        const QnpepsElocConfig* cfg,
        int64_t n_samples,
        const QnpepsElocTermTable* terms,
        uint32_t flags,
        void* stream,
        qnpeps_eloc_ctx** out
    );

    void qnpeps_eloc_ctx_destroy(qnpeps_eloc_ctx* ctx);

    qnpeps_status qnpeps_eloc_ctx_run(qnpeps_eloc_ctx* ctx, const QnpepsElocCtxRunArgs* args);

    qnpeps_status qnpeps_eloc_compact_count(const QnpepsElocConfig* cfg, int64_t* out_count);

    qnpeps_status qnpeps_eloc_gram_tile(
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

    const char* qnpeps_eloc_version(void);
    uint64_t qnpeps_eloc_capabilities(void);

#ifdef __cplusplus
#    pragma GCC visibility pop
}
#endif

#endif
