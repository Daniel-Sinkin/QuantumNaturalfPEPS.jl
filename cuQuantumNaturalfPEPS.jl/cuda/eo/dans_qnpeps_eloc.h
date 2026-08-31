#ifndef QNPEPS_ELOC_H
#define QNPEPS_ELOC_H

#include <stdint.h>

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
    } QnpepsElocConfig;

    typedef struct qnpeps_eloc_cbuf qnpeps_eloc_cbuf;
    typedef struct qnpeps_eloc_ctx qnpeps_eloc_ctx;
    typedef struct qnpeps_eloc_gram_ctx qnpeps_eloc_gram_ctx;

    typedef enum qnpeps_eloc_ctx_flags
    {
        QNPEPS_ELOC_CTX_ENERGY = 0,
        QNPEPS_ELOC_CTX_O_ROWS = 1,
        QNPEPS_ELOC_CTX_GRAM = 2
    } qnpeps_eloc_ctx_flags;

    typedef enum qnpeps_eloc_j2_mode
    {
        QNPEPS_ELOC_J2_EXACT = 0,
        QNPEPS_ELOC_J2_HALF_ROW_PAIRS = 1,
        QNPEPS_ELOC_J2_HALF_COLUMN_PAIRS = 2
    } qnpeps_eloc_j2_mode;

    typedef enum qnpeps_eloc_j2_draw
    {
        QNPEPS_ELOC_J2_BALANCED = 0,
        QNPEPS_ELOC_J2_FORCE_GROUP_0 = 1,
        QNPEPS_ELOC_J2_FORCE_GROUP_1 = 2,
        QNPEPS_ELOC_J2_FORCE_ALL = 3
    } qnpeps_eloc_j2_draw;

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
        uint32_t j2_mode;
        uint32_t j2_draw;
        uint64_t j2_seed;
        uint64_t j2_epoch;
    } QnpepsElocCtxRunArgs;

    typedef struct QnpepsElocCtxStats
    {
        uint32_t struct_size;
        uint32_t graph_enabled;
        uint64_t runs;
        uint64_t bindings;
        uint64_t graph_captures;
        uint64_t graph_replays;
        uint64_t graph_capture_failures;
        uint64_t graph_nodes;
        uint64_t graph_edges;
        uint64_t graph_introspection_failures;
        uint32_t j2_mode_last;
        uint32_t j2_draw_last;
        uint64_t j2_seed_last;
        uint64_t j2_epoch_last;
        uint64_t j2_waves;
        uint64_t j2_group0_waves;
        uint64_t j2_group1_waves;
        uint64_t j2_row_groups_total;
        uint64_t j2_row_groups_retained;
        uint64_t j2_column_groups_total;
        uint64_t j2_column_groups_retained;
        uint64_t j2_diag_terms_total;
        uint64_t j2_diag_terms_retained;
        uint64_t j2_flip_terms_total;
        uint64_t j2_flip_terms_retained;
    } QnpepsElocCtxStats;

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

    qnpeps_eloc_status qnpeps_eloc_chains(
        const QnpepsElocConfig* cfg,
        int64_t n_chains,
        const qnpeps_eloc_cbuf* ma,
        const qnpeps_eloc_cbuf* mb,
        const qnpeps_eloc_cbuf* vin,
        const qnpeps_eloc_cbuf* vend,
        qnpeps_eloc_cbuf* out,
        void* stream
    );

    qnpeps_eloc_status qnpeps_eloc_build_o(
        const QnpepsElocConfig* cfg,
        int64_t n,
        int32_t slice_dim,
        const qnpeps_eloc_cbuf* env,
        const qnpeps_eloc_cbuf* slice_in,
        const qnpeps_eloc_cbuf* gscale,
        qnpeps_eloc_cbuf* out,
        void* stream
    );

    qnpeps_eloc_status qnpeps_eloc_gram(
        const QnpepsElocConfig* cfg,
        int32_t ns,
        int32_t compact_np,
        int32_t n_blocks,
        const qnpeps_eloc_cbuf* compact_rows,
        const int32_t* spins,
        const int32_t* block_offset,
        const int32_t* block_slice,
        qnpeps_eloc_cbuf* out,
        void* stream
    );

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

    qnpeps_eloc_status qnpeps_eloc_ctx_stats(const qnpeps_eloc_ctx* ctx, QnpepsElocCtxStats* out);

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

    qnpeps_eloc_status qnpeps_eloc_gram_ctx_create(
        const QnpepsElocConfig* cfg,
        const uint8_t* device_samples,
        int64_t n_samples,
        void* stream,
        qnpeps_eloc_gram_ctx** out
    );
    void qnpeps_eloc_gram_ctx_destroy(qnpeps_eloc_gram_ctx* ctx);
    qnpeps_eloc_status qnpeps_eloc_gram_ctx_launch(
        qnpeps_eloc_gram_ctx* ctx,
        const qnpeps_eloc_cbuf* rows_a,
        int64_t sample_base_a,
        int64_t ns_a,
        const qnpeps_eloc_cbuf* rows_b,
        int64_t sample_base_b,
        int64_t ns_b,
        qnpeps_eloc_cbuf* out,
        int64_t out_ld,
        void* stream
    );

    qnpeps_eloc_status qnpeps_eloc_run_scratch_bytes(
        const QnpepsElocConfig* cfg,
        int64_t n_samples,
        const QnpepsElocTermTable* terms,
        uint64_t* out_bytes
    );

    const char* qnpeps_eloc_strerror(qnpeps_eloc_status status);
    const char* qnpeps_eloc_version(void);

#ifdef __cplusplus
#    pragma GCC visibility pop
}
#endif

#endif
