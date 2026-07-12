#include "capi/qnpeps.h"
#include "cuda_utils.cuh"
#include "dlenv/build.cuh"
#include "linalg.cuh"
#include "qnpeps_ctx.cuh"
#include "sampler/draw.cuh"

#include <cstdint>
#include <new>
#include <random>
#include <vector>

using namespace qnpeps;

namespace qnpeps::dlenv
{
auto dl_free(qnpeps_ctx& ctx) -> void;
}

namespace
{
auto check_cfg(const QnpepsConfig* config) -> qnpeps_status
{
    if (not config) return QNPEPS_ERR_NULL_ARG;
    if (config->struct_size != sizeof(QnpepsConfig)) return QNPEPS_ERR_BAD_VERSION;
    if (config->dim_phys < 1 or config->lx < 2 or config->ly < 2 or config->dim_bond < 1
        or config->chi_s < 1 or config->chi_dl < 1)
    {
        return QNPEPS_ERR_BAD_CONFIG;
    }
    return QNPEPS_OK;
}
}

extern "C" qnpeps_status
qnpeps_ctx_create(const QnpepsConfig* config, void* stream, qnpeps_ctx** out)
{
    if (not out) return QNPEPS_ERR_NULL_ARG;
    *out = nullptr;
    const auto config_status = check_cfg(config);
    if (config_status != QNPEPS_OK) return config_status;

    qnpeps::err_state() = QNPEPS_OK;
    auto* ctx = new (std::nothrow) qnpeps_ctx{};
    if (not ctx) return QNPEPS_ERR_OOM;
    ctx->cfg = *config;
    if (stream)
    {
        ctx->stream = static_cast<cudaStream_t>(stream);
        ctx->stream_owned = false;
    }
    else
    {
        CUDA_CHECK(cudaStreamCreate(&ctx->stream));
        ctx->stream_owned = true;
    }
    ctx->linalg.create(ctx->stream);
    if (qnpeps::err_state() != QNPEPS_OK)
    {
        ctx->linalg.destroy();
        if (ctx->stream_owned and ctx->stream) CUDA_NOCHECK(cudaStreamDestroy(ctx->stream));
        delete ctx;
        return qnpeps::err_state();
    }
    *out = ctx;
    return QNPEPS_OK;
}

extern "C" void qnpeps_ctx_destroy(qnpeps_ctx* ctx)
{
    if (not ctx) return;
    if (ctx->stream) CUDA_NOCHECK(cudaStreamSynchronize(ctx->stream));
    qnpeps::sampler::ctx_sampler_free(*ctx);
    qnpeps::dlenv::dl_free(*ctx);
    ctx->linalg.destroy();
    if (ctx->stream_owned and ctx->stream) CUDA_NOCHECK(cudaStreamDestroy(ctx->stream));
    delete ctx;
}

extern "C" qnpeps_status
qnpeps_ctx_build_dlenv(qnpeps_ctx* ctx, const qnpeps_device_peps* peps, double* cumulative_row_logs)
{
    if (not ctx or not peps) return QNPEPS_ERR_NULL_ARG;
    qnpeps::err_state() = QNPEPS_OK;
    qnpeps::dlenv::build_dlenv(*ctx, peps, cumulative_row_logs);
    return qnpeps::err_state();
}

extern "C" qnpeps_status qnpeps_ctx_sample(qnpeps_ctx* ctx, const QnpepsCtxSampleArgs* args)
{
    if (not ctx or not args) return QNPEPS_ERR_NULL_ARG;
    if (args->struct_size != sizeof(QnpepsCtxSampleArgs)) return QNPEPS_ERR_BAD_VERSION;
    if (not args->samples_out) return QNPEPS_ERR_NULL_ARG;
    qnpeps::err_state() = QNPEPS_OK;
    return qnpeps::sampler::ctx_sample(
        *ctx,
        {
            .output = args->samples_out,
            .logpc_out = args->log_prob_config,
            .lognorm_out = args->log_gauge,
            .n_samples = args->n_samples,
            .batch_base = args->batch_base,
        }
    );
}

extern "C" qnpeps_status qnpeps_build_dlenv(
    const QnpepsConfig* config,
    const qnpeps_device_peps* device_peps,
    qnpeps_device_dlenv* dlenv_out,
    double* cumulative_row_logs,
    void* stream
)
{
    const auto config_status = check_cfg(config);
    if (config_status != QNPEPS_OK) return config_status;
    if (not device_peps or not dlenv_out) return QNPEPS_ERR_NULL_ARG;

    qnpeps_ctx* ctx{};
    const auto create_status = qnpeps_ctx_create(config, stream, &ctx);
    if (create_status != QNPEPS_OK) return create_status;

    const auto build_status = qnpeps_ctx_build_dlenv(ctx, device_peps, cumulative_row_logs);
    if (build_status == QNPEPS_OK)
    {
        const auto dlenv_bytes = qnpeps_dlenv_bytes(config);
        CUDA_CHECK(cudaMemcpy(
            dlenv_out,
            ctx->dlenv.buf[ctx->dlenv.active],
            static_cast<usize>(dlenv_bytes),
            cudaMemcpyDeviceToDevice
        ));
    }
    const auto final_status = qnpeps::err_state();
    qnpeps_ctx_destroy(ctx);
    return final_status;
}

extern "C" qnpeps_status qnpeps_sample(const QnpepsConfig* config, const QnpepsSampleArgs* args)
{
    if (not config or not args) return QNPEPS_ERR_NULL_ARG;
    if (args->struct_size != sizeof(QnpepsSampleArgs)) return QNPEPS_ERR_BAD_VERSION;
    if (not args->peps or not args->dlenv or not args->samples_out) return QNPEPS_ERR_NULL_ARG;

    qnpeps::err_state() = QNPEPS_OK;

    const auto config_status = check_cfg(config);
    if (config_status != QNPEPS_OK) return config_status;

    if (args->gpus > 1)
    {
        if (args->scratch) return QNPEPS_ERR_BAD_CONFIG;
        if (args->n_samples == 0) return QNPEPS_OK;
        qnpeps::sampler::sample_multigpu(
            *config,
            {
                .device_peps = args->peps,
                .device_dlenv = args->dlenv,
                .gpus = args->gpus,
                .output = args->samples_out,
                .logpc_out = args->log_prob_config,
                .lognorm_out = args->log_gauge,
                .n_samples = args->n_samples,
            }
        );
        return qnpeps::err_state();
    }

    if (args->dim_batch > static_cast<uint64_t>(k_max_batch_size)) return QNPEPS_ERR_BAD_CONFIG;
    if (args->n_samples == 0) return QNPEPS_OK;
    qnpeps::sampler::sample(
        *config,
        {
            .device_peps = args->peps,
            .device_dlenv = args->dlenv,
            .scratch = args->scratch,
            .scratch_bytes = args->scratch_bytes,
            .output = args->samples_out,
            .logpc_out = args->log_prob_config,
            .lognorm_out = args->log_gauge,
            .n_samples = args->n_samples,
            .batch_base = args->batch_base,
            .dim_batch_pin = args->dim_batch,
            .stream = args->stream,
        }
    );
    return qnpeps::err_state();
}

extern "C" qnpeps_status qnpeps_double_layer_row(
    const QnpepsConfig* config,
    int row,
    int maxdim,
    const qnpeps_device_peps* device_peps_row,
    const qnpeps_device_dlenv* device_env_below,
    qnpeps_device_dlenv* dlenv_row_out,
    double* row_log_out,
    void* stream
)
{
    const auto config_status = check_cfg(config);
    if (config_status != QNPEPS_OK) return config_status;
    if (not device_peps_row or not dlenv_row_out) return QNPEPS_ERR_NULL_ARG;
    if (row < 1 or row > config->lx) return QNPEPS_ERR_BAD_CONFIG;
    if (maxdim < 1) return QNPEPS_ERR_BAD_CONFIG;

    qnpeps::err_state() = QNPEPS_OK;
    Linalg linalg{};
    linalg.create(static_cast<cudaStream_t>(stream));
    qnpeps::dlenv::build_dlenv_row(
        linalg, *config, row, maxdim, device_peps_row, device_env_below, dlenv_row_out, row_log_out
    );
    CUDA_CHECK(cudaStreamSynchronize(linalg.stream()));
    linalg.destroy();
    return qnpeps::err_state();
}

extern "C" int64_t qnpeps_dlenv_row_bytes(const QnpepsConfig* config, int maxdim)
{
    if (check_cfg(config) != QNPEPS_OK) return -1;
    if (maxdim < 1) return -1;
    const i64 ly{config->ly};
    const i64 dim_bond{config->dim_bond};
    const i64 maxdim_i64{maxdim};
    const auto int32_bytes = static_cast<i64>(sizeof(int32_t));
    const auto f32_bytes = static_cast<i64>(sizeof(f32));
    const auto bond_pair = dim_bond * dim_bond;
    const auto capped_dim = maxdim_i64 < bond_pair ? maxdim_i64 : bond_pair;
    const auto header = ly * 4 * int32_bytes;
    const auto values = ly * capped_dim * capped_dim * bond_pair;
    return header + values * 2 * f32_bytes;
}

extern "C" int64_t qnpeps_batched_rangefinder_scratch_bytes(int rows, int cols, int rank, int batch)
{
    if (rows < 1 or cols < 1 or rank < 1 or batch < 1) return -1;
    if (rank > rows or rank > cols) return -1;
    const auto rows_u = static_cast<usize>(rows);
    const auto cols_u = static_cast<usize>(cols);
    const auto rank_u = static_cast<usize>(rank);
    const auto batch_u = static_cast<usize>(batch);
    const auto cf_bytes = sizeof(cf);
    const auto ptr_bytes = sizeof(cf*);
    const auto int_bytes = sizeof(int);
    usize total{};
    total += device_align(cf_bytes * rows_u * rank_u * batch_u);
    total += device_align(cf_bytes * cols_u * rank_u * batch_u);
    total += device_align(cf_bytes * rank_u * rank_u * batch_u);
    total += device_align(cf_bytes * cols_u * rank_u);
    total += device_align(ptr_bytes * batch_u);
    total += device_align(ptr_bytes * batch_u);
    total += device_align(int_bytes * batch_u);
    return static_cast<int64_t>(total);
}

extern "C" qnpeps_status qnpeps_batched_rangefinder(
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
)
{
    if (not input or not q_out or not r_out or not scratch) return QNPEPS_ERR_NULL_ARG;
    if (rows < 1 or cols < 1 or rank < 1 or batch < 1) return QNPEPS_ERR_BAD_CONFIG;
    if (rank > rows or rank > cols) return QNPEPS_ERR_BAD_CONFIG;

    const i64 rows_i{rows};
    const i64 cols_i{cols};
    const i64 rank_i{rank};
    const i64 batch_i{batch};
    if (input_stride < rows_i * cols_i or q_stride < rows_i * rank_i or r_stride < rank_i * cols_i)
    {
        return QNPEPS_ERR_BAD_CONFIG;
    }

    const auto needed = qnpeps_batched_rangefinder_scratch_bytes(rows, cols, rank, batch);
    if (needed < 0) return QNPEPS_ERR_BAD_CONFIG;
    if (scratch_bytes < static_cast<uint64_t>(needed)) return QNPEPS_ERR_OOM;

    qnpeps::err_state() = QNPEPS_OK;

    auto* cursor = static_cast<char*>(scratch);
    auto carve = [&cursor](i64 count, usize elem_size) -> void*
    {
        void* slot{cursor};
        cursor += device_align(elem_size * static_cast<usize>(count));
        return slot;
    };
    auto carve_cf = [&carve](i64 count) -> cf*
    { return static_cast<cf*>(carve(count, sizeof(cf))); };
    auto carve_cfp = [&carve](i64 count) -> cf**
    { return static_cast<cf**>(carve(count, sizeof(cf*))); };
    auto* sketch_p = carve_cf(rows_i * rank_i * batch_i);
    auto* proj_p = carve_cf(cols_i * rank_i * batch_i);
    auto* gram_p = carve_cf(rank_i * rank_i * batch_i);
    auto* omega_p = carve_cf(cols_i * rank_i);
    auto* gram_ptrs = carve_cfp(batch_i);
    auto* sketch_ptrs = carve_cfp(batch_i);
    auto* info = static_cast<int*>(carve(batch_i, sizeof(int)));

    std::mt19937_64 rng(seed ^ (static_cast<uint64_t>(cols) << 20));
    std::normal_distribution<f32> gauss(0.0f, 1.0f);
    std::vector<cf> omega_host{};
    omega_host.resize(static_cast<usize>(cols_i * rank_i));
    for (auto& value : omega_host)
        value = cf{gauss(rng), gauss(rng)};
    CUDA_CHECK(cudaMemcpy(
        omega_p, omega_host.data(), omega_host.size() * sizeof(cf), cudaMemcpyHostToDevice
    ));

    const auto batch_u = static_cast<usize>(batch_i);
    std::vector<cf*> gram_ptr_host{};
    gram_ptr_host.resize(batch_u);
    std::vector<cf*> sketch_ptr_host{};
    sketch_ptr_host.resize(batch_u);
    for (auto lane = 0_i64; lane < batch_i; ++lane)
    {
        const auto lane_u = static_cast<usize>(lane);
        gram_ptr_host[lane_u] = gram_p + lane * rank_i * rank_i;
        sketch_ptr_host[lane_u] = sketch_p + lane * rows_i * rank_i;
    }
    CUDA_CHECK(cudaMemcpy(
        gram_ptrs, gram_ptr_host.data(), gram_ptr_host.size() * sizeof(cf*), cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        sketch_ptrs,
        sketch_ptr_host.data(),
        sketch_ptr_host.size() * sizeof(cf*),
        cudaMemcpyHostToDevice
    ));
    if (qnpeps::err_state() != QNPEPS_OK) return qnpeps::err_state();

    Linalg linalg{};
    linalg.create(static_cast<cudaStream_t>(stream));
    batched_rangefinder(
        linalg,
        {
            .input = CuMatrixConstBatched{static_cast<const cf*>(input), input_stride, rows, cols},
            .k = rank,
            .omega = omega_p,
            .q_out = CuMatrixBatched{static_cast<cf*>(q_out), q_stride, rows, rank},
            .r_out = CuMatrixBatched{static_cast<cf*>(r_out), r_stride, rank, cols},
            .dim_batch = batch,
            .sketch = CuArray{sketch_p, rows_i * rank_i},
            .proj = CuArray{proj_p, cols_i * rank_i},
            .gram = CuArray{gram_p, rank_i * rank_i},
            .gram_ptrs = gram_ptrs,
            .sketch_ptrs = sketch_ptrs,
            .info = info,
            .fail_flag = nullptr,
        }
    );
    CUDA_CHECK(cudaStreamSynchronize(linalg.stream()));
    linalg.destroy();
    return qnpeps::err_state();
}

extern "C" int64_t qnpeps_peps_bytes(const QnpepsConfig* config)
{
    if (check_cfg(config) != QNPEPS_OK) return -1;
    const i64 dim_phys{config->dim_phys};
    const i64 dim_bond{config->dim_bond};
    const i64 lx{config->lx};
    const i64 ly{config->ly};
    const i64 ly_chain = ly <= 1 ? ly : 2 * dim_bond + (ly - 2) * dim_bond * dim_bond;
    const i64 lx_chain = lx <= 1 ? lx : 2 * dim_bond + (lx - 2) * dim_bond * dim_bond;
    const auto f32_bytes = static_cast<i64>(sizeof(f32));
    const auto params = dim_phys * ly_chain * lx_chain;
    return params * 2 * f32_bytes;
}

extern "C" int64_t qnpeps_sample_bytes(const QnpepsConfig* config, uint64_t count)
{
    if (check_cfg(config) != QNPEPS_OK) return -1;
    const i64 lx{config->lx};
    const i64 ly{config->ly};
    const auto count_i64 = static_cast<i64>(count);
    const auto u8_bytes = static_cast<i64>(sizeof(uint8_t));
    return count_i64 * lx * ly * u8_bytes;
}

extern "C" int64_t qnpeps_dlenv_bytes(const QnpepsConfig* config)
{
    if (check_cfg(config) != QNPEPS_OK) return -1;
    const i64 lx{config->lx};
    const i64 ly{config->ly};
    const i64 dim_bond{config->dim_bond};
    const i64 chi_dl{config->chi_dl};
    const auto int32_bytes = static_cast<i64>(sizeof(int32_t));
    const auto f32_bytes = static_cast<i64>(sizeof(f32));
    const auto rows_below = lx - 1;
    const auto sites = rows_below * ly;
    const auto bond_pair = dim_bond * dim_bond;
    const auto chi_c = chi_dl < bond_pair ? chi_dl : bond_pair;
    const auto header = sites * 4 * int32_bytes;
    const auto values = sites * chi_c * chi_c * bond_pair;
    return header + values * 2 * f32_bytes;
}

extern "C" int64_t qnpeps_sample_footprint_bytes(const QnpepsConfig* config, uint64_t count)
{
    if (check_cfg(config) != QNPEPS_OK) return -1;
    const auto peps = qnpeps_peps_bytes(config);
    const auto dlenv = qnpeps_dlenv_bytes(config);
    const auto scratch = qnpeps_sample_scratch_bytes(config);
    const auto samples = qnpeps_sample_bytes(config, count);
    const auto count_i64 = static_cast<i64>(count);
    const auto f64_bytes = static_cast<i64>(sizeof(f64));
    const auto log_prob_config = count_i64 * f64_bytes;
    const auto log_norm = count_i64 * f64_bytes;
    return peps + dlenv + scratch + samples + log_prob_config + log_norm;
}

extern "C" int64_t qnpeps_sample_scratch_bytes(const QnpepsConfig* config)
{
    if (check_cfg(config) != QNPEPS_OK) return -1;
    return qnpeps::sampler::sample_arena_bytes(*config, k_max_batch_size);
}

extern "C" void qnpeps_sampler_pool_release(void) {}

extern "C" const char* qnpeps_strerror(qnpeps_status status)
{
    switch (status)
    {
        case QNPEPS_OK:
            return "ok";
        case QNPEPS_ERR_NULL_ARG:
            return "a required pointer was NULL";
        case QNPEPS_ERR_BAD_CONFIG:
            return "descriptor failed validation "
                   "(dim_phys<1/lx<2/ly<2/dim_bond<1/chi_s<1/chi_dl<1) "
                   "or dlenv header inconsistent with the config";
        case QNPEPS_ERR_BAD_VERSION:
            return "descriptor struct_size not recognized";
        case QNPEPS_ERR_CUDA:
            return "an underlying CUDA/cuBLAS/cuSOLVER call failed";
        case QNPEPS_ERR_OOM:
            return "device allocation failed";
        case QNPEPS_ERR_INTERNAL:
            return "internal invariant violated";
    }
    return "unknown status";
}

extern "C" const char* qnpeps_capi_version(void)
{
    return "cuQuantumNaturalfPEPS 0.1 (2026-07-09)";
}
