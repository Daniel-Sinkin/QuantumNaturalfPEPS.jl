#include "capi/qnpeps.h"
#include "cuda_utils.cuh"
#include "dlenv/build.cuh"
#include "linalg.cuh"
#include "peps.cuh"
#include "qnpeps_ctx.cuh"
#include "rangefinder_rng.cuh"
#include "sampler/draw.cuh"

#include <algorithm>
#include <cstdint>
#include <new>
#include <span>
#include <utility>
#include <vector>

#ifndef QNPEPS_C_API_VERSION
#    error "QNPEPS_C_API_VERSION must be provided by CMake"
#endif

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
    if (config->sampling_mode != QNPEPS_SAMPLING_FAST
        and config->sampling_mode != QNPEPS_SAMPLING_FULL)
    {
        return QNPEPS_ERR_BAD_CONFIG;
    }
    if (config->sampling_mode == QNPEPS_SAMPLING_FULL and config->chi_c < 1)
        return QNPEPS_ERR_BAD_CONFIG;
    return QNPEPS_OK;
}
}

extern "C" qnpeps_status
qnpeps_ctx_create(const QnpepsConfig* config, void* stream, qnpeps_ctx** out)
{
    qnpeps::reset_err();
    if (not out) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    *out = nullptr;
    const auto config_status = check_cfg(config);
    if (config_status != QNPEPS_OK) return qnpeps::set_err(config_status);

    cudaStream_t stream_use{};
    bool stream_owned{};
    if (stream)
    {
        stream_use = static_cast<cudaStream_t>(stream);
    }
    else
    {
        const auto stream_status = cudaStreamCreate(&stream_use);
        if (stream_status != cudaSuccess) return qnpeps::set_cuda_err(stream_status);
        stream_owned = true;
    }

    auto linalg = make_linalg(stream_use);
    if (not linalg)
    {
        if (stream_owned) CUDA_NOCHECK(cudaStreamDestroy(stream_use));
        return qnpeps::err_state();
    }

    auto* ctx = new (std::nothrow) qnpeps_ctx{*config, stream_use, stream_owned, std::move(linalg)};
    if (not ctx)
    {
        linalg.reset();
        if (stream_owned) CUDA_NOCHECK(cudaStreamDestroy(stream_use));
        return qnpeps::set_err(QNPEPS_ERR_OOM);
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
    const auto stream = ctx->stream;
    const auto stream_owned = ctx->owns_stream;
    delete ctx;
    if (stream_owned and stream) CUDA_NOCHECK(cudaStreamDestroy(stream));
}

extern "C" qnpeps_status
qnpeps_ctx_build_dlenv(qnpeps_ctx* ctx, const qnpeps_device_peps* peps, double* cumulative_row_logs)
{
    qnpeps::reset_err();
    if (not ctx or not peps) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    qnpeps::dlenv::build_dlenv(*ctx, peps, cumulative_row_logs);
    return qnpeps::err_state();
}

extern "C" qnpeps_status qnpeps_ctx_sample(qnpeps_ctx* ctx, const QnpepsCtxSampleArgs* args)
{
    qnpeps::reset_err();
    if (not ctx or not args) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    if (args->struct_size != sizeof(QnpepsCtxSampleArgs))
        return qnpeps::set_err(QNPEPS_ERR_BAD_VERSION);
    if (not args->samples_out) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
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
    qnpeps::reset_err();
    const auto config_status = check_cfg(config);
    if (config_status != QNPEPS_OK) return qnpeps::set_err(config_status);
    if (not device_peps or not dlenv_out) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);

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
    qnpeps::reset_err();
    if (not config or not args) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    if (args->struct_size != sizeof(QnpepsSampleArgs))
        return qnpeps::set_err(QNPEPS_ERR_BAD_VERSION);
    if (not args->peps or not args->dlenv or not args->samples_out)
        return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);

    const auto config_status = check_cfg(config);
    if (config_status != QNPEPS_OK) return qnpeps::set_err(config_status);
    if (args->dim_batch < 1 or args->dim_batch > static_cast<uint64_t>(k_max_batch_size))
        return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);

    if (args->gpus > 1)
    {
        if (args->scratch) return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
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
                .dim_batch = args->dim_batch,
            }
        );
        return qnpeps::err_state();
    }

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
            .dim_batch = args->dim_batch,
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
    qnpeps::reset_err();
    const auto config_status = check_cfg(config);
    if (config_status != QNPEPS_OK) return qnpeps::set_err(config_status);
    if (not device_peps_row or not dlenv_row_out) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    if (row < 1 or row > config->lx) return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
    if (maxdim < 1) return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);

    auto linalg = make_linalg(static_cast<cudaStream_t>(stream));
    if (not linalg) return qnpeps::err_state();
    qnpeps::dlenv::build_dlenv_row(
        *linalg, *config, row, maxdim, device_peps_row, device_env_below, dlenv_row_out, row_log_out
    );
    CUDA_CHECK(cudaStreamSynchronize(linalg->stream()));
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
    const auto capped_dim = std::min(maxdim_i64, bond_pair);
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
    const auto cf_bytes = sizeof(cuFloatComplex);
    const auto ptr_bytes = sizeof(cuFloatComplex*);
    const auto int_bytes = sizeof(int);
    usize total{};
    total += device_align(cf_bytes * rows_u * rank_u * batch_u);
    total += device_align(cf_bytes * cols_u * rank_u * batch_u);
    total += device_align(cf_bytes * rank_u * rank_u * batch_u);
    total += device_align(cf_bytes * cols_u * rank_u);
    total += device_align(ptr_bytes * batch_u);
    total += device_align(ptr_bytes * batch_u);
    total += device_align(int_bytes * batch_u);
    total += device_align(int_bytes);
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
    qnpeps::reset_err();
    if (not input or not q_out or not r_out or not scratch)
        return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    if (rows < 1 or cols < 1 or rank < 1 or batch < 1)
        return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
    if (rank > rows or rank > cols) return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);

    const auto rows_i64 = static_cast<i64>(rows);
    const auto cols_i64 = static_cast<i64>(cols);
    const auto rank_i64 = static_cast<i64>(rank);
    const auto batch_i64 = static_cast<i64>(batch);
    if (input_stride < rows_i64 * cols_i64 or q_stride < rows_i64 * rank_i64
        or r_stride < rank_i64 * cols_i64)
    {
        return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
    }

    const auto required_scratch_bytes =
        qnpeps_batched_rangefinder_scratch_bytes(rows, cols, rank, batch);
    if (required_scratch_bytes < 0) return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
    if (scratch_bytes < static_cast<uint64_t>(required_scratch_bytes))
        return qnpeps::set_err(QNPEPS_ERR_OOM);

    auto* cursor = static_cast<char*>(scratch);
    const auto carve = [&cursor](i64 count, usize element_size) -> void*
    {
        auto* slot = cursor;
        cursor += device_align(element_size * static_cast<usize>(count));
        return slot;
    };
    const auto carve_complex = [&carve](i64 count) -> cuFloatComplex*
    { return static_cast<cuFloatComplex*>(carve(count, sizeof(cuFloatComplex))); };
    const auto carve_complex_pointers = [&carve](i64 count) -> cuFloatComplex**
    { return static_cast<cuFloatComplex**>(carve(count, sizeof(cuFloatComplex*))); };
    auto* device_sketch = carve_complex(rows_i64 * rank_i64 * batch_i64);
    auto* device_projection = carve_complex(cols_i64 * rank_i64 * batch_i64);
    auto* device_gram = carve_complex(rank_i64 * rank_i64 * batch_i64);
    auto* device_omega = carve_complex(cols_i64 * rank_i64);
    auto* device_gram_pointers = carve_complex_pointers(batch_i64);
    auto* device_sketch_pointers = carve_complex_pointers(batch_i64);
    auto* device_info = static_cast<int*>(carve(batch_i64, sizeof(int)));
    auto* device_fail_flag = static_cast<int*>(carve(1, sizeof(int)));
    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    auto linalg = make_linalg(cuda_stream);
    if (not linalg) return qnpeps::err_state();

    auto rng = RangefinderRng::from_seed_and_width(seed, cols);
    std::vector<cuFloatComplex> host_omega{};
    host_omega.resize(static_cast<usize>(cols_i64 * rank_i64));
    rng.fill_complex_normal(std::span{host_omega});
    copy_h2d_async(device_omega, host_omega.data(), host_omega.size(), cuda_stream);

    const auto batch_size = static_cast<usize>(batch_i64);
    std::vector<cuFloatComplex*> host_gram_pointers{};
    host_gram_pointers.resize(batch_size);
    std::vector<cuFloatComplex*> host_sketch_pointers{};
    host_sketch_pointers.resize(batch_size);
    for (auto lane = 0_i64; lane < batch_i64; ++lane)
    {
        const auto lane_index = static_cast<usize>(lane);
        host_gram_pointers[lane_index] = device_gram + lane * rank_i64 * rank_i64;
        host_sketch_pointers[lane_index] = device_sketch + lane * rows_i64 * rank_i64;
    }
    copy_h2d_async(
        device_gram_pointers, host_gram_pointers.data(), host_gram_pointers.size(), cuda_stream
    );
    copy_h2d_async(
        device_sketch_pointers,
        host_sketch_pointers.data(),
        host_sketch_pointers.size(),
        cuda_stream
    );
    if (qnpeps::err_state() != QNPEPS_OK)
    {
        CUDA_NOCHECK(cudaStreamSynchronize(cuda_stream));
        return qnpeps::err_state();
    }
    CUDA_CHECK(cudaMemsetAsync(device_fail_flag, 0, sizeof(int), linalg->stream()));
    batched_rangefinder(
        *linalg,
        {
            .input =
                CuMatrixConstBatched{
                    static_cast<const cuFloatComplex*>(input), input_stride, rows, cols
                },
            .rank = rank,
            .omega = device_omega,
            .q_out = CuMatrixBatched{static_cast<cuFloatComplex*>(q_out), q_stride, rows, rank},
            .r_out = CuMatrixBatched{static_cast<cuFloatComplex*>(r_out), r_stride, rank, cols},
            .dim_batch = batch,
            .sketch = CuArray{device_sketch, rows_i64 * rank_i64},
            .projection = CuArray{device_projection, cols_i64 * rank_i64},
            .gram = CuArray{device_gram, rank_i64 * rank_i64},
            .gram_ptrs = device_gram_pointers,
            .sketch_ptrs = device_sketch_pointers,
            .info = device_info,
            .fail_flag = device_fail_flag,
        }
    );
    CUDA_CHECK(cudaStreamSynchronize(linalg->stream()));
    int host_fail_flag{};
    CUDA_CHECK(cudaMemcpy(&host_fail_flag, device_fail_flag, sizeof(int), cudaMemcpyDeviceToHost));
    if (host_fail_flag != 0) qnpeps::set_err(QNPEPS_ERR_CUDA);
    return qnpeps::err_state();
}

extern "C" int64_t qnpeps_peps_bytes(const QnpepsConfig* config)
{
    if (check_cfg(config) != QNPEPS_OK) return -1;
    const PepsDims dims{config->lx, config->ly, config->dim_phys, config->dim_bond};
    return peps_elems(dims) * static_cast<i64>(sizeof(cuFloatComplex));
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
    const auto num_sites = rows_below * ly;
    const auto bond_pair = dim_bond * dim_bond;
    const auto chi_c = std::min(chi_dl, bond_pair);
    const auto header = num_sites * 4 * int32_bytes;
    const auto values = num_sites * chi_c * chi_c * bond_pair;
    return header + values * 2 * f32_bytes;
}

extern "C" int64_t
qnpeps_sample_footprint_bytes(const QnpepsConfig* config, uint64_t count, uint64_t dim_batch)
{
    if (check_cfg(config) != QNPEPS_OK or dim_batch < 1
        or dim_batch > static_cast<uint64_t>(k_max_batch_size))
    {
        return -1;
    }
    const auto peps = qnpeps_peps_bytes(config);
    const auto dlenv = qnpeps_dlenv_bytes(config);
    const auto scratch = qnpeps_sample_scratch_bytes(config, dim_batch);
    const auto samples = qnpeps_sample_bytes(config, count);
    const auto count_i64 = static_cast<i64>(count);
    const auto output_scalars = count_i64 * static_cast<i64>(sizeof(f64));
    return peps + dlenv + scratch + samples + 2 * output_scalars;
}

extern "C" int64_t qnpeps_sample_scratch_bytes(const QnpepsConfig* config, uint64_t dim_batch)
{
    if (check_cfg(config) != QNPEPS_OK or dim_batch < 1
        or dim_batch > static_cast<uint64_t>(k_max_batch_size))
    {
        return -1;
    }
    return qnpeps::sampler::sample_arena_bytes(*config, static_cast<int>(dim_batch));
}

extern "C" void qnpeps_sampler_pool_release(void) {}

extern "C" const char* qnpeps_last_error_file(void)
{
    return qnpeps::err_file();
}

extern "C" int32_t qnpeps_last_error_line(void)
{
    return qnpeps::err_line();
}

extern "C" const char* qnpeps_last_error_message(void)
{
    return qnpeps::err_message();
}

extern "C" const char* qnpeps_strerror(qnpeps_status status)
{
    switch (status)
    {
        case QNPEPS_OK:
            return "ok";
        case QNPEPS_ERR_NULL_ARG:
            return "a required pointer was NULL";
        case QNPEPS_ERR_BAD_CONFIG:
            return "descriptor or batch dimensions failed validation, or dlenv header is "
                   "inconsistent with the config";
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
    return "cuQuantumNaturalfPEPS " QNPEPS_C_API_VERSION;
}
