#include "common.cuh"
#include "dans_qnpeps_eloc.h"
#include "eloc_kernels.cuh"
#include "env_build.cuh"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#ifndef QNPEPS_PACKAGE_VERSION
#    error "QNPEPS_PACKAGE_VERSION must be provided by CMake"
#endif

struct qnpeps_eloc_gram_ctx
{
    int device;
    int sites;
    int compact;
    int64_t n_samples;
    int* spins;
    int* block_offset;
    int* block_slice;
};

namespace
{
auto check_cfg(const QnpepsElocConfig* c) -> qnpeps_eloc_status
{
    if (not c) return QNPEPS_ELOC_ERR_NULL_ARG;
    if (c->struct_size != sizeof(QnpepsElocConfig)) return QNPEPS_ELOC_ERR_BAD_VERSION;
    if (c->lx < 2 or c->ly < 2 or c->dim_phys < 1 or c->dim_bond < 1 or c->chi_eo < 1 or c->meo < 1)
        return QNPEPS_ELOC_ERR_BAD_CONFIG;
    if ((c->truncation_route != 0 and c->truncation_route != 6)
        or not std::isfinite(c->density_cutoff) or c->density_cutoff < 0.0)
        return QNPEPS_ELOC_ERR_BAD_CONFIG;
    return QNPEPS_ELOC_OK;
}
auto as_cf(const qnpeps_eloc_cbuf* p) -> const cf*
{
    return reinterpret_cast<const cf*>(p);
}
auto as_cf(qnpeps_eloc_cbuf* p) -> cf*
{
    return reinterpret_cast<cf*>(p);
}

__global__ auto cu_gram_samples_to_i32(int* out, const uint8_t* in, int64_t total) -> void
{
    const auto stride = int64_t{static_cast<int64_t>(gridDim.x) * blockDim.x};
    for (auto i = int64_t{static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x}; i < total;
         i += stride)
        out[i] = static_cast<int>(in[i]);
}

auto gram_bond_dim(int axis_len, int pos, int dim_bond) -> int
{
    return pos <= 0 or pos >= axis_len ? 1 : dim_bond;
}
}

extern "C" qnpeps_eloc_status qnpeps_eloc_chains(
    const QnpepsElocConfig* cfg,
    int64_t n_chains,
    const qnpeps_eloc_cbuf* ma,
    const qnpeps_eloc_cbuf* mb,
    const qnpeps_eloc_cbuf* vin,
    const qnpeps_eloc_cbuf* vend,
    qnpeps_eloc_cbuf* out,
    void* stream
)
{
    const auto v = check_cfg(cfg);
    if (v != QNPEPS_ELOC_OK) return v;
    if (not ma or not mb or not vin or not vend or not out) return QNPEPS_ELOC_ERR_NULL_ARG;
    qn::clear_err();
    const auto s = static_cast<cudaStream_t>(stream);
    qn_eloc::launch_eloc_chains(
        n_chains, cfg->chi_eo, as_cf(ma), as_cf(mb), as_cf(vin), as_cf(vend), as_cf(out), s
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(s));
    return qn::err_state();
}

extern "C" qnpeps_eloc_status qnpeps_eloc_build_o(
    const QnpepsElocConfig* cfg,
    int64_t n,
    int32_t slice_dim,
    const qnpeps_eloc_cbuf* env,
    const qnpeps_eloc_cbuf* slice_in,
    const qnpeps_eloc_cbuf* gscale,
    qnpeps_eloc_cbuf* out,
    void* stream
)
{
    const auto v = check_cfg(cfg);
    if (v != QNPEPS_ELOC_OK) return v;
    if (not env or not slice_in or not gscale or not out) return QNPEPS_ELOC_ERR_NULL_ARG;
    if (slice_dim < 1) return QNPEPS_ELOC_ERR_BAD_CONFIG;
    qn::clear_err();
    const auto s = static_cast<cudaStream_t>(stream);
    qn_eloc::launch_build_o(
        n, slice_dim, as_cf(env), as_cf(slice_in), as_cf(gscale), as_cf(out), s
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(s));
    return qn::err_state();
}

extern "C" qnpeps_eloc_status qnpeps_eloc_gram(
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
)
{
    const auto v = check_cfg(cfg);
    if (v != QNPEPS_ELOC_OK) return v;
    if (not compact_rows or not spins or not block_offset or not block_slice or not out)
        return QNPEPS_ELOC_ERR_NULL_ARG;
    if (ns < 1 or compact_np < 1 or n_blocks < 1) return QNPEPS_ELOC_ERR_BAD_CONFIG;
    qn::clear_err();
    const auto s = static_cast<cudaStream_t>(stream);
    qn_eloc::launch_gram(
        ns,
        compact_np,
        n_blocks,
        as_cf(compact_rows),
        reinterpret_cast<const int*>(spins),
        reinterpret_cast<const int*>(block_offset),
        reinterpret_cast<const int*>(block_slice),
        as_cf(out),
        s
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(s));
    return qn::err_state();
}

auto qn_eloc_envs_logpsi(
    const QnpepsElocConfig& cfg,
    const cf* peps,
    const std::uint8_t* samples,
    std::int64_t n_samples,
    f64* logpsi_out,
    qnpeps::Linalg& linalg
) -> void;

auto qn_eloc_run_impl(
    const QnpepsElocConfig& cfg,
    const cf* peps,
    const std::uint8_t* samples,
    std::int64_t n_samples,
    const QnpepsElocTermTable* tt,
    f64* logpsi_out,
    f64* e_loc_out,
    cf* o_rows_dev,
    cf* o_rows_host,
    cf* T_dev,
    f64 lambda,
    qnpeps::Linalg& linalg
) -> void;

auto qn_eloc_run_scratch(
    const QnpepsElocConfig& cfg, std::int64_t n_samples, const QnpepsElocTermTable* tt
) -> std::uint64_t;

auto qn_eloc_compact_count(const QnpepsElocConfig& cfg) -> std::int64_t;

auto qn_eloc_gram_tile_impl(
    const QnpepsElocConfig& cfg,
    const cf* rows_a,
    const std::uint8_t* samples_a,
    std::int64_t ns_a,
    const cf* rows_b,
    const std::uint8_t* samples_b,
    std::int64_t ns_b,
    cf* tile_out,
    cudaStream_t stream
) -> void;

extern "C" qnpeps_eloc_status qnpeps_eloc_logpsi(
    const QnpepsElocConfig* cfg,
    const qnpeps_eloc_cbuf* device_peps,
    const uint8_t* device_samples,
    int64_t n_samples,
    double* logpsi_out,
    void* stream
)
{
    const auto v = check_cfg(cfg);
    if (v != QNPEPS_ELOC_OK) return v;
    if (not device_peps or not device_samples or not logpsi_out) return QNPEPS_ELOC_ERR_NULL_ARG;
    if (n_samples <= 0) return QNPEPS_ELOC_OK;
    qn::clear_err();
    auto session = qnpeps::make_session(static_cast<cudaStream_t>(stream));
    if (not session) return qn::err_state();
    qn_eloc_envs_logpsi(
        *cfg, as_cf(device_peps), device_samples, n_samples, logpsi_out, session->linalg()
    );
    CUDA_CHECK(cudaStreamSynchronize(session->stream()));
    return qn::err_state();
}

extern "C" qnpeps_eloc_status qnpeps_eloc_run(
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
)
{
    const auto v = check_cfg(cfg);
    if (v != QNPEPS_ELOC_OK) return v;
    if (not device_peps or not device_samples or not terms or not logpsi_out or not e_loc_out)
        return QNPEPS_ELOC_ERR_NULL_ARG;
    if (terms->n_flip > 0 and not terms->flip) return QNPEPS_ELOC_ERR_NULL_ARG;
    if (terms->n_diag > 0 and not terms->diag) return QNPEPS_ELOC_ERR_NULL_ARG;
    if ((o_rows_host or T_dev) and not o_rows_dev) return QNPEPS_ELOC_ERR_NULL_ARG;
    if (n_samples <= 0) return QNPEPS_ELOC_OK;
    qn::clear_err();
    auto session = qnpeps::make_session(static_cast<cudaStream_t>(stream));
    if (not session) return qn::err_state();
    qn_eloc_run_impl(
        *cfg,
        as_cf(device_peps),
        device_samples,
        n_samples,
        terms,
        logpsi_out,
        e_loc_out,
        as_cf(o_rows_dev),
        as_cf(o_rows_host),
        as_cf(T_dev),
        lambda,
        session->linalg()
    );
    CUDA_CHECK(cudaStreamSynchronize(session->stream()));
    return qn::err_state();
}

extern "C" qnpeps_eloc_status qnpeps_eloc_ctx_create(
    const QnpepsElocConfig* cfg,
    int64_t n_samples,
    const QnpepsElocTermTable* terms,
    uint32_t flags,
    void* stream,
    qnpeps_eloc_ctx** out
)
{
    const auto v = check_cfg(cfg);
    if (v != QNPEPS_ELOC_OK) return v;
    if (not terms or not out) return QNPEPS_ELOC_ERR_NULL_ARG;
    *out = nullptr;
    if (terms->n_flip > 0 and not terms->flip) return QNPEPS_ELOC_ERR_NULL_ARG;
    if (terms->n_diag > 0 and not terms->diag) return QNPEPS_ELOC_ERR_NULL_ARG;
    if (n_samples <= 0) return QNPEPS_ELOC_ERR_BAD_CONFIG;
    const auto known = uint32_t{QNPEPS_ELOC_CTX_O_ROWS | QNPEPS_ELOC_CTX_GRAM};
    if ((flags & ~known) != 0
        or ((flags & QNPEPS_ELOC_CTX_GRAM) != 0 and (flags & QNPEPS_ELOC_CTX_O_ROWS) == 0))
        return QNPEPS_ELOC_ERR_BAD_CONFIG;

    qn::clear_err();
    auto session = qnpeps::make_session(static_cast<cudaStream_t>(stream));
    if (session) qn_eloc_ctx_create_impl(*cfg, n_samples, *terms, flags, std::move(session), out);
    return qn::err_state();
}

extern "C" void qnpeps_eloc_ctx_destroy(qnpeps_eloc_ctx* ctx)
{
    qn_eloc_ctx_destroy_impl(ctx);
}

extern "C" qnpeps_eloc_status qnpeps_eloc_ctx_run(
    qnpeps_eloc_ctx* ctx, const QnpepsElocCtxRunArgs* args
)
{
    if (not ctx or not args) return QNPEPS_ELOC_ERR_NULL_ARG;
    if (args->struct_size != sizeof(QnpepsElocCtxRunArgs)) return QNPEPS_ELOC_ERR_BAD_VERSION;
    if (not args->device_peps or not args->device_samples or not args->logpsi_out
        or not args->e_loc_out)
        return QNPEPS_ELOC_ERR_NULL_ARG;
    if ((args->o_rows_host or args->T_dev) and not args->o_rows_dev)
        return QNPEPS_ELOC_ERR_NULL_ARG;
    if (args->j2_mode > QNPEPS_ELOC_J2_HALF_COLUMN_PAIRS or args->j2_draw > QNPEPS_ELOC_J2_FORCE_ALL
        or (args->j2_mode == QNPEPS_ELOC_J2_EXACT and args->j2_draw != QNPEPS_ELOC_J2_BALANCED))
        return QNPEPS_ELOC_ERR_BAD_CONFIG;
    qn::clear_err();
    qn_eloc_ctx_run_impl(
        *ctx,
        as_cf(args->device_peps),
        args->device_samples,
        args->logpsi_out,
        args->e_loc_out,
        as_cf(args->o_rows_dev),
        as_cf(args->o_rows_host),
        as_cf(args->T_dev),
        args->lambda,
        args->j2_mode,
        args->j2_draw,
        args->j2_seed,
        args->j2_epoch
    );
    return qn::err_state();
}

extern "C" qnpeps_eloc_status qnpeps_eloc_ctx_stats(
    const qnpeps_eloc_ctx* ctx, QnpepsElocCtxStats* out
)
{
    if (not ctx or not out) return QNPEPS_ELOC_ERR_NULL_ARG;
    if (out->struct_size != sizeof(QnpepsElocCtxStats)) return QNPEPS_ELOC_ERR_BAD_VERSION;
    qn_eloc_ctx_stats_impl(*ctx, *out);
    return QNPEPS_ELOC_OK;
}

extern "C" qnpeps_eloc_status qnpeps_eloc_gram_tile(
    const QnpepsElocConfig* cfg,
    const qnpeps_eloc_cbuf* rows_a,
    const uint8_t* samples_a,
    int64_t ns_a,
    const qnpeps_eloc_cbuf* rows_b,
    const uint8_t* samples_b,
    int64_t ns_b,
    qnpeps_eloc_cbuf* tile_out,
    void* stream
)
{
    const auto v = check_cfg(cfg);
    if (v != QNPEPS_ELOC_OK) return v;
    if (not rows_a or not samples_a or not rows_b or not samples_b or not tile_out)
        return QNPEPS_ELOC_ERR_NULL_ARG;
    if (ns_a <= 0 or ns_b <= 0) return QNPEPS_ELOC_OK;
    qn::clear_err();
    const auto s = static_cast<cudaStream_t>(stream);
    qn_eloc_gram_tile_impl(
        *cfg, as_cf(rows_a), samples_a, ns_a, as_cf(rows_b), samples_b, ns_b, as_cf(tile_out), s
    );
    CUDA_CHECK(cudaStreamSynchronize(s));
    return qn::err_state();
}

extern "C" qnpeps_eloc_status qnpeps_eloc_gram_ctx_create(
    const QnpepsElocConfig* cfg,
    const uint8_t* device_samples,
    int64_t n_samples,
    void* stream,
    qnpeps_eloc_gram_ctx** out
)
{
    const auto v = qnpeps_eloc_status{check_cfg(cfg)};
    if (v != QNPEPS_ELOC_OK) return v;
    if (not device_samples or not out) return QNPEPS_ELOC_ERR_NULL_ARG;
    *out = nullptr;
    const auto sites64 = int64_t{static_cast<int64_t>(cfg->lx) * cfg->ly};
    if (n_samples < 1 or sites64 < 1 or n_samples > std::numeric_limits<int64_t>::max() / sites64)
        return QNPEPS_ELOC_ERR_BAD_CONFIG;

    qn::clear_err();
    auto* ctx{static_cast<qnpeps_eloc_gram_ctx*>(std::calloc(1, sizeof(qnpeps_eloc_gram_ctx)))};
    if (not ctx) return QNPEPS_ELOC_ERR_OOM;
    ctx->sites = static_cast<int>(sites64);
    ctx->n_samples = n_samples;
    if (cudaGetDevice(&ctx->device) != cudaSuccess) qn::set_err(QNPEPS_ELOC_ERR_CUDA);

    auto offsets = std::vector<int>(static_cast<std::size_t>(ctx->sites));
    auto slices = std::vector<int>(static_cast<std::size_t>(ctx->sites));
    auto compact = int64_t{0};
    for (auto row = int{0}; row < cfg->lx; ++row)
    {
        for (auto col = int{0}; col < cfg->ly; ++col)
        {
            const auto slice =
                int{gram_bond_dim(cfg->ly, col, cfg->dim_bond)
                    * gram_bond_dim(cfg->lx, row + 1, cfg->dim_bond)
                    * gram_bond_dim(cfg->ly, col + 1, cfg->dim_bond)
                    * gram_bond_dim(cfg->lx, row, cfg->dim_bond)};
            const auto site = std::size_t{static_cast<std::size_t>(row) * cfg->ly + col};
            if (compact > std::numeric_limits<int>::max() - slice)
                qn::set_err(QNPEPS_ELOC_ERR_BAD_CONFIG);
            offsets[site] = static_cast<int>(compact);
            slices[site] = slice;
            compact += slice;
        }
    }
    ctx->compact = static_cast<int>(compact);
    if (compact != qn_eloc_compact_count(*cfg)) qn::set_err(QNPEPS_ELOC_ERR_INTERNAL);

    const auto alloc = [](void** ptr, std::size_t bytes) -> void
    {
        if (qn::err_state() != QNPEPS_ELOC_OK) return;
        if (cudaMalloc(ptr, bytes) != cudaSuccess) qn::set_err(QNPEPS_ELOC_ERR_OOM);
    };
    alloc(reinterpret_cast<void**>(&ctx->spins), sizeof(int) * n_samples * sites64);
    alloc(reinterpret_cast<void**>(&ctx->block_offset), sizeof(int) * sites64);
    alloc(reinterpret_cast<void**>(&ctx->block_slice), sizeof(int) * sites64);

    const auto s = cudaStream_t{static_cast<cudaStream_t>(stream)};
    if (qn::err_state() == QNPEPS_ELOC_OK)
    {
        CUDA_CHECK(cudaMemcpyAsync(
            ctx->block_offset,
            offsets.data(),
            sizeof(int) * static_cast<std::size_t>(sites64),
            cudaMemcpyHostToDevice,
            s
        ));
        CUDA_CHECK(cudaMemcpyAsync(
            ctx->block_slice,
            slices.data(),
            sizeof(int) * static_cast<std::size_t>(sites64),
            cudaMemcpyHostToDevice,
            s
        ));
        const auto total = int64_t{n_samples * sites64};
        const auto threads = int{256};
        auto blocks = int{static_cast<int>((total + threads - 1) / threads)};
        if (blocks > 4096) blocks = 4096;
        cu_gram_samples_to_i32<<<blocks, threads, 0, s>>>(ctx->spins, device_samples, total);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(s));
    }

    if (qn::err_state() != QNPEPS_ELOC_OK)
    {
        if (ctx->spins) cudaFree(ctx->spins);
        if (ctx->block_offset) cudaFree(ctx->block_offset);
        if (ctx->block_slice) cudaFree(ctx->block_slice);
        std::free(ctx);
        return qn::err_state();
    }
    *out = ctx;
    return QNPEPS_ELOC_OK;
}

extern "C" void qnpeps_eloc_gram_ctx_destroy(qnpeps_eloc_gram_ctx* ctx)
{
    if (not ctx) return;
    int caller_device{};
    const auto restore =
        bool{cudaGetDevice(&caller_device) == cudaSuccess and caller_device != ctx->device};
    if (restore) cudaSetDevice(ctx->device);
    if (ctx->spins) cudaFree(ctx->spins);
    if (ctx->block_offset) cudaFree(ctx->block_offset);
    if (ctx->block_slice) cudaFree(ctx->block_slice);
    if (restore) cudaSetDevice(caller_device);
    std::free(ctx);
}

extern "C" qnpeps_eloc_status qnpeps_eloc_gram_ctx_launch(
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
)
{
    if (not ctx or not rows_a or not rows_b or not out) return QNPEPS_ELOC_ERR_NULL_ARG;
    if (sample_base_a < 0 or sample_base_b < 0 or ns_a < 1 or ns_b < 1 or out_ld < ns_b
        or ns_a > std::numeric_limits<int>::max() or ns_b > std::numeric_limits<int>::max()
        or out_ld > std::numeric_limits<int>::max() or sample_base_a > ctx->n_samples - ns_a
        or sample_base_b > ctx->n_samples - ns_b)
        return QNPEPS_ELOC_ERR_BAD_CONFIG;
    int device{};
    if (cudaGetDevice(&device) != cudaSuccess or device != ctx->device)
        return QNPEPS_ELOC_ERR_BAD_CONFIG;

    qn::clear_err();
    const auto s = cudaStream_t{static_cast<cudaStream_t>(stream)};
    qn_eloc::launch_gram_tile(
        as_cf(out),
        static_cast<int>(out_ld),
        ctx->compact,
        ctx->sites,
        as_cf(rows_a),
        as_cf(rows_b),
        ctx->spins + sample_base_a * ctx->sites,
        ctx->spins + sample_base_b * ctx->sites,
        ctx->block_offset,
        ctx->block_slice,
        0,
        static_cast<int>(ns_a),
        0,
        static_cast<int>(ns_b),
        s
    );
    CUDA_CHECK(cudaGetLastError());
    return qn::err_state();
}

extern "C" qnpeps_eloc_status qnpeps_eloc_compact_count(
    const QnpepsElocConfig* cfg, int64_t* out_count
)
{
    const auto v = check_cfg(cfg);
    if (v != QNPEPS_ELOC_OK) return v;
    if (not out_count) return QNPEPS_ELOC_ERR_NULL_ARG;
    *out_count = qn_eloc_compact_count(*cfg);
    return QNPEPS_ELOC_OK;
}

extern "C" qnpeps_eloc_status qnpeps_eloc_run_scratch_bytes(
    const QnpepsElocConfig* cfg,
    int64_t n_samples,
    const QnpepsElocTermTable* terms,
    uint64_t* out_bytes
)
{
    const auto v = check_cfg(cfg);
    if (v != QNPEPS_ELOC_OK) return v;
    if (not out_bytes) return QNPEPS_ELOC_ERR_NULL_ARG;
    *out_bytes = qn_eloc_run_scratch(*cfg, n_samples, terms);
    return QNPEPS_ELOC_OK;
}

extern "C" const char* qnpeps_eloc_strerror(qnpeps_eloc_status status)
{
    switch (status)
    {
        case QNPEPS_ELOC_OK:
            return "ok";
        case QNPEPS_ELOC_ERR_NULL_ARG:
            return "a required pointer was NULL";
        case QNPEPS_ELOC_ERR_BAD_CONFIG:
            return "descriptor failed validation";
        case QNPEPS_ELOC_ERR_BAD_VERSION:
            return "descriptor struct_size not recognized";
        case QNPEPS_ELOC_ERR_CUDA:
            {
                auto backend{qnpeps::err_backend()};
                if (not backend or std::strcmp(backend, "none") == 0)
                    return "an underlying CUDA call failed";
                static thread_local qnpeps::CuArray<char, 512> message{};
                std::snprintf(
                    message.data(),
                    sizeof(message),
                    "an underlying CUDA call failed (%s code %d at %s:%d)",
                    backend,
                    qnpeps::err_backend_code(),
                    qnpeps::err_file(),
                    qnpeps::err_line()
                );
                return message.data();
            }
        case QNPEPS_ELOC_ERR_OOM:
            return "device allocation failed";
        case QNPEPS_ELOC_ERR_INTERNAL:
            return "internal invariant violated (please report)";
    }
    return "unknown status";
}

extern "C" const char* qnpeps_eloc_version(void)
{
    return "cuQuantumNaturalfPEPS eo " QNPEPS_PACKAGE_VERSION;
}
