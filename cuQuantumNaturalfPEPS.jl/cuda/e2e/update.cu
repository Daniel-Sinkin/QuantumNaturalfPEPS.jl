#include "core/cuda_utils.cuh"
#include "kernels.cuh"
#include "layout.cuh"
#include "update.cuh"

#include <cmath>
#include <new>

namespace qn_e2e
{

namespace
{

auto map_cuda(cudaError_t error) -> qnpeps_e2e_status
{
    if (error == cudaSuccess) return QNPEPS_E2E_OK;
    if (error == cudaErrorMemoryAllocation) return QNPEPS_E2E_ERR_OOM;
    return QNPEPS_E2E_ERR_CUDA;
}

}

auto build_update_sites(const QnpepsE2eConfig& cfg) -> std::vector<UpdateSite>
{
    std::vector<UpdateSite> sites{};
    const auto row_count = static_cast<usize>(cfg.lx);
    const auto column_count = static_cast<usize>(cfg.ly);
    sites.reserve(row_count * column_count);
    i64 offset{};
    for (int row{}; row < cfg.lx; ++row)
    {
        for (int column{}; column < cfg.ly; ++column)
        {
            const i32 dw{bond_dim(cfg.ly, column, cfg.dim_bond)};
            const i32 ds{bond_dim(cfg.lx, row + 1, cfg.dim_bond)};
            const i32 de{bond_dim(cfg.ly, column + 1, cfg.dim_bond)};
            const i32 dn{bond_dim(cfg.lx, row, cfg.dim_bond)};
            const i32 dp{cfg.dim_phys};
            const i64 count{static_cast<i64>(dw) * ds * de * dn * static_cast<i64>(dp)};
            sites.push_back(UpdateSite{offset, count, dw, ds, de, dn, dp});
            offset += count;
        }
    }
    if (offset != dense_count(cfg)) sites.clear();
    return sites;
}

auto launch_update_f32(
    const UpdateSite* sites,
    int site_count,
    cf* peps_f32_io,
    const cf* theta_dot,
    f64 learning_rate,
    cudaStream_t stream
) -> qnpeps_e2e_status
{
    if (not sites or site_count < 1 or not peps_f32_io or not theta_dot)
        return QNPEPS_E2E_ERR_NULL_ARG;
    const f32 rate{static_cast<f32>(learning_rate)};
    if (not std::isfinite(learning_rate) or not std::isfinite(rate))
        return QNPEPS_E2E_ERR_BAD_CONFIG;
    const auto block_count = static_cast<unsigned int>(site_count);
    const UpdateF32Args args{sites, peps_f32_io, theta_dot, rate};
    cu_update_f32<<<block_count, qnpeps::k_threads_per_block, 0, stream>>>(args);
    return map_cuda(cudaGetLastError());
}

auto launch_update_f64(
    const UpdateSite* sites,
    int site_count,
    zd* state_f64_io,
    const cf* theta_dot,
    cf* peps_f32_out,
    f64 learning_rate,
    cudaStream_t stream
) -> qnpeps_e2e_status
{
    if (not sites or site_count < 1 or not state_f64_io or not theta_dot or not peps_f32_out)
        return QNPEPS_E2E_ERR_NULL_ARG;
    if (not std::isfinite(learning_rate)) return QNPEPS_E2E_ERR_BAD_CONFIG;
    const auto block_count = static_cast<unsigned int>(site_count);
    const UpdateF64Args args{sites, state_f64_io, theta_dot, peps_f32_out, learning_rate};
    cu_update_f64<<<block_count, qnpeps::k_threads_per_block, 0, stream>>>(args);
    return map_cuda(cudaGetLastError());
}

}

struct qnpeps_e2e_update_ctx
{
    qn_e2e::UpdateSite* sites{};
    i64 site_count{};
    i64 dense{};
};

namespace
{

auto update_cuda_status(cudaError_t status) -> qnpeps_e2e_status
{
    if (status == cudaSuccess) return QNPEPS_E2E_OK;
    if (status == cudaErrorMemoryAllocation) return QNPEPS_E2E_ERR_OOM;
    return QNPEPS_E2E_ERR_CUDA;
}

auto update_config_status(const QnpepsE2eConfig& cfg) -> qnpeps_e2e_status
{
    if (cfg.struct_size != sizeof(QnpepsE2eConfig)) return QNPEPS_E2E_ERR_BAD_VERSION;
    if (cfg.lx < 1 or cfg.ly < 1 or cfg.dim_phys < 1 or cfg.dim_bond < 1)
        return QNPEPS_E2E_ERR_BAD_CONFIG;
    return QNPEPS_E2E_OK;
}

}

extern "C" qnpeps_e2e_status qnpeps_e2e_update_ctx_create(
    const QnpepsE2eConfig* cfg, qnpeps_e2e_update_ctx** out
)
{
    if (not cfg or not out) return QNPEPS_E2E_ERR_NULL_ARG;
    *out = nullptr;
    const qnpeps_e2e_status config_status{update_config_status(*cfg)};
    if (config_status != QNPEPS_E2E_OK) return config_status;
    const std::vector<qn_e2e::UpdateSite> sites{qn_e2e::build_update_sites(*cfg)};
    const i64 site_count{static_cast<i64>(cfg->lx) * cfg->ly};
    if (sites.size() != static_cast<usize>(site_count)) return QNPEPS_E2E_ERR_BAD_CONFIG;
    int caller_device{};
    const qnpeps_e2e_status caller_status{update_cuda_status(cudaGetDevice(&caller_device))};
    if (caller_status != QNPEPS_E2E_OK) return caller_status;
    const qnpeps_e2e_status device_status{update_cuda_status(cudaSetDevice(0))};
    if (device_status != QNPEPS_E2E_OK) return device_status;
    auto* ctx{new (std::nothrow) qnpeps_e2e_update_ctx{}};
    if (not ctx)
    {
        cudaSetDevice(caller_device);
        return QNPEPS_E2E_ERR_OOM;
    }
    ctx->site_count = site_count;
    ctx->dense = qn_e2e::dense_count(*cfg);
    qnpeps_e2e_status status{
        update_cuda_status(cudaMalloc(&ctx->sites, sites.size() * sizeof(qn_e2e::UpdateSite)))
    };
    if (status == QNPEPS_E2E_OK)
    {
        status = update_cuda_status(cudaMemcpy(
            ctx->sites,
            sites.data(),
            sites.size() * sizeof(qn_e2e::UpdateSite),
            cudaMemcpyHostToDevice
        ));
    }
    if (status != QNPEPS_E2E_OK)
    {
        cudaFree(ctx->sites);
        delete ctx;
        cudaSetDevice(caller_device);
        return status;
    }
    cudaSetDevice(caller_device);
    *out = ctx;
    return QNPEPS_E2E_OK;
}

extern "C" qnpeps_e2e_status qnpeps_e2e_update_ctx_run(
    qnpeps_e2e_update_ctx* ctx, const QnpepsE2eUpdateArgs* args
)
{
    const auto missing_input = not ctx or not args or not args->state_f64_io or not args->theta_dot
                               or not args->peps_f32_out;
    if (missing_input) return QNPEPS_E2E_ERR_NULL_ARG;
    if (args->struct_size != sizeof(QnpepsE2eUpdateArgs)) return QNPEPS_E2E_ERR_BAD_VERSION;
    const u64 state_bytes{static_cast<u64>(ctx->dense) * sizeof(qn_e2e::zd)};
    const u64 theta_bytes{static_cast<u64>(ctx->dense) * sizeof(qn_e2e::cf)};
    const auto invalid_bytes = args->state_f64_bytes != state_bytes
                               or args->theta_dot_bytes != theta_bytes
                               or args->peps_f32_bytes != theta_bytes;
    if (invalid_bytes) return QNPEPS_E2E_ERR_BAD_CONFIG;
    const auto aliased_buffers = args->state_f64_io == static_cast<void*>(args->peps_f32_out)
                                 or args->state_f64_io == static_cast<const void*>(args->theta_dot)
                                 or args->theta_dot == args->peps_f32_out;
    if (aliased_buffers) return QNPEPS_E2E_ERR_BAD_CONFIG;
    int caller_device{};
    qnpeps_e2e_status status{update_cuda_status(cudaGetDevice(&caller_device))};
    if (status != QNPEPS_E2E_OK) return status;
    status = update_cuda_status(cudaSetDevice(0));
    if (status != QNPEPS_E2E_OK) return status;
    status = qn_e2e::launch_update_f64(
        ctx->sites,
        static_cast<int>(ctx->site_count),
        static_cast<qn_e2e::zd*>(args->state_f64_io),
        reinterpret_cast<const qn_e2e::cf*>(args->theta_dot),
        reinterpret_cast<qn_e2e::cf*>(args->peps_f32_out),
        args->learning_rate,
        reinterpret_cast<cudaStream_t>(args->stream)
    );
    if (status == QNPEPS_E2E_OK)
    {
        status =
            update_cuda_status(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(args->stream)));
    }
    cudaSetDevice(caller_device);
    return status;
}

extern "C" qnpeps_e2e_status qnpeps_e2e_update_ctx_destroy(qnpeps_e2e_update_ctx* ctx)
{
    if (not ctx) return QNPEPS_E2E_OK;
    int caller_device{};
    qnpeps_e2e_status status{update_cuda_status(cudaGetDevice(&caller_device))};
    if (status != QNPEPS_E2E_OK) return status;
    status = update_cuda_status(cudaSetDevice(0));
    if (status != QNPEPS_E2E_OK) return status;
    status = update_cuda_status(cudaFree(ctx->sites));
    cudaSetDevice(caller_device);
    if (status != QNPEPS_E2E_OK) return status;
    delete ctx;
    return status;
}
