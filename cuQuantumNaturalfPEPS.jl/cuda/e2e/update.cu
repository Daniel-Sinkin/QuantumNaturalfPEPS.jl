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
