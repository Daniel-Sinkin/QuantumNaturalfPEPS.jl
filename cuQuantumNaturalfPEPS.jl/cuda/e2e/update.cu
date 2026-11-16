#include "core/cuda_utils.cuh"
#include "kernels.cuh"
#include "layout.cuh"
#include "update.cuh"

#include <cmath>

namespace qn_e2e
{

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
) -> void
{
    const auto has_buffers = sites and site_count >= 1 and peps_f32_io and theta_dot;
    require(
        has_buffers, Status::null_argument, "f32 update over {} sites misses a buffer", site_count
    );
    const f32 rate{static_cast<f32>(learning_rate)};
    require(
        std::isfinite(learning_rate) and std::isfinite(rate),
        Status::bad_config,
        "f32 update learning rate {} is not finite in single precision",
        learning_rate
    );
    const auto block_count = static_cast<unsigned int>(site_count);
    const UpdateF32Args args{sites, peps_f32_io, theta_dot, rate};
    cu_update_f32<<<block_count, qnpeps::k_threads_per_block, 0, stream>>>(args);
    check(cudaGetLastError());
}

auto launch_update_f64(
    const UpdateSite* sites,
    int site_count,
    zd* state_f64_io,
    const cf* theta_dot,
    cf* peps_f32_out,
    f64 learning_rate,
    cudaStream_t stream
) -> void
{
    const auto has_buffers =
        sites and site_count >= 1 and state_f64_io and theta_dot and peps_f32_out;
    require(
        has_buffers, Status::null_argument, "f64 update over {} sites misses a buffer", site_count
    );
    require(
        std::isfinite(learning_rate),
        Status::bad_config,
        "f64 update learning rate {} is not finite",
        learning_rate
    );
    const auto block_count = static_cast<unsigned int>(site_count);
    const UpdateF64Args args{sites, state_f64_io, theta_dot, peps_f32_out, learning_rate};
    cu_update_f64<<<block_count, qnpeps::k_threads_per_block, 0, stream>>>(args);
    check(cudaGetLastError());
}

}
