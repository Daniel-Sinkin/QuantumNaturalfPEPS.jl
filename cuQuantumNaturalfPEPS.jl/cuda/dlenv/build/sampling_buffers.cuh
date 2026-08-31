#ifndef QNPEPS_DLENV_BUILD_SAMPLING_BUFFERS_CUH
#define QNPEPS_DLENV_BUILD_SAMPLING_BUFFERS_CUH

#include "dlenv/build/types.cuh"

namespace qnpeps::dlenv
{

auto ensure_sampling_buffers(qnpeps_ctx& ctx) -> void
{
    auto buffers_ready = true;
    for (const auto& lane : ctx.dlenv.lanes)
        buffers_ready = buffers_ready and lane.sampling;
    const auto density_route = ctx.cfg.dlenv_truncation_route == QNPEPS_TRUNCATION_DENSITY;
    if (buffers_ready and not density_route) return;

    const auto num_env_rows = static_cast<usize>(ctx.cfg.lx - 1);
    const auto num_cols = static_cast<usize>(ctx.cfg.ly);
    ctx.dlenv.env_off.assign(num_env_rows, std::vector<i64>(num_cols, 0));
    ctx.dlenv.sigma_off.assign(num_env_rows, std::vector<i64>(num_cols, 0));
    i64 cursor{};
    for (auto row = 0_uz; row < num_env_rows; ++row)
    {
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const auto site = read_site_dims(ctx.dlenv.dims.data(), row * num_cols + col);
            ctx.dlenv.env_off[row][col] = cursor;
            cursor += site.num_elems();
        }
    }
    const i64 total_env{cursor};
    for (auto row = 0_uz; row < num_env_rows; ++row)
    {
        for (auto col = 0_uz; col < num_cols; ++col)
            ctx.dlenv.sigma_off[row][col] = total_env + ctx.dlenv.env_off[row][col];
    }
    const auto layout_count = static_cast<i64>(k_sampling_layout_count);
    ctx.dlenv.sampling_elements = layout_count * total_env;
    auto sampling_elements = static_cast<usize>(ctx.dlenv.sampling_elements);
    if (density_route)
    {
        const auto chi =
            static_cast<usize>(std::min(ctx.cfg.chi_dl, ctx.cfg.dim_bond * ctx.cfg.dim_bond));
        const auto bond_pair = static_cast<usize>(ctx.cfg.dim_bond * ctx.cfg.dim_bond);
        sampling_elements =
            k_sampling_layout_count * num_env_rows * num_cols * chi * bond_pair * chi;
    }
    const auto sampling_bytes = sampling_elements * sizeof(cuFloatComplex);
    for (auto& lane : ctx.dlenv.lanes)
    {
        if (lane.sampling) continue;
        if (ctx.sampler.execution.host_managed) continue;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&lane.sampling), sampling_bytes));
        lane.sampling_owned = true;
        if (err_state() != QNPEPS_OK) return;
    }
}

auto materialize_sampling_buffer(
    qnpeps_ctx& ctx, const cuFloatComplex* raw_values, cuFloatComplex* sampling_out
) -> void
{
    const auto num_env_rows = static_cast<usize>(ctx.cfg.lx - 1);
    const auto num_cols = static_cast<usize>(ctx.cfg.ly);
    const auto stream = ctx.linalg().stream();
    const auto* device_raw_values = raw_values;
    i64 raw_offset{};
    for (auto row = 0_uz; row < num_env_rows; ++row)
    {
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const auto site_dims = read_site_dims(ctx.dlenv.dims.data(), row * num_cols + col);
            DeviceTensor site{
                {site_dims.bond_left, site_dims.ket, site_dims.bra, site_dims.bond_right},
                const_cast<cuFloatComplex*>(device_raw_values + raw_offset)
            };
            permute_axes(
                site, {1, 3, 2, 0}, false, sampling_out + ctx.dlenv.env_off[row][col], stream
            );
            permute_axes(
                site, {1, 0, 2, 3}, false, sampling_out + ctx.dlenv.sigma_off[row][col], stream
            );
            raw_offset += static_cast<i64>(site.num_elems());
        }
    }
}

}

#endif
