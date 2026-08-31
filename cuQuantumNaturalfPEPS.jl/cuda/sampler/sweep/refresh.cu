#include "sampler/sweep/refresh.cuh"

namespace qnpeps::sampler
{
auto ctx_sample_refresh(qnpeps_ctx& ctx, const void* device_peps, PepsLayout layout) -> void
{
    if (not device_peps)
    {
        set_err(QNPEPS_ERR_NULL_ARG);
        return;
    }

    if (not ctx.sampler.ready())
    {
        set_err(QNPEPS_ERR_INTERNAL);
        return;
    }

    auto& samp = ctx.sampler.samp;
    const auto num_rows = static_cast<usize>(ctx.cfg.lx);
    const auto num_cols = static_cast<usize>(ctx.cfg.ly);
    const auto stream = ctx.linalg().stream();

    CUDA_CHECK(cudaStreamSynchronize(stream));

    Permutation mpo_permutation{};
    Permutation ket_row0_permutation{};
    if (layout == PepsLayout::canonical)
    {
        mpo_permutation = Permutation{0, 3, 4, 1, 2};
        ket_row0_permutation = Permutation{0, 4, 1, 2, 3};
    }
    else
    {
        mpo_permutation = Permutation{4, 1, 0, 3, 2};
        ket_row0_permutation = Permutation{4, 0, 3, 2, 1};
    }

    const auto reversed = Permutation::reverse(k_peps_site_rank);
    const auto* peps_base = static_cast<const cuFloatComplex*>(device_peps);
    usize offset{};
    for (auto row = 0_uz; row < num_rows; ++row)
    {
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const auto& site_shape = samp.peps_shapes()[row][col];
            const DeviceTensor source{
                layout == PepsLayout::canonical ? site_shape : reversed.apply(site_shape),
                const_cast<cuFloatComplex*>(peps_base + offset)
            };
            permute_axes(source, mpo_permutation, false, samp.mpo()[row][col], stream);
            if (row == 0)
            {
                permute_axes(source, ket_row0_permutation, false, samp.ket_row0()[col], stream);
            }
            offset += site_shape.num_elems();
        }
    }
}

}
