#ifndef QNPEPS_SAMPLER_SWEEP_KET_CUH
#define QNPEPS_SAMPLER_SWEEP_KET_CUH

#include "sampler/sweep/common.cuh"

namespace qnpeps::sampler::internal
{
[[nodiscard]] inline auto build_ket_row(
    Sampler& samp,
    const SamplerConfig& cfg,
    int row,
    const std::vector<int>& bond_above,
    const std::vector<int>& ket_bonds,
    int env_above_cur
) -> bool
{
    const auto dim_batch = cfg.dim_batch;
    const auto dim_bond = cfg.dim_bond;
    const auto num_cols = static_cast<usize>(cfg.ly);
    const auto row_u = static_cast<usize>(row);

    auto& la = samp.linalg();
    const auto permute = [&](PermuteOp op) -> bool
    {
        op.batch_count = dim_batch;
        permute_batched(samp.permutation_cache(), op, la.stream());
        return err_state() == QNPEPS_OK;
    };

    const auto fill_first_one_args = CuFillFirstOneArgs{
        .x = samp.rfactor().p,
        .stride = samp.rfactor().stride,
        .n = 1,
        .dim_batch = dim_batch,
    };
    cu_fill_first_one<<<grid_blocks_capped(dim_batch), k_threads_per_block, 0, la.stream()>>>(
        fill_first_one_args
    );
    for (auto col = 0_uz; col < num_cols; ++col)
    {
        const auto col_i = static_cast<i64>(col);
        const auto& peps_shape = samp.peps_shapes()[row_u][col];
        const auto bond_left = peps_shape[0];
        const auto bond_up = peps_shape[3];
        const auto phys_site = peps_shape[4];
        const auto bond_down = peps_shape[1];
        const auto bond_right = peps_shape[2];
        const auto bond_above_l = bond_above[col];
        const auto bond_above_r = bond_above[col + 1];
        const auto ket_bond_l = ket_bonds[col];
        const auto ket_bond_r = ket_bonds[col + 1];

        const auto& environment = samp.env_above()[static_cast<usize>(env_above_cur)];
        const auto environment_offset = col_i * samp.max_env_above_site();
        const auto* environment_site = environment.p + environment_offset;
        {
            const auto res = contract_strided(
                la,
                samp.permutation_cache(),
                {
                    .dims_a = {ket_bond_l, bond_above_l, bond_left},
                    .contracted_a = {1},
                    .dims_b = {bond_above_l, dim_bond, bond_above_r},
                    .contracted_b = {0},
                },
                {.src = samp.rfactor(), .scratch = samp.tmp_a()},
                {.src = {environment_site, environment.stride}},
                {.view = samp.tmp_b()},
                dim_batch
            );
            if (not res) return false;
        }

        {
            const auto res = contract_batched(
                la,
                samp.permutation_cache(),
                {
                    .dims_a = {ket_bond_l, bond_left, bond_up, bond_above_r},
                    .contracted_a = {1, 2},
                    .dims_b = {bond_left, bond_up, phys_site, bond_down, bond_right},
                    .contracted_b = {0, 1},
                },
                {.src = samp.tmp_b(), .scratch = samp.tmp_a(), .ptrs = samp.tmp_a_ptrs()},
                {.ptrs = samp.mpo_ptrs()[row_u][col]},
                {.ptrs = samp.tmp_b_ptrs()},
                dim_batch
            );
            if (not res) return false;
        }

        {
            const auto res = permute({
                .dst = samp.reduce_input(),
                .src = samp.tmp_b(),
                .dims_in = {ket_bond_l, bond_above_r, phys_site, bond_down, bond_right},
                .perm = {0, 2, 3, 1, 4},
            });
            if (not res) return false;
        }
        const int reduce_rows{ket_bond_l * phys_site * bond_down};
        const int reduce_cols{bond_above_r * bond_right};
        auto* ket_site = samp.ket().p + col_i * samp.max_ket_site();
        samp.reduce(
            {samp.reduce_input(), reduce_rows, reduce_cols},
            ket_bond_r,
            {ket_site, samp.ket().stride, reduce_rows, ket_bond_r},
            {samp.rfactor_next(), ket_bond_r, reduce_cols},
            dim_batch
        );
        if (err_state() != QNPEPS_OK) return false;

        const CuNormalizeLogArgs norm_args{
            .x = samp.rfactor_next().p,
            .n = ket_bond_r * reduce_cols,
            .stride = samp.rfactor_next().stride,
            .lognorm_acc = nullptr,
            .dim_batch = dim_batch,
        };
        cu_normalize_log<<<static_cast<u32>(dim_batch), k_tree_reduce_threads, 0, la.stream()>>>(
            norm_args
        );
        std::swap(samp.rfactor().p, samp.rfactor_next().p);
        std::swap(samp.rfactor().stride, samp.rfactor_next().stride);
    }
    return err_state() == QNPEPS_OK;
}

}

#endif
