#ifndef QNPEPS_SAMPLER_SWEEP_ENVIRONMENT_CUH
#define QNPEPS_SAMPLER_SWEEP_ENVIRONMENT_CUH

#include "sampler/sweep/common.cuh"

namespace qnpeps::sampler::internal
{

[[nodiscard]] inline auto build_env_above(
    Sampler& samp,
    const SamplerConfig& cfg,
    int row,
    bool has_below,
    const std::vector<int>& ket_bonds,
    int& env_above_cur,
    std::vector<int>& bond_above_cur
) -> bool
{
    if (not has_below) return true;
    const auto dim_batch = cfg.dim_batch;
    const auto dim_bond = cfg.dim_bond;
    const auto dim_phys = cfg.dim_phys;
    const auto num_cols = static_cast<usize>(cfg.ly);

    auto& la = samp.linalg();

    const int env_above_next{1 - env_above_cur};
    if (cfg.fast_mode or row == 0)
    {
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const auto col_i = static_cast<i64>(col);
            const auto ket_bond_l = ket_bonds[col];
            const auto ket_bond_r = ket_bonds[col + 1];
            const auto* stored_ket = samp.ket().p + col_i * samp.max_ket_site();
            const auto* ket_site = row == 0 ? samp.ket_row0()[col] : stored_ket;
            const i64 ket_stride{row == 0 ? 0 : samp.ket().stride};
            auto* out = samp.env_above()[static_cast<usize>(env_above_next)].p
                        + col_i * samp.max_env_above_site();
            const auto slice_elems = static_cast<i64>(ket_bond_l) * dim_bond * ket_bond_r;
            const CuSliceKetArgs slice_args{
                .out = out,
                .ket = ket_site,
                .chosen_spins = samp.row_spins() + col_i * cfg.row_spin_stride,
                .ket_bond_l = ket_bond_l,
                .dim_phys = dim_phys,
                .slice_elems = slice_elems,
                .stride_out = samp.env_above()[static_cast<usize>(env_above_next)].stride,
                .stride_in = ket_stride,
                .dim_batch = dim_batch,
            };
            cu_slice_ket<<<
                grid_blocks_capped(slice_elems * dim_batch),
                k_threads_per_block,
                0,
                la.stream()>>>(slice_args);
            const CuNormalizeLogArgs norm_args{
                .x = out,
                .n = ket_bond_l * dim_bond * ket_bond_r,
                .stride = samp.env_above()[static_cast<usize>(env_above_next)].stride,
                .lognorm_acc = samp.lognorm(),
                .dim_batch = dim_batch,
            };
            cu_normalize_log<<<
                static_cast<u32>(dim_batch),
                k_tree_reduce_threads,
                0,
                la.stream()>>>(norm_args);
        }
        bond_above_cur = ket_bonds;
    }
    else
    {
        const auto row_u = static_cast<usize>(row);
        std::vector<int> full_bonds{};
        full_bonds.assign(num_cols + 1, 1);
        auto carried_bond = 1;
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const auto& peps_shape = samp.peps_shapes()[row_u][col];
            const auto reduce_rows = carried_bond * peps_shape[1];
            const auto reduce_cols = bond_above_cur[col + 1] * peps_shape[2];
            const auto next_bond = std::max(1, std::min({cfg.chi_c, reduce_rows, reduce_cols}));
            full_bonds[col + 1] = next_bond;
            carried_bond = next_bond;
        }
        full_bonds[num_cols] = 1;

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
            const auto bond_down = peps_shape[1];
            const auto bond_right = peps_shape[2];
            const auto env_left = bond_above_cur[col];
            const auto env_right = bond_above_cur[col + 1];
            const auto full_left = full_bonds[col];
            const auto full_right = full_bonds[col + 1];
            const auto projected_elems = bond_left * bond_up * bond_down * bond_right;
            const CuProjectMpoArgs project_args{
                .out = samp.tmp_b().p,
                .mpo = samp.mpo()[row_u][col],
                .chosen_spins = samp.row_spins() + col_i * cfg.row_spin_stride,
                .spin_block = bond_left * bond_up,
                .dim_phys = dim_phys,
                .output_elems = projected_elems,
                .stride_out = samp.tmp_b().stride,
                .dim_batch = dim_batch,
            };
            cu_project_mpo<<<
                grid_blocks_capped(static_cast<i64>(projected_elems) * dim_batch),
                k_threads_per_block,
                0,
                la.stream()>>>(project_args);

            const auto& environment = samp.env_above()[static_cast<usize>(env_above_cur)];
            const auto environment_offset = col_i * samp.max_env_above_site();
            const auto* environment_site = environment.p + environment_offset;
            {
                const auto res = contract_strided(
                    la,
                    samp.permutation_cache(),
                    {
                        .dims_a = {full_left, env_left, bond_left},
                        .contracted_a = {1},
                        .dims_b = {env_left, bond_up, env_right},
                        .contracted_b = {0},
                    },
                    {.src = samp.rfactor(), .scratch = samp.tmp_a()},
                    {.src = {environment_site, environment.stride}},
                    {.view = samp.reduce_input()},
                    dim_batch
                );
                if (not res) return false;
            }

            {
                const auto res = contract_strided(
                    la,
                    samp.permutation_cache(),
                    {
                        .dims_a = {full_left, bond_left, bond_up, env_right},
                        .contracted_a = {1, 2},
                        .dims_b = {bond_left, bond_up, bond_down, bond_right},
                        .contracted_b = {0, 1},
                    },
                    {.src = samp.reduce_input(), .scratch = samp.tmp_a()},
                    {.src = samp.tmp_b()},
                    {.view = samp.reduce_input()},
                    dim_batch
                );
                if (not res) return false;
            }

            permute_batched(
                samp.permutation_cache(),
                {
                    .dst = samp.tmp_a(),
                    .src = samp.reduce_input(),
                    .dims_in = {full_left, env_right, bond_down, bond_right},
                    .perm = {0, 2, 1, 3},
                    .batch_count = dim_batch,
                },
                la.stream()
            );
            if (err_state() != QNPEPS_OK) return false;
            const auto reduce_rows = full_left * bond_down;
            const auto reduce_cols = env_right * bond_right;
            auto* out = samp.env_above()[static_cast<usize>(env_above_next)].p
                        + col_i * samp.max_env_above_site();
            samp.reduce(
                {samp.tmp_a(), reduce_rows, reduce_cols},
                full_right,
                {out,
                 samp.env_above()[static_cast<usize>(env_above_next)].stride,
                 reduce_rows,
                 full_right},
                {samp.rfactor_next(), full_right, reduce_cols},
                dim_batch
            );
            if (err_state() != QNPEPS_OK) return false;
            const CuNormalizeLogArgs norm_args{
                .x = samp.rfactor_next().p,
                .n = full_right * reduce_cols,
                .stride = samp.rfactor_next().stride,
                .lognorm_acc = samp.lognorm(),
                .dim_batch = dim_batch,
            };
            cu_normalize_log<<<
                static_cast<u32>(dim_batch),
                k_tree_reduce_threads,
                0,
                la.stream()>>>(norm_args);
            std::swap(samp.rfactor().p, samp.rfactor_next().p);
            std::swap(samp.rfactor().stride, samp.rfactor_next().stride);
        }
        bond_above_cur = std::move(full_bonds);
    }
    env_above_cur = env_above_next;
    return err_state() == QNPEPS_OK;
}

}

#endif
