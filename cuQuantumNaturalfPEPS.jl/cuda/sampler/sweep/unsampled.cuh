#ifndef QNPEPS_SAMPLER_SWEEP_UNSAMPLED_CUH
#define QNPEPS_SAMPLER_SWEEP_UNSAMPLED_CUH

#include "sampler/sweep/common.cuh"

namespace qnpeps::sampler::internal
{

[[nodiscard]] inline auto build_env_unsampled(
    Sampler& samp,
    const SamplerConfig& cfg,
    int row,
    bool has_below,
    const std::vector<int>& ket_bonds,
    const std::vector<int>& env_bonds
) -> bool
{
    const auto dim_batch = cfg.dim_batch;
    const auto dim_bond = cfg.dim_bond;
    const auto dim_phys = cfg.dim_phys;
    const auto num_cols = static_cast<usize>(cfg.ly);
    const auto row_u = static_cast<usize>(row);

    auto& la = samp.linalg();

    {
        const auto offset = static_cast<i64>(cfg.ly) * samp.max_env_unsampled();
        auto* env_unsampled_last = samp.env_unsampled().p + offset;
        const auto fill_first_one_args = CuFillFirstOneArgs{
            .x = env_unsampled_last,
            .stride = samp.env_unsampled().stride,
            .n = 1,
            .dim_batch = dim_batch,
        };
        cu_fill_first_one<<<grid_blocks_capped(dim_batch), k_threads_per_block, 0, la.stream()>>>(
            fill_first_one_args
        );
    }
    for (auto col = num_cols - 1; col >= 1; --col)
    {
        const auto col_i = static_cast<i64>(col);
        const auto ket_bond_l = ket_bonds[col];
        const auto ket_bond_r = ket_bonds[col + 1];
        const auto env_bond_l = env_bonds[col];
        const auto env_bond_r = env_bonds[col + 1];
        const int bond_below{has_below ? dim_bond : 1};
        const auto* stored_ket = samp.ket().p + col_i * samp.max_ket_site();
        const auto* ket_site = row == 0 ? samp.ket_row0()[col] : stored_ket;
        const i64 ket_stride{row == 0 ? 0 : samp.ket().stride};
        auto* dlenv_environment = samp.dl_unit_ptrs();
        if (has_below) dlenv_environment = samp.dlenv_env_ptrs()[row_u][col];
        const auto input_offset = (col_i + 1) * samp.max_env_unsampled();
        const auto* env_unsampled_in = samp.env_unsampled().p + input_offset;
        auto* env_unsampled_out = samp.env_unsampled().p + col_i * samp.max_env_unsampled();

        const ContractSpec ket_env_spec{
            .dims_a = {ket_bond_l, dim_phys, bond_below, ket_bond_r},
            .contracted_a = {3},
            .dims_b = {ket_bond_r, env_bond_r, ket_bond_r},
            .contracted_b = {0},
        };
        if (row == 0)
        {
            const auto res = contract_batched(
                la,
                samp.permutation_cache(),
                ket_env_spec,
                {.ptrs = samp.ket_row0_ptrs()[col]},
                {.ptrs = samp.envu_ptrs()[col + 1]},
                {.ptrs = samp.tmp_a_ptrs()},
                dim_batch
            );
            if (not res) return false;
        }
        else
        {
            const auto res = contract_strided(
                la,
                samp.permutation_cache(),
                ket_env_spec,
                {.src = {ket_site, ket_stride}},
                {.src = {env_unsampled_in, samp.env_unsampled().stride}},
                {.view = samp.tmp_a()},
                dim_batch
            );
            if (not res) return false;
        }

        {
            const auto res = contract_batched(
                la,
                samp.permutation_cache(),
                {
                    .dims_a = {ket_bond_l, dim_phys, bond_below, env_bond_r, ket_bond_r},
                    .contracted_a = {2, 3},
                    .dims_b = {bond_below, env_bond_r, bond_below, env_bond_l},
                    .contracted_b = {0, 1},
                },
                {.src = samp.tmp_a(), .scratch = samp.tmp_b(), .ptrs = samp.tmp_b_ptrs()},
                {.ptrs = dlenv_environment},
                {.ptrs = samp.tmp_a_ptrs()},
                dim_batch
            );
            if (not res) return false;
        }

        if (row == 0)
        {
            const auto res = contract_batched(
                la,
                samp.permutation_cache(),
                {
                    .dims_a = {ket_bond_l, dim_phys, ket_bond_r, bond_below, env_bond_l},
                    .contracted_a = {1, 3, 2},
                    .dims_b = {ket_bond_l, dim_phys, bond_below, ket_bond_r},
                    .contracted_b = {1, 2, 3},
                    .transforms = {.conj_b = true},
                },
                {.src = samp.tmp_a(), .scratch = samp.tmp_b(), .ptrs = samp.tmp_b_ptrs()},
                {.ptrs = samp.ket_row0_ptrs()[col]},
                {.ptrs = samp.envu_ptrs()[col]},
                dim_batch
            );
            if (not res) return false;
        }
        else
        {
            const auto res = contract_strided(
                la,
                samp.permutation_cache(),
                {
                    .dims_a = {ket_bond_l, dim_phys, ket_bond_r, bond_below, env_bond_l},
                    .contracted_a = {1, 3, 2},
                    .dims_b = {ket_bond_l, dim_phys, bond_below, ket_bond_r},
                    .contracted_b = {1, 2, 3},
                    .transforms = {.conj_b = true},
                },
                {.src = samp.tmp_a(), .scratch = samp.tmp_b()},
                {.src = {ket_site, ket_stride}},
                {.view = {env_unsampled_out, samp.env_unsampled().stride}},
                dim_batch
            );
            if (not res) return false;
        }

        const CuNormalizeLogArgs norm_args{
            .x = env_unsampled_out,
            .n = ket_bond_l * env_bond_l * ket_bond_l,
            .stride = samp.env_unsampled().stride,
            .lognorm_acc = nullptr,
            .dim_batch = dim_batch,
        };
        cu_normalize_log<<<static_cast<u32>(dim_batch), k_tree_reduce_threads, 0, la.stream()>>>(
            norm_args
        );
    }
    return err_state() == QNPEPS_OK;
}

}

#endif
