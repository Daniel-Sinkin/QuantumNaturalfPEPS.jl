#ifndef QNPEPS_SAMPLER_SWEEP_DRAW_CUH
#define QNPEPS_SAMPLER_SWEEP_DRAW_CUH

#include "sampler/sweep/common.cuh"

namespace qnpeps::sampler::internal
{

[[nodiscard]] inline auto draw_sigma(
    Sampler& samp,
    const SamplerConfig& cfg,
    int row,
    bool has_below,
    const std::vector<int>& ket_bonds,
    const std::vector<int>& env_bonds,
    u64* device_seed
) -> bool
{
    const auto dim_batch = cfg.dim_batch;
    const auto dim_bond = cfg.dim_bond;
    const auto dim_phys = cfg.dim_phys;
    const auto num_cols = static_cast<usize>(cfg.ly);
    const auto row_u = static_cast<usize>(row);

    auto& la = samp.linalg();

    const auto fill_first_one_args = CuFillFirstOneArgs{
        .x = samp.sigma().p,
        .stride = samp.sigma().stride,
        .n = 1,
        .dim_batch = dim_batch,
    };
    cu_fill_first_one<<<grid_blocks_capped(dim_batch), k_threads_per_block, 0, la.stream()>>>(
        fill_first_one_args
    );

    for (auto col = 0_uz; col < num_cols; ++col)
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
        auto* dlenv_sigma = samp.dl_unit_ptrs();
        if (has_below) dlenv_sigma = samp.dlenv_sigma_ptrs()[row_u][col];

        if (row == 0)
        {
            const auto res = contract_batched(
                la,
                samp.permutation_cache(),
                {
                    .dims_a = {ket_bond_l, env_bond_l, ket_bond_l},
                    .contracted_a = {0},
                    .dims_b = {ket_bond_l, dim_phys, bond_below, ket_bond_r},
                    .contracted_b = {0},
                },
                {.src = samp.sigma(), .scratch = samp.tmp_a(), .ptrs = samp.tmp_a_ptrs()},
                {.ptrs = samp.ket_row0_ptrs()[col]},
                {.ptrs = samp.tmp_b_ptrs()},
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
                    .dims_a = {ket_bond_l, env_bond_l, ket_bond_l},
                    .contracted_a = {0},
                    .dims_b = {ket_bond_l, dim_phys, bond_below, ket_bond_r},
                    .contracted_b = {0},
                },
                {.src = samp.sigma(), .scratch = samp.tmp_a()},
                {.src = {ket_site, ket_stride}},
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
                    .dims_a = {env_bond_l, ket_bond_l, dim_phys, bond_below, ket_bond_r},
                    .contracted_a = {3, 0},
                    .dims_b = {bond_below, env_bond_l, bond_below, env_bond_r},
                    .contracted_b = {0, 1},
                },
                {.src = samp.tmp_b(), .scratch = samp.tmp_a(), .ptrs = samp.tmp_a_ptrs()},
                {.ptrs = dlenv_sigma},
                {.ptrs = samp.tmp_b_ptrs()},
                dim_batch
            );
            if (not res) return false;
        }

        {
            const auto res = contract_strided(
                la,
                samp.permutation_cache(),
                {
                    .dims_a = {ket_bond_l, dim_phys, ket_bond_r, bond_below, env_bond_r},
                    .contracted_a = {0, 3},
                    .dims_b = {ket_bond_l, dim_phys, bond_below, ket_bond_r},
                    .contracted_b = {0, 2},
                    .transforms = {.conj_b = true},
                },
                {.src = samp.tmp_b(), .scratch = samp.tmp_a()},
                {.src = {ket_site, ket_stride}, .scratch = samp.tmp_b()},
                {.view = samp.sigma_full()},
                dim_batch
            );
            if (not res) return false;
        }

        const int sigma_elems{ket_bond_r * env_bond_r * ket_bond_r};
        const auto next_offset = (col_i + 1) * samp.max_env_unsampled();
        const auto* env_unsampled_next = samp.env_unsampled().p + next_offset;
        {
            const auto res = contract_strided(
                la,
                samp.permutation_cache(),
                {
                    .dims_a = {dim_phys, ket_bond_r, env_bond_r, dim_phys, ket_bond_r},
                    .contracted_a = {1, 2, 4},
                    .dims_b = {ket_bond_r, env_bond_r, ket_bond_r},
                    .contracted_b = {0, 1, 2},
                },
                {.src = samp.sigma_full(), .scratch = samp.sigma_full_scratch()},
                {.src = {env_unsampled_next, samp.env_unsampled().stride}},
                {.view = samp.rho()},
                dim_batch
            );
            if (not res) return false;
        }
        std::swap(samp.sigma_full().p, samp.sigma_full_scratch().p);

        const int site_counter{row * cfg.ly + static_cast<int>(col)};
        const CuDrawArgs draw_args{
            .rho = samp.rho().p,
            .dim_phys = dim_phys,
            .stride_rho = samp.rho().stride,
            .seed_ptr = device_seed,
            .site_counter = site_counter,
            .lane_base = cfg.lane_base,
            .samples_site = samp.samples() + site_counter,
            .sample_stride = cfg.num_sites(),
            .logpc = samp.logpc(),
            .chosen_spins = samp.drawn_spin(),
            .dim_batch = dim_batch,
        };
        cu_draw<<<grid_blocks_exact(dim_batch), k_threads_per_block, 0, la.stream()>>>(draw_args);
        copy_device_async(
            la,
            samp.row_spins() + col_i * cfg.row_spin_stride,
            samp.drawn_spin(),
            static_cast<usize>(dim_batch)
        );
        const CuProjectArgs project_args{
            .sigma_full = samp.sigma_full().p,
            .rho = samp.rho().p,
            .sigma = samp.sigma().p,
            .chosen_spins = samp.drawn_spin(),
            .dim_phys = dim_phys,
            .sigma_elems = sigma_elems,
            .stride_full = samp.sigma_full().stride,
            .stride_rho = samp.rho().stride,
            .stride_out = samp.sigma().stride,
            .dim_batch = dim_batch,
        };
        cu_project<<<
            grid_blocks_capped(static_cast<i64>(sigma_elems) * dim_batch),
            k_threads_per_block,
            0,
            la.stream()>>>(project_args);
    }
    return err_state() == QNPEPS_OK;
}

}

#endif
