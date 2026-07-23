#include "contraction.cuh"
#include "peps.cuh"
#include "permutation.cuh"
#include "qnpeps_ctx.cuh"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace qnpeps
{
[[nodiscard]] static auto build_ket_row(
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

    cu_fill_first_one<<<grid_blocks_capped(dim_batch), k_threads_per_block, 0, la.stream()>>>(
        samp.rfactor().p, samp.rfactor().stride, 1, dim_batch
    );
    for (auto col = 0_uz; col < num_cols; ++col)
    {
        const auto col_i = static_cast<i64>(col);
        const Shape& peps_shape = samp.peps_shapes()[row_u][col];
        const auto bond_left = peps_shape[0];
        const auto bond_up = peps_shape[3];
        const auto phys_site = peps_shape[4];
        const auto bond_down = peps_shape[1];
        const auto bond_right = peps_shape[2];
        const auto bond_above_l = bond_above[col];
        const auto bond_above_r = bond_above[col + 1];
        const auto ket_bond_l = ket_bonds[col];
        const auto ket_bond_r = ket_bonds[col + 1];

        const auto& environment = samp.env_above()[env_above_cur];
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
        cu_normalize_log<<<static_cast<u32>(dim_batch), k_threads_per_block, 0, la.stream()>>>(
            norm_args
        );
        std::swap(samp.rfactor().p, samp.rfactor_next().p);
        std::swap(samp.rfactor().stride, samp.rfactor_next().stride);
    }
    return err_state() == QNPEPS_OK;
}

[[nodiscard]] static auto build_env_unsampled(
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
        const i64 offset = static_cast<i64>(cfg.ly) * samp.max_env_unsampled();
        auto* env_unsampled_last = samp.env_unsampled().p + offset;
        cu_fill_first_one<<<grid_blocks_capped(dim_batch), k_threads_per_block, 0, la.stream()>>>(
            env_unsampled_last, samp.env_unsampled().stride, 1, dim_batch
        );
    }
    for (usize col{num_cols - 1}; col >= 1; --col)
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
        cu_normalize_log<<<static_cast<u32>(dim_batch), k_threads_per_block, 0, la.stream()>>>(
            norm_args
        );
    }
    return err_state() == QNPEPS_OK;
}

[[nodiscard]] static auto draw_sigma(
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

    cu_fill_first_one<<<grid_blocks_capped(dim_batch), k_threads_per_block, 0, la.stream()>>>(
        samp.sigma().p, samp.sigma().stride, 1, dim_batch
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
            .samples_site = samp.samples() + site_counter,
            .sample_stride = cfg.num_sites(),
            .logpc = samp.logpc(),
            .chosen_spins = samp.drawn_spin(),
            .dim_batch = dim_batch,
        };
        cu_draw<<<grid_blocks_exact(dim_batch), k_threads_per_block, 0, la.stream()>>>(draw_args);
        CUDA_CHECK(cudaMemcpyAsync(
            samp.row_spins() + col_i * dim_batch,
            samp.drawn_spin(),
            static_cast<usize>(dim_batch) * sizeof(int),
            cudaMemcpyDeviceToDevice,
            la.stream()
        ));
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

[[nodiscard]] static auto build_env_above(
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
            auto* out = samp.env_above()[env_above_next].p + col_i * samp.max_env_above_site();
            const auto slice_elems = static_cast<i64>(ket_bond_l) * dim_bond * ket_bond_r;
            const CuSliceKetArgs slice_args{
                .out = out,
                .ket = ket_site,
                .chosen_spins = samp.row_spins() + col_i * dim_batch,
                .ket_bond_l = ket_bond_l,
                .dim_phys = dim_phys,
                .slice_elems = slice_elems,
                .stride_out = samp.env_above()[env_above_next].stride,
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
                .stride = samp.env_above()[env_above_next].stride,
                .lognorm_acc = samp.lognorm(),
                .dim_batch = dim_batch,
            };
            cu_normalize_log<<<static_cast<u32>(dim_batch), k_threads_per_block, 0, la.stream()>>>(
                norm_args
            );
        }
        bond_above_cur = ket_bonds;
    }
    else
    {
        const auto row_u = static_cast<usize>(row);
        auto full_bonds = std::vector<int>(num_cols + 1, 1);
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

        cu_fill_first_one<<<grid_blocks_capped(dim_batch), k_threads_per_block, 0, la.stream()>>>(
            samp.rfactor().p, samp.rfactor().stride, 1, dim_batch
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
            const auto project_args = CuProjectMpoArgs{
                .out = samp.tmp_b().p,
                .mpo = samp.mpo()[row_u][col],
                .chosen_spins = samp.row_spins() + col_i * dim_batch,
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

            const auto& environment = samp.env_above()[env_above_cur];
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
            auto* out = samp.env_above()[env_above_next].p + col_i * samp.max_env_above_site();
            samp.reduce(
                {samp.tmp_a(), reduce_rows, reduce_cols},
                full_right,
                {out, samp.env_above()[env_above_next].stride, reduce_rows, full_right},
                {samp.rfactor_next(), full_right, reduce_cols},
                dim_batch
            );
            if (err_state() != QNPEPS_OK) return false;
            const auto norm_args = CuNormalizeLogArgs{
                .x = samp.rfactor_next().p,
                .n = full_right * reduce_cols,
                .stride = samp.rfactor_next().stride,
                .lognorm_acc = samp.lognorm(),
                .dim_batch = dim_batch,
            };
            cu_normalize_log<<<static_cast<u32>(dim_batch), k_threads_per_block, 0, la.stream()>>>(
                norm_args
            );
            std::swap(samp.rfactor().p, samp.rfactor_next().p);
            std::swap(samp.rfactor().stride, samp.rfactor_next().stride);
        }
        bond_above_cur = std::move(full_bonds);
    }
    env_above_cur = env_above_next;
    return err_state() == QNPEPS_OK;
}
namespace sampler
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

    Sampler& samp = ctx.sampler.samp;
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

auto ctx_sample_run(
    qnpeps_ctx& ctx, const std::vector<int>& batch_ids, HostSampleOutput* host_output
) -> void
{
    if (not ctx.sampler.ready())
    {
        set_err(QNPEPS_ERR_INTERNAL);
        return;
    }
    Sampler& samp = ctx.sampler.samp;
    SamplerConfig& cfg = samp.cfg();

    if (ctx.sampler.execution.dim_batch > 0 and cfg.dim_batch != ctx.sampler.execution.dim_batch)
    {
        if (ctx.sampler.execution.graph)
        {
            CUDA_NOCHECK(cudaGraphExecDestroy(ctx.sampler.execution.graph));
            ctx.sampler.execution.graph = nullptr;
        }
        cfg.dim_batch = ctx.sampler.execution.dim_batch;
    }

    const auto dim_batch = cfg.dim_batch;
    const auto dim_bond = cfg.dim_bond;
    const int chi_s{cfg.chi_s};
    const auto num_rows = static_cast<usize>(cfg.lx);
    const auto num_cols = static_cast<usize>(cfg.ly);
    const i64 lane_samples{static_cast<i64>(dim_batch) * cfg.num_sites()};
    auto& la = samp.linalg();
    auto* device_seed = ctx.sampler.allocation.device_seed;
    auto& all_samples = ctx.sampler.staging.all_samples;
    auto& all_logpc = ctx.sampler.staging.all_logpc;
    auto& all_lognorm = ctx.sampler.staging.all_lognorm;

    const auto lane_samples_u = static_cast<usize>(lane_samples);
    const auto dim_batch_u = static_cast<usize>(dim_batch);
    if (host_output)
    {
        if (not host_output->samples or host_output->n_samples == 0)
        {
            set_err(QNPEPS_ERR_NULL_ARG);
            return;
        }
    }
    else
    {
        all_samples.clear();
        all_logpc.clear();
        all_lognorm.clear();
        all_samples.reserve(lane_samples_u * batch_ids.size());
        all_logpc.reserve(dim_batch_u * batch_ids.size());
        all_lognorm.reserve(dim_batch_u * batch_ids.size());
    }

    {
        const auto num_env_rows = num_rows - 1;
        const auto lane_capacity = static_cast<usize>(ctx.sampler.allocation.dim_batch_capacity);
        auto* device_sampling = ctx.dlenv.lanes[ctx.dlenv.active_lane].sampling;
        const auto layout_count = qnpeps::dlenv::k_sampling_layout_count;
        ctx.dlenv.ptr_host.assign(layout_count * num_env_rows * num_cols * lane_capacity, nullptr);
        usize pointer_slot{};
        const auto fill_broadcast_pointers = [&](i64 value_offset)
        {
            for (auto lane = 0_uz; lane < lane_capacity; ++lane)
                ctx.dlenv.ptr_host[pointer_slot * lane_capacity + lane] =
                    device_sampling + value_offset;
            pointer_slot += 1;
        };
        for (auto row = 0_uz; row < num_env_rows; ++row)
            for (auto col = 0_uz; col < num_cols; ++col)
                fill_broadcast_pointers(ctx.dlenv.env_off[row][col]);
        for (auto row = 0_uz; row < num_env_rows; ++row)
            for (auto col = 0_uz; col < num_cols; ++col)
                fill_broadcast_pointers(ctx.dlenv.sigma_off[row][col]);
        if (num_env_rows > 0)
        {
            copy_h2d_async(
                samp.dlenv_env_ptrs()[0][0],
                ctx.dlenv.ptr_host.data(),
                ctx.dlenv.ptr_host.size(),
                la.stream()
            );
        }
        const auto expected_slots = layout_count * num_env_rows * num_cols;
        if (pointer_slot != expected_slots)
        {
            set_err(QNPEPS_ERR_INTERNAL);
            return;
        }
        assert(pointer_slot == expected_slots);
    }

    const auto ket_bonds_for_row = [&](const std::vector<int>& bond_above, usize row)
    {
        std::vector<int> bonds{};
        bonds.assign(num_cols + 1, 1);
        if (row == 0)
        {
            for (auto col = 0_uz; col <= num_cols; ++col)
                bonds[col] = bond_dim(cfg.ly, static_cast<int>(col), dim_bond);
            return bonds;
        }
        auto carried_bond = 1;
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const auto& peps_shape = samp.peps_shapes()[row][col];
            const auto reduce_rows = carried_bond * peps_shape[4] * peps_shape[1];
            const auto reduce_cols = bond_above[col + 1] * peps_shape[2];
            const auto next_bond = std::max(1, std::min({chi_s, reduce_rows, reduce_cols}));
            bonds[col + 1] = next_bond;
            carried_bond = next_bond;
        }
        bonds[num_cols] = 1;
        return bonds;
    };
    const auto environment_bonds_for_row = [&](usize row)
    {
        std::vector<int> bonds{};
        bonds.assign(num_cols + 1, 1);
        if (row + 1 < num_rows)
        {
            const auto& env_row = samp.dlenv_host()[row];
            for (auto col = 0_uz; col < num_cols; ++col)
            {
                bonds[col] = env_row.site_shapes[col][0];
                bonds[col + 1] = env_row.site_shapes[col][3];
            }
        }
        return bonds;
    };

    constexpr u64 k_seed_multiplier{1000003};
    for (auto batch_index = 0_uz; batch_index < batch_ids.size(); ++batch_index)
    {
        const auto batch_id = batch_ids[batch_index];
        const auto seed_offset = cfg.batch_base + static_cast<u64>(batch_id);
        const auto batch_seed = cfg.seed * k_seed_multiplier + seed_offset;
        ctx.sampler.staging.h_seed = batch_seed;
        copy_h2d_async(device_seed, &ctx.sampler.staging.h_seed, 1, la.stream());

        const auto enqueue_batch = [&]() -> bool
        {
            CUDA_CHECK(cudaMemsetAsync(
                samp.logpc(), 0, static_cast<usize>(dim_batch) * sizeof(f64), la.stream()
            ));
            CUDA_CHECK(cudaMemsetAsync(
                samp.lognorm(), 0, static_cast<usize>(dim_batch) * sizeof(f64), la.stream()
            ));
            CUDA_CHECK(cudaMemsetAsync(samp.fail(), 0, sizeof(int), la.stream()));

            int env_above_cur{0};
            std::vector<int> bond_above_cur{};
            bond_above_cur.assign(num_cols + 1, 1);

            for (auto row = 0; row < cfg.lx; ++row)
            {
                const bool has_below{row + 1 < cfg.lx};
                const std::vector<int> bond_above{bond_above_cur};
                const auto ket_bonds = ket_bonds_for_row(bond_above, static_cast<usize>(row));
                const auto env_bonds = environment_bonds_for_row(static_cast<usize>(row));

                if (row > 0
                    and not build_ket_row(samp, cfg, row, bond_above, ket_bonds, env_above_cur))
                {
                    return false;
                }

                if (not build_env_unsampled(samp, cfg, row, has_below, ket_bonds, env_bonds))
                    return false;

                if (not draw_sigma(samp, cfg, row, has_below, ket_bonds, env_bonds, device_seed))
                    return false;

                if (not build_env_above(
                        samp, cfg, row, has_below, ket_bonds, env_above_cur, bond_above_cur
                    ))
                {
                    return false;
                }
            }
            return err_state() == QNPEPS_OK;
        };

        if (ctx.use_graph and ctx.sampler.execution.graph)
        {
            if (std::getenv("QNPEPS_GRAPH_LOG"))
                std::fprintf(stderr, "[qnpeps] sample_graph replayed\n");
            CUDA_CHECK(cudaGraphLaunch(ctx.sampler.execution.graph, la.stream()));
        }
        else if (ctx.use_graph and ctx.sampler.execution.warmed)
        {
            cudaGraph_t graph{};
            CUDA_CHECK(cudaStreamBeginCapture(la.stream(), cudaStreamCaptureModeThreadLocal));
            const auto enqueued = enqueue_batch();
            const auto capture_status = cudaStreamEndCapture(la.stream(), &graph);
            if (enqueued and capture_status == cudaSuccess
                and instantiate_graph(ctx.sampler.execution.graph, graph) == cudaSuccess)
            {
                CUDA_CHECK(cudaGraphDestroy(graph));
                if (std::getenv("QNPEPS_GRAPH_LOG"))
                    std::fprintf(stderr, "[qnpeps] sample_graph captured\n");
                CUDA_CHECK(cudaGraphLaunch(ctx.sampler.execution.graph, la.stream()));
            }
            else
            {
                if (graph) CUDA_NOCHECK(cudaGraphDestroy(graph));
                cudaGetLastError();
                ctx.sampler.execution.graph = nullptr;
                if (enqueued and not enqueue_batch()) assert(err_state() != QNPEPS_OK);
            }
        }
        else
        {
            if (enqueue_batch()) ctx.sampler.execution.warmed = true;
        }

        if (err_state() != QNPEPS_OK) break;

        copy_d2h_async(
            ctx.sampler.staging.h_samples,
            samp.samples(),
            static_cast<usize>(lane_samples),
            la.stream()
        );
        copy_d2h_async(
            ctx.sampler.staging.h_logpc, samp.logpc(), static_cast<usize>(dim_batch), la.stream()
        );
        copy_d2h_async(
            ctx.sampler.staging.h_lognorm,
            samp.lognorm(),
            static_cast<usize>(dim_batch),
            la.stream()
        );
        CUDA_CHECK(cudaStreamSynchronize(la.stream()));
        int fail_host{};
        CUDA_CHECK(cudaMemcpy(&fail_host, samp.fail(), sizeof(int), cudaMemcpyDeviceToHost));
        if (fail_host != 0)
        {
            set_err(QNPEPS_ERR_CUDA);
            break;
        }
        if (host_output)
        {
            const auto sample_offset =
                static_cast<u64>(batch_id) * static_cast<u64>(dim_batch);
            if (sample_offset >= host_output->n_samples)
            {
                set_err(QNPEPS_ERR_INTERNAL);
                break;
            }
            const auto valid_samples = static_cast<usize>(
                std::min<u64>(static_cast<u64>(dim_batch), host_output->n_samples - sample_offset)
            );
            const auto destination_sample = static_cast<usize>(sample_offset);
            std::memcpy(
                host_output->samples + destination_sample * static_cast<usize>(cfg.num_sites()),
                ctx.sampler.staging.h_samples,
                valid_samples * static_cast<usize>(cfg.num_sites()) * sizeof(u8)
            );
            if (host_output->logpc)
            {
                std::memcpy(
                    host_output->logpc + destination_sample,
                    ctx.sampler.staging.h_logpc,
                    valid_samples * sizeof(f64)
                );
            }
            if (host_output->lognorm)
            {
                std::memcpy(
                    host_output->lognorm + destination_sample,
                    ctx.sampler.staging.h_lognorm,
                    valid_samples * sizeof(f64)
                );
            }
        }
        else
        {
            all_samples.insert(
                all_samples.end(),
                ctx.sampler.staging.h_samples,
                ctx.sampler.staging.h_samples + lane_samples
            );
            all_logpc.insert(
                all_logpc.end(),
                ctx.sampler.staging.h_logpc,
                ctx.sampler.staging.h_logpc + dim_batch
            );
            all_lognorm.insert(
                all_lognorm.end(),
                ctx.sampler.staging.h_lognorm,
                ctx.sampler.staging.h_lognorm + dim_batch
            );
        }
    }

    if (err_state() == QNPEPS_OK and not host_output)
    {
        const auto batch_count = batch_ids.size();
        assert(all_samples.size() == lane_samples_u * batch_count);
        assert(all_logpc.size() == dim_batch_u * batch_count);
        assert(all_lognorm.size() == dim_batch_u * batch_count);
    }
}
}
}
