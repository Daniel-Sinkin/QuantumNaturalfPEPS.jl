#include "core/qnpeps_ctx.cuh"
#include "linalg/transfer.cuh"
#include "peps/peps.cuh"
#include "sampler/kernels.cuh"
#include "tensor/contraction.cuh"
#include "tensor/permutation.cuh"

#include <algorithm>
#include <cstdint>
#include <utility>

struct QnpepsSweepSiteArgs
{
    uint32_t struct_size;
    int32_t row;
    int32_t col;
    int32_t has_below;
    int32_t env_above_cur;
    int32_t bond_above_l;
    int32_t bond_above_r;
    int32_t ket_bond_l;
    int32_t ket_bond_r;
    int32_t env_bond_l;
    int32_t env_bond_r;
    int32_t full_bond_l;
    int32_t full_bond_r;
};

static_assert(sizeof(QnpepsSweepSiteArgs) == 13 * sizeof(uint32_t));

namespace
{
using namespace qnpeps;

[[nodiscard]] auto require_sampler(qnpeps_ctx* ctx) -> Sampler*
{
    if (not ctx)
    {
        set_err(QNPEPS_ERR_NULL_ARG);
        return nullptr;
    }
    const auto invalid_state = not ctx->sampler.ready() or not ctx->sampler.execution.host_managed
                               or not ctx->sampler.execution.host_pointers;
    if (invalid_state)
    {
        set_err(QNPEPS_ERR_BAD_CONFIG);
        return nullptr;
    }
    return &ctx->sampler.samp;
}
[[nodiscard]] auto require_site(qnpeps_ctx* ctx, const QnpepsSweepSiteArgs* args) -> Sampler*
{
    auto* samp = require_sampler(ctx);
    if (not samp) return nullptr;
    if (not args)
    {
        set_err(QNPEPS_ERR_NULL_ARG);
        return nullptr;
    }
    const auto& cfg = samp->cfg();
    if (args->struct_size != sizeof(QnpepsSweepSiteArgs))
    {
        set_err(QNPEPS_ERR_BAD_VERSION);
        return nullptr;
    }
    const auto invalid_site =
        args->row < 0 or args->row >= cfg.lx or args->col < 0 or args->col >= cfg.ly
        or args->has_below != (args->row + 1 < cfg.lx) or args->env_above_cur < 0
        or args->env_above_cur > 1 or args->bond_above_l < 1 or args->bond_above_r < 1
        or args->ket_bond_l < 1 or args->ket_bond_r < 1 or args->env_bond_l < 1
        or args->env_bond_r < 1 or args->full_bond_l < 1 or args->full_bond_r < 1;
    if (invalid_site)
    {
        set_err(QNPEPS_ERR_BAD_CONFIG);
        return nullptr;
    }
    return samp;
}

[[nodiscard]] auto build_ket_site(
    Sampler& samp, const SamplerConfig& cfg, const QnpepsSweepSiteArgs& args
) -> bool
{
    const auto dim_batch = cfg.dim_batch;
    const auto row_u = static_cast<usize>(args.row);
    const auto col_u = static_cast<usize>(args.col);
    const auto col_i = static_cast<i64>(args.col);
    auto& la = samp.linalg();

    if (args.col == 0)
    {
        const auto fill_first_one_args = CuFillFirstOneArgs{
            .x = samp.rfactor().p,
            .stride = samp.rfactor().stride,
            .n = 1,
            .dim_batch = dim_batch,
        };
        cu_fill_first_one<<<grid_blocks_capped(dim_batch), k_threads_per_block, 0, la.stream()>>>(
            fill_first_one_args
        );
    }

    const auto& peps_shape = samp.peps_shapes()[row_u][col_u];
    const auto bond_left = peps_shape[0];
    const auto bond_up = peps_shape[3];
    const auto phys_site = peps_shape[4];
    const auto bond_down = peps_shape[1];
    const auto bond_right = peps_shape[2];
    const auto& environment = samp.env_above()[static_cast<usize>(args.env_above_cur)];
    const auto environment_offset = col_i * samp.max_env_above_site();
    const auto* environment_site = environment.p + environment_offset;

    auto contracted = contract_strided(
        la,
        samp.permutation_cache(),
        {
            .dims_a = {args.ket_bond_l, args.bond_above_l, bond_left},
            .contracted_a = {1},
            .dims_b = {args.bond_above_l, cfg.dim_bond, args.bond_above_r},
            .contracted_b = {0},
        },
        {.src = samp.rfactor(), .scratch = samp.tmp_a()},
        {.src = {environment_site, environment.stride}},
        {.view = samp.tmp_b()},
        dim_batch
    );
    if (not contracted)
    {
        return false;
    }

    contracted = contract_batched(
        la,
        samp.permutation_cache(),
        {
            .dims_a = {args.ket_bond_l, bond_left, bond_up, args.bond_above_r},
            .contracted_a = {1, 2},
            .dims_b = {bond_left, bond_up, phys_site, bond_down, bond_right},
            .contracted_b = {0, 1},
        },
        {.src = samp.tmp_b(), .scratch = samp.tmp_a(), .ptrs = samp.tmp_a_ptrs()},
        {.ptrs = samp.mpo_ptrs()[row_u][col_u]},
        {.ptrs = samp.tmp_b_ptrs()},
        dim_batch
    );
    if (not contracted)
    {
        return false;
    }

    permute_batched(
        samp.permutation_cache(),
        {
            .dst = samp.reduce_input(),
            .src = samp.tmp_b(),
            .dims_in = {args.ket_bond_l, args.bond_above_r, phys_site, bond_down, bond_right},
            .perm = {0, 2, 3, 1, 4},
            .batch_count = dim_batch,
        },
        la.stream()
    );
    if (err_state() != QNPEPS_OK) return false;

    const int reduce_rows{args.ket_bond_l * phys_site * bond_down};
    const int reduce_cols{args.bond_above_r * bond_right};
    auto* ket_site = samp.ket().p + col_i * samp.max_ket_site();
    samp.reduce(
        {samp.reduce_input(), reduce_rows, reduce_cols},
        args.ket_bond_r,
        {ket_site, samp.ket().stride, reduce_rows, args.ket_bond_r},
        {samp.rfactor_next(), args.ket_bond_r, reduce_cols},
        dim_batch
    );
    if (err_state() != QNPEPS_OK) return false;

    const CuNormalizeLogArgs norm_args{
        .x = samp.rfactor_next().p,
        .n = args.ket_bond_r * reduce_cols,
        .stride = samp.rfactor_next().stride,
        .lognorm_acc = nullptr,
        .dim_batch = dim_batch,
    };
    cu_normalize_log<<<static_cast<u32>(dim_batch), k_tree_reduce_threads, 0, la.stream()>>>(
        norm_args
    );
    std::swap(samp.rfactor().p, samp.rfactor_next().p);
    std::swap(samp.rfactor().stride, samp.rfactor_next().stride);
    return err_state() == QNPEPS_OK;
}

[[nodiscard]] auto build_env_unsampled_site(
    Sampler& samp, const SamplerConfig& cfg, const QnpepsSweepSiteArgs& args
) -> bool
{
    if (args.col < 1)
    {
        set_err(QNPEPS_ERR_BAD_CONFIG);
        return false;
    }
    const auto dim_batch = cfg.dim_batch;
    const auto num_cols = static_cast<usize>(cfg.ly);
    const auto row_u = static_cast<usize>(args.row);
    const auto col_u = static_cast<usize>(args.col);
    const auto col_i = static_cast<i64>(args.col);
    const int bond_below{args.has_below ? cfg.dim_bond : 1};
    auto& la = samp.linalg();

    if (col_u + 1 == num_cols)
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

    const auto* stored_ket = samp.ket().p + col_i * samp.max_ket_site();
    const auto* ket_site = args.row == 0 ? samp.ket_row0()[col_u] : stored_ket;
    const i64 ket_stride{args.row == 0 ? 0 : samp.ket().stride};
    auto* dlenv_environment = samp.dl_unit_ptrs();
    if (args.has_below) dlenv_environment = samp.dlenv_env_ptrs()[row_u][col_u];
    const auto input_offset = (col_i + 1) * samp.max_env_unsampled();
    const auto* env_unsampled_in = samp.env_unsampled().p + input_offset;
    auto* env_unsampled_out = samp.env_unsampled().p + col_i * samp.max_env_unsampled();

    const ContractSpec ket_env_spec{
        .dims_a = {args.ket_bond_l, cfg.dim_phys, bond_below, args.ket_bond_r},
        .contracted_a = {3},
        .dims_b = {args.ket_bond_r, args.env_bond_r, args.ket_bond_r},
        .contracted_b = {0},
    };
    bool contracted{};
    if (args.row == 0)
    {
        contracted = contract_batched(
            la,
            samp.permutation_cache(),
            ket_env_spec,
            {.ptrs = samp.ket_row0_ptrs()[col_u]},
            {.ptrs = samp.envu_ptrs()[col_u + 1]},
            {.ptrs = samp.tmp_a_ptrs()},
            dim_batch
        );
        if (not contracted)
        {
            return false;
        }
    }
    else
    {
        contracted = contract_strided(
            la,
            samp.permutation_cache(),
            ket_env_spec,
            {.src = {ket_site, ket_stride}},
            {.src = {env_unsampled_in, samp.env_unsampled().stride}},
            {.view = samp.tmp_a()},
            dim_batch
        );
        if (not contracted) return false;
    }

    contracted = contract_batched(
        la,
        samp.permutation_cache(),
        {
            .dims_a = {args.ket_bond_l, cfg.dim_phys, bond_below, args.env_bond_r, args.ket_bond_r},
            .contracted_a = {2, 3},
            .dims_b = {bond_below, args.env_bond_r, bond_below, args.env_bond_l},
            .contracted_b = {0, 1},
        },
        {.src = samp.tmp_a(), .scratch = samp.tmp_b(), .ptrs = samp.tmp_b_ptrs()},
        {.ptrs = dlenv_environment},
        {.ptrs = samp.tmp_a_ptrs()},
        dim_batch
    );
    if (not contracted)
    {
        return false;
    }

    const ContractSpec close_spec{
        .dims_a = {args.ket_bond_l, cfg.dim_phys, args.ket_bond_r, bond_below, args.env_bond_l},
        .contracted_a = {1, 3, 2},
        .dims_b = {args.ket_bond_l, cfg.dim_phys, bond_below, args.ket_bond_r},
        .contracted_b = {1, 2, 3},
        .transforms = {.conj_b = true},
    };
    if (args.row == 0)
    {
        contracted = contract_batched(
            la,
            samp.permutation_cache(),
            close_spec,
            {.src = samp.tmp_a(), .scratch = samp.tmp_b(), .ptrs = samp.tmp_b_ptrs()},
            {.ptrs = samp.ket_row0_ptrs()[col_u]},
            {.ptrs = samp.envu_ptrs()[col_u]},
            dim_batch
        );
        if (not contracted)
        {
            return false;
        }
    }
    else
    {
        contracted = contract_strided(
            la,
            samp.permutation_cache(),
            close_spec,
            {.src = samp.tmp_a(), .scratch = samp.tmp_b()},
            {.src = {ket_site, ket_stride}},
            {.view = {env_unsampled_out, samp.env_unsampled().stride}},
            dim_batch
        );
        if (not contracted) return false;
    }

    const CuNormalizeLogArgs norm_args{
        .x = env_unsampled_out,
        .n = args.ket_bond_l * args.env_bond_l * args.ket_bond_l,
        .stride = samp.env_unsampled().stride,
        .lognorm_acc = nullptr,
        .dim_batch = dim_batch,
    };
    cu_normalize_log<<<static_cast<u32>(dim_batch), k_tree_reduce_threads, 0, la.stream()>>>(
        norm_args
    );
    return err_state() == QNPEPS_OK;
}

[[nodiscard]] auto draw_sigma_site(
    qnpeps_ctx& ctx, Sampler& samp, const SamplerConfig& cfg, const QnpepsSweepSiteArgs& args
) -> bool
{
    const auto dim_batch = cfg.dim_batch;
    const auto row_u = static_cast<usize>(args.row);
    const auto col_u = static_cast<usize>(args.col);
    const auto col_i = static_cast<i64>(args.col);
    const int bond_below{args.has_below ? cfg.dim_bond : 1};
    auto& la = samp.linalg();

    if (args.col == 0)
    {
        const auto fill_first_one_args = CuFillFirstOneArgs{
            .x = samp.sigma().p,
            .stride = samp.sigma().stride,
            .n = 1,
            .dim_batch = dim_batch,
        };
        cu_fill_first_one<<<grid_blocks_capped(dim_batch), k_threads_per_block, 0, la.stream()>>>(
            fill_first_one_args
        );
    }

    const auto* stored_ket = samp.ket().p + col_i * samp.max_ket_site();
    const auto* ket_site = args.row == 0 ? samp.ket_row0()[col_u] : stored_ket;
    const i64 ket_stride{args.row == 0 ? 0 : samp.ket().stride};
    auto* dlenv_sigma = samp.dl_unit_ptrs();
    if (args.has_below) dlenv_sigma = samp.dlenv_sigma_ptrs()[row_u][col_u];

    const ContractSpec left_spec{
        .dims_a = {args.ket_bond_l, args.env_bond_l, args.ket_bond_l},
        .contracted_a = {0},
        .dims_b = {args.ket_bond_l, cfg.dim_phys, bond_below, args.ket_bond_r},
        .contracted_b = {0},
    };
    bool contracted{};
    if (args.row == 0)
    {
        contracted = contract_batched(
            la,
            samp.permutation_cache(),
            left_spec,
            {.src = samp.sigma(), .scratch = samp.tmp_a(), .ptrs = samp.tmp_a_ptrs()},
            {.ptrs = samp.ket_row0_ptrs()[col_u]},
            {.ptrs = samp.tmp_b_ptrs()},
            dim_batch
        );
        if (not contracted)
        {
            return false;
        }
    }
    else
    {
        contracted = contract_strided(
            la,
            samp.permutation_cache(),
            left_spec,
            {.src = samp.sigma(), .scratch = samp.tmp_a()},
            {.src = {ket_site, ket_stride}},
            {.view = samp.tmp_b()},
            dim_batch
        );
        if (not contracted) return false;
    }

    contracted = contract_batched(
        la,
        samp.permutation_cache(),
        {
            .dims_a = {args.env_bond_l, args.ket_bond_l, cfg.dim_phys, bond_below, args.ket_bond_r},
            .contracted_a = {3, 0},
            .dims_b = {bond_below, args.env_bond_l, bond_below, args.env_bond_r},
            .contracted_b = {0, 1},
        },
        {.src = samp.tmp_b(), .scratch = samp.tmp_a(), .ptrs = samp.tmp_a_ptrs()},
        {.ptrs = dlenv_sigma},
        {.ptrs = samp.tmp_b_ptrs()},
        dim_batch
    );
    if (not contracted)
    {
        return false;
    }

    contracted = contract_strided(
        la,
        samp.permutation_cache(),
        {
            .dims_a = {args.ket_bond_l, cfg.dim_phys, args.ket_bond_r, bond_below, args.env_bond_r},
            .contracted_a = {0, 3},
            .dims_b = {args.ket_bond_l, cfg.dim_phys, bond_below, args.ket_bond_r},
            .contracted_b = {0, 2},
            .transforms = {.conj_b = true},
        },
        {.src = samp.tmp_b(), .scratch = samp.tmp_a()},
        {.src = {ket_site, ket_stride}, .scratch = samp.tmp_b()},
        {.view = samp.sigma_full()},
        dim_batch
    );
    if (not contracted)
    {
        return false;
    }

    const int sigma_elems{args.ket_bond_r * args.env_bond_r * args.ket_bond_r};
    const auto next_offset = (col_i + 1) * samp.max_env_unsampled();
    const auto* env_unsampled_next = samp.env_unsampled().p + next_offset;
    contracted = contract_strided(
        la,
        samp.permutation_cache(),
        {
            .dims_a =
                {cfg.dim_phys, args.ket_bond_r, args.env_bond_r, cfg.dim_phys, args.ket_bond_r},
            .contracted_a = {1, 2, 4},
            .dims_b = {args.ket_bond_r, args.env_bond_r, args.ket_bond_r},
            .contracted_b = {0, 1, 2},
        },
        {.src = samp.sigma_full(), .scratch = samp.sigma_full_scratch()},
        {.src = {env_unsampled_next, samp.env_unsampled().stride}},
        {.view = samp.rho()},
        dim_batch
    );
    if (not contracted)
    {
        return false;
    }
    std::swap(samp.sigma_full().p, samp.sigma_full_scratch().p);

    const int site_counter{args.row * cfg.ly + args.col};
    const CuDrawArgs draw_args{
        .rho = samp.rho().p,
        .dim_phys = cfg.dim_phys,
        .stride_rho = samp.rho().stride,
        .seed_ptr = ctx.sampler.allocation.device_seed,
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
        .dim_phys = cfg.dim_phys,
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
    return err_state() == QNPEPS_OK;
}

[[nodiscard]] auto build_env_above_site(
    Sampler& samp, const SamplerConfig& cfg, const QnpepsSweepSiteArgs& args
) -> bool
{
    if (not args.has_below)
    {
        set_err(QNPEPS_ERR_BAD_CONFIG);
        return false;
    }
    const auto dim_batch = cfg.dim_batch;
    const auto row_u = static_cast<usize>(args.row);
    const auto col_u = static_cast<usize>(args.col);
    const auto col_i = static_cast<i64>(args.col);
    const int env_above_next{1 - args.env_above_cur};
    auto& la = samp.linalg();

    if (cfg.fast_mode or args.row == 0)
    {
        const auto* stored_ket = samp.ket().p + col_i * samp.max_ket_site();
        const auto* ket_site = args.row == 0 ? samp.ket_row0()[col_u] : stored_ket;
        const i64 ket_stride{args.row == 0 ? 0 : samp.ket().stride};
        auto* out = samp.env_above()[static_cast<usize>(env_above_next)].p
                    + col_i * samp.max_env_above_site();
        const auto slice_elems = static_cast<i64>(args.ket_bond_l) * cfg.dim_bond * args.ket_bond_r;
        const CuSliceKetArgs slice_args{
            .out = out,
            .ket = ket_site,
            .chosen_spins = samp.row_spins() + col_i * cfg.row_spin_stride,
            .ket_bond_l = args.ket_bond_l,
            .dim_phys = cfg.dim_phys,
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
            .n = args.ket_bond_l * cfg.dim_bond * args.ket_bond_r,
            .stride = samp.env_above()[static_cast<usize>(env_above_next)].stride,
            .lognorm_acc = samp.lognorm(),
            .dim_batch = dim_batch,
        };
        cu_normalize_log<<<static_cast<u32>(dim_batch), k_tree_reduce_threads, 0, la.stream()>>>(
            norm_args
        );
        return err_state() == QNPEPS_OK;
    }

    if (args.col == 0)
    {
        const auto fill_first_one_args = CuFillFirstOneArgs{
            .x = samp.rfactor().p,
            .stride = samp.rfactor().stride,
            .n = 1,
            .dim_batch = dim_batch,
        };
        cu_fill_first_one<<<grid_blocks_capped(dim_batch), k_threads_per_block, 0, la.stream()>>>(
            fill_first_one_args
        );
    }
    const auto& peps_shape = samp.peps_shapes()[row_u][col_u];
    const auto bond_left = peps_shape[0];
    const auto bond_up = peps_shape[3];
    const auto bond_down = peps_shape[1];
    const auto bond_right = peps_shape[2];
    const auto projected_elems = bond_left * bond_up * bond_down * bond_right;
    const CuProjectMpoArgs project_args{
        .out = samp.tmp_b().p,
        .mpo = samp.mpo()[row_u][col_u],
        .chosen_spins = samp.row_spins() + col_i * cfg.row_spin_stride,
        .spin_block = bond_left * bond_up,
        .dim_phys = cfg.dim_phys,
        .output_elems = projected_elems,
        .stride_out = samp.tmp_b().stride,
        .dim_batch = dim_batch,
    };
    cu_project_mpo<<<
        grid_blocks_capped(static_cast<i64>(projected_elems) * dim_batch),
        k_threads_per_block,
        0,
        la.stream()>>>(project_args);

    const auto& environment = samp.env_above()[static_cast<usize>(args.env_above_cur)];
    const auto environment_offset = col_i * samp.max_env_above_site();
    const auto* environment_site = environment.p + environment_offset;
    auto contracted = contract_strided(
        la,
        samp.permutation_cache(),
        {
            .dims_a = {args.full_bond_l, args.bond_above_l, bond_left},
            .contracted_a = {1},
            .dims_b = {args.bond_above_l, bond_up, args.bond_above_r},
            .contracted_b = {0},
        },
        {.src = samp.rfactor(), .scratch = samp.tmp_a()},
        {.src = {environment_site, environment.stride}},
        {.view = samp.reduce_input()},
        dim_batch
    );
    if (not contracted)
    {
        return false;
    }
    contracted = contract_strided(
        la,
        samp.permutation_cache(),
        {
            .dims_a = {args.full_bond_l, bond_left, bond_up, args.bond_above_r},
            .contracted_a = {1, 2},
            .dims_b = {bond_left, bond_up, bond_down, bond_right},
            .contracted_b = {0, 1},
        },
        {.src = samp.reduce_input(), .scratch = samp.tmp_a()},
        {.src = samp.tmp_b()},
        {.view = samp.reduce_input()},
        dim_batch
    );
    if (not contracted)
    {
        return false;
    }
    permute_batched(
        samp.permutation_cache(),
        {
            .dst = samp.tmp_a(),
            .src = samp.reduce_input(),
            .dims_in = {args.full_bond_l, args.bond_above_r, bond_down, bond_right},
            .perm = {0, 2, 1, 3},
            .batch_count = dim_batch,
        },
        la.stream()
    );
    if (err_state() != QNPEPS_OK) return false;
    const auto reduce_rows = args.full_bond_l * bond_down;
    const auto reduce_cols = args.bond_above_r * bond_right;
    auto* out =
        samp.env_above()[static_cast<usize>(env_above_next)].p + col_i * samp.max_env_above_site();
    samp.reduce(
        {samp.tmp_a(), reduce_rows, reduce_cols},
        args.full_bond_r,
        {out,
         samp.env_above()[static_cast<usize>(env_above_next)].stride,
         reduce_rows,
         args.full_bond_r},
        {samp.rfactor_next(), args.full_bond_r, reduce_cols},
        dim_batch
    );
    if (err_state() != QNPEPS_OK) return false;
    const CuNormalizeLogArgs norm_args{
        .x = samp.rfactor_next().p,
        .n = args.full_bond_r * reduce_cols,
        .stride = samp.rfactor_next().stride,
        .lognorm_acc = samp.lognorm(),
        .dim_batch = dim_batch,
    };
    cu_normalize_log<<<static_cast<u32>(dim_batch), k_tree_reduce_threads, 0, la.stream()>>>(
        norm_args
    );
    std::swap(samp.rfactor().p, samp.rfactor_next().p);
    std::swap(samp.rfactor().stride, samp.rfactor_next().stride);
    return err_state() == QNPEPS_OK;
}
}

#define QNPEPS_INTERNAL_EXPORT __attribute__((visibility("default")))

extern "C" QNPEPS_INTERNAL_EXPORT qnpeps_status
qnpeps_sweep_begin(qnpeps_ctx* ctx, const uint64_t* batch_seed)
{
    qnpeps::reset_err();
    auto* samp = require_sampler(ctx);
    if (not samp) return qnpeps::err_state();
    if (not batch_seed) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    auto& la = samp->linalg();
    const auto dim_batch = static_cast<qnpeps::usize>(samp->cfg().dim_batch);
    upload_async(la, ctx->sampler.allocation.device_seed, batch_seed, 1);
    zero_async(la, samp->logpc(), dim_batch);
    zero_async(la, samp->lognorm(), dim_batch);
    zero_async(la, samp->fail(), 1);
    return qnpeps::err_state();
}

extern "C" QNPEPS_INTERNAL_EXPORT qnpeps_status
qnpeps_sweep_build_ket_site(qnpeps_ctx* ctx, const QnpepsSweepSiteArgs* args)
{
    qnpeps::reset_err();
    auto* samp = require_site(ctx, args);
    if (not samp) return qnpeps::err_state();
    if (args->row == 0) return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
    if (not build_ket_site(*samp, samp->cfg(), *args) and qnpeps::err_state() == QNPEPS_OK)
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
    return qnpeps::err_state();
}

extern "C" QNPEPS_INTERNAL_EXPORT qnpeps_status
qnpeps_sweep_build_env_unsampled_site(qnpeps_ctx* ctx, const QnpepsSweepSiteArgs* args)
{
    qnpeps::reset_err();
    auto* samp = require_site(ctx, args);
    if (not samp) return qnpeps::err_state();
    const auto built = build_env_unsampled_site(*samp, samp->cfg(), *args);
    if (not built and qnpeps::err_state() == QNPEPS_OK)
    {
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
    }
    return qnpeps::err_state();
}

extern "C" QNPEPS_INTERNAL_EXPORT qnpeps_status
qnpeps_sweep_draw_sigma_site(qnpeps_ctx* ctx, const QnpepsSweepSiteArgs* args)
{
    qnpeps::reset_err();
    auto* samp = require_site(ctx, args);
    if (not samp) return qnpeps::err_state();
    if (not draw_sigma_site(*ctx, *samp, samp->cfg(), *args) and qnpeps::err_state() == QNPEPS_OK)
    {
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
    }
    return qnpeps::err_state();
}

extern "C" QNPEPS_INTERNAL_EXPORT qnpeps_status
qnpeps_sweep_build_env_above_site(qnpeps_ctx* ctx, const QnpepsSweepSiteArgs* args)
{
    qnpeps::reset_err();
    auto* samp = require_site(ctx, args);
    if (not samp) return qnpeps::err_state();
    if (not build_env_above_site(*samp, samp->cfg(), *args) and qnpeps::err_state() == QNPEPS_OK)
    {
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
    }
    return qnpeps::err_state();
}

extern "C" QNPEPS_INTERNAL_EXPORT qnpeps_status
qnpeps_sweep_finish(qnpeps_ctx* ctx, uint8_t* samples, double* logpc, double* lognorm)
{
    qnpeps::reset_err();
    auto* samp = require_sampler(ctx);
    if (not samp) return qnpeps::err_state();
    if (not samples or not logpc or not lognorm) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    auto& la = samp->linalg();
    const auto dim_batch = static_cast<qnpeps::usize>(samp->cfg().dim_batch);
    const auto sample_count = dim_batch * static_cast<qnpeps::usize>(samp->cfg().num_sites());
    download_async(la, samples, samp->samples(), sample_count);
    download_async(la, logpc, samp->logpc(), dim_batch);
    download_async(la, lognorm, samp->lognorm(), dim_batch);
    CUDA_CHECK(cudaStreamSynchronize(la.stream()));
    int fail{};
    download(&fail, samp->fail(), 1);
    if (fail != 0) return qnpeps::set_err(QNPEPS_ERR_CUDA);
    return qnpeps::err_state();
}

extern "C" QNPEPS_INTERNAL_EXPORT qnpeps_status
qnpeps_sweep_graph_policy(qnpeps_ctx* ctx, int32_t policy, int32_t* use_graph, int32_t* has_graph)
{
    qnpeps::reset_err();
    auto* samp = require_sampler(ctx);
    if (not samp) return qnpeps::err_state();
    if (policy < -1 or policy > 1) return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
    if (policy >= 0)
    {
        if (ctx->sampler.execution.graph)
        {
            CUDA_CHECK(cudaStreamSynchronize(ctx->linalg().stream()));
            CUDA_CHECK(cudaGraphExecDestroy(ctx->sampler.execution.graph));
            ctx->sampler.execution.graph = nullptr;
        }
        ctx->use_graph = policy == 1;
    }
    if (use_graph) *use_graph = ctx->use_graph ? 1 : 0;
    if (has_graph) *has_graph = ctx->sampler.execution.graph ? 1 : 0;
    return qnpeps::err_state();
}
