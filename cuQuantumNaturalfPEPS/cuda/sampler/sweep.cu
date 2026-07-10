#include "qnpeps_ctx.cuh"

#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

namespace qnpeps
{
static auto build_ket_row(
    Sampler& samp,
    const SamplerConfig& cfg,
    int row,
    const std::vector<int>& bond_above,
    const std::vector<int>& ket_bonds,
    int env_above_cur
) -> void
{
    const auto dim_batch = cfg.dim_batch;
    const auto dim_bond = cfg.dim_bond;
    const auto num_cols = static_cast<usize>(cfg.ly);
    const auto row_u = static_cast<usize>(row);

    auto& la = samp.linalg();
    const auto permute = [&](PermuteOp op)
    {
        op.batch = dim_batch;
        permute_batched(la, samp.permutation_cache(), op);
    };

    cu_fill_first_one<<<grid_blocks_capped(dim_batch), k_threads_per_block, 0, la.stream()>>>(
        samp.rfactor().p, samp.rfactor().stride, 1, dim_batch
    );
    for (auto col = 0_uz; col < num_cols; ++col)
    {
        const auto col_i = static_cast<i64>(col);
        const HostTensor& mpo_site = samp.mpo_host()[row_u][col];
        const auto bond_left = mpo_site.dim[0];
        const auto bond_up = mpo_site.dim[1];
        const auto phys_site = mpo_site.dim[2];
        const auto bond_down = mpo_site.dim[3];
        const auto bond_right = mpo_site.dim[4];
        const auto bond_above_l = bond_above[col];
        const auto bond_above_r = bond_above[col + 1];
        const auto ket_bond_l = ket_bonds[col];
        const auto ket_bond_r = ket_bonds[col + 1];

        permute({
            .dst = samp.tmp_a(),
            .src = samp.rfactor(),
            .dims_in = {ket_bond_l, bond_above_l, bond_left},
            .perm = {0, 2, 1},
        });
        const auto* envA_site =
            samp.env_above()[env_above_cur].p + col_i * samp.max_env_above_site();
        {
            const auto m = ket_bond_l * bond_left;
            const auto k = bond_above_l;
            const auto n = dim_bond * bond_above_r;
            la.matmul_batched(
                {samp.tmp_a(), m, k},
                {envA_site, samp.env_above()[env_above_cur].stride, k, n},
                {samp.tmp_b(), m, n},
                dim_batch
            );
        }

        permute({
            .dst = samp.tmp_a(),
            .src = samp.tmp_b(),
            .dims_in = {ket_bond_l, bond_left, bond_up, bond_above_r},
            .perm = {0, 3, 1, 2},
        });
        {
            const auto m = ket_bond_l * bond_above_r;
            const auto k = bond_left * bond_up;
            const auto n = phys_site * bond_down * bond_right;
            la.matmul_batched_ptr(
                samp.tmp_a_ptrs(),
                m,
                k,
                samp.mpo_ptrs()[row_u][col],
                k,
                n,
                samp.tmp_b_ptrs(),
                m,
                n,
                dim_batch
            );
        }

        permute({
            .dst = samp.reduce_input(),
            .src = samp.tmp_b(),
            .dims_in = {ket_bond_l, bond_above_r, phys_site, bond_down, bond_right},
            .perm = {0, 2, 3, 1, 4},
        });
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
}

static auto build_env_unsampled(
    Sampler& samp,
    const SamplerConfig& cfg,
    int row,
    bool has_below,
    const std::vector<int>& ket_bonds,
    const std::vector<int>& env_bonds
) -> void
{
    const auto dim_batch = cfg.dim_batch;
    const auto dim_bond = cfg.dim_bond;
    const auto dim_phys = cfg.dim_phys;
    const auto num_cols = static_cast<usize>(cfg.ly);
    const auto row_u = static_cast<usize>(row);

    auto& la = samp.linalg();
    const auto permute = [&](PermuteOp op)
    {
        op.batch = dim_batch;
        permute_batched(la, samp.permutation_cache(), op);
    };

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
        const auto* ket_site =
            row == 0 ? samp.ket_row0()[col] : samp.ket().p + col_i * samp.max_ket_site();
        const i64 ket_stride{row == 0 ? 0 : samp.ket().stride};
        cf* const* dlenv_env_arr =
            has_below ? samp.dlenv_env_ptrs()[row_u][col] : samp.dl_unit_ptrs();
        const auto* env_unsampled_in =
            samp.env_unsampled().p + (col_i + 1) * samp.max_env_unsampled();
        auto* env_unsampled_out = samp.env_unsampled().p + col_i * samp.max_env_unsampled();

        {
            const auto m = ket_bond_l * dim_phys * bond_below;
            const auto k = ket_bond_r;
            const auto n = env_bond_r * ket_bond_r;
            if (row == 0)
            {
                la.matmul_batched_ptr(
                    samp.ket_row0_ptrs()[col],
                    m,
                    k,
                    samp.envu_ptrs()[col + 1],
                    k,
                    n,
                    samp.tmp_a_ptrs(),
                    m,
                    n,
                    dim_batch
                );
            }
            else
            {
                la.matmul_batched(
                    {ket_site, ket_stride, m, k},
                    {env_unsampled_in, samp.env_unsampled().stride, k, n},
                    {samp.tmp_a(), m, n},
                    dim_batch
                );
            }
        }

        permute({
            .dst = samp.tmp_b(),
            .src = samp.tmp_a(),
            .dims_in = {ket_bond_l, dim_phys, bond_below, env_bond_r, ket_bond_r},
            .perm = {0, 1, 4, 2, 3},
        });
        {
            const auto m = ket_bond_l * dim_phys * ket_bond_r;
            const auto k = bond_below * env_bond_r;
            const auto n = bond_below * env_bond_l;
            la.matmul_batched_ptr(
                samp.tmp_b_ptrs(), m, k, dlenv_env_arr, k, n, samp.tmp_a_ptrs(), m, n, dim_batch
            );
        }

        permute({
            .dst = samp.tmp_b(),
            .src = samp.tmp_a(),
            .dims_in = {ket_bond_l, dim_phys, ket_bond_r, bond_below, env_bond_l},
            .perm = {0, 4, 1, 3, 2},
        });
        {
            const auto m = ket_bond_l * env_bond_l;
            const auto k = dim_phys * bond_below * ket_bond_r;
            if (row == 0)
            {
                la.matmul_batched_ptr(
                    samp.tmp_b_ptrs(),
                    m,
                    k,
                    samp.ket_row0_ptrs()[col],
                    ket_bond_l,
                    k,
                    samp.envu_ptrs()[col],
                    m,
                    ket_bond_l,
                    dim_batch,
                    {.op_b = BlasOp::conj_trans}
                );
            }
            else
            {
                la.matmul_batched_none_adj(
                    {samp.tmp_b(), m, k},
                    {ket_site, ket_stride, ket_bond_l, k},
                    {env_unsampled_out, samp.env_unsampled().stride, m, ket_bond_l},
                    dim_batch
                );
            }
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
}

static auto draw_sigma(
    Sampler& samp,
    const SamplerConfig& cfg,
    int row,
    bool has_below,
    const std::vector<int>& ket_bonds,
    const std::vector<int>& env_bonds,
    u64* device_seed
) -> void
{
    const auto dim_batch = cfg.dim_batch;
    const auto dim_bond = cfg.dim_bond;
    const auto dim_phys = cfg.dim_phys;
    const auto num_cols = static_cast<usize>(cfg.ly);
    const auto row_u = static_cast<usize>(row);

    auto& la = samp.linalg();
    const auto permute = [&](PermuteOp op)
    {
        op.batch = dim_batch;
        permute_batched(la, samp.permutation_cache(), op);
    };

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
        const auto* ket_site =
            row == 0 ? samp.ket_row0()[col] : samp.ket().p + col_i * samp.max_ket_site();
        const i64 ket_stride{row == 0 ? 0 : samp.ket().stride};
        cf* const* dlenv_sigma_arr =
            has_below ? samp.dlenv_sigma_ptrs()[row_u][col] : samp.dl_unit_ptrs();

        permute({
            .dst = samp.tmp_a(),
            .src = samp.sigma(),
            .dims_in = {ket_bond_l, env_bond_l, ket_bond_l},
            .perm = {1, 2, 0},
        });
        {
            const auto m = env_bond_l * ket_bond_l;
            const auto k = ket_bond_l;
            const auto n = dim_phys * bond_below * ket_bond_r;
            if (row == 0)
            {
                la.matmul_batched_ptr(
                    samp.tmp_a_ptrs(),
                    m,
                    k,
                    samp.ket_row0_ptrs()[col],
                    k,
                    n,
                    samp.tmp_b_ptrs(),
                    m,
                    n,
                    dim_batch
                );
            }
            else
            {
                la.matmul_batched(
                    {samp.tmp_a(), m, k},
                    {ket_site, ket_stride, k, n},
                    {samp.tmp_b(), m, n},
                    dim_batch
                );
            }
        }

        permute({
            .dst = samp.tmp_a(),
            .src = samp.tmp_b(),
            .dims_in = {env_bond_l, ket_bond_l, dim_phys, bond_below, ket_bond_r},
            .perm = {1, 2, 4, 3, 0},
        });
        {
            const auto m = ket_bond_l * dim_phys * ket_bond_r;
            const auto k = bond_below * env_bond_l;
            const auto n = bond_below * env_bond_r;
            la.matmul_batched_ptr(
                samp.tmp_a_ptrs(), m, k, dlenv_sigma_arr, k, n, samp.tmp_b_ptrs(), m, n, dim_batch
            );
        }

        permute({
            .dst = samp.tmp_a(),
            .src = samp.tmp_b(),
            .dims_in = {ket_bond_l, dim_phys, ket_bond_r, bond_below, env_bond_r},
            .perm = {1, 2, 4, 0, 3},
        });
        permute({
            .dst = samp.tmp_b(),
            .src = {ket_site, ket_stride},
            .dims_in = {ket_bond_l, dim_phys, bond_below, ket_bond_r},
            .perm = {0, 2, 1, 3},
            .conj = 1,
        });
        {
            const auto m = dim_phys * ket_bond_r * env_bond_r;
            const auto k = ket_bond_l * bond_below;
            const auto n = dim_phys * ket_bond_r;
            la.matmul_batched(
                {samp.tmp_a(), m, k}, {samp.tmp_b(), k, n}, {samp.sigma_full(), m, n}, dim_batch
            );
        }

        permute({
            .dst = samp.sigma_full_scratch(),
            .src = samp.sigma_full(),
            .dims_in = {dim_phys, ket_bond_r, env_bond_r, dim_phys, ket_bond_r},
            .perm = {0, 3, 1, 2, 4},
        });
        std::swap(samp.sigma_full().p, samp.sigma_full_scratch().p);

        const int sigma_elems{ket_bond_r * env_bond_r * ket_bond_r};
        const auto* env_unsampled_next =
            samp.env_unsampled().p + (col_i + 1) * samp.max_env_unsampled();
        {
            const auto m = dim_phys * dim_phys;
            const auto k = sigma_elems;
            const auto n = 1;
            la.matmul_batched(
                {samp.sigma_full(), m, k},
                {env_unsampled_next, samp.env_unsampled().stride, k, n},
                {samp.rho(), m, n},
                dim_batch
            );
        }

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
}

static auto build_env_above(
    Sampler& samp,
    const SamplerConfig& cfg,
    int row,
    bool has_below,
    const std::vector<int>& ket_bonds,
    int& env_above_cur,
    std::vector<int>& bond_above_cur
) -> void
{
    if (not has_below) return;
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
            const auto* ket_site =
                row == 0 ? samp.ket_row0()[col] : samp.ket().p + col_i * samp.max_ket_site();
            const i64 ket_stride{row == 0 ? 0 : samp.ket().stride};
            auto* out = samp.env_above()[env_above_next].p + col_i * samp.max_env_above_site();
            const auto slice_elems = static_cast<i64>(ket_bond_l) * dim_bond * ket_bond_r;
            const CuSliceKetArgs slice_args{
                .out = out,
                .ket = ket_site,
                .chosen_spins = samp.row_spins() + col_i * dim_batch,
                .ket_bond_l = ket_bond_l,
                .dim_phys = dim_phys,
                .bond_below = dim_bond,
                .ket_bond_r = ket_bond_r,
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
    env_above_cur = env_above_next;
}
namespace sampler
{
auto ctx_sample_refresh(qnpeps_ctx& ctx) -> void
{
    Sampler& samp = ctx.sampler.samp;
    const auto num_rows = static_cast<usize>(ctx.cfg.lx);
    const auto num_cols = static_cast<usize>(ctx.cfg.ly);
    const auto stream = ctx.linalg.stream();

    CUDA_CHECK(cudaStreamSynchronize(stream));

    const auto reupload = [&](cf* dst, const HostTensor& host)
    {
        CUDA_CHECK(cudaMemcpyAsync(
            dst,
            host.v.data(),
            static_cast<usize>(host.n()) * sizeof(cf),
            cudaMemcpyHostToDevice,
            stream
        ));
    };
    for (auto row = 0_uz; row < num_rows; ++row)
    {
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            samp.mpo_host()[row][col] = hpermute(ctx.host_peps[row][col], {0, 3, 4, 1, 2});
            reupload(samp.mpo()[row][col], samp.mpo_host()[row][col]);
        }
    }
    for (auto col = 0_uz; col < num_cols; ++col)
    {
        samp.ket_row0_host()[col] = hpermute(ctx.host_peps[0][col], {0, 4, 1, 2, 3});
        reupload(samp.ket_row0()[col], samp.ket_row0_host()[col]);
    }
}

auto ctx_sample_run(qnpeps_ctx& ctx, const std::vector<int>& batch_ids) -> void
{
    Sampler& samp = ctx.sampler.samp;
    SamplerConfig& cfg = samp.cfg();

    if (ctx.sampler.dim_batch > 0 and cfg.dim_batch != ctx.sampler.dim_batch)
    {
        if (ctx.sampler.graph)
        {
            CUDA_NOCHECK(cudaGraphExecDestroy(ctx.sampler.graph));
            ctx.sampler.graph = nullptr;
        }
        cfg.dim_batch = ctx.sampler.dim_batch;
    }

    const auto dim_batch = cfg.dim_batch;
    const auto dim_bond = cfg.dim_bond;
    const int chi_s{cfg.chi_s};
    const auto num_rows = static_cast<usize>(cfg.lx);
    const auto num_cols = static_cast<usize>(cfg.ly);
    const i64 lane_samples{static_cast<i64>(dim_batch) * cfg.lx * cfg.ly};
    auto& la = samp.linalg();
    auto* device_seed = ctx.sampler.device_seed;
    auto& all_samples = ctx.sampler.all_samples;
    auto& all_logpc = ctx.sampler.all_logpc;
    auto& all_lognorm = ctx.sampler.all_lognorm;

    const auto lane_samples_u = static_cast<usize>(lane_samples);
    const auto dim_batch_u = static_cast<usize>(dim_batch);
    all_samples.clear();
    all_logpc.clear();
    all_lognorm.clear();
    all_samples.reserve(lane_samples_u * batch_ids.size());
    all_logpc.reserve(dim_batch_u * batch_ids.size());
    all_lognorm.reserve(dim_batch_u * batch_ids.size());

    {
        const auto num_env = num_rows - 1;
        const auto cap = static_cast<usize>(k_max_batch_size);
        cf* const views = ctx.dlenv.views[ctx.dlenv.active];
        ctx.dlenv.ptr_host.assign(2 * num_env * num_cols * cap, nullptr);
        usize slot{};
        const auto fill = [&](i64 off)
        {
            for (auto lane = 0_uz; lane < cap; ++lane)
                ctx.dlenv.ptr_host[slot * cap + lane] = views + off;
            ++slot;
        };
        for (auto row = 0_uz; row < num_env; ++row)
            for (auto col = 0_uz; col < num_cols; ++col)
                fill(ctx.dlenv.env_off[row][col]);
        for (auto row = 0_uz; row < num_env; ++row)
            for (auto col = 0_uz; col < num_cols; ++col)
                fill(ctx.dlenv.sigma_off[row][col]);
        if (num_env > 0)
            CUDA_CHECK(cudaMemcpyAsync(
                samp.dlenv_env_ptrs()[0][0],
                ctx.dlenv.ptr_host.data(),
                ctx.dlenv.ptr_host.size() * sizeof(cf*),
                cudaMemcpyHostToDevice,
                la.stream()
            ));
    }

    auto bondsK = [&](const std::vector<int>& bond_above, usize row)
    {
        std::vector<int> b{};
        b.assign(num_cols + 1, 1);
        if (row == 0)
        {
            for (auto col = 0_uz; col <= num_cols; ++col)
                b[col] = bond_dim(cfg.ly, static_cast<int>(col), dim_bond);
            return b;
        }
        int k{1};
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const HostTensor& mpo_site = samp.mpo_host()[row][col];
            const int M{k * mpo_site.dim[2] * mpo_site.dim[3]};
            const int N{bond_above[col + 1] * mpo_site.dim[4]};
            const int k2{std::max(1, std::min({chi_s, M, N}))};
            b[col + 1] = k2;
            k = k2;
        }
        b[num_cols] = 1;
        return b;
    };
    auto bondsE = [&](usize row)
    {
        std::vector<int> b{};
        b.assign(num_cols + 1, 1);
        if (row + 1 < num_rows)
        {
            const auto& env_row = samp.dlenv_host()[row];
            for (auto col = 0_uz; col < num_cols; ++col)
            {
                b[col] = env_row.site[col].dim[0];
                b[col + 1] = env_row.site[col].dim[3];
            }
        }
        return b;
    };

    constexpr u64 k_seed_multiplier{1000003};
    for (auto idx = 0_uz; idx < batch_ids.size(); ++idx)
    {
        const auto batch_id = batch_ids[idx];
        const u64 batch_seed{
            cfg.seed * k_seed_multiplier + cfg.batch_base + static_cast<u64>(batch_id)
        };
        ctx.sampler.h_seed = batch_seed;
        CUDA_CHECK(cudaMemcpyAsync(
            device_seed, &ctx.sampler.h_seed, sizeof(u64), cudaMemcpyHostToDevice, la.stream()
        ));

        auto enqueue_batch = [&]()
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
                const auto ket_bonds = bondsK(bond_above, static_cast<usize>(row));
                const auto env_bonds = bondsE(static_cast<usize>(row));

                if (row > 0) build_ket_row(samp, cfg, row, bond_above, ket_bonds, env_above_cur);

                build_env_unsampled(samp, cfg, row, has_below, ket_bonds, env_bonds);

                draw_sigma(samp, cfg, row, has_below, ket_bonds, env_bonds, device_seed);

                build_env_above(
                    samp, cfg, row, has_below, ket_bonds, env_above_cur, bond_above_cur
                );
            }
        };

        if (ctx.use_graph and ctx.sampler.graph)
        {
            if (std::getenv("QNPEPS_GRAPH_LOG"))
                std::fprintf(stderr, "[qnpeps] sample_graph replayed\n");
            CUDA_CHECK(cudaGraphLaunch(ctx.sampler.graph, la.stream()));
        }
        else if (ctx.use_graph and ctx.sampler.warmed)
        {
            cudaGraph_t graph{};
            CUDA_CHECK(cudaStreamBeginCapture(la.stream(), cudaStreamCaptureModeThreadLocal));
            enqueue_batch();
            const cudaError_t cap_rc = cudaStreamEndCapture(la.stream(), &graph);
            if (cap_rc == cudaSuccess
                and cudaGraphInstantiate(&ctx.sampler.graph, graph, nullptr, nullptr, 0)
                        == cudaSuccess)
            {
                CUDA_CHECK(cudaGraphDestroy(graph));
                if (std::getenv("QNPEPS_GRAPH_LOG"))
                    std::fprintf(stderr, "[qnpeps] sample_graph captured\n");
                CUDA_CHECK(cudaGraphLaunch(ctx.sampler.graph, la.stream()));
            }
            else
            {
                cudaGetLastError();
                ctx.sampler.graph = nullptr;
                enqueue_batch();
            }
        }
        else
        {
            enqueue_batch();
            ctx.sampler.warmed = true;
        }

        CUDA_CHECK(cudaMemcpyAsync(
            ctx.sampler.h_samples,
            samp.samples(),
            static_cast<usize>(lane_samples) * sizeof(u8),
            cudaMemcpyDeviceToHost,
            la.stream()
        ));
        CUDA_CHECK(cudaMemcpyAsync(
            ctx.sampler.h_logpc,
            samp.logpc(),
            static_cast<usize>(dim_batch) * sizeof(f64),
            cudaMemcpyDeviceToHost,
            la.stream()
        ));
        CUDA_CHECK(cudaMemcpyAsync(
            ctx.sampler.h_lognorm,
            samp.lognorm(),
            static_cast<usize>(dim_batch) * sizeof(f64),
            cudaMemcpyDeviceToHost,
            la.stream()
        ));
        CUDA_CHECK(cudaStreamSynchronize(la.stream()));
        int fail_host{};
        CUDA_CHECK(cudaMemcpy(&fail_host, samp.fail(), sizeof(int), cudaMemcpyDeviceToHost));
        if (fail_host != 0)
        {
            set_err(QNPEPS_ERR_CUDA);
            break;
        }
        all_samples.insert(
            all_samples.end(), ctx.sampler.h_samples, ctx.sampler.h_samples + lane_samples
        );
        all_logpc.insert(all_logpc.end(), ctx.sampler.h_logpc, ctx.sampler.h_logpc + dim_batch);
        all_lognorm.insert(
            all_lognorm.end(), ctx.sampler.h_lognorm, ctx.sampler.h_lognorm + dim_batch
        );
    }
}
}
}
