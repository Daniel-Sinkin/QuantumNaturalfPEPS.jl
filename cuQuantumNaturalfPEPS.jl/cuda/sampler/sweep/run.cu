#include "sampler/sweep/run.cuh"

#include "sampler/sweep/draw.cuh"
#include "sampler/sweep/environment.cuh"
#include "sampler/sweep/ket.cuh"
#include "sampler/sweep/unsampled.cuh"

#include <cstdlib>

namespace qnpeps::sampler
{

auto ctx_sample_run(qnpeps_ctx& ctx, std::span<const int> batch_ids, HostSampleOutput* host_output)
    -> void
{
    if (not ctx.sampler.ready())
    {
        require(
            ctx.sampler.execution.host_pointers, "sampler runs before setup without host pointers"
        );
    }
    auto& samp = ctx.sampler.samp;
    auto& cfg = samp.cfg();

    if (ctx.sampler.execution.dim_batch > 0 and cfg.dim_batch != ctx.sampler.execution.dim_batch)
    {
        if (ctx.sampler.execution.graph)
        {
            unchecked(cudaGraphExecDestroy(ctx.sampler.execution.graph));
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
        const auto has_samples = host_output->samples != nullptr and host_output->n_samples != 0;
        require(has_samples, Status::null_argument, "sampler host output has no sample buffer");
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
            {
                ctx.dlenv.ptr_host[pointer_slot * lane_capacity + lane] =
                    device_sampling + value_offset;
            }
            pointer_slot += 1;
        };
        for (auto row = 0_uz; row < num_env_rows; ++row)
        {
            for (auto col = 0_uz; col < num_cols; ++col)
                fill_broadcast_pointers(ctx.dlenv.env_off[row][col]);
        }
        for (auto row = 0_uz; row < num_env_rows; ++row)
        {
            for (auto col = 0_uz; col < num_cols; ++col)
                fill_broadcast_pointers(ctx.dlenv.sigma_off[row][col]);
        }
        if (num_env_rows > 0)
        {
            upload_async(
                la,
                samp.dlenv_env_ptrs()[0][0],
                ctx.dlenv.ptr_host.data(),
                ctx.dlenv.ptr_host.size()
            );
        }
        const auto expected_slots = layout_count * num_env_rows * num_cols;
        require(
            pointer_slot == expected_slots,
            "sampler filled {} environment pointer slots, expected {}",
            pointer_slot,
            expected_slots
        );
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
        upload_async(la, device_seed, &ctx.sampler.staging.h_seed, 1);

        const auto enqueue_batch = [&]
        {
            const auto batch_count = static_cast<usize>(dim_batch);
            zero_async(la, samp.logpc(), batch_count);
            zero_async(la, samp.lognorm(), batch_count);
            zero_async(la, samp.fail(), 1);

            if (cfg.density_route)
            {
                for (auto lane = 0; lane < dim_batch; ++lane)
                {
                    sampler::DensityLane lane_view{samp, lane};
                    auto lane_cfg = cfg;
                    lane_cfg.dim_batch = 1;
                    lane_cfg.lane_base = lane;
                    int env_above_cur{0};
                    std::vector<int> bond_above_cur{};
                    bond_above_cur.assign(num_cols + 1, 1);
                    for (auto row = 0; row < cfg.lx; ++row)
                    {
                        const bool has_below{row + 1 < cfg.lx};
                        const auto bond_above = bond_above_cur;
                        auto ket_bonds = ket_bonds_for_row(bond_above, static_cast<usize>(row));
                        if (row != 0)
                        {
                            sampler::build_density_state_row(
                                ctx, lane_cfg, row, env_above_cur, bond_above, ket_bonds
                            );
                        }
                        const auto env_bonds = environment_bonds_for_row(static_cast<usize>(row));
                        internal::build_env_unsampled(
                            samp, lane_cfg, row, has_below, ket_bonds, env_bonds
                        );
                        internal::draw_sigma(
                            samp, lane_cfg, row, has_below, ket_bonds, env_bonds, device_seed
                        );
                        if (not has_below) continue;
                        if (cfg.fast_mode or row == 0)
                        {
                            auto select_cfg = lane_cfg;
                            select_cfg.fast_mode = true;
                            internal::build_env_above(
                                samp,
                                select_cfg,
                                row,
                                has_below,
                                ket_bonds,
                                env_above_cur,
                                bond_above_cur
                            );
                        }
                        else
                        {
                            const auto next = 1 - env_above_cur;
                            sampler::build_density_projected_row(
                                ctx, lane_cfg, row, env_above_cur, next, bond_above, bond_above_cur
                            );
                            env_above_cur = next;
                        }
                    }
                }
                return;
            }

            int env_above_cur{0};
            std::vector<int> bond_above_cur{};
            bond_above_cur.assign(num_cols + 1, 1);

            for (auto row = 0; row < cfg.lx; ++row)
            {
                const bool has_below{row + 1 < cfg.lx};
                const std::vector<int> bond_above{bond_above_cur};
                const auto ket_bonds = ket_bonds_for_row(bond_above, static_cast<usize>(row));
                const auto env_bonds = environment_bonds_for_row(static_cast<usize>(row));

                if (row != 0)
                    internal::build_ket_row(samp, cfg, row, bond_above, ket_bonds, env_above_cur);
                internal::build_env_unsampled(samp, cfg, row, has_below, ket_bonds, env_bonds);
                internal::draw_sigma(samp, cfg, row, has_below, ket_bonds, env_bonds, device_seed);
                internal::build_env_above(
                    samp, cfg, row, has_below, ket_bonds, env_above_cur, bond_above_cur
                );
            }
        };

        if (ctx.use_graph and ctx.sampler.execution.graph)
        {
            check(cudaGraphLaunch(ctx.sampler.execution.graph, la.stream()));
        }
        else if (ctx.use_graph and ctx.sampler.execution.warmed)
        {
            const auto graph = [&]
            {
                StreamCapture capture{la.stream()};
#ifdef QNPEPS_GATE_HOOKS
                const auto* injected = std::getenv("QNPEPS_INJECT_CAPTURE_FAILURE");
                require(
                    injected == nullptr or injected[0] != '1',
                    Status::internal,
                    "injected failure inside the sampler graph capture of batch {}",
                    batch_id
                );
#endif
                enqueue_batch();
                return capture.finish();
            }();
            const auto graph_instantiated =
                graph and instantiate_graph(ctx.sampler.execution.graph, graph) == cudaSuccess;
            if (graph_instantiated)
            {
                check(cudaGraphDestroy(graph));
                check(cudaGraphLaunch(ctx.sampler.execution.graph, la.stream()));
            }
            else
            {
                if (graph) unchecked(cudaGraphDestroy(graph));
                unchecked(cudaGetLastError());
                ctx.sampler.execution.graph = nullptr;
                enqueue_batch();
            }
        }
        else
        {
            enqueue_batch();
            ctx.sampler.execution.warmed = true;
        }

        download_async(la, ctx.sampler.staging.h_samples, samp.samples(), lane_samples_u);
        download_async(la, ctx.sampler.staging.h_logpc, samp.logpc(), dim_batch_u);
        download_async(la, ctx.sampler.staging.h_lognorm, samp.lognorm(), dim_batch_u);
        check(cudaStreamSynchronize(la.stream()));
        int fail_host{};
        download(&fail_host, samp.fail(), 1);
        require(
            fail_host == 0,
            Status::cuda,
            "sampler batch {} reported device failure flag {}",
            batch_id,
            fail_host
        );
        if (host_output)
        {
            require(
                static_cast<u64>(batch_id) >= host_output->batch_origin,
                "sampler batch {} precedes the host output origin {}",
                batch_id,
                host_output->batch_origin
            );
            const auto sample_offset = (static_cast<u64>(batch_id) - host_output->batch_origin)
                                       * static_cast<u64>(dim_batch);
            require(
                sample_offset < host_output->n_samples,
                "sampler batch {} starts at sample {} past the {} host samples",
                batch_id,
                sample_offset,
                host_output->n_samples
            );
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

    if (not host_output)
    {
        const auto batch_count = batch_ids.size();
        assert(all_samples.size() == lane_samples_u * batch_count);
        assert(all_logpc.size() == dim_batch_u * batch_count);
        assert(all_lognorm.size() == dim_batch_u * batch_count);
    }
}
}
