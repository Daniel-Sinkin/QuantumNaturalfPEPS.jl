#include "dlenv/build/execute.cuh"

namespace qnpeps::dlenv
{

auto dl_free(qnpeps_ctx& ctx) -> void
{
    BuildAllocation allocation{ctx};
    allocation.release();
}

auto build_dlenv(qnpeps_ctx& ctx, const void* device_peps, f64* cumulative_row_logs) -> int
{
    auto& la = ctx.linalg();
    const auto& cfg = ctx.cfg;
    const int lx{cfg.lx};
    const int ly{cfg.ly};
    const auto dim_bond = cfg.dim_bond;
    const auto dim_phys = cfg.dim_phys;
    const auto num_rows = static_cast<usize>(lx);
    const auto num_cols = static_cast<usize>(ly);

    BuildAllocation allocation{ctx};
    allocation.ensure(la);
    if (err_state() != QNPEPS_OK) return err_state();

    const Dims dims{lx, ly, dim_phys, dim_bond};

    std::vector<PepsRow> device_peps_grid{};
    device_peps_grid.resize(num_rows);

    const auto device_peps_base = reinterpret_cast<const cuFloatComplex*>(device_peps);
    i64 source_offset{};
    i64 packed_offset{};
    for (auto row = 0; row < lx; ++row)
    {
        const auto row_u = static_cast<usize>(row);
        device_peps_grid[row_u].resize(num_cols);
        pack_peps_row(
            dims,
            row,
            row + 1,
            device_peps_base,
            source_offset,
            ctx.dl.peps_buf,
            packed_offset,
            device_peps_grid[row_u],
            la.stream()
        );
    }

    ctx.dl.known.rewind();
    ctx.dl.rolling_r.rewind();
    ctx.dl.scratch.rewind();
    if (cfg.dlenv_truncation_route == QNPEPS_TRUNCATION_DENSITY) ctx.dl.density.rewind();
    zero_async(la, ctx.dl.fail, 1);

    const int chi_c{std::min(cfg.chi_dl, dim_bond * dim_bond)};
    const Arenas ar{ctx.dl.known, ctx.dl.rolling_r, ctx.dl.scratch};

    const auto target = static_cast<usize>(ctx.dlenv.build_count) % ctx.dlenv.lanes.size();
    auto& target_lane = ctx.dlenv.lanes[target];
    const auto num_sites = (num_rows - 1) * num_cols;
    auto* device_header = reinterpret_cast<int32_t*>(target_lane.packed);
    const auto header_elements = num_sites * k_dl_axis_count;
    auto* device_values = reinterpret_cast<cuFloatComplex*>(device_header + header_elements);

    const auto build_region = [&]
    {
        EnvironmentRowBuilder row_builder{ctx.dl, la, ar, dims};
        auto env_rows = row_builder.build_rows(
            device_peps_grid,
            chi_c,
            ctx.dl.fail,
            cfg.dlenv_truncation_route,
            cfg.dlenv_density_cutoff
        );
        if (err_state() != QNPEPS_OK) return;

        const auto header_required = cfg.dlenv_truncation_route == QNPEPS_TRUNCATION_DENSITY
                                     or not target_lane.header_written;
        if (header_required)
        {
            ctx.dlenv.dims.resize(num_sites * k_dl_axis_count);
            for (auto row = 0_uz; row < num_rows - 1; ++row)
            {
                for (auto col = 0_uz; col < num_cols; ++col)
                {
                    write_site_dims(
                        ctx.dlenv.dims.data(), row * num_cols + col, env_rows[row][col].dim
                    );
                }
            }
            upload_async(la, device_header, ctx.dlenv.dims.data(), ctx.dlenv.dims.size());
            target_lane.header_written = true;
        }

        i64 values_offset{};
        for (auto row = 0_uz; row < num_rows - 1; ++row)
        {
            for (auto col = 0_uz; col < num_cols; ++col)
            {
                const auto& site = env_rows[row][col];
                copy_device_async(la, device_values + values_offset, site.d, site.num_elems());
                values_offset += static_cast<i64>(site.num_elems());
            }
        }
    };

    const auto graph_capturable =
        ctx.use_graph and cfg.dlenv_truncation_route == QNPEPS_TRUNCATION_DEFAULT
        and trunc_svd::require_dlenv_route() == trunc_svd::Route::rangefinder;

    if (graph_capturable and target_lane.graph)
    {
        CUDA_CHECK(cudaGraphLaunch(target_lane.graph, la.stream()));
    }
    else if (graph_capturable and ctx.dl.warmed)
    {
        cudaGraph_t graph{};
        CUDA_CHECK(cudaStreamBeginCapture(la.stream(), cudaStreamCaptureModeThreadLocal));
        build_region();
        const auto capture_status = cudaStreamEndCapture(la.stream(), &graph);
        const auto capture_succeeded = capture_status == cudaSuccess and err_state() == QNPEPS_OK;
        const auto graph_instantiated =
            capture_succeeded and instantiate_graph(target_lane.graph, graph) == cudaSuccess;
        if (graph_instantiated)
        {
            CUDA_CHECK(cudaGraphDestroy(graph));
            CUDA_CHECK(cudaGraphLaunch(target_lane.graph, la.stream()));
        }
        else
        {
            cudaGetLastError();
            reset_err();
            target_lane.graph = nullptr;
            if (graph) CUDA_NOCHECK(cudaGraphDestroy(graph));
            ctx.dl.known.rewind();
            ctx.dl.rolling_r.rewind();
            ctx.dl.scratch.rewind();
            if (cfg.dlenv_truncation_route == QNPEPS_TRUNCATION_DENSITY) ctx.dl.density.rewind();
            zero_async(la, ctx.dl.fail, 1);
            build_region();
        }
    }
    else
    {
        build_region();
        ctx.dl.warmed = true;
    }
    if (err_state() != QNPEPS_OK) return err_state();

    CUDA_CHECK(cudaStreamSynchronize(la.stream()));

    const auto num_env_rows = num_rows - 1;
    std::vector<f64> scales_host{};
    scales_host.resize(num_env_rows * num_cols);
    download(scales_host.data(), ctx.dl.scales_all, scales_host.size());

    std::vector<f64> row_logs{};
    row_logs.assign(num_env_rows, 0.0);
    f64 total_log{0.0};
    for (auto step = 0_uz; step < num_env_rows; ++step)
    {
        bool aborted{false};
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const auto scale = scales_host[step * num_cols + col];
            if (not std::isfinite(scale))
            {
                set_err(QNPEPS_ERR_INTERNAL);
                aborted = true;
                break;
            }
            if (scale > 0.0) total_log += std::log(scale);
        }
        row_logs[num_env_rows - 1 - step] = total_log;
        if (aborted) break;
    }

    if (cumulative_row_logs and not row_logs.empty())
    {
        upload(cumulative_row_logs, row_logs.data(), row_logs.size());
    }

    int fail_host{};
    download(&fail_host, ctx.dl.fail, 1);
    if (fail_host != 0) set_err(QNPEPS_ERR_CUDA);
    if (err_state() != QNPEPS_OK) return err_state();

    if (ctx.sampler.allocation.allocated)
    {
        ensure_sampling_buffers(ctx);
        if (err_state() != QNPEPS_OK) return err_state();
        materialize_sampling_buffer(ctx, device_values, target_lane.sampling);
    }

    target_lane.valid = true;
    ctx.dlenv.active_lane = target;
    ctx.dlenv.build_count += 1;
    return err_state();
}

}
