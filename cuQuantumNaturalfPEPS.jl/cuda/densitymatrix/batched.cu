#include "core/cuda_utils.cuh"
#include "core/defer.cuh"
#include "densitymatrix/backend.cuh"
#include "densitymatrix/batched.cuh"
#include "linalg/transfer.cuh"

#include <algorithm>

namespace qnpeps::densitymatrix
{
namespace
{
[[nodiscard]] auto combined_left_of(const BatchedSite& site) noexcept -> i64
{
    return site.state_left * site.operator_left;
}

[[nodiscard]] auto combined_right_of(const BatchedSite& site) noexcept -> i64
{
    return site.state_right * site.operator_right;
}

[[nodiscard]] auto product_panel(
    const BatchedWorkspace& workspace, const BatchedSite& site, i64 output_state
) noexcept -> CuMatrixBatchedCF64
{
    const auto combined_left = combined_left_of(site);
    const auto combined_right = combined_right_of(site);
    const auto panel_offset = output_state * combined_left * combined_right;
    return {
        workspace.products + panel_offset,
        workspace.layout.product_lane,
        static_cast<int>(combined_left),
        static_cast<int>(combined_right),
    };
}

[[nodiscard]] auto left_environment(
    const BatchedWorkspace& workspace, usize cut, i64 combined
) noexcept -> CuMatrixBatchedCF64
{
    const auto cut_offset = static_cast<i64>(cut) * workspace.layout.environment_cut;
    const auto extent = static_cast<int>(combined);
    return {
        workspace.left_environments + cut_offset, workspace.layout.environment_lane, extent, extent
    };
}

[[nodiscard]] auto block_matrix(cuDoubleComplex* data, i64 lane_stride, i64 rows, i64 cols) noexcept
    -> CuMatrixBatchedCF64
{
    return {data, lane_stride, static_cast<int>(rows), static_cast<int>(cols)};
}

[[nodiscard]] auto is_open_chain(std::span<const BatchedSite> sites) noexcept -> bool
{
    auto state_bond = 1_i64;
    auto operator_bond = 1_i64;
    for (const auto& site : sites)
    {
        const auto is_linked =
            site.state_left == state_bond and site.operator_left == operator_bond;
        const auto is_positive =
            all_positive(site.physical_input, site.state_right, site.physical_output)
            and site.operator_right > 0 and site.output and site.operator_values;
        if (not is_linked or not is_positive) return false;
        state_bond = site.state_right;
        operator_bond = site.operator_right;
    }
    return state_bond == 1 and operator_bond == 1;
}

[[nodiscard]] auto is_within_geometry(
    const BatchedGeometry& geometry, const BatchedSite& site
) noexcept -> bool
{
    const auto is_state_within =
        site.state_left <= geometry.state_bond and site.state_right <= geometry.state_bond;
    const auto is_operator_within = site.operator_left <= geometry.operator_bond
                                    and site.operator_right <= geometry.operator_bond;
    return is_state_within and is_operator_within
           and site.physical_output <= geometry.output_dimension;
}

auto form_products(
    Linalg& linalg, const BatchedWorkspace& workspace, const BatchedRow& row, usize site_index
) -> void
{
    const auto& site = row.sites[site_index];
    const auto lanes = static_cast<i64>(row.lanes);
    const auto count =
        combined_left_of(site) * combined_right_of(site) * site.physical_output * lanes;
    const auto products_args = BatchedProductArgs{
        .state = site.state,
        .state_lane_stride = row.state_lane_stride,
        .state_left = site.state_left,
        .physical_input = site.physical_input,
        .state_right = site.state_right,
        .operator_values = site.operator_values,
        .operator_lane_stride = row.operator_lane_stride,
        .operator_strides = site.operator_strides,
        .operator_left = site.operator_left,
        .physical_output = site.physical_output,
        .operator_right = site.operator_right,
        .products = workspace.products,
        .product_lane_stride = workspace.layout.product_lane,
        .lanes = lanes,
    };
    cu_batched_products<<<grid_blocks_capped(count), k_threads_per_block, 0, linalg.stream()>>>(
        products_args
    );
    check(cudaGetLastError());
}

auto form_left_environments(
    Linalg& linalg, const BatchedWorkspace& workspace, const BatchedRow& row
) -> void
{
    const auto cuts = row.sites.size() - 1;
    for (auto cut = 0_uz; cut < cuts; ++cut)
    {
        form_products(linalg, workspace, row, cut);
        const auto& site = row.sites[cut];
        const auto combined_left = combined_left_of(site);
        const auto combined_right = combined_right_of(site);
        const auto environment = left_environment(workspace, cut, combined_right);
        const auto weighted = block_matrix(
            workspace.environment_product,
            workspace.layout.block_lane,
            combined_left,
            combined_right
        );
        for (auto output_state = 0_i64; output_state < site.physical_output; ++output_state)
        {
            const auto panel = product_panel(workspace, site, output_state);
            const auto accumulate = MatmulConfig{
                .op_a = BlasOp::conj_trans,
                .beta_real = output_state == 0 ? 0.0f : 1.0f,
            };
            if (cut == 0)
            {
                linalg.matmul_batched(panel, panel, environment, row.lanes, accumulate);
                continue;
            }
            const auto previous = left_environment(workspace, cut - 1, combined_left);
            linalg.matmul_batched(previous, panel, weighted, row.lanes);
            linalg.matmul_batched(panel, weighted, environment, row.lanes, accumulate);
        }
    }
}

auto form_right_block(
    Linalg& linalg,
    const BatchedWorkspace& workspace,
    const BatchedRow& row,
    usize site_index,
    i64 right_rank
) -> void
{
    form_products(linalg, workspace, row, site_index);
    const auto& site = row.sites[site_index];
    const auto lanes = static_cast<i64>(row.lanes);
    const auto count = combined_left_of(site) * site.physical_output * right_rank * lanes;
    const auto right_block_args = BatchedRightBlockArgs{
        .products = workspace.products,
        .product_lane_stride = workspace.layout.product_lane,
        .carried = workspace.carried,
        .carried_lane_stride = workspace.layout.carried_lane,
        .combined_left = combined_left_of(site),
        .combined_right = combined_right_of(site),
        .physical_output = site.physical_output,
        .right_rank = right_rank,
        .right_block = workspace.right_block,
        .block_lane_stride = workspace.layout.block_lane,
        .lanes = lanes,
    };
    cu_batched_right_block<<<grid_blocks_capped(count), k_threads_per_block, 0, linalg.stream()>>>(
        right_block_args
    );
    check(cudaGetLastError());
}

auto truncate_cut(
    Linalg& linalg,
    const BatchedWorkspace& workspace,
    const BatchedRow& row,
    usize site_index,
    i64 right_rank,
    i64 rank
) -> void
{
    const auto& layout = workspace.layout;
    const auto& site = row.sites[site_index];
    const auto lanes = static_cast<i64>(row.lanes);
    const auto stream = linalg.stream();
    const auto combined_left = combined_left_of(site);
    const auto order = site.physical_output * right_rank;
    form_right_block(linalg, workspace, row, site_index, right_rank);
    const auto right_block =
        block_matrix(workspace.right_block, layout.block_lane, combined_left, order);
    const auto weighted =
        block_matrix(workspace.environment_product, layout.block_lane, combined_left, order);
    const auto density_stride = order * order;
    const auto density = block_matrix(workspace.density, density_stride, order, order);
    const auto environment = left_environment(workspace, site_index - 1, combined_left);
    linalg.matmul_batched(environment, right_block, weighted, row.lanes);
    linalg.matmul_batched(right_block, weighted, density, row.lanes, {.op_a = BlasOp::conj_trans});
    linalg.diagonalize_batched(
        density,
        {
            .eigenvalues = workspace.eigenvalues,
            .workspace = workspace.solver_workspace,
            .workspace_bytes = workspace.solver_workspace_bytes,
            .info = workspace.solver_information,
        },
        row.lanes
    );
    const auto select_args = BatchedSelectArgs{
        .eigenvalues = workspace.eigenvalues,
        .eigenvalue_lane_stride = order,
        .order = order,
        .solver_information = workspace.solver_information,
        .sorted_indices = workspace.sorted_indices,
        .index_lane_stride = layout.spectrum_lane,
        .lane_flags = workspace.lane_flags,
    };
    cu_batched_select<<<static_cast<u32>(row.lanes), k_threads_per_block, 0, stream>>>(select_args);
    check(cudaGetLastError());
    const auto basis = BatchedBasisArgs{
        .vectors = workspace.density,
        .vector_lane_stride = density_stride,
        .sorted_indices = workspace.sorted_indices,
        .index_lane_stride = layout.spectrum_lane,
        .order = order,
        .rank = rank,
        .lanes = lanes,
    };
    const auto pack_site_args = BatchedPackSiteArgs{
        .basis = basis,
        .output = site.output,
        .output_lane_stride = row.output_lane_stride,
    };
    cu_batched_pack_site<<<
        grid_blocks_capped(rank * order * lanes),
        k_threads_per_block,
        0,
        stream>>>(pack_site_args);
    check(cudaGetLastError());
    const auto project_args = BatchedProjectArgs{
        .basis = basis,
        .right_block = workspace.right_block,
        .block_lane_stride = layout.block_lane,
        .combined_left = combined_left,
        .carried = workspace.carried,
        .carried_lane_stride = layout.carried_lane,
    };
    cu_batched_project<<<
        grid_blocks_capped(combined_left * rank * lanes),
        k_threads_per_block,
        0,
        stream>>>(project_args);
    check(cudaGetLastError());
}

auto sweep_from_the_right(
    Linalg& linalg,
    const BatchedWorkspace& workspace,
    const BatchedRow& row,
    const std::vector<i64>& ranks
) -> void
{
    const auto lanes = static_cast<i64>(row.lanes);
    const auto stream = linalg.stream();
    cu_batched_unit<<<grid_blocks_capped(lanes), k_threads_per_block, 0, stream>>>(
        workspace.carried, workspace.layout.carried_lane, lanes
    );
    check(cudaGetLastError());
    auto right_rank = 1_i64;
    for (auto site_index = row.sites.size() - 1; site_index > 0; --site_index)
    {
        truncate_cut(linalg, workspace, row, site_index, right_rank, ranks[site_index]);
        right_rank = ranks[site_index];
    }
    form_right_block(linalg, workspace, row, 0, right_rank);
    const auto& first = row.sites[0];
    const auto normalize_args = BatchedNormalizeArgs{
        .values = workspace.right_block,
        .value_lane_stride = workspace.layout.block_lane,
        .count = first.physical_output * right_rank,
        .output = first.output,
        .output_lane_stride = row.output_lane_stride,
        .log_gauge = row.log_gauge,
        .lane_flags = workspace.lane_flags,
    };
    cu_batched_normalize<<<static_cast<u32>(row.lanes), k_threads_per_block, 0, stream>>>(
        normalize_args
    );
    check(cudaGetLastError());
}

[[nodiscard]] auto solver_workspace_bytes(Linalg& linalg, const BatchedWorkspace& workspace)
    -> usize
{
    const auto lanes = static_cast<int>(workspace.geometry.lanes);
    auto bytes = 1_uz;
    for (auto order = 1_i64; order <= workspace.layout.order; ++order)
    {
        const auto matrices = block_matrix(workspace.density, order * order, order, order);
        const auto order_bytes =
            linalg.diagonalize_batched_workspace_bytes(matrices, workspace.eigenvalues, lanes);
        bytes = std::max(bytes, order_bytes);
    }
    return bytes;
}
}

auto batched_layout(const BatchedGeometry& geometry) noexcept -> BatchedLayout
{
    const auto combined = geometry.state_bond * geometry.operator_bond;
    const auto order = geometry.output_dimension * geometry.upper_bond;
    const auto environment_cut = combined * combined;
    return {
        .combined = combined,
        .order = order,
        .product_lane = geometry.output_dimension * environment_cut,
        .environment_cut = environment_cut,
        .environment_lane = (geometry.num_sites - 1) * environment_cut,
        .block_lane = combined * std::max(combined, order),
        .density_lane = order * order,
        .carried_lane = combined * geometry.upper_bond,
        .spectrum_lane = order,
    };
}

auto _take_batched_workspace_check(const BatchedGeometry& geometry) noexcept -> bool
{
    const auto has_extents = all_positive(
        geometry.state_bond,
        geometry.operator_bond,
        geometry.output_dimension,
        geometry.upper_bond,
        geometry.lanes
    );
    return geometry.num_sites >= 2 and has_extents;
}

auto take_batched_workspace(Linalg& linalg, ArenaCursor& arena, const BatchedGeometry& geometry)
    -> BatchedWorkspace
{
    require(
        _take_batched_workspace_check(geometry),
        Status::bad_config,
        "batched density workspace rejects {} sites, bonds {} and {}, output {}, upper bond {}, {} "
        "lanes",
        geometry.num_sites,
        geometry.state_bond,
        geometry.operator_bond,
        geometry.output_dimension,
        geometry.upper_bond,
        geometry.lanes
    );
    const auto layout = batched_layout(geometry);
    const auto lanes = static_cast<usize>(geometry.lanes);
    const auto per_lane = [&](i64 count) -> usize { return lanes * static_cast<usize>(count); };
    BatchedWorkspace workspace{
        .geometry = geometry,
        .layout = layout,
        .products = arena.take<cuDoubleComplex>(per_lane(layout.product_lane)),
        .left_environments = arena.take<cuDoubleComplex>(per_lane(layout.environment_lane)),
        .right_block = arena.take<cuDoubleComplex>(per_lane(layout.block_lane)),
        .environment_product = arena.take<cuDoubleComplex>(per_lane(layout.block_lane)),
        .density = arena.take<cuDoubleComplex>(per_lane(layout.density_lane)),
        .carried = arena.take<cuDoubleComplex>(per_lane(layout.carried_lane)),
        .eigenvalues = arena.take<f64>(per_lane(layout.spectrum_lane)),
        .sorted_indices = arena.take<i32>(per_lane(layout.spectrum_lane)),
        .solver_information = arena.take<i32>(lanes),
        .lane_flags = arena.take<u32>(lanes),
    };
    workspace.solver_workspace_bytes = solver_workspace_bytes(linalg, workspace);
    workspace.solver_workspace = arena.take<char>(workspace.solver_workspace_bytes);
    return workspace;
}

auto kept_ranks(std::span<const BatchedSite> sites, i64 upper_bond) -> std::vector<i64>
{
    auto ranks = std::vector<i64>(sites.size() + 1, 1_i64);
    auto right_rank = 1_i64;
    for (auto site_index = sites.size() - 1; site_index > 0; --site_index)
    {
        const auto& site = sites[site_index];
        const auto order = site.physical_output * right_rank;
        const auto is_last = site_index == sites.size() - 1;
        const auto natural_cap = is_last ? order : combined_left_of(site);
        ranks[site_index] = std::max(1_i64, std::min({order, natural_cap, upper_bond}));
        right_rank = ranks[site_index];
    }
    return ranks;
}

auto _apply_batched_check(
    const BatchedWorkspace& workspace, const BatchedRow& row, bool has_state
) noexcept -> bool
{
    const auto& geometry = workspace.geometry;
    const auto has_valid_lanes = static_cast<i64>(row.lanes) == geometry.lanes;
    const auto has_matching_sites = static_cast<i64>(row.sites.size()) == geometry.num_sites;
    if (not has_valid_lanes or not has_matching_sites or not row.log_gauge) return false;
    if (not is_open_chain(row.sites)) return false;
    for (const auto& site : row.sites)
    {
        const auto has_input = has_state ? site.state != nullptr : site.physical_input == 1;
        if (not has_input or not is_within_geometry(geometry, site)) return false;
    }
    return true;
}

auto apply_batched(Linalg& linalg, const BatchedWorkspace& workspace, const BatchedRow& row)
    -> std::vector<i64>
{
    require(
        _apply_batched_check(workspace, row, true),
        Status::bad_config,
        "batched density apply rejects a row of {} sites over {} lanes for {} sites and {} lanes",
        row.sites.size(),
        row.lanes,
        workspace.geometry.num_sites,
        workspace.geometry.lanes
    );
    HandleState handle_state{};
    enter_handle_state(linalg, handle_state);
    DEFER([&] { restore_handle_state(linalg, handle_state); });
    const auto ranks = kept_ranks(row.sites, workspace.geometry.upper_bond);
    form_left_environments(linalg, workspace, row);
    sweep_from_the_right(linalg, workspace, row, ranks);
    return ranks;
}

auto normalize_batched(Linalg& linalg, const BatchedWorkspace& workspace, const BatchedRow& row)
    -> std::vector<i64>
{
    require(
        _apply_batched_check(workspace, row, false),
        Status::bad_config,
        "batched density boundary rejects a row of {} sites over {} lanes for {} sites and {} "
        "lanes",
        row.sites.size(),
        row.lanes,
        workspace.geometry.num_sites,
        workspace.geometry.lanes
    );
    auto bonds = std::vector<i64>(row.sites.size() + 1, 1_i64);
    for (auto index = 0_uz; index < row.sites.size(); ++index)
    {
        const auto& site = row.sites[index];
        const auto boundary_args = BatchedBoundaryArgs{
            .input = site.operator_values,
            .input_lane_stride = row.operator_lane_stride,
            .input_strides = site.operator_strides,
            .left = site.operator_left,
            .vertical = site.physical_output,
            .right = site.operator_right,
            .output = site.output,
            .output_lane_stride = row.output_lane_stride,
            .log_gauge = row.log_gauge,
            .lane_flags = workspace.lane_flags,
        };
        cu_batched_boundary<<<
            static_cast<u32>(row.lanes),
            k_threads_per_block,
            0,
            linalg.stream()>>>(boundary_args);
        check(cudaGetLastError());
        bonds[index] = site.operator_left;
        bonds[index + 1] = site.operator_right;
    }
    return bonds;
}

auto clear_lane_flags(Linalg& linalg, const BatchedWorkspace& workspace) -> void
{
    zero_async(linalg, workspace.lane_flags, static_cast<usize>(workspace.geometry.lanes));
}

auto read_lane_flags(Linalg& linalg, const BatchedWorkspace& workspace, int lanes)
    -> std::vector<u32>
{
    const auto lane_count = static_cast<usize>(lanes);
    auto flags = std::vector<u32>(lane_count);
    download_async(linalg, flags.data(), workspace.lane_flags, lane_count);
    check(cudaStreamSynchronize(linalg.stream()));
    return flags;
}
}
