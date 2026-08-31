#include "core/cuda_utils.cuh"
#include "core/defer.cuh"
#include "densitymatrix/backend.cuh"
#include "densitymatrix/kernels.cuh"
#include "densitymatrix/types.cuh"
#include "linalg/transfer.cuh"

#include <algorithm>
#include <cmath>

namespace qnpeps::densitymatrix
{
namespace
{
auto launch_count(i64 count_value) -> u32
{
    return grid_blocks_capped(std::max(count_value, 1_i64));
}

auto local_product(Linalg& linalg, const Apply& args, const Site& site, cuDoubleComplex* output)
    -> void
{
    const auto combined_left = site.state_left * site.operator_left;
    const auto combined_right = site.state_right * site.operator_right;
    const auto elements = combined_left * combined_right * site.physical_output;
    const auto local_product_args = LocalProductArgs{
        .state = args.state_values + site.state_offset,
        .operator_values = args.operator_values + site.operator_offset,
        .state_left = site.state_left,
        .physical_input = site.physical_input,
        .state_right = site.state_right,
        .operator_left = site.operator_left,
        .physical_output = site.physical_output,
        .operator_right = site.operator_right,
        .product = output,
    };
    cu_local_product<<<launch_count(elements), k_threads_per_block, 0, linalg.stream()>>>(
        local_product_args
    );
    CUDA_CHECK(cudaGetLastError());
}

auto form_product(
    Linalg& linalg, const Apply& args, const Site& site, i64 output_state, cuDoubleComplex* output
) -> void
{
    if (args.product)
    {
        args.product(
            linalg,
            site,
            output_state,
            args.state_values,
            args.operator_values,
            output,
            args.product_context
        );
        return;
    }
    local_product(linalg, args, site, output);
}

auto validate(const Apply& args, i64& maximum_combined, i64& maximum_physical, i64& maximum_output)
    -> bool
{
    const auto invalid_args =
        args.settings.struct_size != sizeof(QnpepsDensitySettings)
        or args.settings.precision != 1_u32 or args.settings.mindim != 1_u32
        or args.settings.reserved != 0_u32 or not std::isfinite(args.settings.relative_cutoff)
        or args.settings.relative_cutoff < 0.0 or args.upper_bond < 1 or args.sites.size() < 2;
    if (invalid_args)
    {
        set_err(QNPEPS_ERR_BAD_CONFIG);
        return false;
    }
    auto state_bond = 1_i64;
    auto operator_bond = 1_i64;
    maximum_combined = 1_i64;
    maximum_physical = 1_i64;
    maximum_output = 1_i64;
    for (auto index = 0_uz; index < args.sites.size(); ++index)
    {
        const auto& site = args.sites[index];
        const auto valid = site.site == index and site.num_sites == args.sites.size()
                           and site.state_left == state_bond and site.operator_left == operator_bond
                           and site.state_left > 0 and site.physical_input > 0
                           and site.state_right > 0 and site.operator_left > 0
                           and site.physical_output > 0 and site.operator_right > 0;
        if (not valid)
        {
            set_err(QNPEPS_ERR_BAD_CONFIG);
            return false;
        }
        maximum_combined = std::max(
            maximum_combined,
            std::max(site.state_left * site.operator_left, site.state_right * site.operator_right)
        );
        maximum_physical = std::max(maximum_physical, site.physical_output);
        maximum_output = std::max(maximum_output, site.physical_output * args.upper_bond);
        state_bond = site.state_right;
        operator_bond = site.operator_right;
    }
    if (state_bond != 1 or operator_bond != 1)
    {
        set_err(QNPEPS_ERR_BAD_CONFIG);
        return false;
    }
    return true;
}
}

auto take_workspace(Linalg& linalg, ArenaCursor& arena, const Geometry& geometry) -> Workspace
{
    const auto sites = static_cast<usize>(geometry.num_sites);
    const auto cuts = sites - 1;
    const auto input_bond = static_cast<usize>(geometry.input_bond);
    const auto operator_bond = static_cast<usize>(geometry.operator_bond);
    const auto output_dimension = static_cast<usize>(geometry.output_dimension);
    const auto upper_bond = static_cast<usize>(geometry.upper_bond);
    const auto lanes = static_cast<usize>(geometry.lanes);
    const auto combined = input_bond * operator_bond;
    const auto output = output_dimension * upper_bond;
    const auto left = lanes * cuts * combined * combined;
    const auto products = lanes * combined * combined * output_dimension;
    const auto blocks = lanes * combined * output;
    const auto bases = lanes * output * upper_bond;
    const auto right = products + blocks + bases;
    const auto temporary = lanes * 2 * combined * output;
    const auto density = lanes * output * output;
    const auto linear = lanes * output;
    const auto ranks = lanes * cuts;
    const auto solver_bytes = eigen_workspace_bytes(linalg, static_cast<int>(output));
    return {
        .left_environments = arena.take<cuDoubleComplex>(left),
        .right_blocks = arena.take<cuDoubleComplex>(right),
        .temporaries = arena.take<cuDoubleComplex>(temporary),
        .density = arena.take<cuDoubleComplex>(density),
        .eigenvalues = arena.take<f64>(linear),
        .sorted_eigenvalues = arena.take<f64>(linear),
        .sort_indices = arena.take<i32>(linear),
        .active_ranks = arena.take<i32>(ranks),
        .records = arena.take<QnpepsDensityRankRecord>(ranks),
        .solver_workspace = arena.take<char>(solver_bytes),
        .solver_workspace_bytes = solver_bytes,
        .solver_information = arena.take<i32>(1),
    };
}

auto apply(Linalg& linalg, const Apply& args) -> void
{
    auto maximum_combined = 0_i64;
    auto maximum_physical = 0_i64;
    auto maximum_output = 0_i64;
    if (not validate(args, maximum_combined, maximum_physical, maximum_output)) return;
    HandleState handle_state{};
    if (not enter_handle_state(linalg, handle_state)) return;
    DEFER([&] { restore_handle_state(linalg, handle_state); });
    const auto sites = args.sites.size();
    const auto cuts = sites - 1;
    const auto maximum_combined_u = static_cast<usize>(maximum_combined);
    const auto maximum_output_u = static_cast<usize>(maximum_output);
    const auto maximum_physical_u = static_cast<usize>(maximum_physical);
    const auto upper_bond = static_cast<usize>(args.upper_bond);
    const auto combined_square = maximum_combined_u * maximum_combined_u;
    const auto combined_output = maximum_combined_u * maximum_output_u;
    const auto product_count =
        args.product ? combined_square : combined_square * maximum_physical_u;
    const auto basis_count = maximum_output_u * upper_bond;
    const auto stream = linalg.stream();
    auto* product_values = args.workspace.right_blocks;
    auto* right_block = product_values + product_count;
    auto* basis = right_block + combined_output;
    auto* temporary = args.workspace.temporaries;
    zero_async(linalg, args.workspace.left_environments, cuts * combined_square);
    zero_async(linalg, args.workspace.active_ranks, cuts);
    if (err_state() != QNPEPS_OK) return;
    auto* previous_environment = static_cast<cuDoubleComplex*>(nullptr);
    auto previous_combined = 1_i64;
    for (auto index = 0_uz; index < cuts; ++index)
    {
        const auto& site = args.sites[index];
        const auto combined_left = site.state_left * site.operator_left;
        const auto combined_right = site.state_right * site.operator_right;
        const auto combined_left_u = static_cast<usize>(combined_left);
        const auto combined_right_u = static_cast<usize>(combined_right);
        if (combined_left != previous_combined) return set_err(QNPEPS_ERR_INTERNAL), void();
        if (not args.product) form_product(linalg, args, site, 0_i64, product_values);
        auto* next_environment = args.workspace.left_environments + index * combined_square;
        const auto environment_count = combined_right_u * combined_right_u;
        zero_async(linalg, next_environment, environment_count);
        for (auto output_state = 0_i64; output_state < site.physical_output; ++output_state)
        {
            if (args.product) form_product(linalg, args, site, output_state, product_values);
            const auto offset = args.product ? 0_uz
                                             : combined_left_u * combined_right_u
                                                   * static_cast<usize>(output_state);
            const CuMatrixCF64Const panel{
                product_values + offset,
                static_cast<int>(combined_left),
                static_cast<int>(combined_right),
            };
            if (index == 0)
            {
                matmul(
                    linalg,
                    panel,
                    panel,
                    CuMatrixCF64{
                        next_environment,
                        static_cast<int>(combined_right),
                        static_cast<int>(combined_right),
                    },
                    BlasOp::conj_trans,
                    BlasOp::none,
                    1.0
                );
            }
            else
            {
                matmul(
                    linalg,
                    CuMatrixCF64Const{
                        previous_environment,
                        static_cast<int>(combined_left),
                        static_cast<int>(combined_left),
                    },
                    panel,
                    CuMatrixCF64{
                        temporary,
                        static_cast<int>(combined_left),
                        static_cast<int>(combined_right),
                    }
                );
                matmul(
                    linalg,
                    panel,
                    CuMatrixCF64Const{
                        temporary,
                        static_cast<int>(combined_left),
                        static_cast<int>(combined_right),
                    },
                    CuMatrixCF64{
                        next_environment,
                        static_cast<int>(combined_right),
                        static_cast<int>(combined_right),
                    },
                    BlasOp::conj_trans,
                    BlasOp::none,
                    1.0
                );
            }
            if (err_state() != QNPEPS_OK) return;
        }
        previous_environment = next_environment;
        previous_combined = combined_right;
    }
    zero_async(linalg, temporary, 2 * combined_output);
    const auto one = make_cuDoubleComplex(1.0, 0.0);
    upload_async(linalg, temporary, &one, 1);
    auto right_rank = 1_i64;
    for (auto reverse_index = sites - 1; reverse_index > 0; --reverse_index)
    {
        const auto step = sites - 1 - reverse_index;
        const auto& site = args.sites[reverse_index];
        const auto combined_left = site.state_left * site.operator_left;
        const auto combined_right = site.state_right * site.operator_right;
        const auto order = site.physical_output * right_rank;
        const auto right_count = static_cast<usize>(combined_left) * static_cast<usize>(order);
        if (args.product)
        {
            for (auto output_state = 0_i64; output_state < site.physical_output; ++output_state)
            {
                form_product(linalg, args, site, output_state, product_values);
                const auto right_block_slice_args = RightBlockSliceArgs{
                    .product = product_values,
                    .carried = temporary,
                    .combined_left = combined_left,
                    .combined_right = combined_right,
                    .physical_output = site.physical_output,
                    .output_state = output_state,
                    .right_rank = right_rank,
                    .right_block = right_block,
                };
                cu_right_block_slice<<<
                    launch_count(combined_left * right_rank),
                    k_threads_per_block,
                    0,
                    stream>>>(right_block_slice_args);
            }
        }
        else
        {
            form_product(linalg, args, site, 0_i64, product_values);
            const auto right_block_args = RightBlockArgs{
                .product = product_values,
                .carried = temporary,
                .combined_left = combined_left,
                .combined_right = combined_right,
                .physical_output = site.physical_output,
                .right_rank = right_rank,
                .right_block = right_block,
            };
            cu_right_block<<<
                launch_count(static_cast<i64>(right_count)),
                k_threads_per_block,
                0,
                stream>>>(right_block_args);
        }
        CUDA_CHECK(cudaGetLastError());
        auto* environment =
            args.workspace.left_environments + (reverse_index - 1) * combined_square;
        matmul(
            linalg,
            CuMatrixCF64Const{
                environment,
                static_cast<int>(combined_left),
                static_cast<int>(combined_left),
            },
            CuMatrixCF64Const{
                right_block, static_cast<int>(combined_left), static_cast<int>(order)
            },
            CuMatrixCF64{temporary, static_cast<int>(combined_left), static_cast<int>(order)}
        );
        matmul(
            linalg,
            CuMatrixCF64Const{
                right_block, static_cast<int>(combined_left), static_cast<int>(order)
            },
            CuMatrixCF64Const{temporary, static_cast<int>(combined_left), static_cast<int>(order)},
            CuMatrixCF64{args.workspace.density, static_cast<int>(order), static_cast<int>(order)},
            BlasOp::conj_trans
        );
        if (err_state() != QNPEPS_OK) return;
        eigen_hermitian(
            linalg,
            CuMatrixCF64{args.workspace.density, static_cast<int>(order), static_cast<int>(order)},
            args.workspace.eigenvalues,
            args.workspace.solver_workspace,
            args.workspace.solver_workspace_bytes,
            args.workspace.solver_information
        );
        const auto natural_cap = step == 0 ? order : combined_left;
        const auto applied_cap =
            step == 0 ? args.upper_bond : std::min(natural_cap, args.upper_bond);
        const auto filter_args = FilterArgs{
            .solver_values = args.workspace.eigenvalues,
            .order = order,
            .natural_cap = natural_cap,
            .applied_cap = applied_cap,
            .mindim = static_cast<i64>(args.settings.mindim),
            .cutoff = args.settings.relative_cutoff,
            .sorted_values = args.workspace.sorted_eigenvalues,
            .retained_values = args.workspace.eigenvalues,
            .sorted_indices = args.workspace.sort_indices,
            .active_rank = args.workspace.active_ranks + step,
            .record = args.workspace.records + step,
        };
        cu_filter<<<1, 1, 0, stream>>>(filter_args);
        CUDA_CHECK(cudaGetLastError());
        QnpepsDensityRankRecord record{};
        auto solver_information = 0_i32;
        download_async(linalg, &record, args.workspace.records + step, 1);
        download_async(linalg, &solver_information, args.workspace.solver_information, 1);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (err_state() != QNPEPS_OK) return;
        if (solver_information != 0 or (record.flags & 0x80000000_u32) != 0)
            return set_err(QNPEPS_ERR_INTERNAL), void();
        const auto active_rank = record.active_rank;
        zero_async(linalg, basis, basis_count);
        const auto gather_basis_args = GatherBasisArgs{
            .solver_vectors = args.workspace.density,
            .sorted_indices = args.workspace.sort_indices,
            .active_rank = args.workspace.active_ranks + step,
            .order = order,
            .upper_bond = args.upper_bond,
            .basis = basis,
        };
        cu_gather_basis<<<launch_count(order * args.upper_bond), k_threads_per_block, 0, stream>>>(
            gather_basis_args
        );
        CUDA_CHECK(cudaGetLastError());
        matmul(
            linalg,
            CuMatrixCF64Const{basis, static_cast<int>(order), static_cast<int>(active_rank)},
            CuMatrixCF64Const{basis, static_cast<int>(order), static_cast<int>(active_rank)},
            CuMatrixCF64{args.workspace.density, static_cast<int>(order), static_cast<int>(order)},
            BlasOp::none,
            BlasOp::conj_trans
        );
        const auto& output_site = args.sites[reverse_index];
        auto* output = args.result_values + output_site.output_offset;
        const auto output_slot = upper_bond * static_cast<usize>(site.physical_output) * upper_bond;
        zero_async(linalg, output, output_slot);
        const auto pack_site_args = PackSiteArgs{
            .basis = basis,
            .physical_output = site.physical_output,
            .right_rank = right_rank,
            .left_rank = active_rank,
            .output = output,
        };
        cu_pack_site<<<
            launch_count(site.physical_output * right_rank * active_rank),
            k_threads_per_block,
            0,
            stream>>>(pack_site_args);
        const auto dimensions_args = DimensionsArgs{
            .site = static_cast<i64>(reverse_index),
            .left = static_cast<i32>(site.physical_output),
            .physical = static_cast<i32>(right_rank),
            .right = static_cast<i32>(active_rank),
            .dimensions = args.result_dimensions,
        };
        cu_dimensions<<<1, 1, 0, stream>>>(dimensions_args);
        const auto project_right_args = ProjectRightArgs{
            .right_block = right_block,
            .basis = basis,
            .combined_left = combined_left,
            .order = order,
            .active_rank = active_rank,
            .carried = temporary,
        };
        cu_project_right<<<
            launch_count(combined_left * active_rank),
            k_threads_per_block,
            0,
            stream>>>(project_right_args);
        CUDA_CHECK(cudaGetLastError());
        right_rank = active_rank;
    }
    const auto& first = args.sites[0];
    const auto first_right = first.state_right * first.operator_right;
    const auto first_count =
        static_cast<usize>(first.physical_output) * static_cast<usize>(right_rank);
    if (args.product)
    {
        for (auto output_state = 0_i64; output_state < first.physical_output; ++output_state)
        {
            form_product(linalg, args, first, output_state, product_values);
            const auto right_block_slice_args = RightBlockSliceArgs{
                .product = product_values,
                .carried = temporary,
                .combined_left = 1_i64,
                .combined_right = first_right,
                .physical_output = first.physical_output,
                .output_state = output_state,
                .right_rank = right_rank,
                .right_block = right_block,
            };
            cu_right_block_slice<<<launch_count(right_rank), k_threads_per_block, 0, stream>>>(
                right_block_slice_args
            );
        }
    }
    else
    {
        form_product(linalg, args, first, 0_i64, product_values);
        const auto right_block_args = RightBlockArgs{
            .product = product_values,
            .carried = temporary,
            .combined_left = 1_i64,
            .combined_right = first_right,
            .physical_output = first.physical_output,
            .right_rank = right_rank,
            .right_block = right_block,
        };
        cu_right_block<<<
            launch_count(static_cast<i64>(first_count)),
            k_threads_per_block,
            0,
            stream>>>(right_block_args);
    }
    const auto first_slot = upper_bond * static_cast<usize>(first.physical_output) * upper_bond;
    auto* first_output = args.result_values + first.output_offset;
    zero_async(linalg, first_output, first_slot);
    const auto pack_first_args = PackFirstArgs{
        .right_block = right_block,
        .physical_output = first.physical_output,
        .right_rank = right_rank,
        .output = first_output,
    };
    cu_pack_first<<<launch_count(static_cast<i64>(first_count)), k_threads_per_block, 0, stream>>>(
        pack_first_args
    );
    const auto dimensions_args = DimensionsArgs{
        .site = 0_i64,
        .left = static_cast<i32>(first.physical_output),
        .physical = static_cast<i32>(right_rank),
        .right = 1_i32,
        .dimensions = args.result_dimensions,
    };
    cu_dimensions<<<1, 1, 0, stream>>>(dimensions_args);
    const auto normalize_args = NormalizeArgs{
        .values = first_output,
        .count = first_count,
        .normalize = args.normalize,
        .input_gauge = args.input_gauge,
        .normalization_log = args.normalization_log,
        .output_gauge = args.output_gauge,
    };
    cu_normalize<<<1, k_threads_per_block, 0, stream>>>(normalize_args);
    CUDA_CHECK(cudaGetLastError());
    if (args.active_ranks_out)
        copy_device_async(linalg, args.active_ranks_out, args.workspace.active_ranks, cuts);
    if (args.records_out) copy_device_async(linalg, args.records_out, args.workspace.records, cuts);
}
}
