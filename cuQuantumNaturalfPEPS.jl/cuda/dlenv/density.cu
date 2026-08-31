#include "core/complex.cuh"
#include "core/cuda_utils.cuh"
#include "densitymatrix/types.cuh"
#include "dlenv/density.cuh"
#include "dlenv/kernels.cuh"
#include "linalg/transfer.cuh"

#include <algorithm>
#include <cmath>
#include <cuda/std/cmath>
#include <span>
#include <vector>

namespace qnpeps::dlenv
{
namespace
{
struct ProductContext
{
    std::span<const DeviceTensor> row_ket{};
    int dim_phys{};
};

auto launch_count(usize count) -> u32
{
    return grid_blocks_capped(static_cast<i64>(std::max(count, 1_uz)));
}

auto product(
    Linalg& linalg,
    const densitymatrix::Site& site,
    i64 output_state,
    const cuDoubleComplex* state,
    const cuDoubleComplex*,
    cuDoubleComplex* output,
    const void* context_value
) -> void
{
    const auto& context = *static_cast<const ProductContext*>(context_value);
    if (site.site >= context.row_ket.size()) return set_err(QNPEPS_ERR_INTERNAL), void();
    const auto& ket = context.row_ket[site.site];
    const auto combined_left = site.state_left * site.operator_left;
    const auto combined_right = site.state_right * site.operator_right;
    const auto product_args = ProductArgs{
        .state = state + site.state_offset,
        .ket = ket.d,
        .state_left = site.state_left,
        .state_right = site.state_right,
        .down = ket.dim[3],
        .left = ket.dim[4],
        .up = ket.dim[1],
        .right = ket.dim[2],
        .dim_phys = context.dim_phys,
        .output_state = output_state,
        .product = output,
    };
    cu_product<<<
        grid_blocks_capped(combined_left * combined_right),
        k_threads_per_block,
        0,
        linalg.stream()>>>(product_args);
    CUDA_CHECK(cudaGetLastError());
}

}

auto density_row(
    Linalg& linalg,
    ArenaCursor& known,
    ArenaCursor& arena,
    const DensityRowArgs& args,
    f64& output_gauge
) -> std::vector<DeviceTensor>
{
    const auto sites = args.row_ket.size();
    std::vector<DeviceTensor> output{};
    output.resize(sites);
    const auto invalid_args = sites < 2 or args.environment.size() != sites or args.maxdim < 1
                              or not std::isfinite(args.cutoff) or args.cutoff < 0.0
                              or not args.device_scales;
    if (invalid_args)
    {
        set_err(QNPEPS_ERR_BAD_CONFIG);
        return output;
    }
    const auto dim_phys = args.row_ket.front().dim[0];
    const auto dim_bond = std::max({
        args.row_ket.front().dim[1],
        args.row_ket.front().dim[2],
        args.row_ket.front().dim[3],
        args.row_ket.front().dim[4],
    });
    const auto dim_bond_u = static_cast<usize>(dim_bond);
    const auto maxdim = static_cast<usize>(args.maxdim);
    const auto bond_pair = dim_bond_u * dim_bond_u;
    const auto state_slot = maxdim * bond_pair * maxdim;
    const auto state_capacity = sites * state_slot;
    const auto result_capacity = state_capacity;
    arena.rewind();
    auto workspace = densitymatrix::take_workspace(
        linalg,
        arena,
        {.num_sites = static_cast<i64>(sites),
         .input_bond = args.maxdim,
         .operator_bond = static_cast<i64>(bond_pair),
         .output_dimension = static_cast<i64>(bond_pair),
         .upper_bond = args.maxdim,
         .lanes = 1}
    );
    auto* state_values = arena.take<cuDoubleComplex>(state_capacity);
    auto* result_values = arena.take<cuDoubleComplex>(result_capacity);
    auto* result_dimensions = arena.take<i32>(3 * sites);
    auto* normalization_log = arena.take<f64>(1);
    auto* device_output_gauge = arena.take<f64>(1);
    if (err_state() != QNPEPS_OK) return output;

    std::vector<densitymatrix::Site> plans{};
    plans.reserve(sites);
    auto state_offset = 0_uz;
    const auto physical_output =
        static_cast<i64>(args.row_ket.front().dim[1]) * args.row_ket.front().dim[1];
    const auto output_slot = maxdim * static_cast<usize>(physical_output) * maxdim;
    for (auto site_index = 0_uz; site_index < sites; ++site_index)
    {
        const auto& state = args.environment[site_index];
        const auto& ket = args.row_ket[site_index];
        const auto invalid_site = state.dim.rank() != 4 or ket.dim.rank() != 5
                                  or state.dim[1] != ket.dim[3] or state.dim[2] != ket.dim[3];
        if (invalid_site)
        {
            set_err(QNPEPS_ERR_BAD_CONFIG);
            return output;
        }
        const auto promote_args = PromoteArgs{
            .input = state.d,
            .count = state.num_elems(),
            .output = state_values + state_offset,
        };
        cu_promote<<<launch_count(state.num_elems()), k_threads_per_block, 0, linalg.stream()>>>(
            promote_args
        );
        plans.push_back(
            {.site = site_index,
             .num_sites = sites,
             .state_left = state.dim[0],
             .physical_input = state.dim[1] * state.dim[2],
             .state_right = state.dim[3],
             .operator_left = ket.dim[4] * ket.dim[4],
             .physical_output = ket.dim[1] * ket.dim[1],
             .operator_right = ket.dim[2] * ket.dim[2],
             .state_offset = state_offset,
             .operator_offset = 0,
             .output_offset = site_index * output_slot}
        );
        state_offset += state.num_elems();
    }
    CUDA_CHECK(cudaGetLastError());
    if (state_offset > state_capacity or err_state() != QNPEPS_OK) return output;
    const ProductContext context{args.row_ket, dim_phys};
    densitymatrix::apply(
        linalg,
        {.settings =
             {.struct_size = static_cast<u32>(sizeof(QnpepsDensitySettings)),
              .precision = 1_u32,
              .mindim = 1_u32,
              .reserved = 0_u32,
              .relative_cutoff = args.cutoff},
         .upper_bond = args.maxdim,
         .normalize = true,
         .input_gauge = args.input_gauge,
         .sites = plans,
         .workspace = workspace,
         .state_values = state_values,
         .state_value_count = state_offset,
         .operator_values = nullptr,
         .operator_value_count = 0,
         .result_dimensions = result_dimensions,
         .result_dimension_count = 3 * sites,
         .result_values = result_values,
         .result_value_count = result_capacity,
         .normalization_log = normalization_log,
         .output_gauge = device_output_gauge,
         .product = product,
         .product_context = &context}
    );
    if (err_state() != QNPEPS_OK) return output;
    std::vector<i32> dimensions{};
    dimensions.resize(3 * sites);
    download_async(linalg, dimensions.data(), result_dimensions, dimensions.size());
    download_async(linalg, &output_gauge, device_output_gauge, 1);
    CUDA_CHECK(cudaStreamSynchronize(linalg.stream()));
    if (err_state() != QNPEPS_OK) return output;
    for (auto site_index = 0_uz; site_index < sites; ++site_index)
    {
        const auto offset = 3 * site_index;
        const auto physical = static_cast<i64>(dimensions[offset]);
        const auto right = static_cast<i64>(dimensions[offset + 1]);
        const auto left = static_cast<i64>(dimensions[offset + 2]);
        const auto vertical = args.row_ket[site_index].dim[1];
        if (physical != static_cast<i64>(vertical) * vertical or left < 1 or right < 1)
        {
            set_err(QNPEPS_ERR_INTERNAL);
            return output;
        }
        auto* packed = known.take<cuFloatComplex>(static_cast<usize>(left * physical * right));
        const auto pack_args = PackArgs{
            .input = result_values + site_index * output_slot,
            .physical = physical,
            .right = right,
            .left = left,
            .output = packed,
        };
        cu_pack<<<
            grid_blocks_capped(left * physical * right),
            k_threads_per_block,
            0,
            linalg.stream()>>>(pack_args);
        output[site_index] = DeviceTensor{
            {static_cast<int>(left), vertical, vertical, static_cast<int>(right)}, packed
        };
    }
    const auto scale_args = ScaleArgs{
        .normalization_log = normalization_log,
        .count = sites,
        .scales = args.device_scales,
    };
    cu_scales<<<launch_count(sites), k_threads_per_block, 0, linalg.stream()>>>(scale_args);
    CUDA_CHECK(cudaGetLastError());
    return output;
}
}
