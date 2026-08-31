#include "core/complex.cuh"
#include "core/cuda_utils.cuh"
#include "densitymatrix/kernels.cuh"

#include <cmath>
#include <cuda/std/cmath>

namespace qnpeps::densitymatrix
{
__device__ auto add(cuDoubleComplex left, cuDoubleComplex right) -> cuDoubleComplex
{
    return cuCadd(left, right);
}

__device__ auto multiply(cuDoubleComplex left, cuDoubleComplex right) -> cuDoubleComplex
{
    return cuCmul(left, right);
}

__global__ auto cu_local_product(LocalProductArgs args) -> void
{
    const auto combined_left = args.state_left * args.operator_left;
    const auto combined_right = args.state_right * args.operator_right;
    const auto count = combined_left * combined_right * args.physical_output;
    for (auto lane = global_lane(); lane < count; lane += grid_stride())
    {
        const auto left = lane % combined_left;
        const auto tail = lane / combined_left;
        const auto right = tail % combined_right;
        const auto output_state = tail / combined_right;
        const auto state_left = left % args.state_left;
        const auto operator_left = left / args.state_left;
        const auto state_right = right % args.state_right;
        const auto operator_right = right / args.state_right;
        auto value = make_cuDoubleComplex(0.0, 0.0);
        for (auto physical = 0_i64; physical < args.physical_input; ++physical)
        {
            const auto state_index =
                state_left + args.state_left * (physical + args.physical_input * state_right);
            const auto operator_index =
                operator_left
                + args.operator_left
                      * (physical
                         + args.physical_input
                               * (output_state + args.physical_output * operator_right));
            value =
                add(value, multiply(args.state[state_index], args.operator_values[operator_index]));
        }
        args.product[lane] = value;
    }
}

__global__ auto cu_right_block(RightBlockArgs args) -> void
{
    const auto order = args.physical_output * args.right_rank;
    const auto count = args.combined_left * order;
    for (auto lane = global_lane(); lane < count; lane += grid_stride())
    {
        const auto left = lane % args.combined_left;
        const auto q = lane / args.combined_left;
        const auto output_state = q % args.physical_output;
        const auto rank = q / args.physical_output;
        auto value = make_cuDoubleComplex(0.0, 0.0);
        const auto product_offset = args.combined_left * args.combined_right * output_state;
        for (auto right = 0_i64; right < args.combined_right; ++right)
        {
            value =
                add(value,
                    multiply(
                        args.product[product_offset + left + args.combined_left * right],
                        args.carried[right + args.combined_right * rank]
                    ));
        }
        args.right_block[lane] = value;
    }
}

__global__ auto cu_right_block_slice(RightBlockSliceArgs args) -> void
{
    const auto count = args.combined_left * args.right_rank;
    for (auto lane = global_lane(); lane < count; lane += grid_stride())
    {
        const auto left = lane % args.combined_left;
        const auto rank = lane / args.combined_left;
        auto value = make_cuDoubleComplex(0.0, 0.0);
        for (auto right = 0_i64; right < args.combined_right; ++right)
        {
            value =
                add(value,
                    multiply(
                        args.product[left + args.combined_left * right],
                        args.carried[right + args.combined_right * rank]
                    ));
        }
        const auto q = args.output_state + args.physical_output * rank;
        args.right_block[left + args.combined_left * q] = value;
    }
}

__global__ auto cu_filter(FilterArgs args) -> void
{
    if (global_lane() != 0_i64) return;
    auto flags = 0_u32;
    for (auto index = 0_i64; index < args.order; ++index)
        args.sorted_indices[index] = static_cast<i32>(index);
    for (auto index = 1_i64; index < args.order; ++index)
    {
        const auto selected = args.sorted_indices[index];
        const auto selected_value = args.solver_values[selected];
        auto position = index;
        while (position > 0_i64)
        {
            const auto previous = args.sorted_indices[position - 1_i64];
            const auto previous_value = args.solver_values[previous];
            const auto selected_abs = cuda::std::abs(selected_value);
            const auto previous_abs = cuda::std::abs(previous_value);
            const auto ordered = selected_abs < previous_abs
                                 or (selected_abs == previous_abs and selected > previous);
            if (ordered) break;
            args.sorted_indices[position] = previous;
            --position;
        }
        args.sorted_indices[position] = selected;
    }
    for (auto index = 0_i64; index < args.order; ++index)
    {
        const auto value = args.solver_values[args.sorted_indices[index]];
        args.sorted_values[index] = value;
        if (not cuda::std::isfinite(value)) flags |= 0x80000000_u32;
    }
    const auto sign_value = args.sorted_values[0] < 0.0 ? -1.0 : 1.0;
    if (sign_value < 0.0) flags |= 1_u32;
    for (auto index = 0_i64; index < args.order; ++index)
        args.retained_values[index] = args.sorted_values[index] * sign_value;
    for (auto index = args.order - 1_i64; index >= 0_i64; --index)
    {
        if (args.retained_values[index] >= 0.0) break;
        args.retained_values[index] = 0.0;
        flags |= 2_u32;
    }
    auto scale = 0.0;
    for (auto index = 0_i64; index < args.order; ++index)
        scale += args.retained_values[index];
    if (scale == 0.0) scale = 1.0;
    auto rank = args.order < args.applied_cap ? args.order : args.applied_cap;
    auto discarded = 0.0;
    for (auto index = args.order - 1_i64; index >= rank; --index)
        discarded += args.retained_values[index];
    if (rank < args.order) flags |= 4_u32;
    const auto rank_after_cap = rank;
    auto below_cutoff = discarded + args.retained_values[rank - 1_i64] <= args.cutoff * scale;
    while (rank > args.mindim and below_cutoff)
    {
        discarded += args.retained_values[rank - 1_i64];
        --rank;
        flags |= 8_u32;
        below_cutoff = discarded + args.retained_values[rank - 1_i64] <= args.cutoff * scale;
    }
    if (rank < 1_i64) rank = 1_i64;
    auto docut = 0.0;
    if (rank < args.order)
    {
        docut = (args.retained_values[rank - 1_i64] + args.retained_values[rank]) / 2.0;
        const auto near_degenerate =
            cuda::std::abs(args.retained_values[rank - 1_i64] - args.retained_values[rank])
            < 1.0e-3 * args.retained_values[rank - 1_i64];
        if (near_degenerate) docut += 1.0e-3 * args.retained_values[rank - 1_i64];
    }
    for (auto index = 0_i64; index < rank; ++index)
        args.retained_values[index] *= sign_value;
    for (auto index = rank; index < args.order; ++index)
        args.retained_values[index] = 0.0;
    *args.active_rank = static_cast<i32>(rank);
    *args.record = QnpepsDensityRankRecord{
        .struct_size = static_cast<u32>(sizeof(QnpepsDensityRankRecord)),
        .flags = flags,
        .matrix_order = args.order,
        .natural_cap = args.natural_cap,
        .applied_cap = args.applied_cap,
        .rank_after_cap = rank_after_cap,
        .active_rank = rank,
        .hard_discarded = args.order - rank_after_cap,
        .cutoff_discarded = rank_after_cap - rank,
        .truncation_error = discarded / scale,
        .docut = docut,
        .retained_sum = scale,
        .discarded_weight = discarded,
    };
}

__global__ auto cu_gather_basis(GatherBasisArgs args) -> void
{
    const auto count = args.order * args.upper_bond;
    const auto rank = static_cast<i64>(*args.active_rank);
    for (auto lane = global_lane(); lane < count; lane += grid_stride())
    {
        const auto row = lane % args.order;
        const auto column = lane / args.order;
        args.basis[lane] = column < rank
                               ? args.solver_vectors[row + args.order * args.sorted_indices[column]]
                               : make_cuDoubleComplex(0.0, 0.0);
    }
}

__global__ auto cu_project_right(ProjectRightArgs args) -> void
{
    const auto count = args.combined_left * args.active_rank;
    for (auto lane = global_lane(); lane < count; lane += grid_stride())
    {
        const auto left = lane % args.combined_left;
        const auto rank = lane / args.combined_left;
        auto value = make_cuDoubleComplex(0.0, 0.0);
        for (auto q = 0_i64; q < args.order; ++q)
        {
            value =
                add(value,
                    multiply(
                        args.right_block[left + args.combined_left * q],
                        cuConj(args.basis[q + args.order * rank])
                    ));
        }
        args.carried[lane] = value;
    }
}

__global__ auto cu_pack_site(PackSiteArgs args) -> void
{
    const auto count = args.physical_output * args.right_rank * args.left_rank;
    for (auto lane = global_lane(); lane < count; lane += grid_stride())
        args.output[lane] = args.basis[lane];
}

__global__ auto cu_pack_first(PackFirstArgs args) -> void
{
    const auto count = args.physical_output * args.right_rank;
    for (auto lane = global_lane(); lane < count; lane += grid_stride())
        args.output[lane] = args.right_block[lane];
}

__global__ auto cu_dimensions(DimensionsArgs args) -> void
{
    if (global_lane() != 0_i64) return;
    const auto offset = 3_i64 * args.site;
    args.dimensions[offset] = args.left;
    args.dimensions[offset + 1_i64] = args.physical;
    args.dimensions[offset + 2_i64] = args.right;
}

__global__ auto cu_normalize(NormalizeArgs args) -> void
{
    __shared__ qnpeps::CuArray<f64, k_threads_per_block> reduction;
    auto sum = 0.0;
    for (auto index = static_cast<usize>(threadIdx.x); index < args.count; index += blockDim.x)
    {
        const auto value = args.values[index];
        sum += norm2(value);
    }
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (auto stride = blockDim.x / 2_u32; stride > 0_u32; stride /= 2_u32)
    {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    const auto norm = cuda::std::sqrt(reduction[0]);
    if (args.normalize and norm > 0.0)
    {
        for (auto index = static_cast<usize>(threadIdx.x); index < args.count; index += blockDim.x)
        {
            args.values[index].x /= norm;
            args.values[index].y /= norm;
        }
    }
    if (threadIdx.x == 0_u32)
    {
        const auto log_value = args.normalize and norm > 0.0 ? cuda::std::log(norm) : 0.0;
        *args.normalization_log = log_value;
        *args.output_gauge = args.input_gauge + log_value;
    }
}
}
