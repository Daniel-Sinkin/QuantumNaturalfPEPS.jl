#include "core/complex.cuh"
#include "core/cuda_utils.cuh"
#include "densitymatrix/kernels.cuh"

#include <cmath>
#include <cuda/std/cmath>

namespace qnpeps::densitymatrix
{
__global__ auto cu_promote(PromoteArgs args) -> void
{
    for (auto lane = global_lane(); lane < static_cast<i64>(args.count); lane += grid_stride())
    {
        args.output[lane] = make_cuDoubleComplex(
            static_cast<f64>(args.input[lane].x), static_cast<f64>(args.input[lane].y)
        );
    }
}

__global__ auto cu_pack(PackArgs args) -> void
{
    const auto count = args.left * args.physical * args.right;
    for (auto lane = global_lane(); lane < count; lane += grid_stride())
    {
        const auto left = lane % args.left;
        const auto tail = lane / args.left;
        const auto physical = tail % args.physical;
        const auto right = tail / args.physical;
        const auto value = args.input[physical + args.physical * (right + args.right * left)];
        args.output[lane] = to_cf<cuFloatComplex>(value);
    }
}

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
    const auto sign_value = args.solver_values[args.sorted_indices[0]] < 0.0 ? -1.0 : 1.0;
    if (sign_value < 0.0) flags |= 1_u32;
    for (auto index = 1_i64; index < args.order; ++index)
    {
        const auto selected = args.sorted_indices[index];
        const auto selected_signed = args.solver_values[selected] * sign_value;
        const auto selected_mass = selected_signed > 0.0 ? selected_signed : 0.0;
        auto position = index;
        while (position > 0_i64)
        {
            const auto previous = args.sorted_indices[position - 1_i64];
            const auto previous_signed = args.solver_values[previous] * sign_value;
            const auto previous_mass = previous_signed > 0.0 ? previous_signed : 0.0;
            if (selected_mass <= previous_mass) break;
            args.sorted_indices[position] = previous;
            --position;
        }
        args.sorted_indices[position] = selected;
    }
    for (auto index = 0_i64; index < args.order; ++index)
        args.sorted_values[index] = args.solver_values[args.sorted_indices[index]];
    for (auto index = 0_i64; index < args.order; ++index)
    {
        const auto value = args.sorted_values[index];
        const auto signed_value = value * sign_value;
        args.retained_values[index] = signed_value < 0.0 ? 0.0 : signed_value;
        if (signed_value < 0.0) flags |= 2_u32;
        if (not cuda::std::isfinite(value)) flags |= 0x80000000_u32;
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
        args.basis[lane] =
            column < rank
                ? cuConj(args.solver_vectors[row + args.order * args.sorted_indices[column]])
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

__global__ auto cu_batched_unit(cuDoubleComplex* values, i64 lane_stride, i64 lanes) -> void
{
    for (auto lane = global_lane(); lane < lanes; lane += grid_stride())
        values[lane * lane_stride] = cuDoubleComplex{1.0, 0.0};
}

__global__ auto cu_batched_products(BatchedProductArgs args) -> void
{
    const auto combined_left = args.state_left * args.operator_left;
    const auto combined_right = args.state_right * args.operator_right;
    const auto lane_count = combined_left * combined_right * args.physical_output;
    const auto count = lane_count * args.lanes;
    const auto& strides = args.operator_strides;
    for (auto index = global_lane(); index < count; index += grid_stride())
    {
        const auto lane = index / lane_count;
        const auto element = index % lane_count;
        const auto left = element % combined_left;
        const auto tail = element / combined_left;
        const auto right = tail % combined_right;
        const auto output_state = tail / combined_right;
        const auto state_left = left % args.state_left;
        const auto operator_left = left / args.state_left;
        const auto state_right = right % args.state_right;
        const auto operator_right = right / args.state_right;
        const auto* state = args.state + lane * args.state_lane_stride;
        const auto* operator_values = args.operator_values + lane * args.operator_lane_stride;
        const auto operator_base = operator_left * strides.left + output_state * strides.output
                                   + operator_right * strides.right;
        cuDoubleComplex value{};
        for (auto physical = 0_i64; physical < args.physical_input; ++physical)
        {
            const auto state_input = state
                [state_left + args.state_left * (physical + args.physical_input * state_right)];
            const auto operator_input = operator_values[operator_base + physical * strides.input];
            const cuDoubleComplex state_value{
                static_cast<f64>(state_input.x), static_cast<f64>(state_input.y)
            };
            const cuDoubleComplex operator_value{
                static_cast<f64>(operator_input.x), static_cast<f64>(operator_input.y)
            };
            value = cuCadd(value, cuCmul(state_value, operator_value));
        }
        args.products[lane * args.product_lane_stride + element] = value;
    }
}

__global__ auto cu_batched_boundary(BatchedBoundaryArgs args) -> void
{
    __shared__ qnpeps::CuArray<f64, k_threads_per_block> reduction;
    const auto lane = static_cast<i64>(blockIdx.x);
    const auto* input = args.input + lane * args.input_lane_stride;
    auto* output = args.output + lane * args.output_lane_stride;
    const auto& strides = args.input_strides;
    const auto count = args.left * args.vertical * args.right;
    const auto source = [&](i64 index) -> i64
    {
        const auto left = index % args.left;
        const auto tail = index / args.left;
        const auto vertical = tail % args.vertical;
        const auto right = tail / args.vertical;
        return left * strides.left + vertical * strides.output + right * strides.right;
    };
    auto sum = 0.0;
    for (auto index = static_cast<i64>(threadIdx.x); index < count; index += blockDim.x)
    {
        const auto value = input[source(index)];
        sum += static_cast<f64>(value.x) * value.x + static_cast<f64>(value.y) * value.y;
    }
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (auto stride = blockDim.x / 2_u32; stride > 0_u32; stride /= 2_u32)
    {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    const auto norm = cuda::std::sqrt(reduction[0]);
    const auto scale = norm > 0.0 ? 1.0 / norm : 1.0;
    for (auto index = static_cast<i64>(threadIdx.x); index < count; index += blockDim.x)
    {
        const auto value = input[source(index)];
        output[index] =
            cuFloatComplex{static_cast<f32>(value.x * scale), static_cast<f32>(value.y * scale)};
    }
    if (threadIdx.x != 0_u32) return;
    if (norm > 0.0) args.log_gauge[lane] += cuda::std::log(norm);
    if (not cuda::std::isfinite(norm)) args.lane_flags[lane] |= k_lane_nonfinite_norm;
}

__global__ auto cu_batched_right_block(BatchedRightBlockArgs args) -> void
{
    const auto order = args.physical_output * args.right_rank;
    const auto lane_count = args.combined_left * order;
    const auto panel = args.combined_left * args.combined_right;
    const auto count = lane_count * args.lanes;
    for (auto index = global_lane(); index < count; index += grid_stride())
    {
        const auto lane = index / lane_count;
        const auto element = index % lane_count;
        const auto left = element % args.combined_left;
        const auto column = element / args.combined_left;
        const auto output_state = column % args.physical_output;
        const auto rank = column / args.physical_output;
        const auto* product =
            args.products + lane * args.product_lane_stride + panel * output_state;
        const auto* carried = args.carried + lane * args.carried_lane_stride;
        cuDoubleComplex value{};
        for (auto right = 0_i64; right < args.combined_right; ++right)
        {
            value = cuCadd(
                value,
                cuCmul(
                    product[left + args.combined_left * right],
                    carried[right + args.combined_right * rank]
                )
            );
        }
        args.right_block[lane * args.block_lane_stride + element] = value;
    }
}

__global__ auto cu_batched_select(BatchedSelectArgs args) -> void
{
    __shared__ f64 sign_value;
    const auto lane = static_cast<i64>(blockIdx.x);
    const auto* values = args.eigenvalues + lane * args.eigenvalue_lane_stride;
    auto* sorted = args.sorted_indices + lane * args.index_lane_stride;
    for (auto index = static_cast<i64>(threadIdx.x); index < args.order; index += blockDim.x)
        sorted[index] = static_cast<i32>(index);
    if (threadIdx.x == 0_u32)
    {
        auto leading = 0_i64;
        for (auto index = 1_i64; index < args.order; ++index)
        {
            if (cuda::std::abs(values[index]) > cuda::std::abs(values[leading])) leading = index;
        }
        sign_value = values[leading] < 0.0 ? -1.0 : 1.0;
    }
    __syncthreads();
    const auto mass = [&](i64 index) -> f64
    {
        const auto signed_value = values[index] * sign_value;
        return signed_value > 0.0 ? signed_value : 0.0;
    };
    const auto is_ranked_before = [&](i64 candidate, i64 index) -> bool
    {
        const auto candidate_mass = mass(candidate);
        const auto index_mass = mass(index);
        if (candidate_mass != index_mass) return candidate_mass > index_mass;
        const auto candidate_magnitude = cuda::std::abs(values[candidate]);
        const auto index_magnitude = cuda::std::abs(values[index]);
        if (candidate_magnitude != index_magnitude) return candidate_magnitude > index_magnitude;
        return candidate < index;
    };
    auto has_nonfinite = false;
    for (auto index = static_cast<i64>(threadIdx.x); index < args.order; index += blockDim.x)
    {
        has_nonfinite = has_nonfinite or not cuda::std::isfinite(values[index]);
        auto position = 0_i64;
        for (auto candidate = 0_i64; candidate < args.order; ++candidate)
        {
            if (candidate != index and is_ranked_before(candidate, index)) ++position;
        }
        sorted[position] = static_cast<i32>(index);
    }
    const auto is_spectrum_nonfinite = __syncthreads_or(has_nonfinite ? 1 : 0) != 0;
    if (threadIdx.x != 0_u32) return;
    auto flags = 0_u32;
    if (args.solver_information[lane] != 0) flags |= k_lane_solver_failed;
    if (is_spectrum_nonfinite) flags |= k_lane_nonfinite_spectrum;
    args.lane_flags[lane] |= flags;
}

__global__ auto cu_batched_pack_site(BatchedPackSiteArgs args) -> void
{
    const auto& basis = args.basis;
    const auto lane_count = basis.rank * basis.order;
    const auto count = lane_count * basis.lanes;
    for (auto index = global_lane(); index < count; index += grid_stride())
    {
        const auto lane = index / lane_count;
        const auto element = index % lane_count;
        const auto kept = element % basis.rank;
        const auto row = element / basis.rank;
        const auto source = basis.sorted_indices[lane * basis.index_lane_stride + kept];
        const auto* vectors = basis.vectors + lane * basis.vector_lane_stride;
        const auto value = cuConj(vectors[row + basis.order * source]);
        args.output[lane * args.output_lane_stride + element] = to_cf<cuFloatComplex>(value);
    }
}

__global__ auto cu_batched_project(BatchedProjectArgs args) -> void
{
    const auto& basis = args.basis;
    const auto lane_count = args.combined_left * basis.rank;
    const auto count = lane_count * basis.lanes;
    for (auto index = global_lane(); index < count; index += grid_stride())
    {
        const auto lane = index / lane_count;
        const auto element = index % lane_count;
        const auto left = element % args.combined_left;
        const auto kept = element / args.combined_left;
        const auto source = basis.sorted_indices[lane * basis.index_lane_stride + kept];
        const auto* vector = basis.vectors + lane * basis.vector_lane_stride + basis.order * source;
        const auto* block = args.right_block + lane * args.block_lane_stride;
        cuDoubleComplex value{};
        for (auto row = 0_i64; row < basis.order; ++row)
            value = cuCadd(value, cuCmul(block[left + args.combined_left * row], vector[row]));
        args.carried[lane * args.carried_lane_stride + element] = value;
    }
}

__global__ auto cu_batched_normalize(BatchedNormalizeArgs args) -> void
{
    __shared__ qnpeps::CuArray<f64, k_threads_per_block> reduction;
    const auto lane = static_cast<i64>(blockIdx.x);
    const auto* values = args.values + lane * args.value_lane_stride;
    auto* output = args.output + lane * args.output_lane_stride;
    auto sum = 0.0;
    for (auto index = static_cast<i64>(threadIdx.x); index < args.count; index += blockDim.x)
        sum += norm2(values[index]);
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (auto stride = blockDim.x / 2_u32; stride > 0_u32; stride /= 2_u32)
    {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    const auto norm = cuda::std::sqrt(reduction[0]);
    const auto is_scaled = norm > 0.0;
    for (auto index = static_cast<i64>(threadIdx.x); index < args.count; index += blockDim.x)
    {
        const auto value = values[index];
        const cuDoubleComplex scaled{value.x / norm, value.y / norm};
        output[index] = to_cf<cuFloatComplex>(is_scaled ? scaled : value);
    }
    if (threadIdx.x != 0_u32) return;
    args.log_gauge[lane] += is_scaled ? cuda::std::log(norm) : 0.0;
    if (not cuda::std::isfinite(norm)) args.lane_flags[lane] |= k_lane_nonfinite_norm;
}
}
