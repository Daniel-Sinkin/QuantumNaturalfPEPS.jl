#include "core/arena_cursor.cuh"
#include "core/cuda_utils.cuh"
#include "linalg/transfer.cuh"
#include "tensor/kernels.cuh"
#include "tensor/permutation.cuh"

#include <algorithm>
#include <array>
#include <cassert>
#include <cuda_runtime.h>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

namespace qnpeps
{
namespace
{
[[nodiscard]] auto column_major_strides(const Shape& shape) -> std::vector<i64>
{
    std::vector<i64> strides{};
    strides.assign(shape.rank(), 1);
    for (auto axis = 1_uz; axis < shape.rank(); ++axis)
        strides[axis] = strides[axis - 1] * shape[axis - 1];
    return strides;
}

[[nodiscard]] auto permutation_index_map(const Shape& input_shape, const Permutation& permutation)
    -> std::vector<int>
{
    if (not permutation.can_apply_to(input_shape))
    {
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return {};
    }
    const auto output_shape = permutation.apply(input_shape);
    const auto rank = input_shape.rank();
    const auto total = output_shape.num_elems();
    if (total > static_cast<usize>(std::numeric_limits<int>::max()))
    {
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return {};
    }
    const auto input_strides = column_major_strides(input_shape);

    std::vector<int> index_map{};
    index_map.resize(total);
    std::vector<int> coordinates{};
    coordinates.resize(rank);
    for (auto output_position = 0_uz; output_position < total; ++output_position)
    {
        i64 input_position{};
        for (auto axis = 0_uz; axis < rank; ++axis)
        {
            const auto input_axis = static_cast<usize>(permutation[axis]);
            input_position += coordinates[axis] * input_strides[input_axis];
        }
        index_map[output_position] = static_cast<int>(input_position);
        for (auto axis = 0_uz; axis < rank; ++axis)
        {
            coordinates[axis] += 1;
            if (coordinates[axis] < output_shape[axis]) break;
            coordinates[axis] = 0;
        }
    }
    return index_map;
}

}

auto PermutationCache::get_or_create(const Shape& shape, const Permutation& permutation) -> int*
{
    if (not permutation.can_apply_to(shape))
    {
        set_err(QNPEPS_ERR_INTERNAL);
        return nullptr;
    }
    const Key key{shape.get(), permutation.get()};
    if (const auto it = entries_.find(key); it != entries_.end())
    {
        return it->second;
    }

    const auto gather_indices = permutation_index_map(shape, permutation);
    if (err_state() != QNPEPS_OK) return nullptr;
    int* device_ptr{};
    CUDA_CHECK(cudaMalloc(&device_ptr, gather_indices.size() * sizeof(int)));
    if (err_state() != QNPEPS_OK) return nullptr;
    upload(device_ptr, gather_indices.data(), gather_indices.size());
    if (err_state() != QNPEPS_OK)
    {
        CUDA_NOCHECK(cudaFree(device_ptr));
        return nullptr;
    }
    entries_.emplace(key, device_ptr);
    return device_ptr;
}

auto PermutationCache::release() -> void
{
    for (auto& entry : entries_)
        maybe_free_device(entry.second);
    entries_.clear();
}

auto permute_axes(
    const DeviceTensor& tensor,
    const Permutation& permutation,
    bool conjugate,
    cuFloatComplex* output,
    cudaStream_t stream
) -> void
{
    if (err_state() != QNPEPS_OK) return;

    const auto rank = permutation.size();
    const auto permutation_applies = permutation.can_apply_to(tensor.dim);
    const auto pointers_valid = tensor.d and output;
    const std::array validation_conditions{permutation_applies, pointers_valid};
    if (not std::ranges::all_of(validation_conditions, std::identity{}))
    {
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return;
    }
    const auto output_shape = permutation.apply(tensor.dim);
    const auto num_elements = output_shape.num_elems();
    if (not std::in_range<i64>(num_elements))
    {
        set_err(QNPEPS_ERR_INTERNAL);
        return;
    }
    const auto input_strides = column_major_strides(tensor.dim);
    PermutationPlan permutation_plan{};
    for (auto output_axis = 0_uz; output_axis < rank; ++output_axis)
    {
        permutation_plan.output_extents[output_axis] = output_shape[output_axis];
        const auto input_axis = static_cast<usize>(permutation[output_axis]);
        const auto input_stride = input_strides[input_axis];
        permutation_plan.input_strides[output_axis] = input_stride;
    }
    const auto element_count = static_cast<i64>(num_elements);
    assert(element_count == static_cast<i64>(tensor.num_elems()));
    const CuPermuteArgs args{
        .output = output,
        .input = tensor.d,
        .plan = permutation_plan,
        .rank = static_cast<int>(rank),
        .element_count = element_count,
        .conjugate = conjugate ? 1 : 0,
    };
    cu_permute<<<grid_blocks_exact(element_count), k_threads_per_block, 0, stream>>>(args);
    CUDA_CHECK(cudaGetLastError());
}

auto permute_axes(
    ArenaCursor& arena,
    const DeviceTensor& tensor,
    const Permutation& permutation,
    bool conjugate,
    cudaStream_t stream
) -> DeviceTensor
{
    auto result = alloc(arena, permutation.apply(tensor.dim));
    permute_axes(tensor, permutation, conjugate, result.d, stream);
    return result;
}

auto permute_batched(PermutationCache& permutation_cache, const PermuteOp& op, cudaStream_t stream)
    -> void
{
    if (err_state() != QNPEPS_OK) return;

    const auto num_elements = op.dims_in.num_elems();
    const auto permutation_applies = op.perm.can_apply_to(op.dims_in);
    const auto index_map_size_valid = std::in_range<int>(num_elements);
    const auto batch_count_valid = op.batch_count > 0;
    const auto pointers_valid = op.dst.p and op.src.p;
    const auto stride_valid = [&](i64 stride)
    { return std::cmp_greater_equal(stride, num_elements); };
    const auto dst_valid = stride_valid(op.dst.stride);
    const auto src_valid = op.src.stride == 0 or stride_valid(op.src.stride);

    const auto valid = permutation_applies and index_map_size_valid and batch_count_valid
                       and pointers_valid and dst_valid and src_valid;
    if (not valid)
    {
        set_err(QNPEPS_ERR_INTERNAL);
        return;
    }
    const auto element_count = static_cast<i64>(num_elements);
    auto* gather_indices = permutation_cache.get_or_create(op.dims_in, op.perm);
    if (not gather_indices or err_state() != QNPEPS_OK) return;
    const auto blocks = grid_blocks_capped(element_count * op.batch_count);
    const CuGatherArgs gather_args{
        .output = op.dst.p,
        .input = op.src.p,
        .gather_indices = gather_indices,
        .element_count = element_count,
        .output_stride = op.dst.stride,
        .input_stride = op.src.stride,
        .conjugate = op.conjugate ? 1 : 0,
        .batch_count = op.batch_count,
    };
    cu_gather<<<blocks, k_threads_per_block, 0, stream>>>(gather_args);
    CUDA_CHECK(cudaGetLastError());
}
}
