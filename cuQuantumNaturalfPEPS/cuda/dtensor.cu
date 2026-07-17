#include "arena_cursor.cuh"
#include "cuda_utils.cuh"
#include "dtensor.cuh"
#include "linalg.cuh"

#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <limits>
#include <utility>
#include <vector>

namespace qnpeps
{
static thread_local cudaStream_t g_dlenv_stream{};

auto set_stream(cudaStream_t new_stream) -> void
{
    g_dlenv_stream = new_stream;
}
auto stream() -> cudaStream_t
{
    return g_dlenv_stream;
}

auto alloc(ArenaCursor& arena, const Shape& dim) -> DeviceTensor
{
    DeviceTensor tensor{};
    tensor.dim = dim;
    tensor.d = arena.take<cuFloatComplex>(tensor.num_elems());
    return tensor;
}

auto free(DeviceTensor&) -> void {}

auto view(cuFloatComplex* data, Shape dim) -> DeviceTensor
{
    DeviceTensor result{};
    result.dim = std::move(dim);
    result.d = data;
    return result;
}

auto PermutationCache::get_or_create(const Shape& dims, const Permutation& perm) -> int*
{
    if (not perm.can_apply_to(dims))
    {
        set_err(QNPEPS_ERR_INTERNAL);
        return nullptr;
    }
    const Key key{dims.get(), perm.get()};
    const auto cached = entries_.find(key);
    if (cached != entries_.end()) return cached->second;

    const auto gather_indices = permutation_index_map(dims, perm);
    if (err_state() != QNPEPS_OK) return nullptr;
    int* device_ptr{};
    CUDA_CHECK(cudaMalloc(&device_ptr, gather_indices.size() * sizeof(int)));
    if (err_state() != QNPEPS_OK) return nullptr;
    CUDA_CHECK(cudaMemcpy(
        device_ptr,
        gather_indices.data(),
        gather_indices.size() * sizeof(int),
        cudaMemcpyHostToDevice
    ));
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
    for (const auto& entry : entries_)
        if (entry.second) CUDA_NOCHECK(cudaFree(entry.second));
    entries_.clear();
}

struct DLPermutationPlan
{
    int out_dim[k_max_tensor_rank];
    i64 in_stride[k_max_tensor_rank];
};

struct CuGatherArgs
{
    cf* out{};
    const cf* in{};
    const int* gather_indices{};
    i64 element_count{};
    i64 stride_out{};
    i64 stride_in{};
    int conjugate{};
    int batch_count{};
};

__global__ auto cu_gather(CuGatherArgs args) -> void
{
    const auto out = args.out;
    const auto in = args.in;
    const auto gather_indices = args.gather_indices;
    const auto element_count = args.element_count;
    const auto stride_out = args.stride_out;
    const auto stride_in = args.stride_in;
    const auto conjugate = args.conjugate;
    const auto batch_count = args.batch_count;

    const i64 total{element_count * batch_count};
    for (i64 tid{global_lane()}; tid < total; tid += grid_stride())
    {
        const auto lane = static_cast<int>(tid / element_count);
        const auto elem = tid % element_count;
        const auto lane_i64 = static_cast<i64>(lane);
        const i64 in_idx{lane_i64 * stride_in + gather_indices[elem]};
        const i64 out_idx{lane_i64 * stride_out + elem};
        auto value = in[in_idx];
        if (conjugate) value.im = -value.im;
        out[out_idx] = value;
    }
}

__global__ auto cu_permute(
    const cuFloatComplex* in,
    cuFloatComplex* out,
    DLPermutationPlan permute_plan,
    int rank,
    i64 element_count,
    int conj
) -> void
{
    const i64 flat_index{global_lane()};
    if (flat_index >= element_count) return;
    i64 out_stride[k_max_tensor_rank];
    i64 running_stride{1};
    for (auto k = 0; k < rank; ++k)
    {
        out_stride[k] = running_stride;
        running_stride *= permute_plan.out_dim[k];
    }
    i64 src{};
    for (auto k = 0; k < rank; ++k)
    {
        const i64 coord{(flat_index / out_stride[k]) % permute_plan.out_dim[k]};
        src += coord * permute_plan.in_stride[k];
    }
    auto value = in[src];
    if (conj) value.y = -value.y;
    out[flat_index] = value;
}

auto permute_axes(
    const DeviceTensor& tensor, const Permutation& perm, bool conj, cuFloatComplex* out
) -> void
{
    const auto rank = perm.size();
    if (rank > static_cast<usize>(k_max_tensor_rank) or not perm.can_apply_to(tensor.dim)
        or not tensor.d or not out)
    {
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return;
    }
    const auto outdim = perm.apply(tensor.dim);
    std::vector<i64> in_stride{};
    in_stride.resize(tensor.dim.rank());
    {
        i64 acc{1};
        for (auto ax = 0_uz; ax < tensor.dim.rank(); ++ax)
        {
            in_stride[ax] = acc;
            acc *= tensor.dim[ax];
        }
    }
    DLPermutationPlan permute_plan{};
    for (auto k = 0_uz; k < rank; ++k)
    {
        permute_plan.out_dim[k] = outdim[k];
        permute_plan.in_stride[k] = in_stride[static_cast<usize>(perm[k])];
    }
    const auto element_count_u = outdim.num_elems();
    if (err_state() != QNPEPS_OK
        or element_count_u > static_cast<usize>(std::numeric_limits<i64>::max()))
    {
        set_err(QNPEPS_ERR_INTERNAL);
        return;
    }
    const auto element_count = static_cast<i64>(element_count_u);
    assert(element_count == static_cast<i64>(tensor.num_elems()));
    cu_permute<<<grid_blocks_exact(element_count), k_threads_per_block, 0, g_dlenv_stream>>>(
        tensor.d, out, permute_plan, static_cast<int>(rank), element_count, conj ? 1 : 0
    );
    CUDA_CHECK(cudaGetLastError());
}

auto permute_axes(
    ArenaCursor& arena, const DeviceTensor& tensor, const Permutation& perm, bool conj
) -> DeviceTensor
{
    auto result = alloc(arena, perm.apply(tensor.dim));
    permute_axes(tensor, perm, conj, result.d);
    return result;
}

auto permute_batched(Linalg& linalg, PermutationCache& permutation_cache, const PermuteOp& op)
    -> void
{
    const auto element_count_u = op.dims_in.num_elems();
    if (err_state() != QNPEPS_OK
        or element_count_u > static_cast<usize>(std::numeric_limits<int>::max()))
    {
        set_err(QNPEPS_ERR_INTERNAL);
        return;
    }
    const auto element_count = static_cast<i64>(element_count_u);
    const bool valid = op.perm.can_apply_to(op.dims_in) and op.batch_count > 0 and op.dst.p
                       and op.src.p and op.dst.stride >= element_count
                       and (op.src.stride == 0 or op.src.stride >= element_count);
    if (not valid)
    {
        set_err(QNPEPS_ERR_INTERNAL);
        return;
    }
    assert(valid);
    auto* gather_indices = permutation_cache.get_or_create(op.dims_in, op.perm);
    if (not gather_indices or err_state() != QNPEPS_OK) return;
    const auto blocks = grid_blocks_capped(element_count * op.batch_count);
    const CuGatherArgs gather_args{
        .out = op.dst.p,
        .in = op.src.p,
        .gather_indices = gather_indices,
        .element_count = element_count,
        .stride_out = op.dst.stride,
        .stride_in = op.src.stride,
        .conjugate = op.conjugate ? 1 : 0,
        .batch_count = op.batch_count,
    };
    cu_gather<<<blocks, k_threads_per_block, 0, linalg.stream()>>>(gather_args);
    CUDA_CHECK(cudaGetLastError());
}
}
