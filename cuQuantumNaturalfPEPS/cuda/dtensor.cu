#include "cuda_utils.cuh"
#include "dtensor.cuh"
#include "linalg.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
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

auto alloc(Carver& carver, const Shape& dim) -> DeviceTensor
{
    DeviceTensor tensor{};
    tensor.dim = dim;
    tensor.d = carver.take<cuFloatComplex>(tensor.num_elems());
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

struct DLPermutationPlan
{
    int out_dim[k_max_tensor_rank];
    i64 in_stride[k_max_tensor_rank];
};

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
    if (rank > static_cast<usize>(k_max_tensor_rank))
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
    i64 element_count{1};
    for (int d : outdim)
        element_count *= d;
    const int threads{256};
    const auto blocks = static_cast<u32>(ceil_div(element_count, threads));
    const auto shmem_size = 0;
    cu_permute<<<blocks, threads, shmem_size, g_dlenv_stream>>>(
        tensor.d, out, permute_plan, static_cast<int>(rank), element_count, conj ? 1 : 0
    );
    CUDA_CHECK(cudaGetLastError());
}

auto permute_axes(Carver& carver, const DeviceTensor& tensor, const Permutation& perm, bool conj)
    -> DeviceTensor
{
    auto result = alloc(carver, perm.apply(tensor.dim));
    permute_axes(tensor, perm, conj, result.d);
    return result;
}

[[nodiscard]] auto contract_plan(
    const Shape& a_dim,
    const std::vector<int>& contracted_a,
    const Shape& b_dim,
    const std::vector<int>& contracted_b
) -> ContractPlan
{
    const auto free_axes_of = [](const Shape& dims, const std::vector<int>& contracted)
    {
        bool is_contracted[k_max_tensor_rank]{};
        for (const auto ax : contracted)
            is_contracted[static_cast<usize>(ax)] = true;
        std::vector<int> free_axes{};
        free_axes.reserve(dims.rank());
        for (auto ax = 0; ax < static_cast<int>(dims.rank()); ++ax)
            if (not is_contracted[static_cast<usize>(ax)]) free_axes.push_back(ax);
        return free_axes;
    };
    const auto free_a = free_axes_of(a_dim, contracted_a);
    const auto free_b = free_axes_of(b_dim, contracted_b);

    std::vector<int> perm_a{free_a};
    perm_a.insert(perm_a.end(), contracted_a.begin(), contracted_a.end());
    std::vector<int> perm_b{contracted_b};
    perm_b.insert(perm_b.end(), free_b.begin(), free_b.end());

    int result_rows{1};
    int result_cols{1};
    int contracted_elems{1};
    std::vector<int> result_dim{};
    for (const auto ax : free_a)
    {
        result_rows *= a_dim[static_cast<usize>(ax)];
        result_dim.push_back(a_dim[static_cast<usize>(ax)]);
    }
    for (const auto ax : free_b)
    {
        result_cols *= b_dim[static_cast<usize>(ax)];
        result_dim.push_back(b_dim[static_cast<usize>(ax)]);
    }
    for (const auto ax : contracted_a)
        contracted_elems *= a_dim[static_cast<usize>(ax)];
    if (result_dim.empty()) result_dim.push_back(1);

    return ContractPlan{
        .perm_a = Permutation{std::move(perm_a)},
        .perm_b = Permutation{std::move(perm_b)},
        .result_dim = Shape{std::move(result_dim)},
        .M = result_rows,
        .K = contracted_elems,
        .N = result_cols,
    };
}

auto contract(
    Linalg& la,
    const DeviceTensor& tensor_a,
    const DeviceTensor& tensor_b,
    ContractFlags flags,
    const ContractPlan& plan,
    void* scratch,
    cuFloatComplex* out
) -> void
{
    const auto lhs_rows = static_cast<usize>(plan.M);
    const auto inner_dim = static_cast<usize>(plan.K);
    const auto a_perm_bytes = device_align(sizeof(cuFloatComplex) * lhs_rows * inner_dim);
    auto* a_perm = byte_offset<cuFloatComplex>(scratch, 0);
    auto* b_perm = byte_offset<cuFloatComplex>(scratch, a_perm_bytes);
    permute_axes(tensor_a, plan.perm_a, flags.conj_a, a_perm);
    permute_axes(tensor_b, plan.perm_b, flags.conj_b, b_perm);

    la.matmul(
        CuMatrixConst{cf_cast(a_perm), plan.M, plan.K},
        CuMatrixConst{cf_cast(b_perm), plan.K, plan.N},
        CuMatrix{cf_cast(out), plan.M, plan.N}
    );
}

auto contract(
    Linalg& la,
    const DeviceTensor& tensor_a,
    const std::vector<int>& contracted_a,
    const DeviceTensor& tensor_b,
    const std::vector<int>& contracted_b,
    ContractFlags flags,
    void* scratch,
    cuFloatComplex* out
) -> void
{
    const auto plan = contract_plan(tensor_a.dim, contracted_a, tensor_b.dim, contracted_b);
    contract(la, tensor_a, tensor_b, flags, plan, scratch, out);
}

auto contract(
    Carver& carver,
    Linalg& la,
    const DeviceTensor& tensor_a,
    const std::vector<int>& contracted_a,
    const DeviceTensor& tensor_b,
    const std::vector<int>& contracted_b,
    ContractFlags flags
) -> DeviceTensor
{
    const auto plan = contract_plan(tensor_a.dim, contracted_a, tensor_b.dim, contracted_b);
    auto result = alloc(carver, plan.result_dim);

    Carver scratch_frame{carver};
    const auto lhs_rows = static_cast<usize>(plan.M);
    const auto inner_dim = static_cast<usize>(plan.K);
    const auto rhs_cols = static_cast<usize>(plan.N);
    const auto a_perm_bytes = device_align(sizeof(cuFloatComplex) * lhs_rows * inner_dim);
    const auto b_perm_bytes = device_align(sizeof(cuFloatComplex) * inner_dim * rhs_cols);
    const auto scratch_bytes = a_perm_bytes + b_perm_bytes;
    auto* scratch = scratch_frame.take<char>(scratch_bytes);
    contract(la, tensor_a, tensor_b, flags, plan, scratch, result.d);
    return result;
}
}
