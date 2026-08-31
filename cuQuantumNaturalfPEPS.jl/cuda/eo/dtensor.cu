#include "../linalg/eo.cuh"
#include "common.cuh"
#include "core/defer.cuh"
#include "dtensor.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <utility>
#include <vector>

using namespace qnpeps;

namespace dt
{
static thread_local cudaStream_t g_dlenv_stream{};

auto BumpArena::bump(usize bytes, usize align) -> void*
{
    cursor = (cursor + (align - 1)) & ~(align - 1);
    auto* result = base + cursor;
    cursor += bytes;
    if (cursor > cap)
    {
        qn::set_err(QNPEPS_ELOC_ERR_OOM);
        return nullptr;
    }
    return result;
}

auto set_stream(cudaStream_t new_stream) -> void
{
    g_dlenv_stream = new_stream;
}
auto stream() -> cudaStream_t
{
    return g_dlenv_stream;
}

auto alloc(BumpArena& arena, const std::vector<int>& dim) -> DeviceTensor
{
    DeviceTensor tensor{};
    tensor.dim = dim;
    tensor.d = static_cast<cuFloatComplex*>(
        arena.bump(sizeof(cuFloatComplex) * static_cast<usize>(tensor.size()))
    );
    return tensor;
}

auto free(DeviceTensor&) -> void {}

auto view(cuFloatComplex* data, std::vector<int> dim) -> DeviceTensor
{
    DeviceTensor result{};
    result.dim = std::move(dim);
    result.d = data;
    return result;
}

struct DlPerm
{
    qnpeps::CuArray<int, MAX_TENSOR_RANK> out_dim;
    qnpeps::CuArray<i64, MAX_TENSOR_RANK> in_stride;
};

__global__ auto cu_permute(
    const cuFloatComplex* in,
    cuFloatComplex* out,
    DlPerm permute_plan,
    int rank,
    i64 element_count,
    int conj
) -> void
{
    const auto flat_index = i64{static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x};
    if (flat_index >= element_count) return;
    qnpeps::CuArray<i64, MAX_TENSOR_RANK> out_stride;
    auto running_stride = i64{1};
    for (auto k = int{0}; k < rank; ++k)
    {
        out_stride[k] = running_stride;
        running_stride *= permute_plan.out_dim[k];
    }
    i64 src{};
    for (auto k = int{0}; k < rank; ++k)
    {
        const auto coord = i64{(flat_index / out_stride[k]) % permute_plan.out_dim[k]};
        src += coord * permute_plan.in_stride[k];
    }
    auto value = in[src];
    if (conj) value.y = -value.y;
    out[flat_index] = value;
}

auto permute_out_dims(const std::vector<int>& in_dim, const std::vector<int>& perm)
    -> std::vector<int>
{
    auto outdim = std::vector<int>(perm.size());
    for (auto k = usize{0}; k < perm.size(); ++k)
        outdim[k] = in_dim[static_cast<usize>(perm[k])];
    return outdim;
}

auto permute_axes(
    const DeviceTensor& tensor, const std::vector<int>& perm, bool conj, cuFloatComplex* out
) -> void
{
    const auto rank = perm.size();
    if (rank > MAX_TENSOR_RANK)
    {
        qn::set_err(QNPEPS_ELOC_ERR_INTERNAL);
        return;
    }
    const auto outdim = permute_out_dims(tensor.dim, perm);
    auto in_stride = std::vector<i64>(tensor.dim.size());
    {
        auto acc = i64{1};
        for (auto ax = usize{0}; ax < tensor.dim.size(); ++ax)
        {
            in_stride[ax] = acc;
            acc *= tensor.dim[ax];
        }
    }
    DlPerm permute_plan{};
    for (auto k = usize{0}; k < rank; ++k)
    {
        permute_plan.out_dim[k] = outdim[k];
        permute_plan.in_stride[k] = in_stride[static_cast<usize>(perm[k])];
    }
    auto element_count = i64{1};
    for (int d : outdim)
        element_count *= d;
    const auto threads = int{256};
    const auto blocks = i64{(element_count + threads - 1) / threads};
    cu_permute<<<static_cast<u32>(blocks), threads, 0, g_dlenv_stream>>>(
        tensor.d, out, permute_plan, static_cast<int>(rank), element_count, conj ? 1 : 0
    );
    CUDA_CHECK(cudaGetLastError());
}

auto permute_axes(
    BumpArena& arena, const DeviceTensor& tensor, const std::vector<int>& perm, bool conj
) -> DeviceTensor
{
    auto result = alloc(arena, permute_out_dims(tensor.dim, perm));
    permute_axes(tensor, perm, conj, result.d);
    return result;
}

namespace
{
struct ContractPlan
{
    std::vector<int> perm_a{};
    std::vector<int> perm_b{};
    std::vector<int> result_dim{};
    int M{1};
    int K{1};
    int N{1};
};

auto contract_plan(
    const std::vector<int>& a_dim,
    const std::vector<int>& contracted_a,
    const std::vector<int>& b_dim,
    const std::vector<int>& contracted_b
) -> ContractPlan
{
    const auto rank_a = int{static_cast<int>(a_dim.size())};
    const auto rank_b = int{static_cast<int>(b_dim.size())};
    qnpeps::CuArray<bool, MAX_TENSOR_RANK> is_contracted_a{};
    qnpeps::CuArray<bool, MAX_TENSOR_RANK> is_contracted_b{};
    for (int ax : contracted_a)
        is_contracted_a[static_cast<usize>(ax)] = true;
    for (int ax : contracted_b)
        is_contracted_b[static_cast<usize>(ax)] = true;
    std::vector<int> free_a{};
    std::vector<int> free_b{};
    for (auto ax = int{0}; ax < rank_a; ++ax)
        if (not is_contracted_a[static_cast<usize>(ax)]) free_a.push_back(ax);
    for (auto ax = int{0}; ax < rank_b; ++ax)
        if (not is_contracted_b[static_cast<usize>(ax)]) free_b.push_back(ax);

    ContractPlan plan{};
    plan.perm_a = free_a;
    plan.perm_a.insert(plan.perm_a.end(), contracted_a.begin(), contracted_a.end());
    plan.perm_b = contracted_b;
    plan.perm_b.insert(plan.perm_b.end(), free_b.begin(), free_b.end());
    for (int ax : free_a)
        plan.M *= a_dim[static_cast<usize>(ax)];
    for (int ax : contracted_a)
        plan.K *= a_dim[static_cast<usize>(ax)];
    for (int ax : free_b)
        plan.N *= b_dim[static_cast<usize>(ax)];
    for (int ax : free_a)
        plan.result_dim.push_back(a_dim[static_cast<usize>(ax)]);
    for (int ax : free_b)
        plan.result_dim.push_back(b_dim[static_cast<usize>(ax)]);
    if (plan.result_dim.empty()) plan.result_dim.push_back(1);
    return plan;
}
}

auto contract(
    cublasHandle_t blas_handle,
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
    auto* base = static_cast<char*>(scratch);
    auto* a_perm = reinterpret_cast<cuFloatComplex*>(base);
    const auto lhs_rows = static_cast<usize>(plan.M);
    const auto inner_dim = static_cast<usize>(plan.K);
    const auto a_perm_bytes = cuda_align(sizeof(cuFloatComplex) * lhs_rows * inner_dim);
    auto* b_perm = reinterpret_cast<cuFloatComplex*>(base + a_perm_bytes);
    permute_axes(tensor_a, plan.perm_a, flags.conj_a, a_perm);
    permute_axes(tensor_b, plan.perm_b, flags.conj_b, b_perm);

    const auto one = make_cuFloatComplex(1.0f, 0.0f);
    const auto zero = make_cuFloatComplex(0.0f, 0.0f);
    CUBLAS_CHECK(cublasCgemm(
        blas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        plan.M,
        plan.N,
        plan.K,
        &one,
        a_perm,
        plan.M,
        b_perm,
        plan.K,
        &zero,
        out,
        plan.M
    ));
}

auto contract(
    BumpArena& arena,
    cublasHandle_t blas_handle,
    const DeviceTensor& tensor_a,
    const std::vector<int>& contracted_a,
    const DeviceTensor& tensor_b,
    const std::vector<int>& contracted_b,
    ContractFlags flags
) -> DeviceTensor
{
    const auto plan = contract_plan(tensor_a.dim, contracted_a, tensor_b.dim, contracted_b);
    auto result = alloc(arena, plan.result_dim);

    auto scratch_frame = ArenaScope(arena);
    const auto lhs_rows = static_cast<usize>(plan.M);
    const auto inner_dim = static_cast<usize>(plan.K);
    const auto rhs_cols = static_cast<usize>(plan.N);
    const auto a_perm_bytes = cuda_align(sizeof(cuFloatComplex) * lhs_rows * inner_dim);
    const auto b_perm_bytes = cuda_align(sizeof(cuFloatComplex) * inner_dim * rhs_cols);
    const auto scratch_bytes = a_perm_bytes + b_perm_bytes;
    const auto scratch = arena.bump(scratch_bytes);
    contract(blas_handle, tensor_a, contracted_a, tensor_b, contracted_b, flags, scratch, result.d);
    return result;
}
}
