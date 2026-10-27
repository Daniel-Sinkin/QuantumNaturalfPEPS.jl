#include "parts.cuh"

namespace qn_eloc::env
{

__global__ auto cu_gather(
    cf* out,
    const cf* in,
    const int* gather_indices,
    int n,
    i64 stride_out,
    i64 stride_in,
    int conjugate,
    int dim_batch
) -> void
{
    const auto n_i64 = i64{static_cast<i64>(n)};
    const auto total = i64{n_i64 * dim_batch};
    const auto grid_stride = i64{static_cast<i64>(gridDim.x) * blockDim.x};
    for (auto tid = static_cast<i64>(blockIdx.x * blockDim.x + threadIdx.x); tid < total;
         tid += grid_stride)
    {
        const auto lane = int{static_cast<int>(tid / n)};
        const auto elem = int{static_cast<int>(tid % n)};
        const auto lane_i64 = i64{static_cast<i64>(lane)};
        const auto in_idx = i64{lane_i64 * stride_in + gather_indices[elem]};
        const auto out_idx = i64{lane_i64 * stride_out + elem};
        auto value = static_cast<cf>(in[in_idx]);
        if (conjugate) value.im = -value.im;
        out[out_idx] = value;
    }
}

auto perm_key(const std::vector<int>& dims, const std::vector<int>& perm) -> std::string
{
    auto result = static_cast<std::string>("p");
    for (const int dim : dims)
        result += "_" + std::to_string(dim);
    result += "x";
    for (const int axis : perm)
        result += "_" + std::to_string(axis);
    return result;
}

auto device_permute(Linalg& linalg, PermutationCache& lookup_tables, const PermuteOp& op) -> void
{
    auto element_count = i64{1};
    for (const int dim : op.dims_in)
    {
        if (dim < 1 or element_count > std::numeric_limits<int>::max() / dim)
        {
            qn::set_err(QNPEPS_ELOC_ERR_OOM);
            return;
        }
        element_count *= dim;
    }
    if (op.batch < 1 or element_count > std::numeric_limits<i64>::max() / op.batch)
    {
        qn::set_err(QNPEPS_ELOC_ERR_OOM);
        return;
    }
    auto gather_indices = lookup_tables.get(op.dims_in, op.perm);
    if (qn::err_state() != QNPEPS_ELOC_OK or not gather_indices) return;
    const auto threads = int{256};
    const auto blocks = int{
        static_cast<int>(std::min<i64>(4096, (element_count * op.batch + threads - 1) / threads))
    };
    cu_gather<<<blocks, threads, 0, linalg.stream()>>>(
        op.dst.p,
        op.src.p,
        gather_indices,
        static_cast<int>(element_count),
        op.dst.stride,
        op.src.stride,
        0,
        op.batch
    );
    CUDA_CHECK(cudaGetLastError());
}

}
