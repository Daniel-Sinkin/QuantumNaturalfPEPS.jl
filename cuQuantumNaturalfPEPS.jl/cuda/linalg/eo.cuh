#ifndef QNPEPS_LINALG_EO_CUH
#define QNPEPS_LINALG_EO_CUH

#include "../eo/common.cuh"
#include "core/arena_cursor.cuh"
#include "linalg/linalg.cuh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace qnpeps
{

using chost = std::complex<f32>;

static_assert(sizeof(cf) == sizeof(cuFloatComplex));
static_assert(alignof(cf) == alignof(f32));

__global__ auto cu_normalize_log(
    cf* x, int n, i64 stride, f64* lognorm_acc, int dim_batch, f32* scale_out = {}
) -> void;
__global__ auto cu_chol_shift(cf* gram, int k, i64 stride, int dim_batch) -> void;
__global__ auto cu_or_info_bit(const int* info, int n, int bit, int* flag) -> void;
__global__ auto cu_fill_first_one(cf* x, i64 stride, int n, int dim_batch) -> void;

inline auto checked_tensor_elements(const std::vector<int>& dims, i64 limit, i64& result) -> bool
{
    auto total = i64{1};
    for (const int dim : dims)
    {
        if (dim < 1 or total > limit / dim)
        {
            qn::set_err(QNPEPS_ELOC_ERR_OOM);
            result = 0;
            return false;
        }
        total *= dim;
    }
    result = total;
    return true;
}

struct HostTensor
{
    std::vector<int> dim{};
    std::vector<chost> v{};
    auto n() const -> i64
    {
        i64 total{};
        checked_tensor_elements(dim, std::numeric_limits<i64>::max(), total);
        assert(total >= 0);
        return total;
    }
    auto alloc() -> void
    {
        const auto count = static_cast<usize>(n());
        const auto is_positive = count >= 1;
        const auto prod_fits = count <= std::numeric_limits<usize>::max() / sizeof(chost);
        if (not is_positive or not prod_fits)
        {
            qn::set_err(QNPEPS_ELOC_ERR_OOM);
            return;
        }
        v.assign(count, chost{});
    }
};

struct EoDeviceBuffer
{
    cf* p{};
    i64 stride{};
};

inline auto _permutation_index_map_check(
    const std::vector<int>& dims_in, const std::vector<int>& perm
) -> bool
{
    if (not same_size(dims_in, perm)) return false;
    if (not all_nonnegative(perm)) return false;
    if (not all_nonnegative(dims_in)) return false;
    return true;
}

inline auto permutation_index_map(const std::vector<int>& dims_in, const std::vector<int>& perm)
    -> std::vector<int>
{
    if (not _permutation_index_map_check(dims_in, perm))
    {
        qn::set_err(QNPEPS_ELOC_ERR_INTERNAL);
        return {};
    }
    const auto rank = dims_in.size();
    std::vector<int> dims_out{};
    dims_out.resize(rank);
    for (auto k = 0_uz; k < rank; ++k)
    {
        assert(perm[k] >= 0);
        const auto perm_idx = static_cast<usize>(perm[k]);
        if (perm[k] < 0 or perm_idx >= rank)
        {
            qn::set_err(QNPEPS_ELOC_ERR_INTERNAL);
            return {};
        }
        dims_out[k] = dims_in[perm_idx];
    }
    i64 total{};
    if (not checked_tensor_elements(dims_out, std::numeric_limits<int>::max(), total)) return {};
    auto stride_in = std::vector<i64>(rank, 1);
    for (auto k = usize{1}; k < rank; ++k)
    {
        if (dims_in[k - 1] < 1
            or stride_in[k - 1] > std::numeric_limits<int>::max() / dims_in[k - 1])
        {
            qn::set_err(QNPEPS_ELOC_ERR_OOM);
            return {};
        }
        stride_in[k] = stride_in[k - 1] * dims_in[k - 1];
    }
    auto index_map = std::vector<int>(static_cast<usize>(total));
    auto idx = std::vector<int>(rank, 0);
    for (auto out_pos = i64{0}; out_pos < total; ++out_pos)
    {
        i64 src{};
        for (auto k = usize{0}; k < rank; ++k)
        {
            const auto perm_idx = static_cast<usize>(perm[k]);
            const auto stride = i64{stride_in[perm_idx]};
            src += idx[k] * stride;
        }
        const auto out_idx = static_cast<usize>(out_pos);
        const auto src_int = static_cast<int>(src);
        index_map[out_idx] = src_int;
        for (auto k = usize{0}; k < rank; ++k)
        {
            if (++idx[k] < dims_out[k]) break;
            idx[k] = 0;
        }
    }
    return index_map;
}

class DeviceMatrix
{
    cf* data_{};
    int rows_{};
    int cols_{};

  public:
    DeviceMatrix(const cf* data, int rows, int cols)
        : data_(const_cast<cf*>(data)), rows_(rows), cols_(cols)
    {
    }
    [[nodiscard]] auto data() const -> cf* { return data_; }
    [[nodiscard]] auto rows() const -> int { return rows_; }
    [[nodiscard]] auto cols() const -> int { return cols_; }
    [[nodiscard]] auto ld() const -> int { return rows_; }
};

class MatrixBatch
{
    cf* data_{};
    i64 stride_{};
    int rows_{};
    int cols_{};

  public:
    MatrixBatch(const cf* data, i64 stride, int rows, int cols)
        : data_(const_cast<cf*>(data)), stride_(stride), rows_(rows), cols_(cols)
    {
    }
    [[nodiscard]] auto data() const -> cf* { return data_; }
    [[nodiscard]] auto rows() const -> int { return rows_; }
    [[nodiscard]] auto cols() const -> int { return cols_; }
    [[nodiscard]] auto ld() const -> int { return rows_; }
    [[nodiscard]] auto stride() const -> i64 { return stride_; }
};

enum class RangefinderRoute
{
    cholqr,
    householder,
    gesvda
};

auto rangefinder_experimental_carve(
    Linalg& la,
    ArenaCursor& arena,
    int max_dim,
    int rank,
    int batch,
    char*& scratch,
    usize& scratch_bytes
) -> void;

auto batched_rangefinder(
    Linalg& la,
    MatrixBatch panel,
    int k,
    bool exact_factorization,
    RangefinderRoute route,
    f64 svd_cutoff,
    const cf* omega,
    MatrixBatch q_out,
    MatrixBatch r_out,
    int dim_batch,
    EoDeviceBuffer sketch,
    EoDeviceBuffer proj,
    EoDeviceBuffer gram,
    cf** gram_ptrs,
    cf** sketch_ptrs,
    int* info,
    int* fail_flag,
    void* robust_scratch,
    usize robust_scratch_bytes
) -> void;

inline auto cuda_align(usize bytes) -> usize
{
    return (bytes + (CUDA_MALLOC_ALIGN - 1)) & ~(CUDA_MALLOC_ALIGN - 1);
}

inline auto qr_workspace_count(Linalg& linalg, int m, int n) -> int
{
    const auto key = i64{static_cast<i64>(m) << 32 | static_cast<unsigned>(n)};
    static thread_local std::map<i64, int> cache;
    const auto found{cache.find(key)};
    if (found != cache.end()) return found->second;

    auto handle = cusolverDnHandle_t{linalg.cusolver()};
    if (not handle)
    {
        qn::set_err(QNPEPS_ELOC_ERR_INTERNAL);
        return 0;
    }
    int geqrf_size{};
    CUSOLVER_CHECK(cusolverDnCgeqrf_bufferSize(handle, m, n, nullptr, m, &geqrf_size));
    int ungqr_size{};
    CUSOLVER_CHECK(cusolverDnCungqr_bufferSize(handle, m, n, n, nullptr, m, nullptr, &ungqr_size));
    const auto workspace_count = int{std::max({geqrf_size, ungqr_size, 1})};
    cache.emplace(key, workspace_count);
    return workspace_count;
}

inline auto qr_scratch_bytes(Linalg& linalg, int m, int n) -> usize
{
    const auto workspace_size = int{qr_workspace_count(linalg, m, n)};
    const auto reflector_count = static_cast<usize>(n);
    const auto reflector_bytes = cuda_align(sizeof(cuComplex) * reflector_count);
    const auto status_bytes = cuda_align(sizeof(int));
    const auto workspace_count = static_cast<usize>(std::max(workspace_size, 1));
    const auto workspace_bytes = cuda_align(sizeof(cuComplex) * workspace_count);
    const auto total_bytes = reflector_bytes + status_bytes + workspace_bytes;
    return total_bytes;
}

inline auto qr(
    Linalg& linalg, int m, int n, cf* A, int lda, void* scratch, int* fail_flag, int failure_bit = 1
) -> void
{
    auto* base = static_cast<char*>(scratch);
    const auto reflector_count = static_cast<usize>(n);
    const auto reflector_bytes = cuda_align(sizeof(cuComplex) * reflector_count);
    const auto status_bytes = cuda_align(sizeof(int));
    auto* reflector_scalars = reinterpret_cast<cuComplex*>(base);
    auto* device_status = reinterpret_cast<int*>(base + reflector_bytes);
    auto* solver_workspace = reinterpret_cast<cuComplex*>(base + reflector_bytes + status_bytes);
    auto* a_cuda = reinterpret_cast<cuComplex*>(A);
    const auto workspace_size = int{qr_workspace_count(linalg, m, n)};
    CUSOLVER_CHECK(cusolverDnCgeqrf(
        linalg.cusolver(),
        m,
        n,
        a_cuda,
        lda,
        reflector_scalars,
        solver_workspace,
        workspace_size,
        device_status
    ));
    if (fail_flag)
        cu_or_info_bit<<<1, 1, 0, linalg.stream()>>>(device_status, 1, failure_bit, fail_flag);
    CUSOLVER_CHECK(cusolverDnCungqr(
        linalg.cusolver(),
        m,
        n,
        n,
        a_cuda,
        lda,
        reflector_scalars,
        solver_workspace,
        workspace_size,
        device_status
    ));
    if (fail_flag)
        cu_or_info_bit<<<1, 1, 0, linalg.stream()>>>(device_status, 1, failure_bit, fail_flag);
}

inline auto qr_scratch_bytes(Linalg& linalg, DeviceMatrix A) -> usize
{
    return qr_scratch_bytes(linalg, A.rows(), A.cols());
}
inline auto qr(Linalg& linalg, DeviceMatrix A, void* scratch, int* fail_flag = nullptr) -> void
{
    qr(linalg, A.rows(), A.cols(), A.data(), A.ld(), scratch, fail_flag);
}

}

#endif
