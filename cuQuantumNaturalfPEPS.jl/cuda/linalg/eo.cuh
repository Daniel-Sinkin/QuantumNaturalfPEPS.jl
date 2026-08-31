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
__global__ auto cu_or_info(const int* info, int n, int* flag) -> void;
__global__ auto cu_or_info_bit(const int* info, int n, int bit, int* flag) -> void;
__global__ auto cu_record_lane_info(const int* info, int* failure_log, int lane) -> void;
__global__ auto cu_or_unmasked_info(const int* info, const int* mask, int n, int* flag) -> void;
__global__ auto cu_union_info(const int* info, int* mask, int n) -> void;
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

inline auto reupload_ht(cf* dst, const HostTensor& tensor) -> void
{
    const auto count = i64{tensor.n()};
    if (count < 1 or static_cast<u64>(count) > std::numeric_limits<usize>::max() / sizeof(cf))
    {
        qn::set_err(QNPEPS_ELOC_ERR_OOM);
        return;
    }
    const auto element_count = usize{static_cast<usize>(count)};
    const auto byte_count = usize{sizeof(cf) * element_count};
    CUDA_CHECK(cudaMemcpy(dst, tensor.v.data(), byte_count, cudaMemcpyHostToDevice));
}

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

inline auto hpermute(const HostTensor& tensor, const std::vector<int>& perm, bool conj = false)
    -> HostTensor
{
    const auto index_map = permutation_index_map(tensor.dim, perm);
    HostTensor out{};
    out.dim.resize(tensor.dim.size());
    for (auto k = usize{0}; k < perm.size(); ++k)
    {
        const auto perm_idx = static_cast<usize>(perm[k]);
        out.dim[k] = tensor.dim[perm_idx];
    }
    out.alloc();
    for (auto i = usize{0}; i < index_map.size(); ++i)
    {
        const auto in_idx = static_cast<usize>(index_map[i]);
        const auto source_value = tensor.v[in_idx];
        if (conj)
            out.v[i] = std::conj(source_value);
        else
            out.v[i] = source_value;
    }
    return out;
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
    qb_svd,
    gesvda
};

inline auto rangefinder_route_from_env() -> RangefinderRoute
{
    auto value{std::getenv("QNPEPS_E0191_RF_ROUTE")};
    if (not value or value[0] == '\0' or std::strcmp(value, "cholqr") == 0)
        return RangefinderRoute::cholqr;
    if (std::strcmp(value, "householder") == 0) return RangefinderRoute::householder;
    if (std::strcmp(value, "qb_svd") == 0) return RangefinderRoute::qb_svd;
    if (std::strcmp(value, "gesvda_cutoff") == 0) return RangefinderRoute::gesvda;
    qn::set_err(QNPEPS_ELOC_ERR_BAD_CONFIG);
    return RangefinderRoute::cholqr;
}

inline auto rangefinder_force_nonexact_from_env() -> bool
{
    auto value{std::getenv("QNPEPS_E0218_FORCE_NONEXACT")};
    if (not value or value[0] == '\0' or std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    qn::set_err(QNPEPS_ELOC_ERR_BAD_CONFIG);
    return false;
}

inline auto rangefinder_route_uses_householder_scratch() -> bool
{
    const auto route = RangefinderRoute{rangefinder_route_from_env()};
    return route == RangefinderRoute::householder;
}

inline auto rangefinder_route_uses_experimental_scratch() -> bool
{
    const auto route = RangefinderRoute{rangefinder_route_from_env()};
    return route == RangefinderRoute::qb_svd or route == RangefinderRoute::gesvda;
}

inline auto rangefinder_oversampling_from_env() -> int
{
    auto value{std::getenv("QNPEPS_E0191_OVERSAMPLE")};
    if (not value or value[0] == '\0') return 8;
    char* end{};
    const auto parsed = long{std::strtol(value, &end, 10)};
    if (not end or end == value or end[0] != '\0' or parsed < 0 or parsed > 64)
    {
        qn::set_err(QNPEPS_ELOC_ERR_BAD_CONFIG);
        return 0;
    }
    return static_cast<int>(parsed);
}

inline auto rangefinder_sketch_width(int rows, int cols, int k) -> int
{
    const auto route = RangefinderRoute{rangefinder_route_from_env()};
    if (route != RangefinderRoute::qb_svd) return k;
    const auto oversampling = int{rangefinder_oversampling_from_env()};
    return std::min(std::min(rows, cols), k + oversampling);
}

inline auto rangefinder_gesvda_cutoff_from_env() -> f64
{
    auto value{std::getenv("QNPEPS_ELOC_GESVDA_CUTOFF")};
    if (not value or value[0] == '\0') return 1.0e-13;
    char* end{};
    const auto parsed = f64{std::strtod(value, &end)};
    if (not end or end == value or end[0] != '\0' or not std::isfinite(parsed) or parsed < 0.0
        or parsed > 1.0)
    {
        qn::set_err(QNPEPS_ELOC_ERR_BAD_CONFIG);
        return 0.0;
    }
    return parsed;
}

__attribute__((visibility("default"))) auto rangefinder_experimental_carve(
    Linalg& la,
    ArenaCursor& arena,
    int max_dim,
    int rank,
    int batch,
    char*& scratch,
    usize& scratch_bytes
) -> void;

__attribute__((visibility("default"))) auto rangefinder_run_production_gate(
    Linalg& la,
    MatrixBatch panel,
    int rank,
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
    int* failure_log,
    const int* fallback_info,
    void* robust_scratch,
    usize robust_scratch_bytes
) -> bool;

auto batched_rangefinder(
    Linalg& la,
    MatrixBatch panel,
    int k,
    bool exact_factorization,
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
    int* failure_log,
    const int* fallback_info,
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
    Linalg& linalg,
    int m,
    int n,
    cf* A,
    int lda,
    void* scratch,
    int* fail_flag,
    int failure_bit = 1,
    int* failure_log = nullptr,
    int failure_lane = 0
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
    if (failure_log)
        cu_record_lane_info<<<1, 1, 0, linalg.stream()>>>(device_status, failure_log, failure_lane);
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
    if (failure_log)
        cu_record_lane_info<<<1, 1, 0, linalg.stream()>>>(device_status, failure_log, failure_lane);
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
