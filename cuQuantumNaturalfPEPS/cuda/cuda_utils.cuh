#ifndef QNPEPS_CUDA_UTILS_CUH
#define QNPEPS_CUDA_UTILS_CUH

#include "capi/qnpeps.h"
#include "types.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

namespace qnpeps
{
inline auto err_state() -> qnpeps_status&
{
    static thread_local auto state = QNPEPS_OK;
    return state;
}

inline auto set_err(qnpeps_status status) -> qnpeps_status
{
    if (err_state() == QNPEPS_OK) err_state() = status;
    return err_state();
}
}

#define CUDA_CHECK(x)                                                                              \
    do                                                                                             \
    {                                                                                              \
        const auto e_ = (x);                                                                       \
        if (e_ != cudaSuccess) qnpeps::set_err(QNPEPS_ERR_CUDA);                                   \
    } while (0)

#define CUBLAS_CHECK(x)                                                                            \
    do                                                                                             \
    {                                                                                              \
        const auto s_ = (x);                                                                       \
        if (s_ != CUBLAS_STATUS_SUCCESS) qnpeps::set_err(QNPEPS_ERR_CUDA);                         \
    } while (0)

#define CUSOLVER_CHECK(x)                                                                          \
    do                                                                                             \
    {                                                                                              \
        const auto s_ = (x);                                                                       \
        if (s_ != CUSOLVER_STATUS_SUCCESS) qnpeps::set_err(QNPEPS_ERR_CUDA);                       \
    } while (0)

#define CUDA_NOCHECK(x)                                                                            \
    do                                                                                             \
    {                                                                                              \
        (void) (x);                                                                                \
    } while (0)

namespace qnpeps
{
static_assert(sizeof(cf) == sizeof(cuFloatComplex));
static_assert(alignof(cf) == alignof(cuFloatComplex));

template <class T>
[[nodiscard]] inline auto byte_offset(const void* base, usize bytes) noexcept -> const T*
{
    return reinterpret_cast<const T*>(static_cast<const char*>(base) + bytes);
}

template <class T>
[[nodiscard]] inline auto byte_offset(void* base, usize bytes) noexcept -> T*
{
    return reinterpret_cast<T*>(static_cast<char*>(base) + bytes);
}

[[nodiscard]] inline constexpr auto device_align(usize bytes) noexcept -> usize
{
    return (bytes + (k_device_malloc_align - 1)) & ~(k_device_malloc_align - 1);
}

[[nodiscard]] __device__ inline auto global_lane() -> i64
{
    return static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
}

[[nodiscard]] __device__ inline auto grid_stride() -> i64
{
    return static_cast<i64>(gridDim.x) * blockDim.x;
}

[[nodiscard]] __device__ inline auto diag_index(int i, int n) -> i64
{
    return static_cast<i64>(i) * (1 + n);
}

[[nodiscard]] inline auto cu_cast(const cf* p) noexcept -> const cuFloatComplex*
{
    return reinterpret_cast<const cuFloatComplex*>(p);
}
[[nodiscard]] inline auto cu_cast(cf* p) noexcept -> cuFloatComplex*
{
    return reinterpret_cast<cuFloatComplex*>(p);
}
[[nodiscard]] inline auto cu_cast(cf** p) noexcept -> cuFloatComplex**
{
    return reinterpret_cast<cuFloatComplex**>(p);
}
[[nodiscard]] inline auto cu_cast(cf* const* p) noexcept -> cuFloatComplex* const*
{
    return reinterpret_cast<cuFloatComplex* const*>(p);
}
[[nodiscard]] inline auto cu_cast(const cf* const* p) noexcept -> const cuFloatComplex* const*
{
    return reinterpret_cast<const cuFloatComplex* const*>(p);
}

[[nodiscard]] inline auto cf_cast(cuFloatComplex* p) noexcept -> cf*
{
    return reinterpret_cast<cf*>(p);
}
[[nodiscard]] inline auto cf_cast(const cuFloatComplex* p) noexcept -> const cf*
{
    return reinterpret_cast<const cf*>(p);
}
}

#endif
