#ifndef QNPEPS_CUDA_UTILS_CUH
#define QNPEPS_CUDA_UTILS_CUH

#include "capi/qnpeps.h"
#include "types.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cstdint>
#include <source_location>

namespace qnpeps
{
constexpr i64 k_tree_reduce_threads{256};

struct ErrorState
{
    qnpeps_status status{QNPEPS_OK};
    const char* file{};
    int32_t line{};
};

inline auto error_state() -> ErrorState&
{
    static thread_local ErrorState state{};
    return state;
}

inline auto err_state() -> qnpeps_status&
{
    return error_state().status;
}

inline auto reset_err() -> void
{
    error_state() = {};
}

inline auto set_err_at(qnpeps_status status, const char* file, int32_t line) -> qnpeps_status
{
    auto& state = error_state();
    if (state.status == QNPEPS_OK)
    {
        state.status = status;
        state.file = file;
        state.line = line;
    }
    return state.status;
}

inline auto set_err(
    qnpeps_status status,
    std::source_location where = std::source_location::current()
) -> qnpeps_status
{
    return set_err_at(status, where.file_name(), static_cast<int32_t>(where.line()));
}

inline auto err_file() -> const char*
{
    return error_state().file;
}

inline auto err_line() -> int32_t
{
    return error_state().line;
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

[[nodiscard]] inline auto instantiate_graph(cudaGraphExec_t& executable, cudaGraph_t graph)
    -> cudaError_t
{
    return cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0);
}
}

#endif
