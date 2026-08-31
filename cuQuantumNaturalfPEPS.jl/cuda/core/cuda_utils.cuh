#ifndef QNPEPS_CUDA_UTILS_CUH
#define QNPEPS_CUDA_UTILS_CUH

#include "core/error.cuh"
#include "core/types.cuh"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>

#if defined(__clang__) or defined(__GNUC__)
#    define QNPEPS_UNREACHABLE()                                                                   \
        do                                                                                         \
        {                                                                                          \
            assert(false);                                                                         \
            __builtin_unreachable();                                                               \
        } while (false)
#elif defined(_MSC_VER)
#    define QNPEPS_UNREACHABLE()                                                                   \
        do                                                                                         \
        {                                                                                          \
            assert(false);                                                                         \
            __assume(false);                                                                       \
        } while (false)
#else
#    define QNPEPS_UNREACHABLE()                                                                   \
        do                                                                                         \
        {                                                                                          \
            assert(false);                                                                         \
            std::abort();                                                                          \
        } while (false)
#endif

namespace qnpeps
{
constexpr i64 k_tree_reduce_threads{256};
static_assert(
    (k_tree_reduce_threads & (k_tree_reduce_threads - 1)) == 0,
    "the tree reduction halves its offset each step, so the width must be a power of two or "
    "elements are silently dropped from the fold"
);
static_assert(
    k_tree_reduce_threads <= 1024,
    "the reduce width is also the launch block width, which cannot exceed the CUDA maximum"
);
}

namespace qnpeps
{
static_assert(sizeof(cf32) == sizeof(cuFloatComplex));
static_assert(sizeof(cf64) == sizeof(cuDoubleComplex));

inline constexpr u32 k_threads_per_block{256};
inline constexpr i64 k_max_blocks{4096};

[[nodiscard]] inline constexpr auto grid_blocks_capped(i64 work_items) -> u32
{
    return static_cast<u32>(std::min(k_max_blocks, ceil_div(work_items, k_threads_per_block)));
}

[[nodiscard]] inline constexpr auto grid_blocks_exact(i64 work_items) -> u32
{
    return static_cast<u32>(ceil_div(work_items, k_threads_per_block));
}

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

[[nodiscard]] inline auto instantiate_graph(cudaGraphExec_t& executable, cudaGraph_t graph)
    -> cudaError_t
{
    return cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0);
}
}

#endif
