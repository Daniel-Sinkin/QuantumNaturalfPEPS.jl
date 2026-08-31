#ifndef QNPEPS_ELOC_DTENSOR_CUH
#define QNPEPS_ELOC_DTENSOR_CUH

#include "core/types.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

namespace dt
{
using qnpeps::i64;
using qnpeps::usize;

inline constexpr int MAX_TENSOR_RANK{8};
static_assert(MAX_TENSOR_RANK >= 6, "MAX_TENSOR_RANK must cover the rank-6 RAB tensor");

struct DeviceTensor
{
    std::vector<int> dim{};
    cuFloatComplex* d{};
    auto size() const -> i64
    {
        auto total_elements = i64{1};
        for (const auto dim_extent : dim)
            total_elements *= dim_extent;
        return total_elements;
    }
};

struct BumpArena
{
    char* base{};
    usize cap{};
    usize cursor{};

    auto bump(usize bytes, usize align = qnpeps::k_device_malloc_align) -> void*;
    auto mark() const -> usize { return cursor; }
    auto reset(usize marker) -> void { cursor = marker; }
};

class ArenaScope
{
    BumpArena& arena_;
    usize saved_;

  public:
    explicit ArenaScope(BumpArena& arena) : arena_(arena), saved_(arena.mark()) {}
    ~ArenaScope() { arena_.reset(saved_); }
    ArenaScope(const ArenaScope&) = delete;
    auto operator=(const ArenaScope&) -> ArenaScope& = delete;
};

auto set_stream(cudaStream_t new_stream) -> void;
auto stream() -> cudaStream_t;

auto alloc(BumpArena& arena, const std::vector<int>& dim) -> DeviceTensor;
auto free(DeviceTensor& tensor) -> void;

auto view(cuFloatComplex* data, std::vector<int> dim) -> DeviceTensor;

auto permute_out_dims(const std::vector<int>& in_dim, const std::vector<int>& perm)
    -> std::vector<int>;
auto permute_axes(
    const DeviceTensor& tensor, const std::vector<int>& perm, bool conj, cuFloatComplex* out
) -> void;
auto permute_axes(
    BumpArena& arena, const DeviceTensor& tensor, const std::vector<int>& perm, bool conj
) -> DeviceTensor;

struct ContractFlags
{
    bool conj_a{false};
    bool conj_b{false};
};

auto contract(
    cublasHandle_t blas_handle,
    const DeviceTensor& tensor_a,
    const std::vector<int>& contracted_a,
    const DeviceTensor& tensor_b,
    const std::vector<int>& contracted_b,
    ContractFlags flags,
    void* scratch,
    cuFloatComplex* out
) -> void;
auto contract(
    BumpArena& arena,
    cublasHandle_t blas_handle,
    const DeviceTensor& tensor_a,
    const std::vector<int>& contracted_a,
    const DeviceTensor& tensor_b,
    const std::vector<int>& contracted_b,
    ContractFlags flags
) -> DeviceTensor;
}

#endif
