#ifndef QNPEPS_LINALG_SCRATCH_CUH
#define QNPEPS_LINALG_SCRATCH_CUH

#include "core/arena_cursor.cuh"
#include "core/cuda_utils.cuh"
#include "core/types.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <limits>
#include <type_traits>
#include <utility>

namespace qnpeps
{

struct QrScratch
{
    usize reflector_bytes{};
    usize status_bytes{};
    usize workspace_bytes{};

    [[nodiscard]] constexpr auto total() const noexcept -> usize
    {
        return reflector_bytes + status_bytes + workspace_bytes;
    }
};

struct HeevdScratch
{
    usize eigenvalue_bytes{};
    usize status_bytes{};
    usize workspace_bytes{};

    [[nodiscard]] constexpr auto total() const noexcept -> usize
    {
        return eigenvalue_bytes + status_bytes + workspace_bytes;
    }

    [[nodiscard]] auto eigenvalues(void* scratch) const noexcept -> f64*
    {
        return byte_offset<f64>(scratch, 0);
    }

    [[nodiscard]] auto status(void* scratch) const noexcept -> int*
    {
        return byte_offset<int>(scratch, eigenvalue_bytes);
    }
};

[[nodiscard]] inline constexpr auto strided_batch_fits_int(i64 stride, int batch_size) noexcept
    -> bool
{
    if (stride <= 0 or batch_size <= 0) return false;
    return stride <= static_cast<i64>(std::numeric_limits<int>::max()) / batch_size;
}

inline constexpr f64 k_gesvdj_tolerance{1.0e-7};
inline constexpr int k_gesvdj_max_sweeps{100};

struct GesvdaScratch
{
    usize left_bytes{};
    usize right_bytes{};
    usize singular_bytes{};
    usize status_bytes{};
    usize retry_bytes{};
    usize workspace_bytes{};
    int workspace_count{};

    [[nodiscard]] constexpr auto total() const noexcept -> usize
    {
        return left_bytes + right_bytes + singular_bytes + status_bytes + retry_bytes
               + workspace_bytes;
    }

    [[nodiscard]] auto left(void* scratch) const noexcept -> cuFloatComplex*
    {
        return byte_offset<cuFloatComplex>(scratch, 0);
    }

    [[nodiscard]] auto right(void* scratch) const noexcept -> cuFloatComplex*
    {
        return byte_offset<cuFloatComplex>(scratch, left_bytes);
    }

    [[nodiscard]] auto singular(void* scratch) const noexcept -> f32*
    {
        return byte_offset<f32>(scratch, left_bytes + right_bytes);
    }

    [[nodiscard]] auto status(void* scratch) const noexcept -> int*
    {
        return byte_offset<int>(scratch, left_bytes + right_bytes + singular_bytes);
    }

    [[nodiscard]] auto workspace(void* scratch) const noexcept -> cuFloatComplex*
    {
        const auto offset = left_bytes + right_bytes + singular_bytes + status_bytes + retry_bytes;
        return byte_offset<cuFloatComplex>(scratch, offset);
    }

    [[nodiscard]] auto retry(void* scratch) const noexcept -> int*
    {
        return byte_offset<int>(scratch, left_bytes + right_bytes + singular_bytes + status_bytes);
    }
};

struct GesvdjScratch
{
    usize left_bytes{};
    usize right_bytes{};
    usize singular_bytes{};
    usize status_bytes{};
    usize workspace_bytes{};
    int workspace_count{};

    [[nodiscard]] constexpr auto total() const noexcept -> usize
    {
        return left_bytes + right_bytes + singular_bytes + status_bytes + workspace_bytes;
    }

    [[nodiscard]] auto left(void* scratch) const noexcept -> cuFloatComplex*
    {
        return byte_offset<cuFloatComplex>(scratch, 0);
    }

    [[nodiscard]] auto right(void* scratch) const noexcept -> cuFloatComplex*
    {
        return byte_offset<cuFloatComplex>(scratch, left_bytes);
    }

    [[nodiscard]] auto singular(void* scratch) const noexcept -> f32*
    {
        return byte_offset<f32>(scratch, left_bytes + right_bytes);
    }

    [[nodiscard]] auto status(void* scratch) const noexcept -> int*
    {
        return byte_offset<int>(scratch, left_bytes + right_bytes + singular_bytes);
    }

    [[nodiscard]] auto workspace(void* scratch) const noexcept -> cuFloatComplex*
    {
        const auto offset = left_bytes + right_bytes + singular_bytes + status_bytes;
        return byte_offset<cuFloatComplex>(scratch, offset);
    }
};

class DeviceBuffer
{
  public:
    DeviceBuffer() = default;

    ~DeviceBuffer()
    {
        if (not arena_) maybe_free_device(base_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    auto operator=(const DeviceBuffer&) -> DeviceBuffer& = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : base_(std::exchange(other.base_, nullptr)), bytes_(std::exchange(other.bytes_, 0_uz)),
          arena_(std::exchange(other.arena_, nullptr))
    {
    }

    auto operator=(DeviceBuffer&& other) noexcept -> DeviceBuffer&
    {
        if (this == &other) return *this;
        if (not arena_) maybe_free_device(base_);
        base_ = std::exchange(other.base_, nullptr);
        bytes_ = std::exchange(other.bytes_, 0_uz);
        arena_ = std::exchange(other.arena_, nullptr);
        return *this;
    }

    auto bind_arena(ArenaCursor& arena) -> void
    {
        if (not arena_) maybe_free_device(base_);
        base_ = nullptr;
        bytes_ = 0;
        arena_ = &arena;
    }

    [[nodiscard]] auto grow(usize bytes) -> void*
    {
        if (bytes <= bytes_) return base_;
        void* grown = arena_ ? arena_->take<char>(bytes) : nullptr;
        if (not arena_) CUDA_CHECK(cudaMalloc(&grown, bytes));
        if (not grown)
        {
            qnpeps::set_err(QNPEPS_ERR_OOM);
            return nullptr;
        }
        if (not arena_) maybe_free_device(base_);
        base_ = grown;
        bytes_ = bytes;
        return base_;
    }

    [[nodiscard]] auto data() const noexcept -> void* { return base_; }
    [[nodiscard]] auto bytes() const noexcept -> usize { return bytes_; }

  private:
    void* base_{};
    usize bytes_{};
    ArenaCursor* arena_{};
};

static_assert(not std::is_copy_constructible_v<DeviceBuffer>);
static_assert(std::is_move_constructible_v<DeviceBuffer>);
}

#endif
