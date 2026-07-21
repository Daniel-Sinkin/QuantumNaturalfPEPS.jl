#ifndef QNPEPS_TENSOR_CUH
#define QNPEPS_TENSOR_CUH

#include "cuda_utils.cuh"
#include "types.cuh"

#include <cassert>
#include <cuda_runtime.h>
#include <initializer_list>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace qnpeps
{
class ArenaCursor;

inline constexpr int k_max_tensor_rank{8};
static_assert(
    k_max_tensor_rank >= 6, "k_max_tensor_rank must cover the rank-6 double-layer column tensor"
);

class Shape
{
  public:
    Shape() = default;
    Shape(std::initializer_list<int> extents) : Shape(std::vector<int>(extents)) {}
    explicit Shape(std::vector<int> extents) : extents_(ensure_valid(std::move(extents))) {}

    [[nodiscard]] auto get() const noexcept -> const std::vector<int>& { return extents_; }
    [[nodiscard]] auto rank() const noexcept -> usize { return extents_.size(); }
    [[nodiscard]] auto operator[](usize axis) const noexcept -> int { return extents_[axis]; }

    [[nodiscard]] auto num_elems() const noexcept -> usize
    {
        auto total = 1_uz;
        for (const auto extent : extents_)
            total *= static_cast<usize>(extent);
        return total;
    }

    [[nodiscard]] auto operator==(const Shape& other) const noexcept -> bool
    {
        return extents_ == other.extents_;
    }

    [[nodiscard]] auto begin() const noexcept { return extents_.begin(); }
    [[nodiscard]] auto end() const noexcept { return extents_.end(); }

  private:
    [[nodiscard]] static auto is_shape(const std::vector<int>& extents) noexcept -> bool
    {
        if (extents.size() > static_cast<usize>(k_max_tensor_rank)) return false;
        auto total = 1_uz;
        for (const auto extent : extents)
        {
            if (extent <= 0
                or total > std::numeric_limits<usize>::max() / static_cast<usize>(extent))
            {
                return false;
            }
            total *= static_cast<usize>(extent);
        }
        return true;
    }

    [[nodiscard]] static auto ensure_valid(std::vector<int> extents) -> std::vector<int>
    {
        if (is_shape(extents)) return extents;
        assert(false);
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return {};
    }

    std::vector<int> extents_{};
};

class HostTensor
{
  public:
    explicit HostTensor(Shape shape)
        : shape_(std::move(shape)), values_(shape_.num_elems(), cf32{0, 0})
    {
    }

    [[nodiscard]] auto shape() const noexcept -> const Shape& { return shape_; }
    [[nodiscard]] auto num_elems() const noexcept -> usize { return values_.size(); }
    [[nodiscard]] auto data() const noexcept -> const cf32* { return values_.data(); }
    [[nodiscard]] auto values() noexcept -> std::span<cf32> { return values_; }

  private:
    Shape shape_;
    std::vector<cf32> values_;
};

struct DeviceTensor
{
    Shape dim{};
    cuFloatComplex* d{};
    [[nodiscard]] auto num_elems() const noexcept -> usize { return dim.num_elems(); }
};

auto alloc(ArenaCursor& arena, const Shape& dim) -> DeviceTensor;
auto free(DeviceTensor& tensor) -> void;
}

#endif
