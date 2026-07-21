#ifndef QNPEPS_PERMUTATION_CUH
#define QNPEPS_PERMUTATION_CUH

#include "tensor.cuh"
#include "types.cuh"

#include <cassert>
#include <initializer_list>
#include <map>
#include <utility>
#include <vector>

namespace qnpeps
{
class ArenaCursor;

class Axes
{
  public:
    Axes() = default;
    Axes(std::initializer_list<int> axes) : Axes(std::vector<int>(axes)) {}
    explicit Axes(std::vector<int> axes) : axes_(ensure_valid(std::move(axes))) {}

    [[nodiscard]] auto size() const noexcept -> usize { return axes_.size(); }
    [[nodiscard]] auto operator[](usize k) const noexcept -> int { return axes_[k]; }
    [[nodiscard]] auto can_apply_to(const Shape& shape) const noexcept -> bool
    {
        for (const auto axis : axes_)
            if (static_cast<usize>(axis) >= shape.rank()) return false;
        return true;
    }

    [[nodiscard]] auto begin() const noexcept { return axes_.begin(); }
    [[nodiscard]] auto end() const noexcept { return axes_.end(); }

  private:
    [[nodiscard]] static auto is_axes(const std::vector<int>& values) noexcept -> bool
    {
        for (auto i = 0_uz; i < values.size(); ++i)
        {
            if (values[i] < 0) return false;
            for (auto j = 0_uz; j < i; ++j)
                if (values[i] == values[j]) return false;
        }
        return true;
    }

    [[nodiscard]] static auto ensure_valid(std::vector<int> values) -> std::vector<int>
    {
        if (is_axes(values)) return values;
        assert(false);
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return {};
    }

    std::vector<int> axes_{};
};

class Permutation
{
  public:
    Permutation() = default;
    Permutation(std::initializer_list<int> perm) : Permutation(std::vector<int>(perm)) {}
    explicit Permutation(std::vector<int> perm) : perm_(ensure_valid(std::move(perm))) {}

    [[nodiscard]] auto get() const noexcept -> const std::vector<int>& { return perm_; }
    [[nodiscard]] auto size() const noexcept -> usize { return perm_.size(); }
    [[nodiscard]] auto operator[](usize k) const noexcept -> int { return perm_[k]; }

    template <typename T>
    [[nodiscard]] auto can_apply_to(const std::vector<T>& xs) const noexcept -> bool
    {
        return size() == xs.size();
    }

    [[nodiscard]] auto can_apply_to(const Shape& shape) const noexcept -> bool
    {
        return can_apply_to(shape.get());
    }

    template <typename T>
    [[nodiscard]] auto apply(const std::vector<T>& xs) const -> std::vector<T>
    {
        if (not can_apply_to(xs))
        {
            assert(false);
            qnpeps::set_err(QNPEPS_ERR_INTERNAL);
            return {};
        }
        std::vector<T> out{};
        out.reserve(perm_.size());
        for (const auto p : perm_)
            out.push_back(xs[static_cast<usize>(p)]);
        return out;
    }

    [[nodiscard]] auto apply(const Shape& shape) const -> Shape
    {
        return Shape{apply(shape.get())};
    }

    [[nodiscard]] static auto reverse(int n) -> Permutation
    {
        std::vector<int> out{};
        out.reserve(static_cast<usize>(n));
        for (auto i = n; i > 0; --i)
            out.push_back(i - 1);
        return Permutation{std::move(out)};
    }

    [[nodiscard]] auto is_identity() const noexcept -> bool
    {
        for (auto i = 0_uz; i < perm_.size(); ++i)
            if (perm_[i] != static_cast<int>(i)) return false;
        return true;
    }

    [[nodiscard]] static auto is_permutation(const std::vector<int>& values) noexcept -> bool
    {
        for (auto i = 0_uz; i < values.size(); ++i)
        {
            if (values[i] < 0 or values[i] >= static_cast<int>(values.size())) return false;
            for (auto j = 0_uz; j < i; ++j)
                if (values[i] == values[j]) return false;
        }
        return true;
    }

  private:
    [[nodiscard]] static auto ensure_valid(std::vector<int> values) -> std::vector<int>
    {
        if (is_permutation(values)) return values;
        assert(false);
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return {};
    }

    std::vector<int> perm_{};
};

auto permute_axes(
    const DeviceTensor& tensor,
    const Permutation& permutation,
    bool conjugate,
    cuFloatComplex* output,
    cudaStream_t stream
) -> void;

auto permute_axes(
    ArenaCursor& arena,
    const DeviceTensor& tensor,
    const Permutation& permutation,
    bool conjugate,
    cudaStream_t stream
) -> DeviceTensor;

class PermutationCache
{
  public:
    PermutationCache() = default;
    PermutationCache(const PermutationCache&) = delete;
    auto operator=(const PermutationCache&) -> PermutationCache& = delete;
    PermutationCache(PermutationCache&&) noexcept = default;
    auto operator=(PermutationCache&&) noexcept -> PermutationCache& = default;

    auto get_or_create(const Shape& shape, const Permutation& permutation) -> int*;
    auto release() -> void;

  private:
    struct Key
    {
        std::vector<int> dims{};
        std::vector<int> perm{};
        [[nodiscard]] auto operator<(const Key& other) const noexcept -> bool
        {
            if (dims != other.dims) return dims < other.dims;
            return perm < other.perm;
        }
    };

    std::map<Key, int*> entries_{};
};

struct PermuteOp
{
    CuArray dst{};
    CuArrayConst src{};
    Shape dims_in{};
    Permutation perm{};
    int batch_count{};
    bool conjugate{};
};

auto permute_batched(PermutationCache& permutation_cache, const PermuteOp& op, cudaStream_t stream)
    -> void;
}

#endif
