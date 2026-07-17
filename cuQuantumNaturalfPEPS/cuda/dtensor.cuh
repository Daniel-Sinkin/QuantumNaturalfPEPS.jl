#ifndef QNPEPS_DTENSOR_CUH
#define QNPEPS_DTENSOR_CUH

#include "cuda_utils.cuh"
#include "types.cuh"

#include <cassert>
#include <complex>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <initializer_list>
#include <limits>
#include <map>
#include <utility>
#include <vector>

namespace qnpeps
{
class ArenaCursor;
class Linalg;
inline constexpr int k_max_tensor_rank{8};
static_assert(k_max_tensor_rank >= 6, "k_max_tensor_rank must cover the rank-6 RAB tensor");

class Axes
{
  public:
    Axes() = default;
    Axes(std::initializer_list<int> axes) : axes_(axes) {}
    explicit Axes(std::vector<int> axes) : axes_(std::move(axes)) {}

    [[nodiscard]] auto size() const noexcept -> usize { return axes_.size(); }
    [[nodiscard]] auto operator[](usize k) const noexcept -> int { return axes_[k]; }

    [[nodiscard]] auto begin() const noexcept { return axes_.begin(); }
    [[nodiscard]] auto end() const noexcept { return axes_.end(); }

  private:
    std::vector<int> axes_{};
};

class Shape
{
  public:
    Shape() = default;
    Shape(std::initializer_list<int> extents) : extents_(extents) {}
    explicit Shape(std::vector<int> extents) : extents_(std::move(extents)) {}

    [[nodiscard]] auto get() const noexcept -> const std::vector<int>& { return extents_; }
    [[nodiscard]] auto rank() const noexcept -> usize { return extents_.size(); }
    [[nodiscard]] auto operator[](usize axis) const noexcept -> int { return extents_[axis]; }

    [[nodiscard]] auto num_elems() const noexcept -> usize
    {
        usize total{1};
        for (const auto extent : extents_)
        {
            const bool valid{
                extent > 0
                and total <= std::numeric_limits<usize>::max() / static_cast<usize>(extent)
            };
            if (not valid)
            {
                assert(valid);
                qnpeps::set_err(QNPEPS_ERR_INTERNAL);
                return 0;
            }
            total *= static_cast<usize>(extent);
        }
        return total;
    }

    [[nodiscard]] auto operator==(const Shape& other) const noexcept -> bool
    {
        return extents_ == other.extents_;
    }

    [[nodiscard]] auto begin() const noexcept { return extents_.begin(); }
    [[nodiscard]] auto end() const noexcept { return extents_.end(); }

  private:
    std::vector<int> extents_{};
};

class Permutation
{
  public:
    Permutation() = default;
    Permutation(std::initializer_list<int> perm) : perm_(perm) {}
    explicit Permutation(std::vector<int> perm) : perm_(std::move(perm)) {}

    [[nodiscard]] auto get() const noexcept -> const std::vector<int>& { return perm_; }
    [[nodiscard]] auto size() const noexcept -> usize { return perm_.size(); }
    [[nodiscard]] auto operator[](usize k) const noexcept -> int { return perm_[k]; }

    template <typename T>
    [[nodiscard]] auto can_apply_to(const std::vector<T>& xs) const noexcept -> bool
    {
        if (size() != xs.size()) return false;
        for (auto i = 0_uz; i < size(); ++i)
        {
            if (perm_[i] < 0 or perm_[i] >= static_cast<int>(size())) return false;
            for (auto j = 0_uz; j < i; ++j)
                if (perm_[i] == perm_[j]) return false;
        }
        return true;
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
            assert(can_apply_to(xs));
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

  private:
    std::vector<int> perm_{};
};

using chost = std::complex<f32>;

struct HostTensor
{
    Shape dim{};
    std::vector<chost> v{};
    [[nodiscard]] auto num_elems() const noexcept -> usize { return dim.num_elems(); }
    auto alloc() -> void { v.assign(num_elems(), chost{0, 0}); }
};

[[nodiscard]] inline auto permutation_index_map(const Shape& dims_in, const Permutation& perm)
    -> std::vector<int>
{
    if (not perm.can_apply_to(dims_in))
    {
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return {};
    }
    const auto dims_out = perm.apply(dims_in);
    const auto rank = dims_in.rank();

    std::vector<i64> stride_in{};
    stride_in.assign(rank, 1);
    for (auto k = 1_uz; k < rank; ++k)
        stride_in[k] = stride_in[k - 1] * dims_in[k - 1];

    const auto total = dims_out.num_elems();
    if (total > static_cast<usize>(std::numeric_limits<int>::max()))
    {
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return {};
    }

    std::vector<int> index_map{};
    index_map.resize(total);
    std::vector<int> idx{};
    idx.resize(rank);
    for (auto out_pos = 0_uz; out_pos < total; ++out_pos)
    {
        i64 src{};
        for (auto k = 0_uz; k < rank; ++k)
        {
            const auto perm_idx = static_cast<usize>(perm[k]);
            src += idx[k] * stride_in[perm_idx];
        }
        index_map[out_pos] = static_cast<int>(src);
        for (auto k = 0_uz; k < rank; ++k)
        {
            if (++idx[k] < dims_out[k]) break;
            idx[k] = 0;
        }
    }
    return index_map;
}

struct DeviceTensor
{
    Shape dim{};
    cuFloatComplex* d{};
    [[nodiscard]] auto num_elems() const noexcept -> usize { return dim.num_elems(); }
};

auto set_stream(cudaStream_t new_stream) -> void;
auto stream() -> cudaStream_t;

auto alloc(ArenaCursor& arena, const Shape& dim) -> DeviceTensor;
auto free(DeviceTensor& tensor) -> void;

auto view(cuFloatComplex* data, Shape dim) -> DeviceTensor;

auto permute_axes(
    const DeviceTensor& tensor, const Permutation& perm, bool conj, cuFloatComplex* out
) -> void;
auto permute_axes(
    ArenaCursor& arena, const DeviceTensor& tensor, const Permutation& perm, bool conj
) -> DeviceTensor;

class PermutationCache
{
  public:
    PermutationCache() = default;
    PermutationCache(const PermutationCache&) = delete;
    auto operator=(const PermutationCache&) -> PermutationCache& = delete;
    PermutationCache(PermutationCache&&) noexcept = default;
    auto operator=(PermutationCache&&) noexcept -> PermutationCache& = default;

    auto get_or_create(const Shape& dims, const Permutation& perm) -> int*;
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

auto permute_batched(Linalg& linalg, PermutationCache& permutation_cache, const PermuteOp& op)
    -> void;
}

#endif
