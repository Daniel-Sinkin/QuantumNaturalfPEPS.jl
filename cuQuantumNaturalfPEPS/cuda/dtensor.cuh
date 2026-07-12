#ifndef QNPEPS_DTENSOR_CUH
#define QNPEPS_DTENSOR_CUH

#include "cuda_utils.cuh"
#include "types.cuh"

#include <complex>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <initializer_list>
#include <utility>
#include <vector>

namespace qnpeps
{
class Linalg;
inline constexpr int k_max_tensor_rank{8};
static_assert(k_max_tensor_rank >= 6, "k_max_tensor_rank must cover the rank-6 RAB tensor");

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
            total *= static_cast<usize>(extent);
        return total;
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
        return size() == xs.size();
    }

    [[nodiscard]] auto can_apply_to(const Shape& shape) const noexcept -> bool
    {
        return size() == shape.rank();
    }

    template <typename T>
    [[nodiscard]] auto apply(const std::vector<T>& xs) const -> std::vector<T>
    {
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

  private:
    std::vector<int> perm_{};
};

using chost = std::complex<f32>;

struct HostTensor
{
    Shape dim{};
    std::vector<chost> v{};
    [[nodiscard]] auto n() const noexcept -> i64
    {
        i64 total{1};
        for (const auto dim_size : dim)
            total *= dim_size;
        return total;
    }
    auto alloc() -> void { v.assign(static_cast<usize>(n()), chost{0, 0}); }
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

    usize total{1};
    for (const auto dim_size : dims_out)
        total *= static_cast<usize>(dim_size);

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

class Carver
{
  public:
    Carver() = default;
    Carver(char* base, usize capacity) : base_(base), capacity_(capacity) {}

    [[nodiscard]] auto base() const noexcept -> char* { return base_; }
    [[nodiscard]] auto capacity() const noexcept -> usize { return capacity_; }
    [[nodiscard]] auto offset() const noexcept -> usize { return offset_; }
    auto offset() noexcept -> usize& { return offset_; }

    template <typename T>
    auto take(usize count) -> T*
    {
        offset_ = device_align(offset_);
        const auto begin = offset_;
        offset_ += sizeof(T) * count;
        if (not base_) return nullptr;
        if (capacity_ and offset_ > capacity_)
        {
            set_err(QNPEPS_ERR_OOM);
            return nullptr;
        }
        return reinterpret_cast<T*>(base_ + begin);
    }

    [[nodiscard]] auto total() const noexcept -> usize { return device_align(offset_); }

  private:
    char* base_{};
    usize capacity_{};
    usize offset_{};
};

auto set_stream(cudaStream_t new_stream) -> void;
auto stream() -> cudaStream_t;

auto alloc(Carver& carver, const Shape& dim) -> DeviceTensor;
auto free(DeviceTensor& tensor) -> void;

auto view(cuFloatComplex* data, Shape dim) -> DeviceTensor;

auto permute_axes(
    const DeviceTensor& tensor, const Permutation& perm, bool conj, cuFloatComplex* out
) -> void;
auto permute_axes(Carver& carver, const DeviceTensor& tensor, const Permutation& perm, bool conj)
    -> DeviceTensor;

struct ContractFlags
{
    bool conj_a{false};
    bool conj_b{false};
};

struct ContractPlan
{
    Permutation perm_a{};
    Permutation perm_b{};
    Shape result_dim{};
    int M{1};
    int K{1};
    int N{1};
};

[[nodiscard]] auto contract_plan(
    const Shape& a_dim,
    const std::vector<int>& contracted_a,
    const Shape& b_dim,
    const std::vector<int>& contracted_b
) -> ContractPlan;

auto contract(
    Linalg& la,
    const DeviceTensor& tensor_a,
    const DeviceTensor& tensor_b,
    ContractFlags flags,
    const ContractPlan& plan,
    void* scratch,
    cuFloatComplex* out
) -> void;
auto contract(
    Linalg& la,
    const DeviceTensor& tensor_a,
    const std::vector<int>& contracted_a,
    const DeviceTensor& tensor_b,
    const std::vector<int>& contracted_b,
    ContractFlags flags,
    void* scratch,
    cuFloatComplex* out
) -> void;
auto contract(
    Carver& carver,
    Linalg& la,
    const DeviceTensor& tensor_a,
    const std::vector<int>& contracted_a,
    const DeviceTensor& tensor_b,
    const std::vector<int>& contracted_b,
    ContractFlags flags
) -> DeviceTensor;
}

#endif
