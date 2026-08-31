#ifndef QNPEPS_TYPES_CUH
#define QNPEPS_TYPES_CUH

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cuda/std/array>
#include <cuComplex.h>
#include <type_traits>

namespace qnpeps
{
using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using i32 = std::int32_t;
using i64 = std::int64_t;

using usize = std::size_t;

using f32 = float;
using f64 = double;
using cf32 = std::complex<f32>;
using cf64 = std::complex<f64>;

struct ComplexF32
{
    f32 re;
    f32 im;
};

static_assert(sizeof(ComplexF32) == sizeof(cuFloatComplex));
static_assert(alignof(ComplexF32) == alignof(f32));
static_assert(std::is_standard_layout_v<ComplexF32>);
static_assert(std::is_trivially_copyable_v<ComplexF32>);

[[nodiscard]] __host__ __device__ constexpr auto operator""_i32(unsigned long long value) -> i32
{
    return static_cast<i32>(value);
}

[[nodiscard]] __host__ __device__ constexpr auto operator""_i64(unsigned long long value) -> i64
{
    return static_cast<i64>(value);
}

[[nodiscard]] __host__ __device__ constexpr auto operator""_u32(unsigned long long value) -> u32
{
    return static_cast<u32>(value);
}

[[nodiscard]] __host__ __device__ constexpr auto operator""_u64(unsigned long long value) -> u64
{
    return static_cast<u64>(value);
}

[[nodiscard]] __host__ __device__ constexpr auto operator""_uz(unsigned long long value) -> usize
{
    return static_cast<usize>(value);
}

inline constexpr usize k_device_malloc_align{256};
inline constexpr int k_max_batch_size{2048};

struct Dims
{
    i32 lx{};
    i32 ly{};
    i32 dim_phys{};
    i32 dim_bond{};
};

[[nodiscard]] inline constexpr auto bond_dim(int axis_len, int pos, int dim_bond) noexcept -> int
{
    return (1 <= pos and pos < axis_len) ? dim_bond : 1;
}

[[nodiscard]] inline constexpr auto ceil_div(i64 num, i64 den) noexcept -> i64
{
    return (num + den - 1) / den;
}

template <typename T>
struct CuSpan
{
    T* p{};
    i64 stride{};

    CuSpan() = default;
    CuSpan(const CuSpan&) = default;
    CuSpan(T* p_, i64 stride_) noexcept : p(p_), stride(stride_) {}
    CuSpan(const CuSpan<std::remove_const_t<T>>& buffer) noexcept
        requires std::is_const_v<T>
        : p(buffer.p), stride(buffer.stride)
    {
    }
};

template <typename T, usize N>
using CuArray = cuda::std::array<T, N>;

template <typename T>
class CuMatrix
{
  public:
    CuMatrix(const CuMatrix&) = default;
    CuMatrix(T* data, int rows, int cols) noexcept : data_(data), rows_(rows), cols_(cols) {}
    CuMatrix(const CuMatrix<std::remove_const_t<T>>& matrix) noexcept
        requires std::is_const_v<T>
        : data_(matrix.data()), rows_(matrix.rows()), cols_(matrix.cols())
    {
    }
    [[nodiscard]] auto data() const noexcept -> T* { return data_; }
    [[nodiscard]] auto rows() const noexcept -> int { return rows_; }
    [[nodiscard]] auto cols() const noexcept -> int { return cols_; }
    [[nodiscard]] auto ld() const noexcept -> int { return rows_; }

  private:
    T* data_{};
    int rows_{};
    int cols_{};
};

template <typename T>
class CuMatrixBatched
{
  public:
    CuMatrixBatched() = default;
    CuMatrixBatched(const CuMatrixBatched&) = default;
    CuMatrixBatched(T* data, i64 stride, int rows, int cols) noexcept
        : data_(data), stride_(stride), rows_(rows), cols_(cols)
    {
    }
    CuMatrixBatched(const CuSpan<T>& buffer, int rows, int cols) noexcept
        : CuMatrixBatched(buffer.p, buffer.stride, rows, cols)
    {
    }
    CuMatrixBatched(const CuSpan<std::remove_const_t<T>>& buffer, int rows, int cols) noexcept
        requires std::is_const_v<T>
        : CuMatrixBatched(buffer.p, buffer.stride, rows, cols)
    {
    }
    CuMatrixBatched(const CuMatrixBatched<std::remove_const_t<T>>& matrix) noexcept
        requires std::is_const_v<T>
        : data_(matrix.data()), stride_(matrix.stride()), rows_(matrix.rows()), cols_(matrix.cols())
    {
    }
    [[nodiscard]] auto data() const noexcept -> T* { return data_; }
    [[nodiscard]] auto rows() const noexcept -> int { return rows_; }
    [[nodiscard]] auto cols() const noexcept -> int { return cols_; }
    [[nodiscard]] auto ld() const noexcept -> int { return rows_; }
    [[nodiscard]] auto stride() const noexcept -> i64 { return stride_; }

  private:
    T* data_{};
    i64 stride_{};
    int rows_{};
    int cols_{};
};

using CuMatrixF32 = CuMatrix<f32>;
using CuMatrixF32Const = CuMatrix<const f32>;
using CuMatrixCF32 = CuMatrix<cuFloatComplex>;
using CuMatrixCF32Const = CuMatrix<const cuFloatComplex>;
using CuMatrixCF64 = CuMatrix<cuDoubleComplex>;
using CuMatrixCF64Const = CuMatrix<const cuDoubleComplex>;
using CuMatrixBatchedCF32 = CuMatrixBatched<cuFloatComplex>;
using CuMatrixBatchedCF32Const = CuMatrixBatched<const cuFloatComplex>;
using CuSpanCF32 = CuSpan<cuFloatComplex>;
using CuSpanCF32Const = CuSpan<const cuFloatComplex>;
}

#endif
