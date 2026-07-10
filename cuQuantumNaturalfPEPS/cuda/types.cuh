#ifndef QNPEPS_TYPES_CUH
#define QNPEPS_TYPES_CUH

#include <cmath>
#include <cstddef>
#include <cstdint>

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

struct alignas(8) cf
{
    f32 re;
    f32 im;

    [[nodiscard]] __host__ __device__ constexpr auto norm() const noexcept -> f32
    {
        return re * re + im * im;
    }
    [[nodiscard]] __host__ __device__ auto abs() const noexcept -> f32 { return sqrtf(norm()); }
};

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

struct CuArray
{
    cf* p{};
    i64 stride{};
};

struct CuArrayConst
{
    const cf* p{};
    i64 stride{};

    CuArrayConst() = default;
    CuArrayConst(const cf* p_, i64 stride_) noexcept : p(p_), stride(stride_) {}
    CuArrayConst(const CuArray& buffer) noexcept : p(buffer.p), stride(buffer.stride) {}
};

class CuMatrix
{
  public:
    CuMatrix(cf* data, int rows, int cols) noexcept : data_(data), rows_(rows), cols_(cols) {}
    [[nodiscard]] auto data() const noexcept -> cf* { return data_; }
    [[nodiscard]] auto rows() const noexcept -> int { return rows_; }
    [[nodiscard]] auto cols() const noexcept -> int { return cols_; }
    [[nodiscard]] auto ld() const noexcept -> int { return rows_; }

  private:
    cf* data_{};
    int rows_{};
    int cols_{};
};

class CuMatrixConst
{
  public:
    CuMatrixConst(const cf* data, int rows, int cols) noexcept
        : data_(data), rows_(rows), cols_(cols)
    {
    }
    CuMatrixConst(const CuMatrix& m) noexcept : data_(m.data()), rows_(m.rows()), cols_(m.cols()) {}
    [[nodiscard]] auto data() const noexcept -> const cf* { return data_; }
    [[nodiscard]] auto rows() const noexcept -> int { return rows_; }
    [[nodiscard]] auto cols() const noexcept -> int { return cols_; }
    [[nodiscard]] auto ld() const noexcept -> int { return rows_; }

  private:
    const cf* data_{};
    int rows_{};
    int cols_{};
};

class CuMatrixBatched
{
  public:
    CuMatrixBatched() = default;
    CuMatrixBatched(cf* data, i64 stride, int rows, int cols) noexcept
        : data_(data), stride_(stride), rows_(rows), cols_(cols)
    {
    }
    CuMatrixBatched(const CuArray& buffer, int rows, int cols) noexcept
        : CuMatrixBatched(buffer.p, buffer.stride, rows, cols)
    {
    }
    [[nodiscard]] auto data() const noexcept -> cf* { return data_; }
    [[nodiscard]] auto rows() const noexcept -> int { return rows_; }
    [[nodiscard]] auto cols() const noexcept -> int { return cols_; }
    [[nodiscard]] auto ld() const noexcept -> int { return rows_; }
    [[nodiscard]] auto stride() const noexcept -> i64 { return stride_; }

  private:
    cf* data_{};
    i64 stride_{};
    int rows_{};
    int cols_{};
};

class CuMatrixConstBatched
{
  public:
    CuMatrixConstBatched() = default;
    CuMatrixConstBatched(const cf* data, i64 stride, int rows, int cols) noexcept
        : data_(data), stride_(stride), rows_(rows), cols_(cols)
    {
    }
    CuMatrixConstBatched(const CuArray& buffer, int rows, int cols) noexcept
        : CuMatrixConstBatched(buffer.p, buffer.stride, rows, cols)
    {
    }
    CuMatrixConstBatched(const CuArrayConst& buffer, int rows, int cols) noexcept
        : CuMatrixConstBatched(buffer.p, buffer.stride, rows, cols)
    {
    }
    CuMatrixConstBatched(const CuMatrixBatched& m) noexcept
        : data_(m.data()), stride_(m.stride()), rows_(m.rows()), cols_(m.cols())
    {
    }
    [[nodiscard]] auto data() const noexcept -> const cf* { return data_; }
    [[nodiscard]] auto rows() const noexcept -> int { return rows_; }
    [[nodiscard]] auto cols() const noexcept -> int { return cols_; }
    [[nodiscard]] auto ld() const noexcept -> int { return rows_; }
    [[nodiscard]] auto stride() const noexcept -> i64 { return stride_; }

  private:
    const cf* data_{};
    i64 stride_{};
    int rows_{};
    int cols_{};
};
}

#endif
