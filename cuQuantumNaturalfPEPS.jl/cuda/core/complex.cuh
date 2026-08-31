#ifndef QNPEPS_CORE_COMPLEX_CUH
#define QNPEPS_CORE_COMPLEX_CUH

#include "core/types.cuh"

#include <cuda/std/cmath>

namespace qnpeps
{
[[nodiscard]] __host__ __device__ inline auto to_cu(ComplexF32 value) -> cuFloatComplex
{
    return make_cuFloatComplex(value.re, value.im);
}

template <typename Result = ComplexF32>
[[nodiscard]] __host__ __device__ inline auto to_cf(cuDoubleComplex value) -> Result
{
    static_assert(std::is_same_v<Result, ComplexF32> or std::is_same_v<Result, cuFloatComplex>);
    return Result{static_cast<f32>(cuCreal(value)), static_cast<f32>(cuCimag(value))};
}

template <typename Result = ComplexF32>
[[nodiscard]] __host__ __device__ inline auto to_cf(cuFloatComplex value) -> Result
{
    static_assert(std::is_same_v<Result, ComplexF32> or std::is_same_v<Result, cuFloatComplex>);
    return Result{cuCrealf(value), cuCimagf(value)};
}

[[nodiscard]] __host__ __device__ inline auto norm2(cuDoubleComplex value) -> f64
{
    return cuCreal(value) * cuCreal(value) + cuCimag(value) * cuCimag(value);
}

[[nodiscard]] __host__ __device__ inline auto norm2(cuFloatComplex value) -> f32
{
    return value.x * value.x + value.y * value.y;
}

[[nodiscard]] __host__ __device__ inline auto norm2(ComplexF32 value) -> f32
{
    return value.re * value.re + value.im * value.im;
}

[[nodiscard]] __host__ __device__ inline auto norm2(f64 real, f64 imaginary) -> f64
{
    return real * real + imaginary * imaginary;
}

[[nodiscard]] __host__ __device__ inline auto magnitude(cuFloatComplex value) noexcept -> f32
{
    return cuda::std::sqrt(norm2(value));
}
}

#endif
