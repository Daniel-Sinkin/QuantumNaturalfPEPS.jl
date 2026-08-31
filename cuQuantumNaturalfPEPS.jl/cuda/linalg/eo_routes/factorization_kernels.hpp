#pragma once

#include "../experimental_svd.cuh"
#include "../eo.cuh"

#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda/std/cmath>

#include "core/complex.cuh"

namespace qnpeps
{

#line 447 "cuda/linalg/eo_routes.cu"

__global__ auto cu_normalize_log(
    cf* x, int n, i64 stride, f64* lognorm_acc, int dim_batch, f32* scale_out
) -> void
{
    __shared__ qnpeps::CuArray<f32, 256> smax;
    const auto lane = int{static_cast<int>(blockIdx.x)};
    if (lane >= dim_batch) return;
    auto row_ptr = x + lane * stride;
    f32 max_abs{};
    for (auto elem = static_cast<int>(threadIdx.x); elem < n; elem += blockDim.x)
    {
        const auto abs_sum =
            f32{cuda::std::abs(row_ptr[elem].re) + cuda::std::abs(row_ptr[elem].im)};
        max_abs = abs_sum > max_abs ? abs_sum : max_abs;
    }
    smax[threadIdx.x] = max_abs;
    __syncthreads();
    for (auto step = static_cast<int>(blockDim.x / 2); step > 0; step >>= 1)
    {
        if (threadIdx.x < step)
            smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + step]);
        __syncthreads();
    }
    const auto scale = f32{smax[0] > 0.0f ? smax[0] : 1.0f};
    const auto inv_scale = f32{1.0f / scale};
    for (auto elem = static_cast<int>(threadIdx.x); elem < n; elem += blockDim.x)
    {
        row_ptr[elem].re *= inv_scale;
        row_ptr[elem].im *= inv_scale;
    }
    if (threadIdx.x == 0)
    {
        if (lognorm_acc) lognorm_acc[lane] += log(static_cast<f64>(scale));
        if (scale_out) scale_out[lane] = scale;
    }
}

__global__ auto cu_chol_shift(cf* gram, int k, i64 stride, int dim_batch) -> void
{
    const auto lane = int{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
    if (lane >= dim_batch) return;
    auto gram_lane = gram + lane * stride;
    f32 trace{};
    for (auto diag = int{0}; diag < k; ++diag)
        trace += gram_lane[diag + diag * k].re;
    const auto shift = f32{1.0e-5f * (trace > 0.0f ? trace / k : 1.0f) + 1.0e-30f};
    for (auto diag = int{0}; diag < k; ++diag)
        gram_lane[diag + diag * k].re += shift;
}

__global__ auto cu_or_info(const int* info, int n, int* flag) -> void
{
    const auto idx = int{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
    const auto grid_stride = int{static_cast<int>(gridDim.x * blockDim.x)};
    for (auto i = idx; i < n; i += grid_stride)
        if (info[i] != 0) atomicOr(flag, 1);
}

__global__ auto cu_or_info_bit(const int* info, int n, int bit, int* flag) -> void
{
    const auto idx = int{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
    const auto grid_stride = int{static_cast<int>(gridDim.x * blockDim.x)};
    for (auto i = idx; i < n; i += grid_stride)
        if (info[i] != 0) atomicOr(flag, bit);
}

__global__ auto cu_record_lane_info(const int* info, int* failure_log, int lane) -> void
{
    if (info[0] == 0) return;
    const auto encoded = int{info[0] > 0 ? info[0] : 1000000 - info[0]};
    atomicCAS(failure_log + lane, 0, encoded);
}

__global__ auto cu_or_unmasked_info(const int* info, const int* mask, int n, int* flag) -> void
{
    const auto idx = int{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
    const auto grid_stride = int{static_cast<int>(gridDim.x * blockDim.x)};
    for (auto i = idx; i < n; i += grid_stride)
        if (info[i] != 0 and mask[i] == 0) atomicOr(flag, 1);
}

__global__ auto cu_union_info(const int* info, int* mask, int n) -> void
{
    const auto idx = int{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
    const auto grid_stride = int{static_cast<int>(gridDim.x * blockDim.x)};
    for (auto i = idx; i < n; i += grid_stride)
        if (info[i] != 0) mask[i] = info[i];
}

__global__ auto cu_fill_identity(
    cf* matrix, i64 stride, int n, int dim_batch, const f32* diagonal_scale
) -> void
{
    const auto lane_elems = i64{static_cast<i64>(n) * n};
    const auto total = i64{lane_elems * dim_batch};
    const auto grid_stride = i64{static_cast<i64>(gridDim.x) * blockDim.x};
    for (auto index = i64{static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x}; index < total;
         index += grid_stride)
    {
        const auto lane = i64{index / lane_elems};
        const auto elem = i64{index % lane_elems};
        const auto row = int{static_cast<int>(elem % n)};
        const auto col = int{static_cast<int>(elem / n)};
        const auto diagonal = f32{diagonal_scale ? diagonal_scale[lane] : 1.0f};
        matrix[lane * stride + elem] = row == col ? cf{diagonal, 0.0f} : cf{0.0f, 0.0f};
    }
}

}
