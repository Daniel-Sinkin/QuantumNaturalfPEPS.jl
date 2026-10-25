#pragma once

#include "core/cuda_utils.cuh"
#include "core/types.cuh"

#include <cuda/std/cmath>

namespace qnpeps::zipup
{
inline constexpr int k_power_nonfinite_flag{2};
inline constexpr int k_power_product_exponent_limit{120};
inline constexpr int k_power_qr_exponent_limit{32};

struct InputExponentArgs
{
    const cuFloatComplex* values;
    i64 count;
    int* exponent;
    int* fail_flag;
};

struct PowerScaleArgs
{
    cuFloatComplex* values;
    i64 count;
    const int* input_exponent;
    int next_inner_dimension;
    int* fail_flag;
};

__global__ auto cu_input_exponent(InputExponentArgs args) -> void
{
    __shared__ CuArray<f32, k_tree_reduce_threads> shared_max;
    __shared__ CuArray<int, k_tree_reduce_threads> shared_nonfinite;

    auto local_max = 0.0f;
    auto local_nonfinite = 0;
    for (auto index = static_cast<i64>(threadIdx.x); index < args.count; index += blockDim.x)
    {
        const auto value = args.values[index];
        if (not cuda::std::isfinite(value.x) or not cuda::std::isfinite(value.y))
        {
            local_nonfinite = 1;
            continue;
        }
        const auto real_abs = cuda::std::abs(value.x);
        const auto imaginary_abs = cuda::std::abs(value.y);
        if (real_abs > local_max) local_max = real_abs;
        if (imaginary_abs > local_max) local_max = imaginary_abs;
    }
    shared_max[threadIdx.x] = local_max;
    shared_nonfinite[threadIdx.x] = local_nonfinite;
    __syncthreads();

    for (auto offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
        {
            if (shared_max[threadIdx.x + offset] > shared_max[threadIdx.x])
                shared_max[threadIdx.x] = shared_max[threadIdx.x + offset];
            shared_nonfinite[threadIdx.x] |= shared_nonfinite[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x != 0) return;
    *args.exponent = 0;
    if (shared_nonfinite[0] != 0)
    {
        if (args.fail_flag) atomicOr(args.fail_flag, k_power_nonfinite_flag);
        return;
    }
    if (shared_max[0] > 0.0f) frexpf(shared_max[0], args.exponent);
}

__global__ auto cu_scale_power_intermediate(PowerScaleArgs args) -> void
{
    __shared__ CuArray<f32, k_tree_reduce_threads> shared_max;
    __shared__ CuArray<int, k_tree_reduce_threads> shared_nonfinite;
    __shared__ int shared_shift;

    auto local_max = 0.0f;
    auto local_nonfinite = 0;
    for (auto index = static_cast<i64>(threadIdx.x); index < args.count; index += blockDim.x)
    {
        const auto value = args.values[index];
        if (not cuda::std::isfinite(value.x) or not cuda::std::isfinite(value.y))
        {
            local_nonfinite = 1;
            continue;
        }
        const auto real_abs = cuda::std::abs(value.x);
        const auto imaginary_abs = cuda::std::abs(value.y);
        if (real_abs > local_max) local_max = real_abs;
        if (imaginary_abs > local_max) local_max = imaginary_abs;
    }
    shared_max[threadIdx.x] = local_max;
    shared_nonfinite[threadIdx.x] = local_nonfinite;
    __syncthreads();

    for (auto offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
        {
            if (shared_max[threadIdx.x + offset] > shared_max[threadIdx.x])
                shared_max[threadIdx.x] = shared_max[threadIdx.x + offset];
            shared_nonfinite[threadIdx.x] |= shared_nonfinite[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        shared_shift = 0;
        if (shared_nonfinite[0] != 0)
        {
            if (args.fail_flag) atomicOr(args.fail_flag, k_power_nonfinite_flag);
        }
        else if (shared_max[0] > 0.0f)
        {
            auto value_exponent = 0;
            frexpf(shared_max[0], &value_exponent);
            auto safe_exponent = k_power_qr_exponent_limit;
            if (args.next_inner_dimension > 0)
            {
                auto inner_exponent = 0;
                auto inner_remainder = static_cast<u32>(args.next_inner_dimension - 1);
                while (inner_remainder != 0)
                {
                    ++inner_exponent;
                    inner_remainder >>= 1;
                }
                safe_exponent =
                    k_power_product_exponent_limit - 1 - inner_exponent - *args.input_exponent;
            }
            if (value_exponent > safe_exponent) shared_shift = safe_exponent - value_exponent;
        }
    }
    __syncthreads();

    if (shared_nonfinite[0] != 0 or shared_shift == 0) return;
    for (auto index = static_cast<i64>(threadIdx.x); index < args.count; index += blockDim.x)
    {
        args.values[index].x = ldexpf(args.values[index].x, shared_shift);
        args.values[index].y = ldexpf(args.values[index].y, shared_shift);
    }
}

}
