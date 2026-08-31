#include "kernels.cuh"

#include <cuda/std/cmath>

namespace qnpeps::zipup
{

__global__ auto cu_or_status(const int* status, int* flag) -> void
{
    if (*status != 0) atomicOr(flag, 1);
}

__global__ auto cu_absmax_inverse(AbsmaxInverseArgs args) -> void
{
    __shared__ qnpeps::CuArray<f64, k_tree_reduce_threads> shared_max;
    f64 local_max{0.0};
    for (auto index = threadIdx.x; index < args.element_count; index += blockDim.x)
    {
        const auto real_abs = cuda::std::abs(static_cast<f64>(args.factor[index].x));
        const auto imaginary_abs = cuda::std::abs(static_cast<f64>(args.factor[index].y));
        const auto component_abs_sum = real_abs + imaginary_abs;
        if (component_abs_sum > local_max) local_max = component_abs_sum;
    }
    shared_max[threadIdx.x] = local_max;
    __syncthreads();
    for (auto offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        const auto has_larger_peer =
            threadIdx.x < offset and shared_max[threadIdx.x + offset] > shared_max[threadIdx.x];
        if (has_larger_peer) shared_max[threadIdx.x] = shared_max[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        const auto scale = shared_max[0];
        *args.scale = scale;
        const auto valid_scale = scale > 0.0 and cuda::std::isfinite(scale);
        *args.inverse_scale = valid_scale ? static_cast<f32>(1.0 / scale) : 1.0f;
    }
}

__global__ auto cu_apply_inverse_scale(
    cuFloatComplex* factor, i64 element_count, const f32* inverse_scale
) -> void
{
    const auto scale = *inverse_scale;
    for (auto index = global_lane(); index < element_count; index += grid_stride())
    {
        factor[index].x *= scale;
        factor[index].y *= scale;
    }
}

}
