#ifndef QNPEPS_LINALG_RANDOM_CUH
#define QNPEPS_LINALG_RANDOM_CUH

#include "core/types.cuh"

#include <curand_kernel.h>

namespace qnpeps
{

using DeviceRandomState = curandStatePhilox4_32_10_t;

__device__ inline auto initialize_random(
    DeviceRandomState& state, u64 seed, u64 sequence, u64 offset
) -> void
{
    curand_init(seed, sequence, offset, &state);
}

[[nodiscard]] __device__ inline auto random_normal_pair(DeviceRandomState& state) -> float2
{
    return curand_normal2(&state);
}

[[nodiscard]] __device__ inline auto random_uniform(DeviceRandomState& state) -> f32
{
    return curand_uniform(&state);
}

}

#endif
