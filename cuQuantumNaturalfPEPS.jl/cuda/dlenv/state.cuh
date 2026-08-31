#ifndef QNPEPS_DLENV_STATE_CUH
#define QNPEPS_DLENV_STATE_CUH

#include "core/arena_cursor.cuh"
#include "linalg/rangefinder_rng.cuh"

#include <cuda_runtime.h>
#include <map>
#include <utility>

namespace qnpeps::dlenv
{
inline constexpr u64 k_rangefinder_seed{777};

struct BuildState
{
    bool allocated{};
    cuFloatComplex* peps_buf{};
    ArenaCursor known{};
    ArenaCursor rolling_r{};
    ArenaCursor scratch{};
    ArenaCursor density{};
    int* fail{};
    f64* scales_all{};
    cuFloatComplex* unit_environment{};
    cuFloatComplex* initial_factor{};
    bool warmed{};
    std::map<std::pair<int, int>, cuFloatComplex*> omegas{};
    RangefinderRng rangefinder_rng{k_rangefinder_seed};
};
}

#endif
