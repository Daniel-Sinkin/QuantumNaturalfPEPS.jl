#ifndef QNPEPS_E2E_COMMON_CUH
#define QNPEPS_E2E_COMMON_CUH

#include "capi/qnpeps.h"
#include "core/error.hpp"
#include "core/types.cuh"
#include "dans_qnpeps_e2e.h"
#include "eo/dans_qnpeps_eloc.h"
#include "eo/routes.cuh"

#include <algorithm>
#include <cmath>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

using qnpeps::f32;
using qnpeps::f64;
using qnpeps::i32;
using qnpeps::i64;
using qnpeps::operator""_uz;
using qnpeps::u32;
using qnpeps::u64;
using qnpeps::u8;
using qnpeps::usize;

using qnpeps::check;
using qnpeps::fail;
using qnpeps::guarded;
using qnpeps::guarded_keeping_message;
using qnpeps::guarded_labelled;
using qnpeps::require;
using qnpeps::require_struct_size;
using qnpeps::Status;
using qnpeps::unchecked;

namespace qn_e2e
{

using cf = qnpeps::ComplexF32;
using cd = cuDoubleComplex;

[[nodiscard]] inline auto is_valid_density_cutoff(f64 cutoff) noexcept -> bool
{
    return std::isfinite(cutoff) and cutoff >= 0.0;
}

[[nodiscard]] inline auto is_local_energy_only_route(int32_t route) noexcept -> bool
{
    return route == QNPEPS_TRUNCATION_HOUSEHOLDER or route == QNPEPS_TRUNCATION_DENSITY_BATCHED
           or route == QNPEPS_TRUNCATION_RANGEFINDER;
}

inline constexpr int32_t k_density_batched_wave_lanes{256};

[[nodiscard]] inline auto local_energy_wave_lanes(const QnpepsE2eConfig& config) noexcept -> int32_t
{
    if (not qn_eloc::is_density_batched_route(config.truncation_route)) return config.meo;
    return std::max(config.meo, k_density_batched_wave_lanes);
}

[[nodiscard]] inline auto has_valid_truncation(const QnpepsE2eConfig& config) noexcept -> bool
{
    return qn_eloc::is_local_energy_route(config.truncation_route)
           and is_valid_density_cutoff(config.dlenv_density_cutoff)
           and is_valid_density_cutoff(config.sampler_density_cutoff)
           and is_valid_density_cutoff(config.projected_density_cutoff)
           and is_valid_density_cutoff(config.local_energy_density_cutoff)
           and is_valid_density_cutoff(config.local_energy_svd_cutoff)
           and config.local_energy_svd_cutoff <= 1.0;
}

inline auto apply_truncation(const QnpepsE2eConfig& config, QnpepsConfig& sampler) -> void
{
    const auto route = is_local_energy_only_route(config.truncation_route)
                           ? int32_t{QNPEPS_TRUNCATION_DEFAULT}
                           : config.truncation_route;
    sampler.dlenv_truncation_route = route;
    sampler.sampler_truncation_route = route;
    sampler.dlenv_density_cutoff = config.dlenv_density_cutoff;
    sampler.sampler_density_cutoff = config.sampler_density_cutoff;
    sampler.projected_density_cutoff = config.projected_density_cutoff;
}

inline auto apply_truncation(const QnpepsE2eConfig& config, QnpepsElocConfig& local_energy) -> void
{
    local_energy.truncation_route = config.truncation_route;
    local_energy.density_cutoff = config.local_energy_density_cutoff;
    local_energy.svd_cutoff = config.local_energy_svd_cutoff;
}

}

#endif
