#ifndef QNPEPS_CAPI_VALIDATION_CUH
#define QNPEPS_CAPI_VALIDATION_CUH

#include "capi/internal/format.cuh"
#include "capi/qnpeps.h"
#include "core/error.hpp"
#include "core/types.cuh"

#include <cmath>

namespace qnpeps::capi
{
[[nodiscard]] inline auto is_sampler_route(int32_t route) -> bool
{
    return route == QNPEPS_TRUNCATION_DEFAULT or route == QNPEPS_TRUNCATION_SVD
           or route == QNPEPS_TRUNCATION_DENSITY;
}

[[nodiscard]] inline auto is_density_cutoff(f64 cutoff) -> bool
{
    return std::isfinite(cutoff) and cutoff >= 0.0;
}

inline auto validate_config(const QnpepsConfig* config) -> void
{
    require(config != nullptr, Status::null_argument, "config is null");
    require_struct_size(*config, "config");
    const auto invalid_dimensions = config->dim_phys < 1 or config->lx < 2 or config->ly < 2
                                    or config->dim_bond < 1 or config->chi_s < 1
                                    or config->chi_dl < 1;
    require(not invalid_dimensions, Status::bad_config, "config rejects {}", *config);
    const auto known_sampling_mode = config->sampling_mode == QNPEPS_SAMPLING_FAST
                                     or config->sampling_mode == QNPEPS_SAMPLING_FULL;
    require(
        known_sampling_mode,
        Status::bad_config,
        "config sampling mode {} is unknown",
        config->sampling_mode
    );
    require(
        config->sampling_mode != QNPEPS_SAMPLING_FULL or config->chi_c >= 1,
        Status::bad_config,
        "config full sampling needs a positive chi_c, {}",
        *config
    );
    const auto valid_truncation = is_sampler_route(config->dlenv_truncation_route)
                                  and is_sampler_route(config->sampler_truncation_route)
                                  and is_density_cutoff(config->dlenv_density_cutoff)
                                  and is_density_cutoff(config->sampler_density_cutoff)
                                  and is_density_cutoff(config->projected_density_cutoff);
    require(valid_truncation, Status::bad_config, "config rejects the truncation of {}", *config);
}

}

#endif
