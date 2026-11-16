#ifndef QNPEPS_ELOC_ROUTES_CUH
#define QNPEPS_ELOC_ROUTES_CUH

#include "capi/qnpeps.h"

#include <cstdint>

namespace qn_eloc
{
[[nodiscard]] constexpr auto is_local_energy_route(int32_t route) noexcept -> bool
{
    return route == QNPEPS_TRUNCATION_DEFAULT or route == QNPEPS_TRUNCATION_RANGEFINDER
           or route == QNPEPS_TRUNCATION_SVD or route == QNPEPS_TRUNCATION_HOUSEHOLDER
           or route == QNPEPS_TRUNCATION_DENSITY or route == QNPEPS_TRUNCATION_DENSITY_BATCHED;
}

[[nodiscard]] constexpr auto is_density_batched_route(int32_t route) noexcept -> bool
{
    return route == QNPEPS_TRUNCATION_DEFAULT or route == QNPEPS_TRUNCATION_DENSITY_BATCHED;
}
}

#endif
