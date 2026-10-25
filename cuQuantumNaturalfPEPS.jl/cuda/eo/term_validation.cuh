#ifndef QNPEPS_ELOC_TERM_VALIDATION_CUH
#define QNPEPS_ELOC_TERM_VALIDATION_CUH

#include "dans_qnpeps_eloc.h"

#include <cmath>
#include <cstdint>

namespace qnpeps::eo
{
enum class TermValidation : std::uint8_t
{
    ok,
    null_argument,
    bad_config
};

[[nodiscard]] inline auto validate_terms(
    int32_t lx, int32_t ly, const QnpepsElocTermTable* terms
) noexcept -> TermValidation
{
    if (not terms) return TermValidation::null_argument;
    if (terms->n_diag < 0 or terms->n_flip < 0) return TermValidation::bad_config;
    if ((terms->n_diag > 0 and not terms->diag) or (terms->n_flip > 0 and not terms->flip))
        return TermValidation::null_argument;
    const auto sites = int64_t{static_cast<int64_t>(lx) * ly};
    for (auto index = int32_t{0}; index < terms->n_diag; ++index)
    {
        const auto& term = terms->diag[index];
        if (term.site_a < 0 or term.site_a >= sites
            or (term.site_b != QNPEPS_ELOC_DIAG_ONE_SITE_Z
                and (term.site_b < 0 or term.site_b >= sites))
            or not std::isfinite(term.coeff))
            return TermValidation::bad_config;
    }
    return TermValidation::ok;
}
}

#endif
