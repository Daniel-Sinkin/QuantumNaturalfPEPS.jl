#ifndef QNPEPS_ELOC_ENV_BUILD_SLAB_CUH
#define QNPEPS_ELOC_ENV_BUILD_SLAB_CUH

#include "parts.cuh"

namespace qn_eloc::env
{
[[nodiscard]] inline auto is_full_top_needed(const Shape& shape, bool want_h, bool want_fb) noexcept
    -> bool
{
    return want_h or want_fb or shape.slab_rows > 0;
}

[[nodiscard]] auto slab_scratch_bytes(const Shape& shape) -> usize;

auto eval_slab_compact(
    Worker& w, const TermPiece& piece, const int* active_indices, int active, cf* value_out
) -> f64*;
}

#endif
