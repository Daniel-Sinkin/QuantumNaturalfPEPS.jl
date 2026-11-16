#ifndef QNPEPS_ELOC_TERM_VALIDATION_CUH
#define QNPEPS_ELOC_TERM_VALIDATION_CUH

#include "core/predicates.cuh"
#include "core/types.cuh"
#include "dans_qnpeps_eloc.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

static_assert(sizeof(QnpepsElocComplex) == 16);
static_assert(sizeof(QnpepsElocDiagonalTerm) == 32);
static_assert(offsetof(QnpepsElocDiagonalTerm, site_count) == 8);
static_assert(offsetof(QnpepsElocDiagonalTerm, site) == 16);
static_assert(sizeof(QnpepsElocOffDiagonalPiece) == 40);
static_assert(offsetof(QnpepsElocOffDiagonalPiece, weight_offset) == 8);
static_assert(offsetof(QnpepsElocOffDiagonalPiece, site_count) == 16);
static_assert(offsetof(QnpepsElocOffDiagonalPiece, site) == 24);
static_assert(sizeof(QnpepsElocTermTable) == 88);
static_assert(offsetof(QnpepsElocTermTable, diagonal) == 16);
static_assert(offsetof(QnpepsElocTermTable, diagonal_value_count) == 32);
static_assert(offsetof(QnpepsElocTermTable, piece_state_count) == 48);
static_assert(offsetof(QnpepsElocTermTable, piece_weight_count) == 72);
static_assert(offsetof(QnpepsElocTermTable, piece_weights) == 80);

namespace qnpeps::eo
{
enum class TermValidation : std::uint8_t
{
    ok,
    null_argument,
    bad_version,
    bad_config
};

inline constexpr i32 k_max_local_dimension{256};

[[nodiscard]] inline auto local_state_count(i32 dim_phys, i32 site_count) noexcept -> i64
{
    auto count = i64{1};
    for (auto i = 0_uz; i < static_cast<usize>(site_count); ++i)
        count *= dim_phys;
    return count;
}

[[nodiscard]] inline auto is_finite(const QnpepsElocComplex& value) noexcept -> bool
{
    return std::isfinite(value.re) and std::isfinite(value.im);
}

[[nodiscard]] inline auto is_zero(const QnpepsElocComplex& value) noexcept -> bool
{
    return value.re == 0.0 and value.im == 0.0;
}

[[nodiscard]] inline auto has_pool(i64 count, const void* pool) noexcept -> bool
{
    return count >= 0 and (count == 0 or pool != nullptr);
}

[[nodiscard]] inline auto has_valid_support(
    i64 sites, i32 dim_phys, i32 site_count, i32 state_count, const i32* support
) noexcept -> bool
{
    if (not in_range(site_count, 1, QNPEPS_ELOC_TERM_MAX_SITES)) return false;
    if (state_count != local_state_count(dim_phys, site_count)) return false;
    const auto count = static_cast<usize>(site_count);
    for (auto i = 0_uz; i < count; ++i)
    {
        if (not in_range<i64>(support[i], 0, sites - 1)) return false;
        for (auto j = 0_uz; j < i; ++j)
            if (support[i] == support[j]) return false;
    }
    return true;
}

[[nodiscard]] inline auto has_extent(i64 offset, i64 extent, i64 pool_count) noexcept -> bool
{
    return offset >= 0 and extent <= pool_count and offset <= pool_count - extent;
}

[[nodiscard]] inline auto has_valid_diagonal(
    const QnpepsElocTermTable& table, i64 sites, const QnpepsElocDiagonalTerm& term
) noexcept -> bool
{
    if (not has_valid_support(sites, table.dim_phys, term.site_count, term.state_count, term.site))
        return false;
    return has_extent(term.value_offset, term.state_count, table.diagonal_value_count);
}

[[nodiscard]] inline auto has_nonzero_weights(
    const QnpepsElocTermTable& table, const QnpepsElocOffDiagonalPiece& piece, usize state
) noexcept -> bool
{
    const auto dim_phys = static_cast<usize>(table.dim_phys);
    const auto site_count = static_cast<usize>(piece.site_count);
    const auto first = static_cast<usize>(piece.weight_offset);
    for (auto position = 0_uz; position < site_count; ++position)
    {
        const auto base = first + (state * site_count + position) * dim_phys;
        auto has_nonzero = false;
        for (auto value = 0_uz; value < dim_phys; ++value)
            has_nonzero = has_nonzero or not is_zero(table.piece_weights[base + value]);
        if (not has_nonzero) return false;
    }
    return true;
}

[[nodiscard]] inline auto has_valid_piece(
    const QnpepsElocTermTable& table, i64 sites, const QnpepsElocOffDiagonalPiece& piece
) noexcept -> bool
{
    if (not has_valid_support(
            sites, table.dim_phys, piece.site_count, piece.state_count, piece.site
        ))
        return false;
    const auto states = static_cast<i64>(piece.state_count);
    const auto weights = states * piece.site_count * table.dim_phys;
    if (not has_extent(piece.state_offset, states, table.piece_state_count)) return false;
    if (not has_extent(piece.weight_offset, weights, table.piece_weight_count)) return false;
    const auto state_count = static_cast<usize>(states);
    const auto first = static_cast<usize>(piece.state_offset);
    auto active_states = 0_uz;
    for (auto state = 0_uz; state < state_count; ++state)
    {
        if (table.piece_active[first + state] == 0) continue;
        ++active_states;
        if (not has_nonzero_weights(table, piece, state)) return false;
    }
    return active_states > 0;
}

[[nodiscard]] inline auto has_valid_pools(const QnpepsElocTermTable& table) noexcept -> bool
{
    const auto value_count = static_cast<usize>(table.diagonal_value_count);
    const auto state_count = static_cast<usize>(table.piece_state_count);
    const auto weight_count = static_cast<usize>(table.piece_weight_count);
    for (auto i = 0_uz; i < value_count; ++i)
        if (not is_finite(table.diagonal_values[i])) return false;
    for (auto i = 0_uz; i < state_count; ++i)
    {
        if (table.piece_active[i] > 1) return false;
        if (not is_finite(table.piece_coefficients[i])) return false;
    }
    for (auto i = 0_uz; i < weight_count; ++i)
        if (not is_finite(table.piece_weights[i])) return false;
    return true;
}

[[nodiscard]] inline auto has_pool_pointers(const QnpepsElocTermTable& table) noexcept -> bool
{
    const auto has_diagonal = table.diagonal_count == 0 or table.diagonal != nullptr;
    const auto has_piece = table.piece_count == 0 or table.piece != nullptr;
    const auto has_values = has_pool(table.diagonal_value_count, table.diagonal_values);
    const auto has_active = has_pool(table.piece_state_count, table.piece_active);
    const auto has_coefficients = has_pool(table.piece_state_count, table.piece_coefficients);
    const auto has_weights = has_pool(table.piece_weight_count, table.piece_weights);
    return has_diagonal and has_piece and has_values and has_active and has_coefficients
           and has_weights;
}

[[nodiscard]] auto is_routable(i32 lx, i32 ly, const QnpepsElocTermTable& table) noexcept -> bool;

[[nodiscard]] inline auto validate_terms(
    i32 lx, i32 ly, i32 dim_phys, const QnpepsElocTermTable* terms
) noexcept -> TermValidation
{
    if (not terms) return TermValidation::null_argument;
    if (terms->struct_size != sizeof(QnpepsElocTermTable)) return TermValidation::bad_version;
    if (terms->dim_phys != dim_phys) return TermValidation::bad_config;
    if (not in_range(dim_phys, 1, k_max_local_dimension)) return TermValidation::bad_config;
    if (terms->diagonal_count < 0 or terms->piece_count < 0) return TermValidation::bad_config;
    if (not has_pool_pointers(*terms)) return TermValidation::null_argument;
    if (not has_valid_pools(*terms)) return TermValidation::bad_config;
    const auto sites = static_cast<i64>(lx) * ly;
    for (auto i = 0_uz; i < static_cast<usize>(terms->diagonal_count); ++i)
        if (not has_valid_diagonal(*terms, sites, terms->diagonal[i]))
            return TermValidation::bad_config;
    for (auto i = 0_uz; i < static_cast<usize>(terms->piece_count); ++i)
        if (not has_valid_piece(*terms, sites, terms->piece[i])) return TermValidation::bad_config;
    return TermValidation::ok;
}
}

#endif
