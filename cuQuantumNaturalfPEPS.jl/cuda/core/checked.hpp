#ifndef QNPEPS_CHECKED_HPP
#define QNPEPS_CHECKED_HPP

#include "core/error_type.hpp"
#include "core/types.cuh"

#include <concepts>
#include <limits>
#include <optional>
#include <type_traits>

namespace qnpeps
{
template <std::signed_integral Count>
[[nodiscard]] constexpr auto to_count(Count value) -> usize
{
    if (value < 0) throw Error{Status::invalid_argument, "negative element count"};
    if constexpr (sizeof(Count) > sizeof(usize))
    {
        using UnsignedCount = std::make_unsigned_t<Count>;
        const auto maximum = static_cast<UnsignedCount>(std::numeric_limits<usize>::max());
        if (static_cast<UnsignedCount>(value) > maximum)
            throw Error{Status::invalid_argument, "element count exceeds size range"};
    }
    return static_cast<usize>(value);
}

template <std::unsigned_integral Count>
[[nodiscard]] constexpr auto to_count(Count value) -> usize
{
    if constexpr (sizeof(Count) > sizeof(usize))
    {
        const auto maximum = static_cast<Count>(std::numeric_limits<usize>::max());
        if (value > maximum)
            throw Error{Status::invalid_argument, "element count exceeds size range"};
    }
    return static_cast<usize>(value);
}

[[nodiscard]] constexpr auto checked_mul(usize left, usize right) noexcept -> std::optional<usize>
{
    if (left == 0 or right == 0) return usize{};
    if (left > std::numeric_limits<usize>::max() / right) return std::nullopt;
    return left * right;
}

template <std::integral Left, std::integral Right>
[[nodiscard]] constexpr auto checked_add(Left left, Right right) -> std::optional<usize>
{
    const auto checked_left = to_count(left);
    const auto checked_right = to_count(right);
    if (checked_left > std::numeric_limits<usize>::max() - checked_right) return std::nullopt;
    return checked_left + checked_right;
}

template <std::integral Count>
[[nodiscard]] constexpr auto checked_product(Count value) -> std::optional<usize>
{
    return to_count(value);
}

template <std::integral First, std::integral Second, std::integral... Rest>
[[nodiscard]] constexpr auto checked_product(First first, Second second, Rest... rest)
    -> std::optional<usize>
{
    const auto pair = checked_mul(to_count(first), to_count(second));
    if (not pair.has_value()) return std::nullopt;
    return checked_product(*pair, rest...);
}

template <std::integral Count>
[[nodiscard]] constexpr auto checked_ceil_div(Count numerator, Count denominator) -> Count
{
    if (denominator <= 0) throw Error{Status::invalid_argument, "nonpositive divisor"};
    if constexpr (std::signed_integral<Count>)
    {
        if (numerator < 0) throw Error{Status::invalid_argument, "negative dividend"};
    }
    return numerator / denominator + static_cast<Count>(numerator % denominator != 0);
}

[[nodiscard]] constexpr auto checked_align_up(usize value, usize alignment) -> std::optional<usize>
{
    const auto alignment_is_power_of_two = alignment != 0 and (alignment & (alignment - 1)) == 0;
    if (not alignment_is_power_of_two)
        throw Error{Status::invalid_argument, "alignment is not a power of two"};
    const auto padded = checked_add(value, alignment - 1);
    if (not padded.has_value()) return std::nullopt;
    return *padded & ~(alignment - 1);
}
}

#endif
