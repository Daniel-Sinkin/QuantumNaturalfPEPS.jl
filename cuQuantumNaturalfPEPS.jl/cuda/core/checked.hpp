#ifndef QNPEPS_CHECKED_HPP
#define QNPEPS_CHECKED_HPP

#include "core/error.hpp"
#include "core/types.cuh"

#include <complex>
#include <concepts>
#include <limits>
#include <optional>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>

namespace qnpeps
{
template <std::signed_integral Count>
[[nodiscard]] constexpr auto to_count(Count value) -> usize
{
    if (value < 0) fail(Status::bad_config, "negative element count {}", value);
    if constexpr (sizeof(Count) > sizeof(usize))
    {
        using UnsignedCount = std::make_unsigned_t<Count>;
        const auto maximum = static_cast<UnsignedCount>(std::numeric_limits<usize>::max());
        if (static_cast<UnsignedCount>(value) > maximum)
            fail(Status::bad_config, "element count {} exceeds size range", value);
    }
    return static_cast<usize>(value);
}

template <std::unsigned_integral Count>
[[nodiscard]] constexpr auto to_count(Count value) -> usize
{
    if constexpr (sizeof(Count) > sizeof(usize))
    {
        const auto maximum = static_cast<Count>(std::numeric_limits<usize>::max());
        if (value > maximum) fail(Status::bad_config, "element count {} exceeds size range", value);
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
    if (denominator <= 0) fail(Status::bad_config, "nonpositive divisor {}", denominator);
    if constexpr (std::signed_integral<Count>)
    {
        if (numerator < 0) fail(Status::bad_config, "negative dividend {}", numerator);
    }
    return numerator / denominator + static_cast<Count>(numerator % denominator != 0);
}

[[nodiscard]] constexpr auto checked_align_up(usize value, usize alignment) -> std::optional<usize>
{
    const auto alignment_is_power_of_two = alignment != 0 and (alignment & (alignment - 1)) == 0;
    if (not alignment_is_power_of_two)
        fail(Status::bad_config, "alignment {} is not a power of two", alignment);
    const auto padded = checked_add(value, alignment - 1);
    if (not padded.has_value()) return std::nullopt;
    return *padded & ~(alignment - 1);
}

namespace detail
{
template <typename Value>
[[nodiscard]] auto value_text(Value value) -> std::string
{
    return std::to_string(value);
}

template <typename Scalar>
[[nodiscard]] auto value_text(const std::complex<Scalar>& value) -> std::string
{
    return "(" + std::to_string(value.real()) + ", " + std::to_string(value.imag()) + ")";
}

[[nodiscard]] inline auto value_text(ComplexF32 value) -> std::string
{
    return "(" + std::to_string(value.re) + ", " + std::to_string(value.im) + ")";
}
}

[[noreturn]] inline auto throw_size_overflow(
    std::string_view context, std::source_location location = std::source_location::current()
) -> void
{
    fail_at(Status::bad_config, location, std::format("{} size computation overflowed", context));
}

template <std::integral... Dimensions>
[[nodiscard]] auto element_count(std::string_view name, Dimensions... dimensions) -> usize
{
    const auto result = checked_product(dimensions...);
    if (not result.has_value()) throw_size_overflow(name);
    return *result;
}

inline auto require_bytes(
    usize actual,
    usize expected,
    std::string_view context,
    std::source_location location = std::source_location::current()
) -> void
{
    if (actual >= expected) return;
    fail_at(
        Status::bad_config,
        location,
        std::format("{} too small (got {} bytes, expected {} bytes)", context, actual, expected)
    );
}

inline auto require_bytes(
    usize actual,
    i64 expected,
    std::string_view context,
    std::source_location location = std::source_location::current()
) -> void
{
    require_bytes(actual, to_count(expected), context, location);
}

inline auto require_count(
    usize actual,
    usize expected,
    std::string_view context,
    std::source_location location = std::source_location::current()
) -> void
{
    if (actual >= expected) return;
    fail_at(
        Status::bad_config,
        location,
        std::format(
            "{} too small (got {} elements, expected {} elements)", context, actual, expected
        )
    );
}

inline auto require_count(
    usize actual,
    i64 expected,
    std::string_view context,
    std::source_location location = std::source_location::current()
) -> void
{
    require_count(actual, to_count(expected), context, location);
}

template <typename Value>
auto require_equal(
    Value actual,
    Value expected,
    std::string_view unit,
    std::string_view context,
    std::source_location location = std::source_location::current()
) -> void
{
    if (actual == expected) return;
    fail_at(
        Status::bad_config,
        location,
        std::format(
            "{} mismatch (got {} {}, expected {} {})",
            context,
            detail::value_text(actual),
            unit,
            detail::value_text(expected),
            unit
        )
    );
}

template <typename Value>
auto require_less_equal(
    Value actual,
    Value maximum,
    std::string_view unit,
    std::string_view context,
    std::source_location location = std::source_location::current()
) -> void
{
    if (actual <= maximum) return;
    fail_at(
        Status::bad_config,
        location,
        std::format(
            "{} exceeds limit (got {} {}, expected at most {} {})",
            context,
            detail::value_text(actual),
            unit,
            detail::value_text(maximum),
            unit
        )
    );
}

template <typename Value>
auto require_at_least(
    Value actual,
    Value minimum,
    std::string_view unit,
    std::string_view context,
    std::source_location location = std::source_location::current()
) -> void
{
    if (actual >= minimum) return;
    fail_at(
        Status::bad_config,
        location,
        std::format(
            "{} below limit (got {} {}, expected at least {} {})",
            context,
            detail::value_text(actual),
            unit,
            detail::value_text(minimum),
            unit
        )
    );
}

[[nodiscard]] inline auto require_fits_int(
    i64 value,
    std::string_view unit,
    std::string_view context,
    std::source_location location = std::source_location::current()
) -> int
{
    const i64 minimum{std::numeric_limits<int>::min()};
    const i64 maximum{std::numeric_limits<int>::max()};
    require_at_least(value, minimum, unit, context, location);
    require_less_equal(value, maximum, unit, context, location);
    return static_cast<int>(value);
}
}

#endif
