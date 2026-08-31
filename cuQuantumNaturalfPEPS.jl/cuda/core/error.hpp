#ifndef QNPEPS_INTERNAL_ERROR_HPP
#define QNPEPS_INTERNAL_ERROR_HPP

#include "core/checked.hpp"
#include "core/error_type.hpp"
#include "core/types.cuh"

#include <complex>
#include <concepts>
#include <limits>
#include <source_location>
#include <string>
#include <string_view>

namespace qnpeps
{
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

inline auto require(
    bool condition, std::string_view message, Status status = Status::invalid_argument
) -> void
{
    if (condition) return;
    throw Error{status, std::string{message}};
}

[[noreturn]] inline auto throw_size_overflow(
    std::string_view context, std::source_location location = std::source_location::current()
) -> void
{
    throw Error{
        Status::invalid_argument, std::string{context} + " size computation overflowed", location
    };
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
    throw Error{
        Status::invalid_argument,
        std::string{context} + " too small (got " + std::to_string(actual) + " bytes, expected "
            + std::to_string(expected) + " bytes)",
        location
    };
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
    throw Error{
        Status::invalid_argument,
        std::string{context} + " too small (got " + std::to_string(actual) + " elements, expected "
            + std::to_string(expected) + " elements)",
        location
    };
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
    throw Error{
        Status::invalid_argument,
        std::string{context} + " mismatch (got " + detail::value_text(actual) + " "
            + std::string{unit} + ", expected " + detail::value_text(expected) + " "
            + std::string{unit} + ")",
        location
    };
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
    throw Error{
        Status::invalid_argument,
        std::string{context} + " exceeds limit (got " + detail::value_text(actual) + " "
            + std::string{unit} + ", expected at most " + detail::value_text(maximum) + " "
            + std::string{unit} + ")",
        location
    };
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
    throw Error{
        Status::invalid_argument,
        std::string{context} + " below limit (got " + detail::value_text(actual) + " "
            + std::string{unit} + ", expected at least " + detail::value_text(minimum) + " "
            + std::string{unit} + ")",
        location
    };
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
