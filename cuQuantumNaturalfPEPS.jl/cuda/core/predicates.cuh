#ifndef QNPEPS_PREDICATES_CUH
#define QNPEPS_PREDICATES_CUH

#include "core/types.cuh"

#include <vector>

namespace qnpeps
{
namespace predicates
{
template <typename Value, typename Allocator, typename Predicate>
[[nodiscard]] constexpr auto all_vector(
    const std::vector<Value, Allocator>& values, Predicate predicate
) noexcept -> bool
{
    for (const auto& value : values)
        if (not predicate(value)) return false;
    return true;
}
}

template <typename Value>
[[nodiscard]] __host__ __device__ constexpr auto in_range(
    const Value& value, const Value& lower, const Value& upper
) noexcept -> bool
{
    return value >= lower and value <= upper;
}

template <typename... Values>
[[nodiscard]] __host__ __device__ constexpr auto all_negative(const Values&... values) noexcept
    -> bool
{
    return ((values < Values{}) and ...);
}

template <typename Value, typename Allocator>
[[nodiscard]] constexpr auto all_negative(const std::vector<Value, Allocator>& values) noexcept
    -> bool
{
    return predicates::all_vector(values, [](const Value& value) { return value < Value{}; });
}

template <typename... Values>
[[nodiscard]] __host__ __device__ constexpr auto all_nonnegative(const Values&... values) noexcept
    -> bool
{
    return ((values >= Values{}) and ...);
}

template <typename Value, typename Allocator>
[[nodiscard]] constexpr auto all_nonnegative(const std::vector<Value, Allocator>& values) noexcept
    -> bool
{
    return predicates::all_vector(values, [](const Value& value) { return value >= Value{}; });
}

template <typename... Values>
[[nodiscard]] __host__ __device__ constexpr auto all_positive(const Values&... values) noexcept
    -> bool
{
    return ((values > Values{}) and ...);
}

template <typename Value, typename Allocator>
[[nodiscard]] constexpr auto all_positive(const std::vector<Value, Allocator>& values) noexcept
    -> bool
{
    return predicates::all_vector(values, [](const Value& value) { return value > Value{}; });
}

template <typename... Values>
[[nodiscard]] __host__ __device__ constexpr auto all_nonpositive(const Values&... values) noexcept
    -> bool
{
    return ((values <= Values{}) and ...);
}

template <typename Value, typename Allocator>
[[nodiscard]] constexpr auto all_nonpositive(const std::vector<Value, Allocator>& values) noexcept
    -> bool
{
    return predicates::all_vector(values, [](const Value& value) { return value <= Value{}; });
}

template <typename... Values>
[[nodiscard]] __host__ __device__ constexpr auto all_zero(const Values&... values) noexcept -> bool
{
    return ((values == Values{}) and ...);
}

template <typename Value, typename Allocator>
[[nodiscard]] constexpr auto all_zero(const std::vector<Value, Allocator>& values) noexcept -> bool
{
    return predicates::all_vector(values, [](const Value& value) { return value == Value{}; });
}

template <typename... Values>
[[nodiscard]] __host__ __device__ constexpr auto all_nonzero(const Values&... values) noexcept
    -> bool
{
    return ((values != Values{}) and ...);
}

template <typename Value, typename Allocator>
[[nodiscard]] constexpr auto all_nonzero(const std::vector<Value, Allocator>& values) noexcept
    -> bool
{
    return predicates::all_vector(values, [](const Value& value) { return value != Value{}; });
}

template <typename Left, typename Right>
[[nodiscard]] __host__ __device__ constexpr auto same_size(
    const Left& left, const Right& right
) noexcept -> bool
{
    return left.size() == right.size();
}

template <typename Left, typename LeftAllocator, typename Right, typename RightAllocator>
[[nodiscard]] constexpr auto same_size(
    const std::vector<Left, LeftAllocator>& left, const std::vector<Right, RightAllocator>& right
) noexcept -> bool
{
    return left.size() == right.size();
}
}

#endif
