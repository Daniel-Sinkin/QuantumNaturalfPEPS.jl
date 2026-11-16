#include "parts.cuh"

namespace qn_eloc::env
{

auto arena_product(std::initializer_list<std::uint64_t> factors, usize& result) -> void
{
    auto product = std::uint64_t{1};
    for (const std::uint64_t factor : factors)
    {
        const auto overflows =
            factor != 0 and product > std::numeric_limits<std::uint64_t>::max() / factor;
        require(
            not overflows,
            Status::out_of_memory,
            "arena size product {} times {} overflows",
            product,
            factor
        );
        product *= factor;
    }
    require(
        product <= std::numeric_limits<usize>::max(),
        Status::out_of_memory,
        "arena size {} exceeds the size range",
        product
    );
    result = static_cast<usize>(product);
}

auto arena_sum(usize left, usize right, usize& result) -> void
{
    require(
        right <= std::numeric_limits<usize>::max() - left,
        Status::out_of_memory,
        "arena size sum {} plus {} overflows",
        left,
        right
    );
    result = left + right;
}

auto arena_accumulate(usize& total, usize term) -> void
{
    usize result{};
    arena_sum(total, term, result);
    total = result;
}

auto arena_align(usize value, usize& result) -> void
{
    usize padded{};
    arena_sum(value, static_cast<usize>(CUDA_MALLOC_ALIGN - 1), padded);
    result = padded & ~static_cast<usize>(CUDA_MALLOC_ALIGN - 1);
}

auto arena_slot(std::initializer_list<std::uint64_t> factors, i64& result) -> void
{
    usize product{};
    arena_product(factors, product);
    require(
        product <= static_cast<usize>(std::numeric_limits<i64>::max()),
        Status::out_of_memory,
        "arena slot of {} elements exceeds the i64 range",
        product
    );
    result = static_cast<i64>(product);
}

auto arena_int(std::initializer_list<std::uint64_t> factors, int& result) -> void
{
    usize product{};
    arena_product(factors, product);
    require(
        product <= static_cast<usize>(std::numeric_limits<int>::max()),
        Status::out_of_memory,
        "arena extent {} exceeds the int range",
        product
    );
    result = static_cast<int>(product);
}

auto arena_i64_sum(i64 left, i64 right, i64& result) -> void
{
    const auto valid = left >= 0 and right >= 0 and right <= std::numeric_limits<i64>::max() - left;
    require(valid, Status::out_of_memory, "arena offset sum {} plus {} is invalid", left, right);
    result = left + right;
}

}
