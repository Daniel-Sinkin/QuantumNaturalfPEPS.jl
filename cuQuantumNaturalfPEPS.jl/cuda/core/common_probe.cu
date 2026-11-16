#include "core/checked.hpp"
#include "core/error.hpp"

#include <cstdlib>
#include <limits>

namespace
{
using qnpeps::cf32;
using qnpeps::i32;
using qnpeps::i64;
using qnpeps::operator""_i32;
using qnpeps::operator""_i64;
using qnpeps::operator""_u64;
using qnpeps::operator""_uz;
using qnpeps::u64;
using qnpeps::usize;

[[nodiscard]] auto require_probe() -> bool
{
    const auto status = qnpeps::guarded(
        []
        {
            qnpeps::require(true, "require probe");
            qnpeps::require_bytes(16_uz, 8_uz, "byte probe");
            qnpeps::require_bytes(16_uz, 8_i64, "signed byte probe");
            qnpeps::require_count(8_uz, 4_uz, "count probe");
            qnpeps::require_count(8_uz, 4_i64, "signed count probe");
            qnpeps::require_equal(4_i64, 4_i64, "elements", "equal probe");
            qnpeps::require_equal(cf32{1.0f, 2.0f}, cf32{1.0f, 2.0f}, "values", "complex probe");
            qnpeps::require_less_equal(4_i64, 8_i64, "elements", "maximum probe");
            qnpeps::require_at_least(8_i64, 4_i64, "elements", "minimum probe");
            qnpeps::require(
                qnpeps::require_fits_int(8_i64, "elements", "integer probe") == 8, "fit probe"
            );
            qnpeps::require(
                qnpeps::element_count("element probe", 2_i64, 3_i64, 4_uz) == 24_uz, "product probe"
            );
        }
    );
    if (status != QNPEPS_OK) return false;

    const auto overflow_status = qnpeps::guarded([] { qnpeps::throw_size_overflow("probe"); });
    return overflow_status == QNPEPS_ERR_BAD_CONFIG;
}

[[nodiscard]] auto checked_probe() -> bool
{
    const auto count_signed = qnpeps::to_count(8_i64);
    const auto count_unsigned = qnpeps::to_count(8_u64);
    const auto product = qnpeps::checked_mul(3_uz, 4_uz);
    const auto sum = qnpeps::checked_add(3_i32, 4_u64);
    const auto variadic_product = qnpeps::checked_product(2_i64, 3_i32, 4_uz);
    const auto quotient = qnpeps::checked_ceil_div(9_i64, 4_i64);
    const auto aligned = qnpeps::checked_align_up(17_uz, 16_uz);
    const auto overflow = qnpeps::checked_mul(std::numeric_limits<usize>::max(), 2_uz);
    return count_signed == 8_uz and count_unsigned == 8_uz and product == 12_uz and sum == 7_uz
           and variadic_product == 24_uz and quotient == 3_i64 and aligned == 32_uz
           and not overflow.has_value();
}

}

auto main() -> int
{
    const auto passed = require_probe() and checked_probe();
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
