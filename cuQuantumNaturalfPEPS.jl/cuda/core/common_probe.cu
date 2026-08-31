#include "core/error.hpp"
#include "core/protocol_call.hpp"

#include <array>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <string_view>

namespace
{
using qnpeps::cf32;
using qnpeps::Error;
using qnpeps::i32;
using qnpeps::i64;
using qnpeps::Status;
using qnpeps::operator""_i32;
using qnpeps::operator""_i64;
using qnpeps::operator""_u64;
using qnpeps::operator""_uz;
using qnpeps::u64;
using qnpeps::usize;

struct StatusMapping
{
    Status internal;
    qnpeps_status protocol;
};

[[nodiscard]] auto require_probe() -> bool
{
    const auto status = qnpeps::protocol::protocol_call(
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

    const auto overflow_status =
        qnpeps::protocol::protocol_call([] { qnpeps::throw_size_overflow("probe"); });
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

[[nodiscard]] auto mapping_probe() -> bool
{
    constexpr std::array mappings{
        StatusMapping{Status::success, QNPEPS_OK},
        StatusMapping{Status::null_pointer, QNPEPS_ERR_NULL_ARG},
        StatusMapping{Status::invalid_struct_size, QNPEPS_ERR_BAD_VERSION},
        StatusMapping{Status::invalid_argument, QNPEPS_ERR_BAD_CONFIG},
        StatusMapping{Status::cuda_error, QNPEPS_ERR_CUDA},
        StatusMapping{Status::out_of_memory, QNPEPS_ERR_OOM},
        StatusMapping{Status::not_implemented, QNPEPS_ERR_INTERNAL},
        StatusMapping{Status::internal_error, QNPEPS_ERR_INTERNAL}
    };
    for (const auto& mapping : mappings)
    {
        if (qnpeps::protocol::to_protocol(mapping.internal) != mapping.protocol) return false;
    }
    return true;
}

[[nodiscard]] auto adapter_probe() -> bool
{
    const auto error_status =
        qnpeps::protocol::protocol_call([] { throw Error{Status::null_pointer, "error probe"}; });
    const auto error_recorded = qnpeps::error_message_state().find("error probe at ") == 0;

    const auto allocation_status = qnpeps::protocol::protocol_call([] { throw std::bad_alloc{}; });
    const auto allocation_recorded = not qnpeps::error_message_state().empty();
    const auto exception_status =
        qnpeps::protocol::protocol_call([] { throw std::runtime_error{"exception probe"}; });
    const auto exception_recorded = qnpeps::error_message_state() == "exception probe";
    const auto unknown_status = qnpeps::protocol::protocol_call([] { throw 1; });
    const auto unknown_recorded = qnpeps::error_message_state() == "unknown internal failure";

    return error_status == QNPEPS_ERR_NULL_ARG and error_recorded
           and allocation_status == QNPEPS_ERR_OOM and allocation_recorded
           and exception_status == QNPEPS_ERR_INTERNAL and exception_recorded
           and unknown_status == QNPEPS_ERR_INTERNAL and unknown_recorded;
}
}

auto main() -> int
{
    const auto passed = require_probe() and checked_probe() and mapping_probe() and adapter_probe();
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
