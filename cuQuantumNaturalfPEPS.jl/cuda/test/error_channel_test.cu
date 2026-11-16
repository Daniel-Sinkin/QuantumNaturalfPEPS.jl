#include "capi/internal/format.cuh"
#include "capi/qnpeps_status.h"
#include "core/error.hpp"

#include <array>

#include <cstdio>
#include <cuda_runtime.h>
#include <cstdlib>
#include <format>
#include <new>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>

namespace
{
using qnpeps::Status;

int failures{};

auto expect(bool condition, std::string_view name, std::string_view detail = {}) -> void
{
    std::printf(
        "[error_channel_test] %s %.*s",
        condition ? "ok" : "FAIL",
        static_cast<int>(name.size()),
        name.data()
    );
    if (not condition and not detail.empty())
        std::printf(" (%.*s)", static_cast<int>(detail.size()), detail.data());
    std::printf("\n");
    if (not condition) ++failures;
}

[[nodiscard]] auto message() -> std::string
{
    return std::string{qnpeps_last_error()};
}

[[nodiscard]] auto ends_with(std::string_view text, std::string_view suffix) -> bool
{
    return text.size() >= suffix.size() and text.substr(text.size() - suffix.size()) == suffix;
}

[[nodiscard]] auto contains(std::string_view text, std::string_view part) -> bool
{
    return text.find(part) != std::string_view::npos;
}

auto deepest_failure(int requested_bytes) -> void
{
    qnpeps::fail(Status::out_of_memory, "nested request of {} bytes", requested_bytes);
}

auto middle_layer(int requested_bytes) -> void
{
    deepest_failure(requested_bytes);
}

auto fail_message_format() -> void
{
    unsigned line{};
    const auto status = qnpeps::guarded(
        [&]
        {
            line = std::source_location::current().line() + 1;
            qnpeps::fail(
                Status::out_of_memory, "arena needs {} bytes, {} of {} in use", 1024, 512, 768
            );
        }
    );
    const auto text = message();
    const auto expected_tail =
        std::format("error_channel_test.cu:{}] arena needs 1024 bytes, 512 of 768 in use", line);
    expect(status == QNPEPS_ERR_OOM, "fail returns the out-of-memory status", text);
    expect(text.front() == '[', "fail message opens with the location bracket", text);
    expect(ends_with(text, expected_tail), "fail message carries file, line and cause", text);
}

auto require_forms() -> void
{
    const auto passing = qnpeps::guarded([] { qnpeps::require(true, "never raised {}", 1); });
    expect(passing == QNPEPS_OK and message().empty(), "require passes a true condition");

    const auto default_status =
        qnpeps::guarded([] { qnpeps::require(false, "bond {} rejected", 7); });
    expect(default_status == QNPEPS_ERR_INTERNAL, "require defaults to the internal status");
    expect(ends_with(message(), "] bond 7 rejected"), "require formats its arguments", message());

    const auto explicit_status = qnpeps::guarded(
        [] { qnpeps::require(false, Status::null_argument, "pointer {} is null", "peps"); }
    );
    expect(explicit_status == QNPEPS_ERR_NULL_ARG, "require takes an explicit status");
    expect(
        ends_with(message(), "] pointer peps is null"),
        "require formats string arguments",
        message()
    );
}

auto check_overloads() -> void
{
    expect(
        qnpeps::guarded([] { qnpeps::check(cudaSuccess); }) == QNPEPS_OK, "check passes cudaSuccess"
    );

    const auto cuda_status = qnpeps::guarded([] { qnpeps::check(cudaErrorInvalidValue); });
    expect(cuda_status == QNPEPS_ERR_CUDA, "check maps a CUDA error to the CUDA status");
    expect(contains(message(), "cudaErrorInvalidValue"), "CUDA message names the error", message());

    const auto allocation_status =
        qnpeps::guarded([] { qnpeps::check(cudaErrorMemoryAllocation); });
    expect(
        allocation_status == QNPEPS_ERR_OOM, "check maps cudaErrorMemoryAllocation to out of memory"
    );

    const auto forced_status =
        qnpeps::guarded([] { qnpeps::check_allocation(cudaErrorInvalidValue); });
    expect(forced_status == QNPEPS_ERR_OOM, "check_allocation maps any failure to out of memory");

    const auto blas_status = qnpeps::guarded([] { qnpeps::check(CUBLAS_STATUS_NOT_INITIALIZED); });
    expect(blas_status == QNPEPS_ERR_CUDA, "check maps a cuBLAS error to the CUDA status");
    expect(
        contains(message(), "CUBLAS_STATUS_NOT_INITIALIZED"),
        "cuBLAS message names the error",
        message()
    );

    const auto solver_status =
        qnpeps::guarded([] { qnpeps::check(CUSOLVER_STATUS_INVALID_VALUE, "zheevd"); });
    expect(solver_status == QNPEPS_ERR_CUDA, "check maps a cuSOLVER error to the CUDA status");
    expect(
        contains(message(), "CUSOLVER_STATUS_INVALID_VALUE (3) in zheevd"),
        "cuSOLVER message names the error and the routine",
        message()
    );
}

auto status_mapping() -> void
{
    constexpr Status statuses[]{
        Status::null_argument,
        Status::bad_config,
        Status::bad_version,
        Status::cuda,
        Status::out_of_memory,
        Status::internal
    };
    auto integer = 1;
    for (const auto status : statuses)
    {
        const auto returned = qnpeps::guarded([&] { qnpeps::fail(status, "status {}", integer); });
        expect(
            static_cast<int>(returned) == integer,
            std::format("status {} keeps its integer through the guard", integer),
            message()
        );
        ++integer;
    }
    expect(static_cast<int>(qnpeps::to_c_status(Status::ok)) == 0, "ok maps to 0");
}

auto nested_throw() -> void
{
    const auto status = qnpeps::guarded([] { middle_layer(4096); });
    expect(status == QNPEPS_ERR_OOM, "a throw two frames down reaches the guard");
    expect(
        ends_with(message(), "] nested request of 4096 bytes"),
        "the nested cause survives",
        message()
    );

    const auto collapsed = qnpeps::guarded(
        []
        {
            qnpeps::collapse_to_internal(
                {Status::out_of_memory}, [] { qnpeps::check(cudaErrorInvalidValue); }
            );
        }
    );
    expect(
        collapsed == QNPEPS_ERR_INTERNAL, "a failure outside the kept set collapses to internal"
    );
    expect(
        contains(message(), "cudaErrorInvalidValue"),
        "the collapsed failure keeps its message",
        message()
    );

    const auto kept = qnpeps::guarded(
        [] { qnpeps::collapse_to_internal({Status::out_of_memory}, [] { middle_layer(8); }); }
    );
    expect(kept == QNPEPS_ERR_OOM, "a failure in the kept set keeps its status");

    const auto relayed = qnpeps::guarded([] { qnpeps::relay(Status::cuda, "verbatim text"); });
    expect(
        relayed == QNPEPS_ERR_CUDA and message() == "verbatim text",
        "relay keeps the message verbatim",
        message()
    );
}

auto exception_kinds() -> void
{
    const auto allocation = qnpeps::guarded([] { throw std::bad_alloc{}; });
    expect(
        allocation == QNPEPS_ERR_OOM and not message().empty(),
        "std::bad_alloc maps to out of memory",
        message()
    );

    const auto runtime = qnpeps::guarded([] { throw std::runtime_error{"runtime cause"}; });
    expect(
        runtime == QNPEPS_ERR_INTERNAL and message() == "runtime cause",
        "std::exception maps to internal",
        message()
    );

    const auto system = qnpeps::guarded(
        []
        { throw std::system_error{std::make_error_code(std::errc::resource_deadlock_would_occur)}; }
    );
    expect(
        system == QNPEPS_ERR_INTERNAL and not message().empty(),
        "std::system_error maps to internal",
        message()
    );

    const auto unknown = qnpeps::guarded([] { throw 17; });
    expect(
        unknown == QNPEPS_ERR_INTERNAL and message() == "unknown exception",
        "a non-exception throw maps to internal",
        message()
    );

    const auto error =
        qnpeps::guarded([] { throw qnpeps::Error{Status::bad_version, "direct error"}; });
    expect(
        error == QNPEPS_ERR_BAD_VERSION and ends_with(message(), "] direct error"),
        "a direct Error keeps its status",
        message()
    );
}

auto message_lifetime() -> void
{
    static_cast<void>(qnpeps::guarded([] { qnpeps::fail(Status::internal, "stale"); }));
    expect(not message().empty(), "a failure leaves a message");
    static_cast<void>(qnpeps::guarded([] {}));
    expect(message().empty(), "the next guarded call clears the message");
    expect(
        qnpeps_last_error() == qnpeps::last_error_message().c_str(),
        "the accessor returns the thread message"
    );

    static_cast<void>(qnpeps::guarded([] { qnpeps::fail(Status::cuda, "prior cause"); }));
    const auto fallback =
        qnpeps::guarded_or(-1, []() -> int { qnpeps::fail(Status::bad_config, "no value"); });
    expect(
        fallback == -1 and ends_with(message(), "] prior cause"),
        "guarded_or returns the fallback and keeps the stored message",
        message()
    );
    const auto value = qnpeps::guarded_or(-1, [] { return 42; });
    expect(
        value == 42 and ends_with(message(), "] prior cause"),
        "guarded_or returns the value and keeps the stored message",
        message()
    );
    static_cast<void>(qnpeps::guarded([] {}));

    static_cast<void>(qnpeps::guarded([] { qnpeps::fail(Status::cuda, "main thread cause"); }));
    std::string worker_message{};
    std::thread worker{[&]
                       {
                           static_cast<void>(qnpeps::guarded(
                               [] { qnpeps::fail(Status::internal, "worker cause"); }
                           ));
                           worker_message = qnpeps_last_error();
                       }};
    worker.join();
    expect(
        ends_with(worker_message, "] worker cause"),
        "a worker thread holds its own message",
        worker_message
    );
    expect(
        ends_with(message(), "] main thread cause"),
        "the calling thread keeps its message",
        message()
    );
}

auto default_statuses() -> void
{
    const auto failed = qnpeps::guarded([] { qnpeps::fail("cause {}", 3); });
    expect(failed == QNPEPS_ERR_INTERNAL, "fail defaults to the internal status");
    expect(
        ends_with(message(), "] cause 3"), "fail without a status formats its arguments", message()
    );
    auto header = QnpepsConfig{};
    header.struct_size = sizeof(QnpepsConfig) + 4;
    unsigned line{};
    const auto version = qnpeps::guarded(
        [&]
        {
            line = std::source_location::current().line() + 1;
            qnpeps::require_struct_size(header, "config");
        }
    );
    const auto expected_tail = std::format(
        "error_channel_test.cu:{}] config struct_size {}, expected {}",
        line,
        sizeof(QnpepsConfig) + 4,
        sizeof(QnpepsConfig)
    );
    expect(version == QNPEPS_ERR_BAD_VERSION, "a struct size mismatch is a version failure");
    expect(
        ends_with(message(), expected_tail), "a struct size failure names its caller", message()
    );
    header.struct_size = sizeof(QnpepsConfig);
    const auto matching = qnpeps::guarded([&] { qnpeps::require_struct_size(header, "config"); });
    expect(matching == QNPEPS_OK, "a matching struct size passes");
}

auto pending_cuda_error() -> void
{
    const auto provoked = cudaSetDevice(-1);
    static_cast<void>(qnpeps::guarded([] { qnpeps::fail(Status::cuda, "provoked"); }));
    const auto left = cudaGetLastError();
    expect(
        provoked != cudaSuccess and left == cudaSuccess,
        "a recorded failure clears the pending CUDA error",
        std::format("provoked {} left {}", static_cast<int>(provoked), static_cast<int>(left))
    );
}

auto described_values() -> void
{
    auto storage = std::array<cuFloatComplex, 1>{};
    const auto matrix = qnpeps::CuMatrixCF32{storage.data(), 3, 5};
    const auto batch = qnpeps::CuMatrixBatchedCF32{storage.data(), 17, 3, 5};
    const auto matrix_text = std::format("{}", matrix);
    const auto batch_text = std::format("{}", batch);
    expect(matrix_text == "3x5", "a matrix formats as rows by columns", matrix_text);
    expect(batch_text == "3x5 stride=17", "a batched matrix adds its stride", batch_text);
    auto config = QnpepsConfig{};
    config.lx = 3;
    config.ly = 5;
    config.dim_phys = 2;
    config.dim_bond = 4;
    config.chi_s = 8;
    config.chi_dl = 16;
    const auto config_text = std::format("{}", config);
    expect(
        config_text.starts_with("lattice=3x5 dim_phys=2 dim_bond=4 chi_s=8 chi_dl=16 "),
        "a config formats as name=value pairs",
        config_text
    );
}
}

auto main() -> int
{
    fail_message_format();
    require_forms();
    check_overloads();
    status_mapping();
    nested_throw();
    exception_kinds();
    message_lifetime();
    default_statuses();
    described_values();
    pending_cuda_error();
    std::printf("[error_channel_test] %d failures\n", failures);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
