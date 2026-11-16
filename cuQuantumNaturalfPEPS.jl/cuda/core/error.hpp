#ifndef QNPEPS_ERROR_HPP
#define QNPEPS_ERROR_HPP

#include "capi/qnpeps_status.h"
#include "core/types.cuh"

#include <concepts>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <exception>
#include <format>
#include <initializer_list>
#include <new>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace qnpeps
{
enum class Status : i32
{
    ok = QNPEPS_OK,
    null_argument = QNPEPS_ERR_NULL_ARG,
    bad_config = QNPEPS_ERR_BAD_CONFIG,
    bad_version = QNPEPS_ERR_BAD_VERSION,
    cuda = QNPEPS_ERR_CUDA,
    out_of_memory = QNPEPS_ERR_OOM,
    internal = QNPEPS_ERR_INTERNAL
};

[[nodiscard]] constexpr auto to_c_status(Status status) noexcept -> qnpeps_status
{
    return static_cast<qnpeps_status>(status);
}

class Error : public std::exception
{
  public:
    Error(
        Status status,
        std::string_view text,
        std::source_location location = std::source_location::current()
    )
        : status_{status},
          message_{std::format("[{}:{}] {}", location.file_name(), location.line(), text)}
    {
    }

    [[nodiscard]] static auto relayed(Status status, std::string message) -> Error
    {
        return Error{status, std::move(message), Relayed{}};
    }

    [[nodiscard]] auto status() const noexcept -> Status { return status_; }
    [[nodiscard]] auto what() const noexcept -> const char* override { return message_.c_str(); }

  private:
    struct Relayed
    {
    };

    Error(Status status, std::string message, Relayed)
        : status_{status}, message_{std::move(message)}
    {
    }

    Status status_{};
    std::string message_{};
};

template <typename... Args>
struct LocatedFormat
{
    template <typename Text>
        requires std::convertible_to<const Text&, std::string_view>
    consteval LocatedFormat(
        const Text& text, std::source_location location = std::source_location::current()
    )
        : format{text}, where{location}
    {
    }

    std::format_string<Args...> format;
    std::source_location where;
};

template <typename... Args>
using Format = LocatedFormat<std::type_identity_t<Args>...>;

template <typename... Args>
[[noreturn]] auto fail(Status status, Format<Args...> format, const Args&... args) -> void
{
    throw Error{
        status, std::vformat(format.format.get(), std::make_format_args(args...)), format.where
    };
}

template <typename... Args>
[[noreturn]] auto fail(Format<Args...> format, const Args&... args) -> void
{
    fail(Status::internal, format, args...);
}

[[noreturn]] inline auto fail_at(
    Status status, std::source_location location, std::string_view text
) -> void
{
    throw Error{status, text, location};
}

[[noreturn]] inline auto relay(Status status, std::string message) -> void
{
    throw Error::relayed(status, std::move(message));
}

template <typename... Args>
auto require(bool condition, Format<Args...> format, const Args&... args) -> void
{
    if (condition) return;
    fail(Status::internal, format, args...);
}

template <typename... Args>
auto require(bool condition, Status status, Format<Args...> format, const Args&... args) -> void
{
    if (condition) return;
    fail(status, format, args...);
}

template <typename Header>
auto require_struct_size(
    const Header& header,
    std::string_view name,
    std::source_location where = std::source_location::current()
) -> void
{
    if (header.struct_size == sizeof(Header)) return;
    fail_at(
        Status::bad_version,
        where,
        std::format("{} struct_size {}, expected {}", name, header.struct_size, sizeof(Header))
    );
}

[[nodiscard]] auto cuda_failure_text(cudaError_t status) -> std::string;
[[nodiscard]] auto cublas_failure_text(cublasStatus_t status) -> std::string;
[[nodiscard]] auto cusolver_failure_text(cusolverStatus_t status) -> std::string;

inline auto check(
    cudaError_t status, std::source_location location = std::source_location::current()
) -> void
{
    if (status == cudaSuccess) return;
    const auto mapped = status == cudaErrorMemoryAllocation ? Status::out_of_memory : Status::cuda;
    throw Error{mapped, cuda_failure_text(status), location};
}

inline auto check(
    cublasStatus_t status, std::source_location location = std::source_location::current()
) -> void
{
    if (status == CUBLAS_STATUS_SUCCESS) return;
    throw Error{Status::cuda, cublas_failure_text(status), location};
}

inline auto check(
    cusolverStatus_t status, std::source_location location = std::source_location::current()
) -> void
{
    if (status == CUSOLVER_STATUS_SUCCESS) return;
    throw Error{Status::cuda, cusolver_failure_text(status), location};
}

inline auto check(
    cusolverStatus_t status,
    std::string_view routine,
    std::source_location location = std::source_location::current()
) -> void
{
    if (status == CUSOLVER_STATUS_SUCCESS) return;
    throw Error{
        Status::cuda, std::format("{} in {}", cusolver_failure_text(status), routine), location
    };
}

inline auto check_allocation(
    cudaError_t status, std::source_location location = std::source_location::current()
) -> void
{
    if (status == cudaSuccess) return;
    throw Error{Status::out_of_memory, cuda_failure_text(status), location};
}

template <typename BackendStatus>
auto unchecked(BackendStatus) noexcept -> void
{
}

[[nodiscard]] auto last_error_message() noexcept -> std::string&;
auto record_failure(const char* message) noexcept -> void;

template <typename Body>
[[nodiscard]] auto guarded(Body&& body) noexcept -> qnpeps_status
{
    last_error_message().clear();
    try
    {
        std::forward<Body>(body)();
        return QNPEPS_OK;
    }
    catch (const Error& error)
    {
        record_failure(error.what());
        return to_c_status(error.status());
    }
    catch (const std::bad_alloc& error)
    {
        record_failure(error.what());
        return QNPEPS_ERR_OOM;
    }
    catch (const std::exception& error)
    {
        record_failure(error.what());
        return QNPEPS_ERR_INTERNAL;
    }
    catch (...)
    {
        record_failure("unknown exception");
        return QNPEPS_ERR_INTERNAL;
    }
}

template <typename Body>
[[nodiscard]] auto guarded_keeping_message(Body&& body) noexcept -> qnpeps_status
{
    auto kept = std::move(last_error_message());
    const auto status = guarded(std::forward<Body>(body));
    last_error_message() = std::move(kept);
    return status;
}

template <typename Value, typename Body>
[[nodiscard]] auto guarded_or(Value fallback, Body&& body) noexcept -> Value
{
    auto result = fallback;
    const auto status = guarded_keeping_message([&] { result = std::forward<Body>(body)(); });
    return status == QNPEPS_OK ? result : fallback;
}

template <typename Body>
[[nodiscard]] auto guarded_labelled(std::string_view label, Body&& body) noexcept -> qnpeps_status
{
    return guarded(
        [&]
        {
            const auto status = guarded(std::forward<Body>(body));
            if (status == QNPEPS_OK) return;
            relay(static_cast<Status>(status), std::format("[{}] {}", label, last_error_message()));
        }
    );
}

template <typename Body>
[[nodiscard]] auto guarded_labelled(std::string_view label, int gpu, Body&& body) noexcept
    -> qnpeps_status
{
    return guarded(
        [&]
        {
            const auto status = guarded(std::forward<Body>(body));
            if (status == QNPEPS_OK) return;
            relay(
                static_cast<Status>(status),
                std::format("[{}] [GPU {}] {}", label, gpu, last_error_message())
            );
        }
    );
}

template <typename Body>
auto collapse_to_internal(std::initializer_list<Status> kept, Body&& body) -> void
{
    const auto status = guarded(std::forward<Body>(body));
    if (status == QNPEPS_OK) return;
    auto mapped = Status::internal;
    for (const auto keep : kept)
    {
        if (to_c_status(keep) == status) mapped = keep;
    }
    relay(mapped, last_error_message());
}
}

#endif
