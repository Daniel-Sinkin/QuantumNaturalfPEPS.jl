#ifndef QNPEPS_ERROR_TYPE_HPP
#define QNPEPS_ERROR_TYPE_HPP

#include "core/types.cuh"

#include <source_location>
#include <string>
#include <string_view>
#include <utility>

namespace qnpeps
{
enum class Status : u32
{
    success,
    null_pointer,
    invalid_struct_size,
    invalid_argument,
    cuda_error,
    out_of_memory,
    not_implemented,
    internal_error
};

inline thread_local std::string thread_error_message{};

[[nodiscard]] inline auto error_message_state() noexcept -> std::string&
{
    return thread_error_message;
}

[[nodiscard]] constexpr auto to_string(Status status) noexcept -> std::string_view
{
    switch (status)
    {
        case Status::success:
            return "success";
        case Status::null_pointer:
            return "null pointer";
        case Status::invalid_struct_size:
            return "invalid struct size";
        case Status::invalid_argument:
            return "invalid argument";
        case Status::cuda_error:
            return "CUDA error";
        case Status::out_of_memory:
            return "out of memory";
        case Status::not_implemented:
            return "not implemented";
        case Status::internal_error:
            return "internal error";
    }
    return "unknown status";
}

class Error
{
  public:
    explicit Error(
        Status status,
        std::string message = {},
        std::source_location location = std::source_location::current()
    )
        : status_{status},
          message_{message.empty() ? std::string{to_string(status)} : std::move(message)},
          location_{location}
    {
    }

    [[nodiscard]] auto status() const noexcept -> Status { return status_; }
    [[nodiscard]] auto message() const noexcept -> std::string_view { return message_; }
    [[nodiscard]] auto location() const noexcept -> const std::source_location&
    {
        return location_;
    }
    [[nodiscard]] auto what() const noexcept -> const char* { return message_.c_str(); }

  private:
    Status status_{};
    std::string message_{};
    std::source_location location_{};
};
}

#endif
