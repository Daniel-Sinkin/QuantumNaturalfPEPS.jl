#ifndef QNPEPS_PROTOCOL_CALL_HPP
#define QNPEPS_PROTOCOL_CALL_HPP

#include "capi/qnpeps.h"
#include "core/error_type.hpp"

#include <exception>
#include <new>
#include <string>
#include <utility>

namespace qnpeps::protocol
{
[[nodiscard]] constexpr auto to_protocol(Status status) noexcept -> qnpeps_status
{
    switch (status)
    {
        case Status::success:
            return QNPEPS_OK;
        case Status::null_pointer:
            return QNPEPS_ERR_NULL_ARG;
        case Status::invalid_struct_size:
            return QNPEPS_ERR_BAD_VERSION;
        case Status::invalid_argument:
            return QNPEPS_ERR_BAD_CONFIG;
        case Status::cuda_error:
            return QNPEPS_ERR_CUDA;
        case Status::out_of_memory:
            return QNPEPS_ERR_OOM;
        case Status::not_implemented:
            return QNPEPS_ERR_INTERNAL;
        case Status::internal_error:
            return QNPEPS_ERR_INTERNAL;
    }
    return QNPEPS_ERR_INTERNAL;
}

template <typename Function>
auto protocol_call(Function&& function) noexcept -> qnpeps_status
{
    try
    {
        std::forward<Function>(function)();
        return QNPEPS_OK;
    }
    catch (const Error& error)
    {
        error_message_state() = [&]
        {
            auto message = std::string{error.what()};
            message += " at ";
            message += error.location().file_name();
            message += ":";
            message += std::to_string(error.location().line());
            return message;
        }();
        return to_protocol(error.status());
    }
    catch (const std::bad_alloc& error)
    {
        error_message_state() = error.what();
        return QNPEPS_ERR_OOM;
    }
    catch (const std::exception& error)
    {
        error_message_state() = error.what();
        return QNPEPS_ERR_INTERNAL;
    }
    catch (...)
    {
        error_message_state() = "unknown internal failure";
        return QNPEPS_ERR_INTERNAL;
    }
}
}

#endif
