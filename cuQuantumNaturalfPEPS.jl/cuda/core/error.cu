#include "core/error.cuh"
#include "linalg/status.cuh"
#include "core/types.cuh"

#include <array>
#include <cstdio>

namespace qnpeps
{
namespace
{
inline constexpr usize k_error_message_capacity{256};

struct ErrorState
{
    qnpeps_status status{QNPEPS_OK};
    const char* file{};
    i32 line{};
    const char* backend{};
    i32 backend_code{};
    std::array<char, k_error_message_capacity> message{};
};

[[nodiscard]] auto error_state() noexcept -> ErrorState&
{
    static thread_local ErrorState state{};
    return state;
}

[[nodiscard]] auto cusolver_status_name(cusolverStatus_t status) noexcept -> const char*
{
    switch (status)
    {
        case CUSOLVER_STATUS_SUCCESS:
            return "CUSOLVER_STATUS_SUCCESS";
        case CUSOLVER_STATUS_NOT_INITIALIZED:
            return "CUSOLVER_STATUS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_ALLOC_FAILED:
            return "CUSOLVER_STATUS_ALLOC_FAILED";
        case CUSOLVER_STATUS_INVALID_VALUE:
            return "CUSOLVER_STATUS_INVALID_VALUE";
        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "CUSOLVER_STATUS_ARCH_MISMATCH";
        case CUSOLVER_STATUS_MAPPING_ERROR:
            return "CUSOLVER_STATUS_MAPPING_ERROR";
        case CUSOLVER_STATUS_EXECUTION_FAILED:
            return "CUSOLVER_STATUS_EXECUTION_FAILED";
        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "CUSOLVER_STATUS_INTERNAL_ERROR";
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        case CUSOLVER_STATUS_NOT_SUPPORTED:
            return "CUSOLVER_STATUS_NOT_SUPPORTED";
        case CUSOLVER_STATUS_ZERO_PIVOT:
            return "CUSOLVER_STATUS_ZERO_PIVOT";
        case CUSOLVER_STATUS_INVALID_LICENSE:
            return "CUSOLVER_STATUS_INVALID_LICENSE";
        case CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED:
            return "CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_IRS_PARAMS_INVALID:
            return "CUSOLVER_STATUS_IRS_PARAMS_INVALID";
        case CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC:
            return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC";
        case CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE:
            return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE";
        case CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER:
            return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER";
        case CUSOLVER_STATUS_IRS_INTERNAL_ERROR:
            return "CUSOLVER_STATUS_IRS_INTERNAL_ERROR";
        case CUSOLVER_STATUS_IRS_NOT_SUPPORTED:
            return "CUSOLVER_STATUS_IRS_NOT_SUPPORTED";
        case CUSOLVER_STATUS_IRS_OUT_OF_RANGE:
            return "CUSOLVER_STATUS_IRS_OUT_OF_RANGE";
        case CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES:
            return "CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES";
        case CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED:
            return "CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED:
            return "CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED";
        case CUSOLVER_STATUS_IRS_MATRIX_SINGULAR:
            return "CUSOLVER_STATUS_IRS_MATRIX_SINGULAR";
        case CUSOLVER_STATUS_INVALID_WORKSPACE:
            return "CUSOLVER_STATUS_INVALID_WORKSPACE";
    }
    return "CUSOLVER_STATUS_UNKNOWN";
}
}

auto reset_err() noexcept -> void
{
    error_state() = {};
}

auto err_state() noexcept -> qnpeps_status&
{
    return error_state().status;
}

auto err_file() noexcept -> const char*
{
    return error_state().file;
}

auto err_line() noexcept -> i32
{
    return error_state().line;
}

auto err_message() noexcept -> const char*
{
    const auto& message = error_state().message;
    return message.front() == '\0' ? nullptr : message.data();
}

auto err_backend() noexcept -> const char*
{
    return error_state().backend;
}

auto err_backend_code() noexcept -> i32
{
    return error_state().backend_code;
}

auto set_err_at(qnpeps_status status, const char* file, i32 line, const char* message) noexcept
    -> qnpeps_status
{
    auto& state = error_state();
    if (state.status != QNPEPS_OK) return state.status;

    state.status = status;
    state.file = file;
    state.line = line;
    if (message) std::snprintf(state.message.data(), state.message.size(), "%s", message);
    return state.status;
}

auto set_backend_err_at(
    qnpeps_status status,
    const char* backend,
    i32 backend_code,
    const char* file,
    i32 line,
    const char* message
) noexcept -> qnpeps_status
{
    auto& state = error_state();
    if (state.status != QNPEPS_OK) return state.status;

    state.status = status;
    state.file = file;
    state.line = line;
    state.backend = backend;
    state.backend_code = backend_code;
    if (message) std::snprintf(state.message.data(), state.message.size(), "%s", message);
    return state.status;
}

auto set_err(qnpeps_status status, std::source_location where) noexcept -> qnpeps_status
{
    return set_err_at(status, where.file_name(), static_cast<i32>(where.line()));
}

auto set_cuda_err(
    cudaError_t backend_status, qnpeps_status status, std::source_location where
) noexcept -> qnpeps_status
{
    std::array<char, k_error_message_capacity> message{};
    std::snprintf(
        message.data(),
        message.size(),
        "CUDA runtime %s (%d): %s",
        cudaGetErrorName(backend_status),
        static_cast<int>(backend_status),
        cudaGetErrorString(backend_status)
    );
    return set_backend_err_at(
        status,
        "cuda",
        static_cast<i32>(backend_status),
        where.file_name(),
        static_cast<i32>(where.line()),
        message.data()
    );
}

auto set_cublas_err(cublasStatus_t backend_status, std::source_location where) noexcept
    -> qnpeps_status
{
    std::array<char, k_error_message_capacity> message{};
    std::snprintf(
        message.data(),
        message.size(),
        "cuBLAS %s (%d): %s",
        blas_status_name(backend_status),
        static_cast<int>(backend_status),
        blas_status_description(backend_status)
    );
    return set_backend_err_at(
        QNPEPS_ERR_CUDA,
        "cublas",
        static_cast<i32>(backend_status),
        where.file_name(),
        static_cast<i32>(where.line()),
        message.data()
    );
}

auto set_cusolver_err(cusolverStatus_t backend_status, std::source_location where) noexcept
    -> qnpeps_status
{
    std::array<char, k_error_message_capacity> message{};
    std::snprintf(
        message.data(),
        message.size(),
        "cuSOLVER %s (%d)",
        cusolver_status_name(backend_status),
        static_cast<int>(backend_status)
    );
    return set_backend_err_at(
        QNPEPS_ERR_CUDA,
        "cusolver",
        static_cast<i32>(backend_status),
        where.file_name(),
        static_cast<i32>(where.line()),
        message.data()
    );
}
}
