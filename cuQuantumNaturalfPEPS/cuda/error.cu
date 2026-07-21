#include "error.cuh"
#include "types.cuh"

#include <array>
#include <cstdio>

namespace qnpeps
{
namespace
{
inline constexpr auto k_error_message_capacity = usize{256};

struct ErrorState
{
    qnpeps_status status{QNPEPS_OK};
    const char* file{};
    i32 line{};
    std::array<char, k_error_message_capacity> message{};
};

auto error_state() -> ErrorState&
{
    static thread_local ErrorState state{};
    return state;
}

auto cusolver_status_name(cusolverStatus_t status) -> const char*
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

auto reset_err() -> void
{
    error_state() = {};
}

auto err_state() -> qnpeps_status&
{
    return error_state().status;
}

auto err_file() -> const char*
{
    return error_state().file;
}

auto err_line() -> i32
{
    return error_state().line;
}

auto err_message() -> const char*
{
    const auto& message = error_state().message;
    return message.front() == '\0' ? nullptr : message.data();
}

auto set_err_at(qnpeps_status status, const char* file, i32 line, const char* message)
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

auto set_err(qnpeps_status status, std::source_location where) -> qnpeps_status
{
    return set_err_at(status, where.file_name(), static_cast<i32>(where.line()));
}

// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html
auto set_cuda_err(cudaError_t backend_status, qnpeps_status status, std::source_location where)
    -> qnpeps_status
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
    return set_err_at(status, where.file_name(), static_cast<i32>(where.line()), message.data());
}

// https://docs.nvidia.com/cuda/cublas/#cublasstatus-t
auto set_cublas_err(cublasStatus_t backend_status, std::source_location where) -> qnpeps_status
{
    std::array<char, k_error_message_capacity> message{};
    std::snprintf(
        message.data(),
        message.size(),
        "cuBLAS %s (%d): %s",
        cublasGetStatusName(backend_status),
        static_cast<int>(backend_status),
        cublasGetStatusString(backend_status)
    );
    return set_err_at(
        QNPEPS_ERR_CUDA, where.file_name(), static_cast<i32>(where.line()), message.data()
    );
}

// https://docs.nvidia.com/cuda/cusolver/index.html#cusolverstatus-t
auto set_cusolver_err(cusolverStatus_t backend_status, std::source_location where) -> qnpeps_status
{
    std::array<char, k_error_message_capacity> message{};
    std::snprintf(
        message.data(),
        message.size(),
        "cuSOLVER %s (%d)",
        cusolver_status_name(backend_status),
        static_cast<int>(backend_status)
    );
    return set_err_at(
        QNPEPS_ERR_CUDA, where.file_name(), static_cast<i32>(where.line()), message.data()
    );
}
}
