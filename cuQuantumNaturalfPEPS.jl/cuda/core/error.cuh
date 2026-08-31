#ifndef QNPEPS_ERROR_CUH
#define QNPEPS_ERROR_CUH

#include "capi/qnpeps.h"
#include "core/types.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <source_location>

namespace qnpeps
{
auto reset_err() noexcept -> void;
[[nodiscard]] auto err_state() noexcept -> qnpeps_status&;
[[nodiscard]] auto err_file() noexcept -> const char*;
[[nodiscard]] auto err_line() noexcept -> i32;
[[nodiscard]] auto err_message() noexcept -> const char*;
[[nodiscard]] auto err_backend() noexcept -> const char*;
[[nodiscard]] auto err_backend_code() noexcept -> i32;

auto set_err_at(
    qnpeps_status status, const char* file, i32 line, const char* message = nullptr
) noexcept -> qnpeps_status;

auto set_backend_err_at(
    qnpeps_status status,
    const char* backend,
    i32 backend_code,
    const char* file,
    i32 line,
    const char* message = nullptr
) noexcept -> qnpeps_status;

auto set_err(
    qnpeps_status status, std::source_location where = std::source_location::current()
) noexcept -> qnpeps_status;

auto set_cuda_err(
    cudaError_t backend_status,
    qnpeps_status status = QNPEPS_ERR_CUDA,
    std::source_location where = std::source_location::current()
) noexcept -> qnpeps_status;

auto set_cublas_err(
    cublasStatus_t backend_status, std::source_location where = std::source_location::current()
) noexcept -> qnpeps_status;

auto set_cusolver_err(
    cusolverStatus_t backend_status, std::source_location where = std::source_location::current()
) noexcept -> qnpeps_status;

[[nodiscard]] inline auto cuda_status(
    cudaError_t backend_status, std::source_location where = std::source_location::current()
) noexcept -> qnpeps_status
{
    if (backend_status == cudaSuccess) return QNPEPS_OK;
    const auto mapped =
        backend_status == cudaErrorMemoryAllocation ? QNPEPS_ERR_OOM : QNPEPS_ERR_CUDA;
    set_cuda_err(backend_status, mapped, where);
    return mapped;
}

[[nodiscard]] inline auto cublas_status(
    cublasStatus_t backend_status, std::source_location where = std::source_location::current()
) noexcept -> qnpeps_status
{
    if (backend_status == CUBLAS_STATUS_SUCCESS) return QNPEPS_OK;
    set_cublas_err(backend_status, where);
    return QNPEPS_ERR_CUDA;
}
}

#define CUDA_CHECK(x)                                                                              \
    do                                                                                             \
    {                                                                                              \
        const auto status_ = (x);                                                                  \
        if (status_ != cudaSuccess) qnpeps::set_cuda_err(status_);                                 \
    } while (0)

#define CUBLAS_CHECK(x)                                                                            \
    do                                                                                             \
    {                                                                                              \
        const auto status_ = (x);                                                                  \
        if (status_ != CUBLAS_STATUS_SUCCESS) qnpeps::set_cublas_err(status_);                     \
    } while (0)

#define CUSOLVER_CHECK(x)                                                                          \
    do                                                                                             \
    {                                                                                              \
        const auto status_ = (x);                                                                  \
        if (status_ != CUSOLVER_STATUS_SUCCESS) qnpeps::set_cusolver_err(status_);                 \
    } while (0)

#define CUDA_NOCHECK(x)                                                                            \
    do                                                                                             \
    {                                                                                              \
        (void) (x);                                                                                \
    } while (0)

#endif
