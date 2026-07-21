#ifndef QNPEPS_ERROR_CUH
#define QNPEPS_ERROR_CUH

#include "capi/qnpeps.h"
#include "types.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <source_location>

namespace qnpeps
{
auto reset_err() -> void;
auto err_state() -> qnpeps_status&;
auto err_file() -> const char*;
auto err_line() -> i32;
auto err_message() -> const char*;

auto set_err_at(qnpeps_status status, const char* file, i32 line, const char* message = nullptr)
    -> qnpeps_status;

auto set_err(qnpeps_status status, std::source_location where = std::source_location::current())
    -> qnpeps_status;

auto set_cuda_err(
    cudaError_t backend_status,
    qnpeps_status status = QNPEPS_ERR_CUDA,
    std::source_location where = std::source_location::current()
) -> qnpeps_status;

auto set_cublas_err(
    cublasStatus_t backend_status, std::source_location where = std::source_location::current()
) -> qnpeps_status;

auto set_cusolver_err(
    cusolverStatus_t backend_status, std::source_location where = std::source_location::current()
) -> qnpeps_status;
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
