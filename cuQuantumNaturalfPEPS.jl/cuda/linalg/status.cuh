#ifndef QNPEPS_LINALG_STATUS_CUH
#define QNPEPS_LINALG_STATUS_CUH

#include <cublas_v2.h>

namespace qnpeps
{

[[nodiscard]] inline auto blas_status_name(cublasStatus_t status) noexcept -> const char*
{
    return cublasGetStatusName(status);
}

[[nodiscard]] inline auto blas_status_description(cublasStatus_t status) noexcept -> const char*
{
    return cublasGetStatusString(status);
}

}

#endif
