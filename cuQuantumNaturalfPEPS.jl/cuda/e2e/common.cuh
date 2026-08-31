#ifndef QNPEPS_E2E_COMMON_CUH
#define QNPEPS_E2E_COMMON_CUH

#include "core/error.cuh"
#include "core/types.cuh"
#include "dans_qnpeps_e2e.h"

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

using qnpeps::f32;
using qnpeps::f64;
using qnpeps::i32;
using qnpeps::i64;
using qnpeps::operator""_uz;
using qnpeps::u32;
using qnpeps::u64;
using qnpeps::u8;
using qnpeps::usize;

namespace qn_e2e
{

using cf = qnpeps::ComplexF32;
using cd = cuDoubleComplex;

static_assert(QNPEPS_E2E_OK == static_cast<int>(QNPEPS_OK));
static_assert(QNPEPS_E2E_ERR_NULL_ARG == static_cast<int>(QNPEPS_ERR_NULL_ARG));
static_assert(QNPEPS_E2E_ERR_BAD_CONFIG == static_cast<int>(QNPEPS_ERR_BAD_CONFIG));
static_assert(QNPEPS_E2E_ERR_BAD_VERSION == static_cast<int>(QNPEPS_ERR_BAD_VERSION));
static_assert(QNPEPS_E2E_ERR_CUDA == static_cast<int>(QNPEPS_ERR_CUDA));
static_assert(QNPEPS_E2E_ERR_OOM == static_cast<int>(QNPEPS_ERR_OOM));
static_assert(QNPEPS_E2E_ERR_INTERNAL == static_cast<int>(QNPEPS_ERR_INTERNAL));

inline auto root_status(qnpeps_e2e_status status) -> qnpeps_status
{
    return static_cast<qnpeps_status>(status);
}

inline auto err_state() -> qnpeps_e2e_status
{
    return static_cast<qnpeps_e2e_status>(qnpeps::err_state());
}
inline void clear_err()
{
    qnpeps::reset_err();
}
inline void set_err(qnpeps_e2e_status s)
{
    qnpeps::set_err(root_status(s));
}

}

#define QN_E2E_CUDA_CHECK(x)                                                                       \
    do                                                                                             \
    {                                                                                              \
        const cudaError_t e_ = (x);                                                                \
        if (e_ != cudaSuccess) qnpeps::set_cuda_err(e_);                                           \
    } while (0)

#define QN_E2E_CUBLAS_CHECK(x)                                                                     \
    do                                                                                             \
    {                                                                                              \
        const cublasStatus_t s_ = (x);                                                             \
        if (s_ != CUBLAS_STATUS_SUCCESS) qnpeps::set_cublas_err(s_);                               \
    } while (0)

#define QN_E2E_CUSOLVER_CHECK(x)                                                                   \
    do                                                                                             \
    {                                                                                              \
        const cusolverStatus_t s_ = (x);                                                           \
        if (s_ != CUSOLVER_STATUS_SUCCESS) qnpeps::set_cusolver_err(s_);                           \
    } while (0)

#endif
