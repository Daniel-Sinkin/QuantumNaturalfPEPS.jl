#ifndef QNPEPS_LINALG_STATE_CUH
#define QNPEPS_LINALG_STATE_CUH

#include "core/error.cuh"
#include "linalg/linalg.cuh"

#include <cublas_v2.h>
#include <cusolverDn.h>

namespace qnpeps
{
[[nodiscard]] inline auto Linalg::enter_default_state(LinalgState& state) noexcept -> bool
{
    CUBLAS_CHECK(cublasGetMathMode(blas_->get(), &state.math));
    CUBLAS_CHECK(cublasGetPointerMode(blas_->get(), &state.pointer));
    CUBLAS_CHECK(cublasGetAtomicsMode(blas_->get(), &state.atomics));
    CUSOLVER_CHECK(cusolverDnGetDeterministicMode(solver_->get(), &state.deterministic));
    CUBLAS_CHECK(cublasSetMathMode(blas_->get(), CUBLAS_DEFAULT_MATH));
    CUBLAS_CHECK(cublasSetPointerMode(blas_->get(), CUBLAS_POINTER_MODE_HOST));
    CUBLAS_CHECK(cublasSetAtomicsMode(blas_->get(), CUBLAS_ATOMICS_NOT_ALLOWED));
    return err_state() == QNPEPS_OK;
}

inline auto Linalg::restore_state(const LinalgState& state) noexcept -> void
{
    CUBLAS_CHECK(cublasSetMathMode(blas_->get(), state.math));
    CUBLAS_CHECK(cublasSetPointerMode(blas_->get(), state.pointer));
    CUBLAS_CHECK(cublasSetAtomicsMode(blas_->get(), state.atomics));
    CUSOLVER_CHECK(cusolverDnSetDeterministicMode(solver_->get(), state.deterministic));
}
}

#endif
