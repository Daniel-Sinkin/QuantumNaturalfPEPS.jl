#ifndef QNPEPS_LINALG_STATE_CUH
#define QNPEPS_LINALG_STATE_CUH

#include "core/error.hpp"
#include "linalg/linalg.cuh"

#include <cublas_v2.h>
#include <cusolverDn.h>

namespace qnpeps
{
inline auto Linalg::enter_default_state(LinalgState& state) -> void
{
    check(cublasGetMathMode(blas_->get(), &state.math));
    check(cublasGetPointerMode(blas_->get(), &state.pointer));
    check(cublasGetAtomicsMode(blas_->get(), &state.atomics));
    check(cusolverDnGetDeterministicMode(solver_->get(), &state.deterministic));
    check(cublasSetMathMode(blas_->get(), CUBLAS_DEFAULT_MATH));
    check(cublasSetPointerMode(blas_->get(), CUBLAS_POINTER_MODE_HOST));
    check(cublasSetAtomicsMode(blas_->get(), CUBLAS_ATOMICS_NOT_ALLOWED));
}

inline auto Linalg::restore_state(const LinalgState& state) -> void
{
    unchecked(cublasSetMathMode(blas_->get(), state.math));
    unchecked(cublasSetPointerMode(blas_->get(), state.pointer));
    unchecked(cublasSetAtomicsMode(blas_->get(), state.atomics));
    unchecked(cusolverDnSetDeterministicMode(solver_->get(), state.deterministic));
}
}

#endif
