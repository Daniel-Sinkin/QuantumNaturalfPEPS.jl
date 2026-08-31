#include "linalg/factorization.cuh"

#include "core/cuda_utils.cuh"
#include "core/types.cuh"

#include <cublas_v2.h>
#include <cusolverDn.h>

namespace qnpeps
{

auto Linalg::cholesky_lower_batched(int n, cuFloatComplex** as, int lda, int* info, int batch_size)
    -> void
{
    const auto valid = n > 0 and as and lda >= n and info and batch_size > 0;
    if (not valid)
    {
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return;
    }
    CUSOLVER_CHECK(cusolverDnCpotrfBatched(
        solver_->get(), CUBLAS_FILL_MODE_LOWER, n, as, lda, info, batch_size
    ));
}

auto Linalg::triangular_solve_batched(
    cuFloatComplex* const* as,
    int lda,
    cuFloatComplex* const* bs,
    int ldb,
    int m,
    int n,
    int batch_size,
    const TriangularSolveConfig& cfg
) -> void
{
    const auto triangular_dim = cfg.side_right ? n : m;
    const auto pointers_valid = as and bs;
    const auto dimensions_valid = m > 0 and n > 0;
    const auto leading_dimensions_valid = lda >= triangular_dim and ldb >= m;
    const auto batch_valid = batch_size > 0;
    const auto shape_valid = dimensions_valid and leading_dimensions_valid;
    const auto valid = pointers_valid and shape_valid and batch_valid;
    if (not valid)
    {
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return;
    }
    const auto alpha = cuFloatComplex{cfg.alpha_real, cfg.alpha_imag};
    CUBLAS_CHECK(cublasCtrsmBatched(
        blas_->get(),
        cfg.side_right ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT,
        to_cublas(cfg.fill_mode),
        to_cublas(cfg.op),
        cfg.has_diag ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT,
        m,
        n,
        &alpha,
        as,
        lda,
        bs,
        ldb,
        batch_size
    ));
}
}
