#ifndef QNPEPS_DENSITYMATRIX_BACKEND_CUH
#define QNPEPS_DENSITYMATRIX_BACKEND_CUH

#include "linalg/linalg.cuh"

namespace qnpeps::densitymatrix
{
using HandleState = LinalgState;

[[nodiscard]] auto enter_handle_state(Linalg& linalg, HandleState& state) -> bool;
auto restore_handle_state(Linalg& linalg, const HandleState& state) -> void;
auto matmul(
    Linalg& linalg,
    CuMatrixCF64Const a,
    CuMatrixCF64Const b,
    CuMatrixCF64 c,
    BlasOp op_a = BlasOp::none,
    BlasOp op_b = BlasOp::none,
    f64 beta = 0.0
) -> void;
[[nodiscard]] auto eigen_workspace_bytes(Linalg& linalg, int order) -> usize;
auto eigen_hermitian(
    Linalg& linalg,
    CuMatrixCF64 matrix,
    f64* eigenvalues,
    void* workspace,
    usize workspace_bytes,
    i32* information
) -> void;
}

#endif
