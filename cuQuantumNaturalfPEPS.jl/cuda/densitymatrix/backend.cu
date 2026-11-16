#include "densitymatrix/backend.cuh"

namespace qnpeps::densitymatrix
{
auto enter_handle_state(Linalg& linalg, HandleState& state) -> void
{
    linalg.enter_default_state(state);
}

auto restore_handle_state(Linalg& linalg, const HandleState& state) -> void
{
    linalg.restore_state(state);
}

auto matmul(
    Linalg& linalg,
    CuMatrixCF64Const a,
    CuMatrixCF64Const b,
    CuMatrixCF64 c,
    BlasOp op_a,
    BlasOp op_b,
    f64 beta_value
) -> void
{
    linalg.matmul(
        a,
        b,
        c,
        {
            .op_a = op_a,
            .op_b = op_b,
            .beta_real = static_cast<f32>(beta_value),
        }
    );
}

auto eigen_workspace_bytes(Linalg& linalg, int order) -> usize
{
    require(
        order >= 1,
        Status::bad_config,
        "density eigensolver workspace needs a positive order, got {}",
        order
    );
    const auto count = linalg.diagonalize_workspace_count(order);
    const auto bytes =
        device_align(sizeof(cuDoubleComplex) * static_cast<usize>(std::max(count, 1)));
    return bytes;
}

auto eigen_hermitian(
    Linalg& linalg,
    CuMatrixCF64 matrix,
    f64* eigenvalues,
    void* workspace,
    usize workspace_bytes,
    i32* information
) -> void
{
    const auto valid = matrix.data() and matrix.rows() > 0 and matrix.rows() == matrix.cols()
                       and eigenvalues and workspace and information
                       and workspace_bytes >= sizeof(cuDoubleComplex);
    require(
        valid,
        "density eigensolve rejects matrix {} with {} workspace bytes",
        matrix,
        workspace_bytes
    );
    linalg.diagonalize(
        matrix,
        {
            .eigenvalues = eigenvalues,
            .workspace = static_cast<cuDoubleComplex*>(workspace),
            .workspace_count = static_cast<int>(workspace_bytes / sizeof(cuDoubleComplex)),
            .info = information,
        }
    );
}
}
