#include "linalg/qr.cuh"

#include "core/cuda_utils.cuh"
#include "core/types.cuh"
#include "linalg/scratch.cuh"

#include <algorithm>
#include <cassert>
#include <cusolverDn.h>

namespace qnpeps
{

auto Linalg::qr_workspace_count(int rows, int cols) -> int
{
    int factor_count{};
    CUSOLVER_CHECK(
        cusolverDnCgeqrf_bufferSize(solver_->get(), rows, cols, nullptr, rows, &factor_count)
    );
    int form_count{};
    CUSOLVER_CHECK(cusolverDnCungqr_bufferSize(
        solver_->get(), rows, cols, cols, nullptr, rows, nullptr, &form_count
    ));
    return std::max({factor_count, form_count, 1});
}

auto Linalg::qr_scratch(int rows, int cols) -> QrScratch
{
    if (rows <= 0 or cols <= 0 or rows < cols)
    {
        assert(false);
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return {};
    }
    const auto workspace_count = qr_workspace_count(rows, cols);
    const auto num_cols = static_cast<usize>(cols);
    const auto workspace_size = static_cast<usize>(workspace_count);
    return QrScratch{
        .reflector_bytes = device_align(num_cols * sizeof(cuFloatComplex)),
        .status_bytes = device_align(sizeof(int)),
        .workspace_bytes = device_align(sizeof(cuFloatComplex) * workspace_size),
    };
}

auto Linalg::qr_scratch(CuMatrixCF32 matrix) -> QrScratch
{
    return qr_scratch(matrix.rows(), matrix.cols());
}

auto Linalg::qr(CuMatrixCF32 matrix, void* scratch, const QrScratch& layout) -> void
{
    const auto rows = matrix.rows();
    const auto cols = matrix.cols();
    const auto valid_dimensions = rows > 0 and cols > 0 and rows >= cols;
    const auto valid_matrix = matrix.data() != nullptr;
    const auto valid_scratch = scratch and layout.total() > 0;
    const auto valid = valid_dimensions and valid_matrix and valid_scratch;
    if (not valid)
    {
        assert(false);
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return;
    }
    auto* reflector_scalars = byte_offset<cuFloatComplex>(scratch, 0);
    auto* device_status = byte_offset<int>(scratch, layout.reflector_bytes);
    const auto workspace_offset = layout.reflector_bytes + layout.status_bytes;
    auto* solver_workspace = byte_offset<cuFloatComplex>(scratch, workspace_offset);
    const auto workspace_elements = layout.workspace_bytes / sizeof(cuFloatComplex);
    const auto workspace_size = static_cast<int>(workspace_elements);
    CUSOLVER_CHECK(cusolverDnCgeqrf(
        solver_->get(),
        rows,
        cols,
        matrix.data(),
        matrix.ld(),
        reflector_scalars,
        solver_workspace,
        workspace_size,
        device_status
    ));
    CUSOLVER_CHECK(cusolverDnCungqr(
        solver_->get(),
        rows,
        cols,
        cols,
        matrix.data(),
        matrix.ld(),
        reflector_scalars,
        solver_workspace,
        workspace_size,
        device_status
    ));
}

auto Linalg::qr_factor(CuMatrixCF32 matrix, const QrStageConfig& config) -> void
{
    CUSOLVER_CHECK(cusolverDnCgeqrf(
        solver_->get(),
        matrix.rows(),
        matrix.cols(),
        matrix.data(),
        matrix.ld(),
        config.reflector_scalars,
        config.workspace,
        config.workspace_count,
        config.info
    ));
}

auto Linalg::qr_form(CuMatrixCF32 matrix, const QrStageConfig& config) -> void
{
    CUSOLVER_CHECK(cusolverDnCungqr(
        solver_->get(),
        matrix.rows(),
        matrix.cols(),
        matrix.cols(),
        matrix.data(),
        matrix.ld(),
        config.reflector_scalars,
        config.workspace,
        config.workspace_count,
        config.info
    ));
}
}
