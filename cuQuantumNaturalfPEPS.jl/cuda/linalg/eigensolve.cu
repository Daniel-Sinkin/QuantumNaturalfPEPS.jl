#include "linalg/eigensolve.cuh"

#include "core/cuda_utils.cuh"
#include "core/predicates.cuh"
#include "core/types.cuh"
#include "linalg/scratch.cuh"

#include <algorithm>
#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

namespace qnpeps
{

auto Linalg::diagonalize_workspace_count(int order) -> int
{
    if (order < 1)
    {
        qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
        return 0;
    }
    cusolverDeterministicMode_t deterministic{};
    CUSOLVER_CHECK(cusolverDnGetDeterministicMode(solver_->get(), &deterministic));
    if (err_state() != QNPEPS_OK) return 0;
    CUSOLVER_CHECK(
        cusolverDnSetDeterministicMode(solver_->get(), CUSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS)
    );
    int workspace_count{};
    if (err_state() == QNPEPS_OK)
    {
        CUSOLVER_CHECK(cusolverDnZheevd_bufferSize(
            solver_->get(),
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER,
            order,
            nullptr,
            order,
            nullptr,
            &workspace_count
        ));
    }
    CUSOLVER_CHECK(cusolverDnSetDeterministicMode(solver_->get(), deterministic));
    return workspace_count;
}

auto Linalg::diagonalize_workspace_count(CuMatrixCF64 matrix, f64* eigenvalues) -> int
{
    const auto order = matrix.rows();
    const auto valid =
        all_positive(order) and matrix.cols() == order and matrix.data() and eigenvalues;
    if (not valid)
    {
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return 0;
    }
    cusolverDeterministicMode_t deterministic{};
    CUSOLVER_CHECK(cusolverDnGetDeterministicMode(solver_->get(), &deterministic));
    if (err_state() != QNPEPS_OK) return 0;
    CUSOLVER_CHECK(
        cusolverDnSetDeterministicMode(solver_->get(), CUSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS)
    );
    int workspace_count{};
    if (err_state() == QNPEPS_OK)
    {
        CUSOLVER_CHECK(cusolverDnZheevd_bufferSize(
            solver_->get(),
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER,
            order,
            matrix.data(),
            matrix.ld(),
            eigenvalues,
            &workspace_count
        ));
    }
    CUSOLVER_CHECK(cusolverDnSetDeterministicMode(solver_->get(), deterministic));
    return workspace_count;
}

auto Linalg::diagonalize(CuMatrixCF64 matrix, const DiagonalizeConfig& config) -> void
{
    const auto order = matrix.rows();
    const auto valid_shape = all_positive(order) and matrix.cols() == order;
    const auto valid_workspace = config.workspace and all_positive(config.workspace_count);
    const auto valid =
        valid_shape and matrix.data() and config.eigenvalues and valid_workspace and config.info;
    if (not valid)
    {
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return;
    }
    CUSOLVER_CHECK(cusolverDnSetDeterministicMode(solver_->get(), CUSOLVER_DETERMINISTIC_RESULTS));
    CUDA_CHECK(cudaMemsetAsync(
        config.workspace,
        0,
        sizeof(cuDoubleComplex) * static_cast<usize>(config.workspace_count),
        stream_
    ));
    CUSOLVER_CHECK(cusolverDnZheevd(
        solver_->get(),
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER,
        order,
        matrix.data(),
        matrix.ld(),
        config.eigenvalues,
        config.workspace,
        config.workspace_count,
        config.info
    ));
}

auto Linalg::heevd_scratch(int order) -> HeevdScratch
{
    if (order <= 0)
    {
        assert(false);
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return {};
    }
    int heevd_size{};
    CUSOLVER_CHECK(cusolverDnZheevd_bufferSize(
        solver_->get(),
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER,
        order,
        nullptr,
        order,
        nullptr,
        &heevd_size
    ));
    if (err_state() != QNPEPS_OK) return {};
    const auto workspace_count = static_cast<usize>(std::max(heevd_size, 1));
    return HeevdScratch{
        .eigenvalue_bytes = device_align(static_cast<usize>(order) * sizeof(f64)),
        .status_bytes = device_align(sizeof(int)),
        .workspace_bytes = device_align(sizeof(cuDoubleComplex) * workspace_count),
    };
}

auto Linalg::zheevd(CuMatrixCF64 matrix, void* scratch, const HeevdScratch& layout) -> void
{
    const auto order = matrix.rows();
    const auto valid_shape = order > 0 and matrix.cols() == order;
    const auto valid_scratch = scratch != nullptr and layout.total() > 0
                               and layout.workspace_bytes >= sizeof(cuDoubleComplex);
    const auto valid = valid_shape and matrix.data() != nullptr and valid_scratch;
    if (not valid)
    {
        assert(false);
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return;
    }
    const auto workspace_offset = layout.eigenvalue_bytes + layout.status_bytes;
    auto* const solver_workspace = byte_offset<cuDoubleComplex>(scratch, workspace_offset);
    const auto workspace_elements = layout.workspace_bytes / sizeof(cuDoubleComplex);
    CUSOLVER_CHECK(cusolverDnSetDeterministicMode(solver_->get(), CUSOLVER_DETERMINISTIC_RESULTS));
    CUDA_CHECK(cudaMemsetAsync(solver_workspace, 0, layout.workspace_bytes, stream_));
    if (err_state() != QNPEPS_OK) return;
    CUSOLVER_CHECK(cusolverDnZheevd(
        solver_->get(),
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER,
        order,
        matrix.data(),
        matrix.ld(),
        layout.eigenvalues(scratch),
        solver_workspace,
        static_cast<int>(workspace_elements),
        layout.status(scratch)
    ));
}
}
