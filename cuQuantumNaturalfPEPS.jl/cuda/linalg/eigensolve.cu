#include "linalg/eigensolve.cuh"

#include "core/cuda_utils.cuh"
#include "core/defer.cuh"
#include "core/predicates.cuh"
#include "core/types.cuh"
#include "linalg/scratch.cuh"

#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

namespace qnpeps
{

namespace
{
template <typename Body>
auto with_deterministic_results(cusolverDnHandle_t solver, Body&& body) -> void
{
    cusolverDeterministicMode_t deterministic{};
    check(cusolverDnGetDeterministicMode(solver, &deterministic));
    check(cusolverDnSetDeterministicMode(solver, CUSOLVER_DETERMINISTIC_RESULTS));
    auto finished = false;
    const Defer restore_after_failure{
        [&]
        {
            if (not finished) unchecked(cusolverDnSetDeterministicMode(solver, deterministic));
        }
    };
    body();
    finished = true;
    check(cusolverDnSetDeterministicMode(solver, deterministic));
}

[[nodiscard]] auto is_packed_batch(CuMatrixBatchedCF64 matrices, int batch_size) noexcept -> bool
{
    const auto order = matrices.rows();
    const auto is_square = all_positive(order, batch_size) and matrices.cols() == order;
    return is_square and matrices.data() and matrices.stride() == static_cast<i64>(order) * order;
}
}

auto Linalg::diagonalize_workspace_count(int order) -> int
{
    require(
        order >= 1,
        Status::bad_config,
        "eigensolver workspace query needs a positive order, got {}",
        order
    );
    int workspace_count{};
    with_deterministic_results(
        solver_->get(),
        [&]
        {
            check(
                cusolverDnZheevd_bufferSize(
                    solver_->get(),
                    CUSOLVER_EIG_MODE_VECTOR,
                    CUBLAS_FILL_MODE_LOWER,
                    order,
                    nullptr,
                    order,
                    nullptr,
                    &workspace_count
                ),
                "zheevd workspace query"
            );
        }
    );
    return workspace_count;
}

auto Linalg::diagonalize_workspace_count(CuMatrixCF64 matrix, f64* eigenvalues) -> int
{
    const auto order = matrix.rows();
    const auto valid =
        all_positive(order) and matrix.cols() == order and matrix.data() and eigenvalues;
    require(valid, "eigensolver workspace query rejects matrix {}", matrix);
    int workspace_count{};
    with_deterministic_results(
        solver_->get(),
        [&]
        {
            check(
                cusolverDnZheevd_bufferSize(
                    solver_->get(),
                    CUSOLVER_EIG_MODE_VECTOR,
                    CUBLAS_FILL_MODE_LOWER,
                    order,
                    matrix.data(),
                    matrix.ld(),
                    eigenvalues,
                    &workspace_count
                ),
                "zheevd workspace query"
            );
        }
    );
    return workspace_count;
}

auto Linalg::diagonalize(CuMatrixCF64 matrix, const DiagonalizeConfig& config) -> void
{
    const auto order = matrix.rows();
    const auto valid_shape = all_positive(order) and matrix.cols() == order;
    const auto valid_workspace = config.workspace and all_positive(config.workspace_count);
    const auto valid =
        valid_shape and matrix.data() and config.eigenvalues and valid_workspace and config.info;
    require(
        valid,
        "diagonalize rejects matrix {} with workspace count {}",
        matrix,
        config.workspace_count
    );
    check(cusolverDnSetDeterministicMode(solver_->get(), CUSOLVER_DETERMINISTIC_RESULTS));
    check(cudaMemsetAsync(
        config.workspace,
        0,
        sizeof(cuDoubleComplex) * static_cast<usize>(config.workspace_count),
        stream_
    ));
    check(
        cusolverDnZheevd(
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
        ),
        "zheevd"
    );
}

auto Linalg::batched_eigen_parameters() -> cusolverDnParams_t
{
    if (not batched_eigen_parameters_) check(cusolverDnCreateParams(&batched_eigen_parameters_));
    return batched_eigen_parameters_;
}

auto Linalg::diagonalize_batched_workspace_bytes(
    CuMatrixBatchedCF64 matrices, f64* eigenvalues, int batch_size
) -> usize
{
    require(
        is_packed_batch(matrices, batch_size) and eigenvalues,
        "batched eigensolve workspace query rejects matrices {} over {} lanes",
        matrices,
        batch_size
    );
    const auto order = static_cast<i64>(matrices.rows());
    usize device_bytes{};
    usize host_bytes{};
    check(
        cusolverDnXsyevBatched_bufferSize(
            solver_->get(),
            batched_eigen_parameters(),
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER,
            order,
            CUDA_C_64F,
            matrices.data(),
            order,
            CUDA_R_64F,
            eigenvalues,
            CUDA_C_64F,
            &device_bytes,
            &host_bytes,
            static_cast<i64>(batch_size)
        ),
        "xsyev batched workspace query"
    );
    require(
        host_bytes == 0,
        "batched eigensolve of order {} over {} lanes asks for {} host workspace bytes",
        order,
        batch_size,
        host_bytes
    );
    return device_bytes;
}

auto Linalg::diagonalize_batched(
    CuMatrixBatchedCF64 matrices, const DiagonalizeBatchedConfig& config, int batch_size
) -> void
{
    const auto has_buffers = config.eigenvalues and config.workspace and config.info;
    require(
        is_packed_batch(matrices, batch_size) and has_buffers,
        "batched diagonalize rejects matrices {} over {} lanes with {} workspace bytes",
        matrices,
        batch_size,
        config.workspace_bytes
    );
    const auto order = static_cast<i64>(matrices.rows());
    check(cusolverDnSetDeterministicMode(solver_->get(), CUSOLVER_DETERMINISTIC_RESULTS));
    check(
        cusolverDnXsyevBatched(
            solver_->get(),
            batched_eigen_parameters(),
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER,
            order,
            CUDA_C_64F,
            matrices.data(),
            order,
            CUDA_R_64F,
            config.eigenvalues,
            CUDA_C_64F,
            config.workspace,
            config.workspace_bytes,
            nullptr,
            0,
            config.info,
            static_cast<i64>(batch_size)
        ),
        "xsyev batched"
    );
}

auto Linalg::heevd_scratch(int order) -> HeevdScratch
{
    require(order > 0, "eigensolver scratch needs a positive order, got {}", order);
    int heevd_size{};
    check(
        cusolverDnZheevd_bufferSize(
            solver_->get(),
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER,
            order,
            nullptr,
            order,
            nullptr,
            &heevd_size
        ),
        "zheevd workspace query"
    );
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
    require(
        valid, "zheevd rejects matrix {} with {} workspace bytes", matrix, layout.workspace_bytes
    );
    const auto workspace_offset = layout.eigenvalue_bytes + layout.status_bytes;
    auto* const solver_workspace = byte_offset<cuDoubleComplex>(scratch, workspace_offset);
    const auto workspace_elements = layout.workspace_bytes / sizeof(cuDoubleComplex);
    check(cusolverDnSetDeterministicMode(solver_->get(), CUSOLVER_DETERMINISTIC_RESULTS));
    check(cudaMemsetAsync(solver_workspace, 0, layout.workspace_bytes, stream_));
    check(
        cusolverDnZheevd(
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
        ),
        "zheevd"
    );
}
}
