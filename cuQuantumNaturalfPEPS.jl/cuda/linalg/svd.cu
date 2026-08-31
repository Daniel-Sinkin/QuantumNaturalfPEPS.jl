#include "linalg/svd.cuh"

#include "core/cuda_utils.cuh"
#include "core/types.cuh"
#include "linalg/scratch.cuh"

#include <algorithm>
#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <map>
#include <tuple>

namespace qnpeps
{

auto Linalg::approximate_svd_scratch(int rows, int rank, int batch_size) -> GesvdaScratch
{
    if (rows <= 0 or rank <= 0 or rank > rows or batch_size <= 0)
    {
        assert(false);
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return {};
    }
    const auto left_stride = static_cast<i64>(rows) * rank;
    const auto right_stride = static_cast<i64>(rank) * rank;
    const auto batched_extents_fit = strided_batch_fits_int(left_stride, batch_size)
                                     and strided_batch_fits_int(right_stride, batch_size)
                                     and strided_batch_fits_int(rank, batch_size);
    if (not batched_extents_fit)
    {
        qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
        return {};
    }
    const auto rows_count = static_cast<usize>(rows);
    const auto rank_count = static_cast<usize>(rank);
    const auto batch_count = static_cast<usize>(batch_size);
    GesvdaScratch layout{};
    layout.left_bytes =
        device_align(sizeof(cuFloatComplex) * rows_count * rank_count * batch_count);
    layout.right_bytes =
        device_align(sizeof(cuFloatComplex) * rank_count * rank_count * batch_count);
    layout.singular_bytes = device_align(sizeof(f32) * rank_count * batch_count);
    layout.status_bytes = device_align(sizeof(int) * batch_count);
    layout.retry_bytes = device_align(sizeof(int));

    const auto key = std::make_tuple(rows, rank, batch_size);
    if (const auto known = gesvda_workspace_counts_.find(key);
        known != gesvda_workspace_counts_.end())
    {
        layout.workspace_count = known->second;
        layout.workspace_bytes =
            device_align(sizeof(cuFloatComplex) * static_cast<usize>(known->second));
        return layout;
    }

    auto* probe = svd_workspace_.grow(layout.total());
    if (not probe) return {};
    int workspace_size{};
    CUSOLVER_CHECK(cusolverDnCgesvdaStridedBatched_bufferSize(
        solver_->get(),
        CUSOLVER_EIG_MODE_VECTOR,
        rank,
        rows,
        rank,
        layout.left(probe),
        rows,
        left_stride,
        layout.singular(probe),
        rank,
        layout.left(probe),
        rows,
        left_stride,
        layout.right(probe),
        rank,
        right_stride,
        &workspace_size,
        batch_size
    ));
    if (err_state() != QNPEPS_OK) return {};
    layout.workspace_count = std::max(workspace_size, 1);
    layout.workspace_bytes =
        device_align(sizeof(cuFloatComplex) * static_cast<usize>(layout.workspace_count));
    gesvda_workspace_counts_.emplace(key, layout.workspace_count);
    return layout;
}

auto Linalg::approximate_svd(
    CuMatrixBatchedCF32Const input,
    void* scratch,
    const GesvdaScratch& layout,
    int* status,
    int batch_size
) -> void
{
    const auto rows = input.rows();
    const auto rank = input.cols();
    const auto valid_shape = rows > 0 and rank > 0 and rank <= rows and batch_size > 0;
    const auto valid_stride = input.stride() >= static_cast<i64>(rows) * rank;
    const auto valid_scratch =
        scratch != nullptr and layout.workspace_count > 0 and layout.total() > 0;
    const auto valid = valid_shape and valid_stride and input.data() and status and valid_scratch;
    if (not valid)
    {
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return;
    }
    if (not strided_batch_fits_int(input.stride(), batch_size))
    {
        qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
        return;
    }
    CUDA_CHECK(cudaMemsetAsync(status, 0, sizeof(int) * static_cast<usize>(batch_size), stream_));
    CUSOLVER_CHECK(cusolverDnCgesvdaStridedBatched(
        solver_->get(),
        CUSOLVER_EIG_MODE_VECTOR,
        rank,
        rows,
        rank,
        input.data(),
        input.ld(),
        input.stride(),
        layout.singular(scratch),
        rank,
        layout.left(scratch),
        rows,
        static_cast<i64>(rows) * rank,
        layout.right(scratch),
        rank,
        static_cast<i64>(rank) * rank,
        layout.workspace(scratch),
        layout.workspace_count,
        status,
        nullptr,
        batch_size
    ));
}

auto Linalg::jacobi_svd_scratch(int rows, int rank, int batch_size) -> GesvdjScratch
{
    if (rows <= 0 or rank <= 0 or rank > rows or batch_size <= 0)
    {
        assert(false);
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return {};
    }
    const auto left_stride = static_cast<i64>(rows) * rank;
    const auto right_stride = static_cast<i64>(rank) * rank;
    const auto batched_extents_fit = strided_batch_fits_int(left_stride, batch_size)
                                     and strided_batch_fits_int(right_stride, batch_size)
                                     and strided_batch_fits_int(rank, batch_size);
    if (not batched_extents_fit)
    {
        qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
        return {};
    }
    if (not gesvdj_parameters_)
    {
        CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdj_parameters_));
        if (err_state() != QNPEPS_OK) return {};
        CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(gesvdj_parameters_, k_gesvdj_tolerance));
        CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(gesvdj_parameters_, k_gesvdj_max_sweeps));
        CUSOLVER_CHECK(cusolverDnXgesvdjSetSortEig(gesvdj_parameters_, 1));
        if (err_state() != QNPEPS_OK) return {};
    }

    const auto rows_count = static_cast<usize>(rows);
    const auto rank_count = static_cast<usize>(rank);
    const auto batch_count = static_cast<usize>(batch_size);
    GesvdjScratch layout{};
    layout.left_bytes =
        device_align(sizeof(cuFloatComplex) * rows_count * rank_count * batch_count);
    layout.right_bytes =
        device_align(sizeof(cuFloatComplex) * rank_count * rank_count * batch_count);
    layout.singular_bytes = device_align(sizeof(f32) * rank_count * batch_count);
    layout.status_bytes = device_align(sizeof(int) * batch_count);

    const auto key = std::make_tuple(rows, rank, batch_size);
    if (const auto known = gesvdj_workspace_counts_.find(key);
        known != gesvdj_workspace_counts_.end())
    {
        layout.workspace_count = known->second;
        layout.workspace_bytes =
            device_align(sizeof(cuFloatComplex) * static_cast<usize>(known->second));
        return layout;
    }

    auto* probe = svd_workspace_.grow(layout.total());
    if (not probe) return {};
    int workspace_size{};
    CUSOLVER_CHECK(cusolverDnCgesvdj_bufferSize(
        solver_->get(),
        CUSOLVER_EIG_MODE_VECTOR,
        1,
        rows,
        rank,
        layout.left(probe),
        rows,
        layout.singular(probe),
        layout.left(probe),
        rows,
        layout.right(probe),
        rank,
        &workspace_size,
        gesvdj_parameters_
    ));
    if (err_state() != QNPEPS_OK) return {};
    layout.workspace_count = std::max(workspace_size, 1);
    layout.workspace_bytes =
        device_align(sizeof(cuFloatComplex) * static_cast<usize>(layout.workspace_count));
    gesvdj_workspace_counts_.emplace(key, layout.workspace_count);
    return layout;
}

auto Linalg::jacobi_svd_batched(
    CuMatrixBatchedCF32 input, void* scratch, const GesvdjScratch& layout, int batch_size
) -> void
{
    const auto rows = input.rows();
    const auto rank = input.cols();
    const auto valid_shape = rows > 0 and rank > 0 and rank <= rows and batch_size > 0;
    const auto valid_stride = input.stride() >= static_cast<i64>(rows) * rank;
    const auto valid_scratch =
        scratch != nullptr and layout.workspace_count > 0 and layout.total() > 0;
    const auto valid =
        valid_shape and valid_stride and input.data() and valid_scratch and gesvdj_parameters_;
    if (not valid)
    {
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return;
    }
    if (not strided_batch_fits_int(input.stride(), batch_size))
    {
        qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
        return;
    }
    CUDA_CHECK(cudaMemsetAsync(
        layout.status(scratch), 0, sizeof(int) * static_cast<usize>(batch_size), stream_
    ));
    const auto left_stride = static_cast<i64>(rows) * rank;
    const auto right_stride = static_cast<i64>(rank) * rank;
    for (auto lane = 0; lane < batch_size; ++lane)
    {
        CUSOLVER_CHECK(cusolverDnCgesvdj(
            solver_->get(),
            CUSOLVER_EIG_MODE_VECTOR,
            1,
            rows,
            rank,
            input.data() + static_cast<i64>(lane) * input.stride(),
            rows,
            layout.singular(scratch) + static_cast<i64>(lane) * rank,
            layout.left(scratch) + static_cast<i64>(lane) * left_stride,
            rows,
            layout.right(scratch) + static_cast<i64>(lane) * right_stride,
            rank,
            layout.workspace(scratch),
            layout.workspace_count,
            layout.status(scratch) + lane,
            gesvdj_parameters_
        ));
        if (err_state() != QNPEPS_OK) return;
    }
}

auto Linalg::svd_workspace() -> DeviceBuffer&
{
    return svd_workspace_;
}
}
