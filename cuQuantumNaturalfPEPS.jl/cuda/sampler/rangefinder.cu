#include "core/cuda_utils.cuh"
#include "linalg/linalg.cuh"
#include "linalg/transfer.cuh"
#include "sampler/kernels.cuh"
#include "sampler/trunc_svd.cuh"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace qnpeps
{
namespace trunc_svd
{
namespace
{
[[nodiscard]] auto resolve_route(const char* variable) -> Route
{
    auto value = static_cast<const char*>(std::getenv(variable));
    if (value == nullptr) return Route::rangefinder;
    if (std::strcmp(value, "rf") == 0) return Route::rangefinder;
    if (std::strcmp(value, "svd") == 0) return Route::svd;
    return Route::invalid;
}

[[nodiscard]] auto reject_route(const char* variable) -> Route
{
    std::array<char, 128> message{};
    std::snprintf(
        message.data(), message.size(), "%s accepts only the values rf and svd", variable
    );
    set_err_at(QNPEPS_ERR_BAD_CONFIG, __FILE__, __LINE__, message.data());
    return Route::invalid;
}

[[nodiscard]] auto copy_left_factors(
    Linalg& la, const cuFloatComplex* left, CuMatrixBatchedCF32 q_out, int dim_batch
) -> bool
{
    const auto rows = static_cast<usize>(q_out.rows());
    const auto rank = static_cast<usize>(q_out.cols());
    const auto copy_width = rows * rank;
    const auto destination_stride = static_cast<usize>(q_out.stride());
    copy_device_2d_async(
        la,
        q_out.data(),
        destination_stride,
        left,
        copy_width,
        copy_width,
        static_cast<usize>(dim_batch)
    );
    return err_state() == QNPEPS_OK;
}

[[nodiscard]] auto primary_failed(Linalg& la, const int* status, int* retry, int dim_batch) -> bool
{
    zero_async(la, retry, 1);
    const auto record_gesvda_failures_args = CuRecordGesvdaFailuresArgs{
        .status = status,
        .retry = retry,
        .dim_batch = dim_batch,
    };
    cu_record_gesvda_failures<<<
        grid_blocks_exact(dim_batch),
        k_threads_per_block,
        0,
        la.stream()>>>(record_gesvda_failures_args);
    CUDA_CHECK(cudaGetLastError());
    int host_retry{};
    download_async(la, &host_retry, retry, 1);
    CUDA_CHECK(cudaStreamSynchronize(la.stream()));
    return err_state() == QNPEPS_OK and host_retry != 0;
}
}

__global__ auto cu_record_gesvda_failures(CuRecordGesvdaFailuresArgs args) -> void
{
    const auto lane = static_cast<int>(global_lane());
    if (lane >= args.dim_batch or args.status[lane] == 0) return;
    atomicExch(args.retry, 1);
}

__global__ auto cu_pad_unconverged_lanes(CuPadUnconvergedLanesArgs args) -> void
{
    const auto lane = static_cast<int>(blockIdx.x);
    if (lane >= args.dim_batch) return;
    if (args.status[lane] == 0) return;

    auto* left_lane = args.left + static_cast<i64>(lane) * args.left_stride;
    const auto column_begin = static_cast<int>(threadIdx.x);
    const auto column_stride = static_cast<int>(blockDim.x);
    for (int column{column_begin}; column < args.rank; column += column_stride)
    {
        auto* column_base = left_lane + static_cast<i64>(column) * args.rows;
        for (auto row = 0; row < args.rows; ++row)
        {
            column_base[row] = cuFloatComplex{0.0f, 0.0f};
        }
    }
}

auto require_sampler_route() -> Route
{
    static const auto route = resolve_route(k_sampler_variable);
    if (route == Route::invalid) return reject_route(k_sampler_variable);
    return route;
}

auto require_dlenv_route() -> Route
{
    static const auto route = resolve_route(k_dlenv_variable);
    if (route == Route::invalid) return reject_route(k_dlenv_variable);
    return route;
}

auto batched_rangefinder_svd(Linalg& la, const RangefinderArgs& args) -> void
{
    const auto input = args.input;
    const auto rows = input.rows();
    const auto cols = input.cols();
    const auto rank = args.rank;
    const auto dim_batch = args.dim_batch;
    const CuMatrixBatchedCF32Const omega_matrix{args.omega, 0, cols, rank};
    const CuMatrixBatchedCF32 sketch_matrix{args.sketch, rows, rank};
    const CuMatrixBatchedCF32 projection_matrix{args.projection, cols, rank};

    la.matmul_batched(input, omega_matrix, sketch_matrix, dim_batch);
    la.matmul_batched_left_adj(input, sketch_matrix, projection_matrix, dim_batch);
    la.matmul_batched(input, projection_matrix, sketch_matrix, dim_batch);

    const auto layout = la.approximate_svd_scratch(rows, rank, dim_batch);
    if (err_state() != QNPEPS_OK) return;
    auto* scratch = la.svd_workspace().grow(layout.total());
    if (not scratch) return;
    la.approximate_svd(sketch_matrix, scratch, layout, args.info, dim_batch);
    if (err_state() != QNPEPS_OK) return;

    auto left = static_cast<const cuFloatComplex*>(layout.left(scratch));
    const auto retry = primary_failed(la, args.info, layout.retry(scratch), dim_batch);
    if (err_state() != QNPEPS_OK) return;
    if (retry)
    {
        const auto jacobi_layout = la.jacobi_svd_scratch(rows, rank, dim_batch);
        if (err_state() != QNPEPS_OK) return;
        auto* jacobi_scratch = la.svd_workspace().grow(jacobi_layout.total());
        if (not jacobi_scratch) return;
        la.jacobi_svd_batched(sketch_matrix, jacobi_scratch, jacobi_layout, dim_batch);
        if (err_state() != QNPEPS_OK) return;
        const auto pad_unconverged_lanes_args = CuPadUnconvergedLanesArgs{
            .left = jacobi_layout.left(jacobi_scratch),
            .status = jacobi_layout.status(jacobi_scratch),
            .left_stride = static_cast<i64>(rows) * rank,
            .rows = rows,
            .rank = rank,
            .dim_batch = dim_batch,
        };
        cu_pad_unconverged_lanes<<<
            static_cast<u32>(dim_batch),
            k_tree_reduce_threads,
            0,
            la.stream()>>>(pad_unconverged_lanes_args);
        CUDA_CHECK(cudaGetLastError());
        left = jacobi_layout.left(jacobi_scratch);
    }
    if (not copy_left_factors(la, left, args.q_out, dim_batch)) return;
    la.matmul_batched_left_adj(args.q_out, input, args.r_out, dim_batch);
}

auto orthonormalize_panel(Linalg& la, CuMatrixCF32 panel) -> void
{
    const auto rows = panel.rows();
    const auto rank = panel.cols();
    const auto valid = panel.data() != nullptr and rows > 0 and rank > 0 and rank <= rows;
    if (not valid)
    {
        assert(false);
        set_err(QNPEPS_ERR_INTERNAL);
        return;
    }
    const auto layout = la.approximate_svd_scratch(rows, rank, 1);
    if (err_state() != QNPEPS_OK) return;
    auto* scratch = la.svd_workspace().grow(layout.total());
    if (not scratch) return;
    const auto panel_elements = static_cast<i64>(rows) * rank;
    const CuMatrixBatchedCF32 panel_batch{panel.data(), panel_elements, rows, rank};
    la.approximate_svd(panel_batch, scratch, layout, layout.status(scratch), 1);
    if (err_state() != QNPEPS_OK) return;

    auto left = static_cast<const cuFloatComplex*>(layout.left(scratch));
    const auto retry = primary_failed(la, layout.status(scratch), layout.retry(scratch), 1);
    if (err_state() != QNPEPS_OK) return;
    if (retry)
    {
        const auto jacobi_layout = la.jacobi_svd_scratch(rows, rank, 1);
        if (err_state() != QNPEPS_OK) return;
        auto* jacobi_scratch = la.svd_workspace().grow(jacobi_layout.total());
        if (not jacobi_scratch) return;
        la.jacobi_svd_batched(panel_batch, jacobi_scratch, jacobi_layout, 1);
        if (err_state() != QNPEPS_OK) return;
        const auto pad_unconverged_lanes_args = CuPadUnconvergedLanesArgs{
            .left = jacobi_layout.left(jacobi_scratch),
            .status = jacobi_layout.status(jacobi_scratch),
            .left_stride = panel_elements,
            .rows = rows,
            .rank = rank,
            .dim_batch = 1,
        };
        cu_pad_unconverged_lanes<<<1, k_tree_reduce_threads, 0, la.stream()>>>(
            pad_unconverged_lanes_args
        );
        CUDA_CHECK(cudaGetLastError());
        left = jacobi_layout.left(jacobi_scratch);
    }
    copy_device_async(la, panel.data(), left, static_cast<usize>(panel_elements));
}
}

auto batched_rangefinder(Linalg& la, const RangefinderArgs& args) -> void
{
    const auto route = trunc_svd::require_sampler_route();
    if (route == trunc_svd::Route::invalid) return;
    if (route == trunc_svd::Route::svd)
    {
        trunc_svd::batched_rangefinder_svd(la, args);
        return;
    }

    const auto input = args.input;
    const auto rows = input.rows();
    const auto cols = input.cols();
    const auto rank = args.rank;
    const auto dim_batch = args.dim_batch;
    const CuMatrixBatchedCF32Const omega_matrix{args.omega, 0, cols, rank};
    const CuMatrixBatchedCF32 sketch_matrix{args.sketch, rows, rank};
    const CuMatrixBatchedCF32 projection_matrix{args.projection, cols, rank};
    const CuMatrixBatchedCF32 gram_matrix{args.gram, rank, rank};

    la.matmul_batched(input, omega_matrix, sketch_matrix, dim_batch);
    la.matmul_batched_left_adj(input, sketch_matrix, projection_matrix, dim_batch);
    la.matmul_batched(input, projection_matrix, sketch_matrix, dim_batch);

    for (auto pass = 0; pass < 2; ++pass)
    {
        la.matmul_batched_left_adj(sketch_matrix, sketch_matrix, gram_matrix, dim_batch);
        const auto chol_shift_roundoff_args = CuCholShiftRoundoffArgs{
            .gram = args.gram.p,
            .k = rank,
            .stride = args.gram.stride,
            .rows = rows,
            .dim_batch = dim_batch,
            .conservative_bound = false,
        };
        cu_chol_shift_roundoff<<<
            grid_blocks_exact(dim_batch),
            k_threads_per_block,
            0,
            la.stream()>>>(chol_shift_roundoff_args);
        la.cholesky_lower_batched(rank, args.gram_ptrs, rank, args.info, dim_batch);
        if (args.fail_flag)
        {
            cu_any_chol_failed<<<
                grid_blocks_exact(dim_batch),
                k_threads_per_block,
                0,
                la.stream()>>>(args.info, dim_batch, args.fail_flag);
        }
        la.triangular_solve_batched(
            args.gram_ptrs,
            rank,
            args.sketch_ptrs,
            rows,
            rows,
            rank,
            dim_batch,
            {.op = BlasOp::conj_trans}
        );
    }

    const auto destination = args.q_out.data();
    const auto destination_stride = static_cast<usize>(args.q_out.stride());
    const auto* source = args.sketch.p;
    const auto source_stride = static_cast<usize>(args.sketch.stride);
    const auto matrix_elements = static_cast<usize>(rows) * static_cast<usize>(rank);
    const auto copy_height = static_cast<usize>(dim_batch);
    copy_device_2d_async(
        la, destination, destination_stride, source, source_stride, matrix_elements, copy_height
    );
    la.matmul_batched_left_adj(sketch_matrix, input, args.r_out, dim_batch);
}
}
