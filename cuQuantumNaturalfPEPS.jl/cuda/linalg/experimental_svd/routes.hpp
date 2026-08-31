#pragma once

#include "../eo.cuh"

#include <algorithm>
#include <cstdint>

#include "workspace.hpp"
#include "kernels.hpp"

namespace qn_eloc::e0191
{

#line 317 "cuda/linalg/experimental_svd.cuh"

struct GesvdaQueryArgs
{
    Linalg& linalg;
    const cf* input;
    i64 input_stride;
    int rows;
    int cols;
    int rank;
    int dim_batch;
    cf* left;
    cf* right;
    f32* singular;
};

inline auto query_gesvda_solver_count(const GesvdaQueryArgs& args) -> int
{
    auto handle = cusolverDnHandle_t{args.linalg.cusolver()};
    if (not handle)
    {
        qn::set_err(QNPEPS_ELOC_ERR_INTERNAL);
        return 0;
    }
    int solver_count{};
    const auto query_status = cusolverStatus_t{cusolverDnCgesvdaStridedBatched_bufferSize(
        handle,
        CUSOLVER_EIG_MODE_VECTOR,
        args.rank,
        args.rows,
        args.cols,
        reinterpret_cast<const cuComplex*>(args.input),
        args.rows,
        args.input_stride,
        args.singular,
        args.rank,
        reinterpret_cast<const cuComplex*>(args.left),
        args.rows,
        static_cast<i64>(args.rows) * args.rank,
        reinterpret_cast<const cuComplex*>(args.right),
        args.cols,
        static_cast<i64>(args.cols) * args.rank,
        &solver_count,
        args.dim_batch
    )};
    if (query_status != CUSOLVER_STATUS_SUCCESS)
    {
        qn::set_backend_err(
            QNPEPS_ELOC_ERR_CUDA, "cusolver", static_cast<int>(query_status), __FILE__, __LINE__
        );
        return 0;
    }
    return std::max(solver_count, 1);
}

inline auto carve_rank_workspace(
    GesvdaWorkspace& work, WorkspaceCursor& cursor, int k, int dim_batch
) -> void
{
    const auto batch = usize{static_cast<usize>(dim_batch)};
    work.total_weight = cursor.take<f64>(batch);
    work.component_weight = cursor.take<f64>(static_cast<usize>(k) * batch);
    work.effective_rank = cursor.take<int>(batch);
}

inline auto carve_direct_workspace(
    Linalg& la,
    GesvdaWorkspace& work,
    WorkspaceCursor& cursor,
    int rows,
    int cols,
    int k,
    int storage_rank,
    int dim_batch,
    int solver_count = 0
) -> void
{
    const auto adjoint = bool{rows < cols};
    const auto svd_rows = int{adjoint ? cols : rows};
    const auto svd_cols = int{adjoint ? rows : cols};
    const auto batch = usize{static_cast<usize>(dim_batch)};
    work.direct_input = cursor.take<cf>(static_cast<usize>(svd_rows) * svd_cols * batch);
    work.left = cursor.take<cf>(static_cast<usize>(svd_rows) * storage_rank * batch);
    work.right = cursor.take<cf>(static_cast<usize>(svd_cols) * storage_rank * batch);
    work.q_candidate = cursor.take<cf>(static_cast<usize>(rows) * k * batch);
    work.r_candidate = cursor.take<cf>(static_cast<usize>(k) * cols * batch);
    work.singular = cursor.take<f32>(static_cast<usize>(storage_rank) * batch);
    work.info = cursor.take<int>(batch);
    carve_rank_workspace(work, cursor, k, dim_batch);
    if (solver_count < 1)
    {
        auto args = GesvdaQueryArgs{
            .linalg = la,
            .input = work.direct_input,
            .input_stride = static_cast<i64>(svd_rows) * svd_cols,
            .rows = svd_rows,
            .cols = svd_cols,
            .rank = storage_rank,
            .dim_batch = dim_batch,
            .left = work.left,
            .right = work.right,
            .singular = work.singular,
        };
        solver_count = query_gesvda_solver_count(args);
        if (storage_rank != k)
        {
            args.rank = k;
            solver_count = std::max(solver_count, query_gesvda_solver_count(args));
        }
    }
    work.solver_scratch_count = static_cast<usize>(solver_count);
    work.solver_scratch = cursor.take<cf>(work.solver_scratch_count);
}

inline auto carve_qb_workspace(
    Linalg& la,
    GesvdaWorkspace& work,
    WorkspaceCursor& cursor,
    int rows,
    int cols,
    int k,
    int width,
    int storage_rank,
    int dim_batch,
    int solver_count = 0
) -> void
{
    const auto batch = usize{static_cast<usize>(dim_batch)};
    work.sketch = cursor.take<cf>(static_cast<usize>(rows) * width * batch);
    work.projection = cursor.take<cf>(static_cast<usize>(cols) * width * batch);
    work.left = cursor.take<cf>(static_cast<usize>(cols) * storage_rank * batch);
    work.right = cursor.take<cf>(static_cast<usize>(width) * storage_rank * batch);
    work.q_candidate = cursor.take<cf>(static_cast<usize>(rows) * k * batch);
    work.r_candidate = cursor.take<cf>(static_cast<usize>(k) * cols * batch);
    work.singular = cursor.take<f32>(static_cast<usize>(storage_rank) * batch);
    work.info = cursor.take<int>(batch);
    work.qr_scratch = cursor.take<char>(qr_scratch_bytes(la, rows, width));
    if (solver_count < 1)
    {
        auto args = GesvdaQueryArgs{
            .linalg = la,
            .input = work.projection,
            .input_stride = static_cast<i64>(cols) * width,
            .rows = cols,
            .cols = width,
            .rank = storage_rank,
            .dim_batch = dim_batch,
            .left = work.left,
            .right = work.right,
            .singular = work.singular,
        };
        solver_count = query_gesvda_solver_count(args);
        if (storage_rank != k)
        {
            args.rank = k;
            solver_count = std::max(solver_count, query_gesvda_solver_count(args));
        }
    }
    work.solver_scratch_count = static_cast<usize>(solver_count);
    work.solver_scratch = cursor.take<cf>(work.solver_scratch_count);
}

inline auto carve_rangefinder_workspace(
    Linalg& la,
    WorkspaceCursor& cursor,
    int max_dim,
    int k,
    int dim_batch,
    char*& scratch,
    usize& scratch_bytes
) -> void
{
    scratch = cursor.take<char>(0);
    GesvdaWorkspace work{};
    const auto route = RangefinderRoute{rangefinder_route_from_env()};
    if (route == RangefinderRoute::gesvda)
    {
        const auto storage_rank = int{condition_spectrum_rank(max_dim, k)};
        carve_direct_workspace(la, work, cursor, max_dim, max_dim, k, storage_rank, dim_batch);
    }
    else if (route == RangefinderRoute::qb_svd)
    {
        const auto width = int{rangefinder_sketch_width(max_dim, max_dim, k)};
        const auto storage_rank = int{condition_spectrum_rank(width, k)};
        carve_qb_workspace(la, work, cursor, max_dim, max_dim, k, width, storage_rank, dim_batch);
    }
    if (qn::err_state() != QNPEPS_ELOC_OK) return;
    auto* end = cursor.take<char>(0);
    if (qn::err_state() != QNPEPS_ELOC_OK) return;
    scratch_bytes = static_cast<usize>(end - scratch);
}

inline auto run_gesvda(
    Linalg& la,
    GesvdaWorkspace& work,
    const cf* input,
    i64 input_stride,
    int rows,
    int cols,
    int k,
    int dim_batch
) -> bool
{
    const auto solver_count = int{query_gesvda_solver_count({
        .linalg = la,
        .input = input,
        .input_stride = input_stride,
        .rows = rows,
        .cols = cols,
        .rank = k,
        .dim_batch = dim_batch,
        .left = work.left,
        .right = work.right,
        .singular = work.singular,
    })};
    if (solver_count < 1 or static_cast<usize>(solver_count) > work.solver_scratch_count)
    {
        if (qn::err_state() == QNPEPS_ELOC_OK) qn::set_err(QNPEPS_ELOC_ERR_OOM);
        return false;
    }
    CUDA_CHECK(
        cudaMemsetAsync(work.info, 0, sizeof(int) * static_cast<usize>(dim_batch), la.stream())
    );
    const auto status = cusolverStatus_t{cusolverDnCgesvdaStridedBatched(
        la.cusolver(),
        CUSOLVER_EIG_MODE_VECTOR,
        k,
        rows,
        cols,
        reinterpret_cast<const cuComplex*>(input),
        rows,
        input_stride,
        work.singular,
        k,
        reinterpret_cast<cuComplex*>(work.left),
        rows,
        static_cast<i64>(rows) * k,
        reinterpret_cast<cuComplex*>(work.right),
        cols,
        static_cast<i64>(cols) * k,
        reinterpret_cast<cuComplex*>(work.solver_scratch),
        solver_count,
        work.info,
        nullptr,
        dim_batch
    )};
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        qn::set_backend_err(
            QNPEPS_ELOC_ERR_CUDA, "cusolver", static_cast<int>(status), __FILE__, __LINE__
        );
        return false;
    }
    return true;
}

inline auto apply_component_mask(
    Linalg& la,
    GesvdaWorkspace& work,
    MatrixBatch panel,
    MatrixBatch q,
    MatrixBatch r,
    const f32* singular,
    f64 cutoff,
    int dim_batch
) -> bool
{
    const auto k = int{q.cols()};
    const auto panel_total_weight_args = CuPanelTotalWeightArgs{
        .panel = panel.data(),
        .panel_stride = panel.stride(),
        .elements = panel.rows() * panel.cols(),
        .total_weight = work.total_weight,
        .dim_batch = dim_batch,
    };
    cu_panel_total_weight<<<dim_batch, 256, 0, la.stream()>>>(panel_total_weight_args);
    const auto singular_count = int{k * dim_batch};
    const auto singular_component_weight_args = CuSingularComponentWeightArgs{
        .singular = singular,
        .k = k,
        .component_weight = work.component_weight,
        .dim_batch = dim_batch,
    };
    cu_singular_component_weight<<<(singular_count + 255) / 256, 256, 0, la.stream()>>>(
        singular_component_weight_args
    );
    const auto select_effective_rank_args = CuSelectEffectiveRankArgs{
        .total_weight = work.total_weight,
        .component_weight = work.component_weight,
        .effective_rank = work.effective_rank,
        .k = k,
        .cutoff = cutoff,
        .dim_batch = dim_batch,
    };
    cu_select_effective_rank<<<(dim_batch + 255) / 256, 256, 0, la.stream()>>>(
        select_effective_rank_args
    );
    const auto total =
        i64{static_cast<i64>(dim_batch)
            * (static_cast<i64>(q.rows()) * k + static_cast<i64>(k) * r.cols())};
    const auto blocks =
        int{static_cast<int>(std::max<i64>(1, std::min<i64>(4096, (total + 255) / 256)))};
    const auto mask_factor_tails_args = CuMaskFactorTailsArgs{
        .q = q.data(),
        .q_stride = q.stride(),
        .r = r.data(),
        .r_stride = r.stride(),
        .effective_rank = work.effective_rank,
        .rows = q.rows(),
        .cols = r.cols(),
        .k = k,
        .dim_batch = dim_batch,
    };
    cu_mask_factor_tails<<<blocks, 256, 0, la.stream()>>>(mask_factor_tails_args);
    CUDA_CHECK(cudaGetLastError());
    return qn::err_state() == QNPEPS_ELOC_OK;
}

inline auto run_direct_gesvda_cutoff(
    Linalg& la,
    MatrixBatch panel,
    int k,
    MatrixBatch q_out,
    MatrixBatch r_out,
    int dim_batch,
    int* fail_flag,
    int* failure_log,
    const int* fallback_info,
    void* scratch,
    usize scratch_bytes
) -> bool
{
    const auto rows = int{panel.rows()};
    const auto cols = int{panel.cols()};
    const auto adjoint = bool{rows < cols};
    const auto svd_rows = int{adjoint ? cols : rows};
    const auto svd_cols = int{adjoint ? rows : cols};
    if (k < 1 or k > svd_cols)
    {
        qn::set_err(QNPEPS_ELOC_ERR_BAD_CONFIG);
        return false;
    }
    GesvdaWorkspace work{};
    auto cursor = WorkspaceCursor::carve(scratch, scratch_bytes);
    const auto storage_rank = int{condition_spectrum_rank(svd_cols, k)};
    carve_direct_workspace(la, work, cursor, rows, cols, k, storage_rank, dim_batch);
    if (qn::err_state() != QNPEPS_ELOC_OK) return false;
    const auto batch = usize{static_cast<usize>(dim_batch)};
    const auto input_stride = usize{static_cast<usize>(svd_rows) * svd_cols};
    const auto q_stride = usize{static_cast<usize>(rows) * k};
    const auto r_stride = usize{static_cast<usize>(k) * cols};
    auto svd_input{panel.data()};
    auto svd_input_stride = i64{panel.stride()};
    if (adjoint)
    {
        const auto total = i64{static_cast<i64>(input_stride) * dim_batch};
        const auto blocks =
            int{static_cast<int>(std::max<i64>(1, std::min<i64>(4096, (total + 255) / 256)))};
        const auto adjoint_panels_args = CuAdjointPanelsArgs{
            .input = panel.data(),
            .input_stride = panel.stride(),
            .output = work.direct_input,
            .output_stride = static_cast<i64>(input_stride),
            .rows = rows,
            .cols = cols,
            .dim_batch = dim_batch,
        };
        cu_adjoint_panels<<<blocks, 256, 0, la.stream()>>>(adjoint_panels_args);
        svd_input = work.direct_input;
        svd_input_stride = static_cast<i64>(input_stride);
    }
    if (not run_gesvda(la, work, svd_input, svd_input_stride, svd_rows, svd_cols, k, dim_batch))
        return false;
    const auto record_gesvda_info_args = CuRecordGesvdaInfoArgs{
        .info = work.info,
        .failure_log = failure_log,
        .fail_flag = fail_flag,
        .dim_batch = dim_batch,
    };
    cu_record_gesvda_info<<<(dim_batch + 255) / 256, 256, 0, la.stream()>>>(
        record_gesvda_info_args
    );
    auto q_source{adjoint ? work.right : work.left};
    CUDA_CHECK(cudaMemcpyAsync(
        work.q_candidate,
        q_source,
        sizeof(cf) * q_stride * batch,
        cudaMemcpyDeviceToDevice,
        la.stream()
    ));
    auto q_candidate = MatrixBatch{work.q_candidate, static_cast<i64>(q_stride), rows, k};
    auto r_candidate = MatrixBatch{work.r_candidate, static_cast<i64>(r_stride), k, cols};
    la.matmul_batched(
        CuMatrixBatchedCF32Const{
            reinterpret_cast<const cuFloatComplex*>(q_candidate.data()),
            q_candidate.stride(),
            q_candidate.rows(),
            q_candidate.cols()
        },
        CuMatrixBatchedCF32Const{
            reinterpret_cast<const cuFloatComplex*>(panel.data()),
            panel.stride(),
            panel.rows(),
            panel.cols()
        },
        CuMatrixBatchedCF32{
            reinterpret_cast<cuFloatComplex*>(r_candidate.data()),
            r_candidate.stride(),
            r_candidate.rows(),
            r_candidate.cols()
        },
        dim_batch,
        {.op_a = BlasOp::conj_trans, .op_b = BlasOp::none}
    );
    if (not apply_component_mask(
            la,
            work,
            panel,
            q_candidate,
            r_candidate,
            work.singular,
            rangefinder_gesvda_cutoff_from_env(),
            dim_batch
        ))
        return false;
    const auto total =
        i64{static_cast<i64>(dim_batch)
            * (static_cast<i64>(rows) * k + static_cast<i64>(k) * cols)};
    const auto blocks =
        int{static_cast<int>(std::max<i64>(1, std::min<i64>(4096, (total + 255) / 256)))};
    const auto copy_factor_outputs_args = CuCopyFactorOutputsArgs{
        .q_candidate = work.q_candidate,
        .q_candidate_stride = static_cast<i64>(q_stride),
        .r_candidate = work.r_candidate,
        .r_candidate_stride = static_cast<i64>(r_stride),
        .r_out = r_out.data(),
        .r_stride = r_out.stride(),
        .q_out = q_out.data(),
        .q_stride = q_out.stride(),
        .info = work.info,
        .replay_mask = fallback_info ? failure_log : nullptr,
        .rows = rows,
        .cols = cols,
        .k = k,
        .dim_batch = dim_batch,
    };
    cu_copy_factor_outputs<<<blocks, 256, 0, la.stream()>>>(copy_factor_outputs_args);
    CUDA_CHECK(cudaGetLastError());
    if (storage_rank > k
        and not run_gesvda(
            la, work, svd_input, svd_input_stride, svd_rows, svd_cols, storage_rank, dim_batch
        ))
        return false;
    if (storage_rank > k)
    {
        const auto record_gesvda_info_args = CuRecordGesvdaInfoArgs{
            .info = work.info,
            .failure_log = failure_log,
            .fail_flag = fail_flag,
            .dim_batch = dim_batch,
        };
        cu_record_gesvda_info<<<(dim_batch + 255) / 256, 256, 0, la.stream()>>>(
            record_gesvda_info_args
        );
        CUDA_CHECK(cudaGetLastError());
    }
    return true;
}

inline auto run_qb_gesvda(
    Linalg& la,
    MatrixBatch panel,
    int k,
    const cf* omega,
    MatrixBatch q_out,
    MatrixBatch r_out,
    int dim_batch,
    int* fail_flag,
    int* failure_log,
    const int* fallback_info,
    void* scratch,
    usize scratch_bytes
) -> bool
{
    const auto rows = int{panel.rows()};
    const auto cols = int{panel.cols()};
    const auto width = int{rangefinder_sketch_width(rows, cols, k)};
    if (rows < width or cols < width or width < k)
    {
        qn::set_err(QNPEPS_ELOC_ERR_BAD_CONFIG);
        return false;
    }

    GesvdaWorkspace work{};
    auto cursor = WorkspaceCursor::carve(scratch, scratch_bytes);
    const auto storage_rank = int{condition_spectrum_rank(width, k)};
    carve_qb_workspace(la, work, cursor, rows, cols, k, width, storage_rank, dim_batch);
    if (qn::err_state() != QNPEPS_ELOC_OK) return false;
    const auto sketch_stride = usize{static_cast<usize>(rows) * width};
    const auto projection_stride = usize{static_cast<usize>(cols) * width};
    const auto right_stride = usize{static_cast<usize>(width) * k};
    const auto q_candidate_stride = usize{static_cast<usize>(rows) * k};
    const auto r_candidate_stride = usize{static_cast<usize>(k) * cols};
    auto sketch = MatrixBatch{work.sketch, static_cast<i64>(sketch_stride), rows, width};
    auto projection =
        MatrixBatch{work.projection, static_cast<i64>(projection_stride), cols, width};
    auto right = MatrixBatch{work.right, static_cast<i64>(right_stride), width, k};
    auto q_candidate = MatrixBatch{work.q_candidate, static_cast<i64>(q_candidate_stride), rows, k};
    auto r_candidate = MatrixBatch{work.r_candidate, static_cast<i64>(r_candidate_stride), k, cols};
    const auto omega_batch = MatrixBatch{omega, 0, cols, width};
    const auto panel_input = CuMatrixBatchedCF32Const{
        reinterpret_cast<const cuFloatComplex*>(panel.data()),
        panel.stride(),
        panel.rows(),
        panel.cols()
    };
    const auto omega_input = CuMatrixBatchedCF32Const{
        reinterpret_cast<const cuFloatComplex*>(omega_batch.data()),
        omega_batch.stride(),
        omega_batch.rows(),
        omega_batch.cols()
    };
    const auto sketch_output = CuMatrixBatchedCF32{
        reinterpret_cast<cuFloatComplex*>(sketch.data()),
        sketch.stride(),
        sketch.rows(),
        sketch.cols()
    };
    const auto projection_output = CuMatrixBatchedCF32{
        reinterpret_cast<cuFloatComplex*>(projection.data()),
        projection.stride(),
        projection.rows(),
        projection.cols()
    };
    const auto right_factor_input = CuMatrixBatchedCF32Const{
        reinterpret_cast<const cuFloatComplex*>(right.data()),
        right.stride(),
        right.rows(),
        right.cols()
    };
    const auto orthogonal_factor_output = CuMatrixBatchedCF32{
        reinterpret_cast<cuFloatComplex*>(q_candidate.data()),
        q_candidate.stride(),
        q_candidate.rows(),
        q_candidate.cols()
    };
    const auto remainder_factor_output = CuMatrixBatchedCF32{
        reinterpret_cast<cuFloatComplex*>(r_candidate.data()),
        r_candidate.stride(),
        r_candidate.rows(),
        r_candidate.cols()
    };

    la.matmul_batched(panel_input, omega_input, sketch_output, dim_batch);
    la.matmul_batched(
        panel_input,
        sketch_output,
        projection_output,
        dim_batch,
        {.op_a = BlasOp::conj_trans, .op_b = BlasOp::none}
    );
    la.matmul_batched(panel_input, projection_output, sketch_output, dim_batch);
    for (int lane{}; lane < dim_batch; ++lane)
    {
        qr(la,
           rows,
           width,
           work.sketch + static_cast<i64>(lane) * sketch_stride,
           rows,
           work.qr_scratch,
           fail_flag,
           2,
           failure_log,
           lane);
    }
    la.matmul_batched(
        panel_input,
        sketch_output,
        projection_output,
        dim_batch,
        {.op_a = BlasOp::conj_trans, .op_b = BlasOp::none}
    );
    if (not run_gesvda(
            la,
            work,
            work.projection,
            static_cast<i64>(projection_stride),
            cols,
            width,
            k,
            dim_batch
        ))
        return false;
    const auto record_gesvda_info_args = CuRecordGesvdaInfoArgs{
        .info = work.info,
        .failure_log = failure_log,
        .fail_flag = fail_flag,
        .dim_batch = dim_batch,
    };
    cu_record_gesvda_info<<<(dim_batch + 255) / 256, 256, 0, la.stream()>>>(
        record_gesvda_info_args
    );
    la.matmul_batched(sketch_output, right_factor_input, orthogonal_factor_output, dim_batch);
    la.matmul_batched(
        orthogonal_factor_output,
        panel_input,
        remainder_factor_output,
        dim_batch,
        {.op_a = BlasOp::conj_trans, .op_b = BlasOp::none}
    );
    const auto total =
        i64{static_cast<i64>(dim_batch)
            * (static_cast<i64>(rows) * k + static_cast<i64>(k) * cols)};
    const auto blocks =
        int{static_cast<int>(std::max<i64>(1, std::min<i64>(4096, (total + 255) / 256)))};
    const auto copy_factor_outputs_args = CuCopyFactorOutputsArgs{
        .q_candidate = work.q_candidate,
        .q_candidate_stride = static_cast<i64>(q_candidate_stride),
        .r_candidate = work.r_candidate,
        .r_candidate_stride = static_cast<i64>(r_candidate_stride),
        .r_out = r_out.data(),
        .r_stride = r_out.stride(),
        .q_out = q_out.data(),
        .q_stride = q_out.stride(),
        .info = work.info,
        .replay_mask = fallback_info ? failure_log : nullptr,
        .rows = rows,
        .cols = cols,
        .k = k,
        .dim_batch = dim_batch,
    };
    cu_copy_factor_outputs<<<blocks, 256, 0, la.stream()>>>(copy_factor_outputs_args);
    CUDA_CHECK(cudaGetLastError());
    if (storage_rank > k
        and not run_gesvda(
            la,
            work,
            work.projection,
            static_cast<i64>(projection_stride),
            cols,
            width,
            storage_rank,
            dim_batch
        ))
        return false;
    if (storage_rank > k)
    {
        const auto record_gesvda_info_args = CuRecordGesvdaInfoArgs{
            .info = work.info,
            .failure_log = failure_log,
            .fail_flag = fail_flag,
            .dim_batch = dim_batch,
        };
        cu_record_gesvda_info<<<(dim_batch + 255) / 256, 256, 0, la.stream()>>>(
            record_gesvda_info_args
        );
        CUDA_CHECK(cudaGetLastError());
    }
    return true;
}
}
