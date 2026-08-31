#pragma once

#include "../experimental_svd.cuh"
#include "../eo.cuh"

#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda/std/cmath>

#include "core/complex.cuh"
#include "factorization_kernels.hpp"

namespace qnpeps
{

#line 971 "cuda/linalg/eo_routes.cu"

auto batched_rangefinder(
    Linalg& la,
    MatrixBatch panel,
    int k,
    bool exact_factorization,
    const cf* omega,
    MatrixBatch q_out,
    MatrixBatch r_out,
    int dim_batch,
    EoDeviceBuffer sketch,
    EoDeviceBuffer proj,
    EoDeviceBuffer gram,
    cf** gram_ptrs,
    cf** sketch_ptrs,
    int* info,
    int* fail_flag,
    int* failure_log,
    const int* fallback_info,
    void* robust_scratch,
    usize robust_scratch_bytes
) -> void
{
    const auto rows = panel.rows();
    const auto cols = panel.cols();
    const auto route = RangefinderRoute{rangefinder_route_from_env()};
    if (exact_factorization and route != RangefinderRoute::gesvda and k == std::min(rows, cols))
    {
        const auto copy_batch = [&](MatrixBatch dst, MatrixBatch src)
        {
            CUDA_CHECK(cudaMemcpy2DAsync(
                dst.data(),
                static_cast<usize>(dst.stride()) * sizeof(cf),
                src.data(),
                static_cast<usize>(src.stride()) * sizeof(cf),
                static_cast<usize>(src.rows()) * src.cols() * sizeof(cf),
                static_cast<usize>(dim_batch),
                cudaMemcpyDeviceToDevice,
                la.stream()
            ));
        };
        const auto identity_elems = i64{static_cast<i64>(k) * k * dim_batch};
        const auto identity_blocks = int{
            static_cast<int>(std::max<i64>(1, std::min<i64>(4096, (identity_elems + 255) / 256)))
        };
        if (rows > cols)
        {
            copy_batch(q_out, panel);
            auto* scale_out{reinterpret_cast<f32*>(info)};
            cu_normalize_log<<<dim_batch, 256, 0, la.stream()>>>(
                q_out.data(), rows * cols, q_out.stride(), nullptr, dim_batch, scale_out
            );
            cu_fill_identity<<<identity_blocks, 256, 0, la.stream()>>>(
                r_out.data(), r_out.stride(), k, dim_batch, scale_out
            );
        }
        else
        {
            cu_fill_identity<<<identity_blocks, 256, 0, la.stream()>>>(
                q_out.data(), q_out.stride(), k, dim_batch, nullptr
            );
            copy_batch(r_out, panel);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (route == RangefinderRoute::qb_svd)
    {
        qn_eloc::e0191::run_qb_gesvda(
            la,
            panel,
            k,
            omega,
            q_out,
            r_out,
            dim_batch,
            fail_flag,
            failure_log,
            fallback_info,
            robust_scratch,
            robust_scratch_bytes
        );
        return;
    }
    if (route == RangefinderRoute::gesvda)
    {
        qn_eloc::e0191::run_direct_gesvda_cutoff(
            la,
            panel,
            k,
            q_out,
            r_out,
            dim_batch,
            fail_flag,
            failure_log,
            fallback_info,
            robust_scratch,
            robust_scratch_bytes
        );
        return;
    }
    auto mat_omega = MatrixBatch{omega, 0, cols, k};
    auto sketch_mat = MatrixBatch{sketch.p, sketch.stride, rows, k};
    auto proj_mat = MatrixBatch{proj.p, proj.stride, cols, k};
    auto gram_mat = MatrixBatch{gram.p, gram.stride, k, k};
    const auto panel_input = CuMatrixBatchedCF32Const{
        reinterpret_cast<const cuFloatComplex*>(panel.data()),
        panel.stride(),
        panel.rows(),
        panel.cols()
    };
    const auto omega_input = CuMatrixBatchedCF32Const{
        reinterpret_cast<const cuFloatComplex*>(mat_omega.data()),
        mat_omega.stride(),
        mat_omega.rows(),
        mat_omega.cols()
    };
    const auto sketch_output = CuMatrixBatchedCF32{
        reinterpret_cast<cuFloatComplex*>(sketch_mat.data()),
        sketch_mat.stride(),
        sketch_mat.rows(),
        sketch_mat.cols()
    };
    const auto projection_output = CuMatrixBatchedCF32{
        reinterpret_cast<cuFloatComplex*>(proj_mat.data()),
        proj_mat.stride(),
        proj_mat.rows(),
        proj_mat.cols()
    };
    const auto gram_output = CuMatrixBatchedCF32{
        reinterpret_cast<cuFloatComplex*>(gram_mat.data()),
        gram_mat.stride(),
        gram_mat.rows(),
        gram_mat.cols()
    };
    const auto factor_output = CuMatrixBatchedCF32{
        reinterpret_cast<cuFloatComplex*>(r_out.data()), r_out.stride(), r_out.rows(), r_out.cols()
    };
    const auto householder_orthogonalize = [&](MatrixBatch matrix)
    {
        const auto required_scratch = usize{qr_scratch_bytes(la, matrix.rows(), matrix.cols())};
        if (required_scratch > robust_scratch_bytes)
        {
            qn::set_err(QNPEPS_ELOC_ERR_OOM);
            return false;
        }
        for (auto lane = int{0}; lane < dim_batch; ++lane)
        {
            qr(la,
               matrix.rows(),
               matrix.cols(),
               matrix.data() + static_cast<i64>(lane) * matrix.stride(),
               matrix.rows(),
               robust_scratch,
               fail_flag,
               2);
        }
        return true;
    };
    const auto finish_householder = [&]()
    {
        const auto dst_pitch = static_cast<usize>(q_out.stride()) * sizeof(cf);
        const auto src_pitch = static_cast<usize>(sketch.stride) * sizeof(cf);
        const auto copy_width = static_cast<usize>(rows) * k * sizeof(cf);
        const auto copy_height = static_cast<usize>(dim_batch);
        CUDA_CHECK(cudaMemcpy2DAsync(
            q_out.data(),
            dst_pitch,
            sketch.p,
            src_pitch,
            copy_width,
            copy_height,
            cudaMemcpyDeviceToDevice,
            la.stream()
        ));
        la.matmul_batched(
            sketch_output,
            panel_input,
            factor_output,
            dim_batch,
            {.op_a = BlasOp::conj_trans, .op_b = BlasOp::none}
        );
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

    if (route == RangefinderRoute::householder)
    {
        if (not householder_orthogonalize(sketch_mat)) return;
        finish_householder();
        return;
    }

    bool has_fallback{};
    if (fallback_info)
    {
        for (auto pass = int{0}; pass < 2; ++pass)
        {
            for (auto lane = int{0}; lane < dim_batch; ++lane)
                has_fallback = has_fallback or fallback_info[pass * dim_batch + lane] != 0;
        }
    }
    if (has_fallback and qr_scratch_bytes(la, rows, k) > robust_scratch_bytes)
    {
        qn::set_err(QNPEPS_ELOC_ERR_OOM);
        return;
    }
    for (auto pass = int{0}; pass < 2; ++pass)
    {
        la.matmul_batched(
            sketch_output,
            sketch_output,
            gram_output,
            dim_batch,
            {.op_a = BlasOp::conj_trans, .op_b = BlasOp::none}
        );
        cu_chol_shift<<<(dim_batch + 255) / 256, 256, 0, la.stream()>>>(
            gram.p, k, gram.stride, dim_batch
        );
        la.cholesky_batched(k, gram_ptrs, k, info, dim_batch);
        auto pass_log{failure_log + pass * dim_batch};
        if (fallback_info)
        {
            if (fail_flag)
            {
                cu_or_unmasked_info<<<(dim_batch + 255) / 256, 256, 0, la.stream()>>>(
                    info, pass_log, dim_batch, fail_flag
                );
            }
            cu_union_info<<<(dim_batch + 255) / 256, 256, 0, la.stream()>>>(
                info, pass_log, dim_batch
            );
            CUDA_CHECK(cudaGetLastError());
            for (auto lane = int{0}; lane < dim_batch; ++lane)
            {
                if (fallback_info[pass * dim_batch + lane] != 0)
                {
                    qr(la,
                       rows,
                       k,
                       sketch.p + static_cast<i64>(lane) * sketch.stride,
                       rows,
                       robust_scratch,
                       fail_flag,
                       2);
                    const auto blocks = int{std::max(1, (k * k + 255) / 256)};
                    cu_fill_identity<<<blocks, 256, 0, la.stream()>>>(
                        gram.p + static_cast<i64>(lane) * gram.stride, gram.stride, k, 1, nullptr
                    );
                }
            }
        }
        else
        {
            CUDA_CHECK(cudaMemcpyAsync(
                pass_log,
                info,
                sizeof(int) * static_cast<usize>(dim_batch),
                cudaMemcpyDeviceToDevice,
                la.stream()
            ));
            if (fail_flag)
            {
                cu_or_info<<<(dim_batch + 255) / 256, 256, 0, la.stream()>>>(
                    info, dim_batch, fail_flag
                );
            }
        }
        la.solve_triangular_batched(rows, k, gram_ptrs, k, sketch_ptrs, rows, dim_batch);
    }

    const auto dst_pitch = static_cast<usize>(q_out.stride()) * sizeof(cf);
    const auto src_pitch = static_cast<usize>(sketch.stride) * sizeof(cf);
    const auto copy_width = static_cast<usize>(rows) * k * sizeof(cf);
    const auto copy_height = static_cast<usize>(dim_batch);
    CUDA_CHECK(cudaMemcpy2DAsync(
        q_out.data(),
        dst_pitch,
        sketch.p,
        src_pitch,
        copy_width,
        copy_height,
        cudaMemcpyDeviceToDevice,
        la.stream()
    ));

    la.matmul_batched(
        sketch_output,
        panel_input,
        factor_output,
        dim_batch,
        {.op_a = BlasOp::conj_trans, .op_b = BlasOp::none}
    );
}

}
