#pragma once

#include "../linalg.cuh"

namespace qnpeps
{

#line 960 "cuda/linalg/linalg.cuh"

inline auto Linalg::gemv(
    CuMatrixCF64Const a, const cuDoubleComplex* x, cuDoubleComplex* y, const GemvConfig& cfg
) -> void
{
    const auto valid =
        a.data() != nullptr and x != nullptr and y != nullptr and a.rows() > 0 and a.cols() > 0;
    if (not valid)
    {
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return;
    }
    const cuDoubleComplex alpha{cfg.alpha_real, cfg.alpha_imag};
    const cuDoubleComplex beta{cfg.beta_real, cfg.beta_imag};
    CUBLAS_CHECK(cublasZgemv(
        blas_->get(),
        to_cublas(cfg.op),
        a.rows(),
        a.cols(),
        &alpha,
        a.data(),
        a.ld(),
        x,
        1,
        &beta,
        y,
        1
    ));
}

inline auto Linalg::matmul(
    CuMatrixCF32Const a, CuMatrixCF32Const b, CuMatrixCF32 c, const MatmulConfig& cfg
) -> void
{
    const auto shape = matmul_shape(cfg, a, b, c);
    const auto valid = shape.has_value() and a.data() and b.data() and c.data();
    if (not valid)
    {
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return;
    }
    const cuFloatComplex alpha{cfg.alpha_real, cfg.alpha_imag};
    const cuFloatComplex beta{cfg.beta_real, cfg.beta_imag};
    CUBLAS_CHECK(cublasCgemm(
        blas_->get(),
        to_cublas(cfg.op_a),
        to_cublas(cfg.op_b),
        shape->m,
        shape->n,
        shape->k,
        &alpha,
        a.data(),
        a.ld(),
        b.data(),
        b.ld(),
        &beta,
        c.data(),
        c.ld()
    ));
}

inline auto Linalg::matmul(
    CuMatrixCF64Const a, CuMatrixCF64Const b, CuMatrixCF64 c, const MatmulConfig& cfg
) -> void
{
    const auto shape = matmul_shape(cfg, a, b, c);
    const auto valid = shape.has_value() and a.data() and b.data() and c.data();
    if (not valid)
    {
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return;
    }
    const cuDoubleComplex alpha{static_cast<f64>(cfg.alpha_real), static_cast<f64>(cfg.alpha_imag)};
    const cuDoubleComplex beta{static_cast<f64>(cfg.beta_real), static_cast<f64>(cfg.beta_imag)};
    CUBLAS_CHECK(cublasZgemm(
        blas_->get(),
        to_cublas(cfg.op_a),
        to_cublas(cfg.op_b),
        shape->m,
        shape->n,
        shape->k,
        &alpha,
        a.data(),
        a.ld(),
        b.data(),
        b.ld(),
        &beta,
        c.data(),
        c.ld()
    ));
}

inline auto Linalg::matmul_left_adj(CuMatrixCF32Const a, CuMatrixCF32Const b, CuMatrixCF32 c)
    -> void
{
    matmul(a, b, c, {.op_a = BlasOp::conj_trans, .op_b = BlasOp::none});
}

inline auto Linalg::matmul_batched(
    CuMatrixBatchedCF32Const a,
    CuMatrixBatchedCF32Const b,
    CuMatrixBatchedCF32 c,
    int batch_size,
    const MatmulConfig& cfg
) -> void
{
    const auto shape = matmul_shape(cfg, a, b, c);
    const auto stored_elements = [](CuMatrixBatchedCF32Const x) -> i64
    { return static_cast<i64>(x.rows()) * x.cols(); };
    const auto valid_shape = shape.has_value();
    const auto valid_batch = batch_size > 0;
    const auto valid_a = a.data() and (a.stride() == 0 or a.stride() >= stored_elements(a));
    const auto valid_b = b.data() and (b.stride() == 0 or b.stride() >= stored_elements(b));
    const auto valid_c = c.data() and c.stride() >= stored_elements(c);
    const auto valid = valid_shape and valid_batch and valid_a and valid_b and valid_c;
    if (not valid)
    {
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return;
    }
    const cuFloatComplex alpha{cfg.alpha_real, cfg.alpha_imag};
    const cuFloatComplex beta{cfg.beta_real, cfg.beta_imag};
    CUBLAS_CHECK(cublasCgemmStridedBatched(
        blas_->get(),
        to_cublas(cfg.op_a),
        to_cublas(cfg.op_b),
        shape->m,
        shape->n,
        shape->k,
        &alpha,
        a.data(),
        a.ld(),
        a.stride(),
        b.data(),
        b.ld(),
        b.stride(),
        &beta,
        c.data(),
        c.ld(),
        c.stride(),
        batch_size
    ));
}

inline auto Linalg::matmul_batched(
    CuMatrixBatchedCF32Const a, CuMatrixBatchedCF32Const b, CuMatrixBatchedCF32 c, int batch_size
) -> void
{
    matmul_batched(a, b, c, batch_size, MatmulConfig{});
}

inline auto Linalg::matmul_batched_left_adj(
    CuMatrixBatchedCF32Const a, CuMatrixBatchedCF32Const b, CuMatrixBatchedCF32 c, int batch_size
) -> void
{
    matmul_batched(a, b, c, batch_size, {.op_a = BlasOp::conj_trans, .op_b = BlasOp::none});
}

inline auto Linalg::matmul_batched_right_adj(
    CuMatrixBatchedCF32Const a, CuMatrixBatchedCF32Const b, CuMatrixBatchedCF32 c, int batch_size
) -> void
{
    matmul_batched(a, b, c, batch_size, {.op_a = BlasOp::none, .op_b = BlasOp::conj_trans});
}

inline auto Linalg::matmul_batched_both_adj(
    CuMatrixBatchedCF32Const a, CuMatrixBatchedCF32Const b, CuMatrixBatchedCF32 c, int batch_size
) -> void
{
    matmul_batched(a, b, c, batch_size, {.op_a = BlasOp::conj_trans, .op_b = BlasOp::conj_trans});
}

inline auto Linalg::matmul_batched_ptr(
    cuFloatComplex* const* a_array,
    int a_rows,
    int a_cols,
    cuFloatComplex* const* b_array,
    int b_rows,
    int b_cols,
    cuFloatComplex* const* c_array,
    int c_rows,
    int c_cols,
    int batch_size,
    const MatmulConfig& cfg
) -> void
{
    const CuMatrixCF32Const a{nullptr, a_rows, a_cols};
    const CuMatrixCF32Const b{nullptr, b_rows, b_cols};
    const CuMatrixCF32 c{nullptr, c_rows, c_cols};
    const auto shape = matmul_shape(cfg, a, b, c);
    const auto valid = shape.has_value() and batch_size > 0 and a_array and b_array and c_array;
    if (not valid)
    {
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return;
    }
    const cuFloatComplex alpha{cfg.alpha_real, cfg.alpha_imag};
    const cuFloatComplex beta{cfg.beta_real, cfg.beta_imag};
    CUBLAS_CHECK(cublasCgemmBatched(
        blas_->get(),
        to_cublas(cfg.op_a),
        to_cublas(cfg.op_b),
        shape->m,
        shape->n,
        shape->k,
        &alpha,
        a_array,
        a_rows,
        b_array,
        b_rows,
        &beta,
        c_array,
        c_rows,
        batch_size
    ));
}

}
