#include "linalg/gram.cuh"

#include "core/types.cuh"

#include <cublas_v2.h>

namespace qnpeps
{

auto Linalg::blas_state(BlasState& state) noexcept -> cublasStatus_t
{
    auto status = cublasGetMathMode(blas_->get(), &state.math);
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasGetPointerMode(blas_->get(), &state.pointer);
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasGetAtomicsMode(blas_->get(), &state.atomics);
    return status;
}

auto Linalg::set_blas_state(const BlasState& state) noexcept -> cublasStatus_t
{
    auto status = cublasSetMathMode(blas_->get(), state.math);
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasSetPointerMode(blas_->get(), state.pointer);
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasSetAtomicsMode(blas_->get(), state.atomics);
    return status;
}

auto Linalg::set_gram_state() noexcept -> cublasStatus_t
{
    auto status = cublasSetMathMode(blas_->get(), CUBLAS_PEDANTIC_MATH);
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasSetPointerMode(blas_->get(), CUBLAS_POINTER_MODE_HOST);
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasSetAtomicsMode(blas_->get(), CUBLAS_ATOMICS_NOT_ALLOWED);
    return status;
}

auto Linalg::accumulate_hermitian(
    int rows, int dense_width, const cuFloatComplex* dense, cuFloatComplex* output, int output_ld
) noexcept -> cublasStatus_t
{
    constexpr f32 alpha{1.0f};
    constexpr f32 beta{1.0f};
    return cublasCherkEx(
        blas_->get(),
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_C,
        rows,
        dense_width,
        &alpha,
        dense,
        CUDA_C_32F,
        dense_width,
        &beta,
        output,
        CUDA_C_32F,
        output_ld
    );
}

auto Linalg::accumulate_gram_block(
    int rows_a,
    int rows_b,
    int dense_width,
    const cuFloatComplex* dense_a,
    const cuFloatComplex* dense_b,
    cuFloatComplex* output,
    int output_ld
) noexcept -> cublasStatus_t
{
    constexpr cuComplex alpha{1.0f, 0.0f};
    constexpr cuComplex beta{1.0f, 0.0f};
    return cublasGemmEx(
        blas_->get(),
        CUBLAS_OP_C,
        CUBLAS_OP_N,
        rows_b,
        rows_a,
        dense_width,
        &alpha,
        dense_b,
        CUDA_C_32F,
        dense_width,
        dense_a,
        CUDA_C_32F,
        dense_width,
        &beta,
        output,
        CUDA_C_32F,
        output_ld,
        CUBLAS_COMPUTE_32F_PEDANTIC,
        CUBLAS_GEMM_DEFAULT
    );
}
}
