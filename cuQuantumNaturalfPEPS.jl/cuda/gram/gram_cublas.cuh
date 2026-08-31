#ifndef QNPEPS_GRAM_CUBLAS_CUH
#define QNPEPS_GRAM_CUBLAS_CUH

#include "core/types.cuh"
#include "linalg/linalg.cuh"

#include <cstdint>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace qnpeps::gram_cublas
{
auto expand(
    cuFloatComplex* dense,
    const cuFloatComplex* rows,
    const std::uint8_t* samples,
    i64 row_count,
    i64 compact,
    i64 sample_base,
    int sites,
    int first_site,
    int compact_begin,
    int compact_count,
    int dense_width,
    const int* slot,
    const int* offsets,
    const int* slices,
    cudaStream_t stream
) -> cudaError_t;

auto accumulate_diagonal_block(
    Linalg& linalg,
    int rows,
    int dense_width,
    const cuFloatComplex* dense,
    cuFloatComplex* output,
    int output_ld
) -> cublasStatus_t;

auto accumulate_offdiagonal_block(
    Linalg& linalg,
    int rows_a,
    int rows_b,
    int dense_width,
    const cuFloatComplex* dense_a,
    const cuFloatComplex* dense_b,
    cuFloatComplex* output,
    int output_ld
) -> cublasStatus_t;

auto finalize(
    cuFloatComplex* output,
    int output_ld,
    int row_count,
    int local_base,
    int samples,
    cudaStream_t stream
) -> cudaError_t;
}

#endif
