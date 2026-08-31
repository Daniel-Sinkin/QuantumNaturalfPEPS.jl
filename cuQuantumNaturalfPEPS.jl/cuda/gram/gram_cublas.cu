#include "core/cuda_utils.cuh"
#include "gram/gram_cublas.cuh"
#include "gram/kernels.cuh"

#include <algorithm>

namespace qnpeps::gram_cublas
{
namespace
{
auto blocks_for(i64 count) -> u32
{
    return std::max(grid_blocks_capped(count), u32{1});
}
}

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
) -> cudaError_t
{
    const usize dense_bytes{
        sizeof(cuFloatComplex) * static_cast<usize>(row_count) * static_cast<usize>(dense_width)
    };
    cudaError_t status{cudaMemsetAsync(dense, 0, dense_bytes, stream)};
    if (status != cudaSuccess) return status;
    const i64 total{row_count * compact_count};
    const ExpandArgs args{
        dense,
        rows,
        samples,
        row_count,
        compact,
        sample_base,
        sites,
        first_site,
        compact_begin,
        compact_count,
        dense_width,
        slot,
        offsets,
        slices,
    };
    cu_expand<<<blocks_for(total), k_threads_per_block, 0, stream>>>(args);
    return cudaGetLastError();
}

auto accumulate_diagonal_block(
    Linalg& linalg,
    int rows,
    int dense_width,
    const cuFloatComplex* dense,
    cuFloatComplex* output,
    int output_ld
) -> cublasStatus_t
{
    return linalg.accumulate_hermitian(rows, dense_width, dense, output, output_ld);
}

auto accumulate_offdiagonal_block(
    Linalg& linalg,
    int rows_a,
    int rows_b,
    int dense_width,
    const cuFloatComplex* dense_a,
    const cuFloatComplex* dense_b,
    cuFloatComplex* output,
    int output_ld
) -> cublasStatus_t
{
    return linalg.accumulate_gram_block(
        rows_a, rows_b, dense_width, dense_a, dense_b, output, output_ld
    );
}

auto finalize(
    cuFloatComplex* output,
    int output_ld,
    int row_count,
    int local_base,
    int samples,
    cudaStream_t stream
) -> cudaError_t
{
    const i64 offdiagonal{static_cast<i64>(row_count) * samples};
    const FinalizeOffdiagonalArgs offdiagonal_args{
        output, output_ld, row_count, local_base, samples
    };
    cu_finalize_offdiagonal<<<blocks_for(offdiagonal), k_threads_per_block, 0, stream>>>(
        offdiagonal_args
    );
    cudaError_t status{cudaGetLastError()};
    if (status != cudaSuccess) return status;
    const i64 diagonal{static_cast<i64>(row_count) * row_count};
    const FinalizeDiagonalArgs diagonal_args{output, output_ld, row_count, local_base};
    cu_finalize_diagonal<<<blocks_for(diagonal), k_threads_per_block, 0, stream>>>(diagonal_args);
    return cudaGetLastError();
}
}
