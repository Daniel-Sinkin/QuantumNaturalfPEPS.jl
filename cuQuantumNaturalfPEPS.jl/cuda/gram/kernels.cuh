#ifndef QNPEPS_GRAM_KERNELS_CUH
#define QNPEPS_GRAM_KERNELS_CUH

#include "core/cuda_utils.cuh"

#include <cstdint>

namespace qnpeps::gram_cublas
{

struct ExpandArgs
{
    cuFloatComplex* dense;
    const cuFloatComplex* rows;
    const std::uint8_t* samples;
    i64 row_count;
    i64 compact;
    i64 sample_base;
    int sites;
    int first_site;
    int compact_begin;
    int compact_count;
    int dense_width;
    const int* slot;
    const int* offsets;
    const int* slices;
};

struct FinalizeOffdiagonalArgs
{
    cuFloatComplex* output;
    int output_ld;
    int row_count;
    int local_base;
    int samples;
};

struct FinalizeDiagonalArgs
{
    cuFloatComplex* output;
    int output_ld;
    int row_count;
    int local_base;
};

__global__ auto cu_expand(ExpandArgs args) -> void;
__global__ auto cu_finalize_offdiagonal(FinalizeOffdiagonalArgs args) -> void;
__global__ auto cu_finalize_diagonal(FinalizeDiagonalArgs args) -> void;

}

#endif
