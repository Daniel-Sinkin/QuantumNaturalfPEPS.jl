#ifndef QNPEPS_TENSOR_KERNELS_CUH
#define QNPEPS_TENSOR_KERNELS_CUH

#include "core/cuda_utils.cuh"
#include "tensor/tensor.cuh"

namespace qnpeps
{

struct PermutationPlan
{
    qnpeps::CuArray<int, k_max_tensor_rank> output_extents;
    qnpeps::CuArray<i64, k_max_tensor_rank> input_strides;
};

struct CuPermuteArgs
{
    cuFloatComplex* output{};
    const cuFloatComplex* input{};
    PermutationPlan plan{};
    int rank{};
    i64 element_count{};
    int conjugate{};
};

struct CuGatherArgs
{
    cuFloatComplex* output{};
    const cuFloatComplex* input{};
    const int* gather_indices{};
    i64 element_count{};
    i64 output_stride{};
    i64 input_stride{};
    int conjugate{};
    int batch_count{};
};

__global__ auto cu_gather(CuGatherArgs args) -> void;
__global__ auto cu_permute(CuPermuteArgs args) -> void;

}

#endif
