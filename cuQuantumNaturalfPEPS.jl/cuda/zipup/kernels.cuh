#ifndef QNPEPS_ZIPUP_KERNELS_CUH
#define QNPEPS_ZIPUP_KERNELS_CUH

#include "core/cuda_utils.cuh"

namespace qnpeps::zipup
{

struct AbsmaxInverseArgs
{
    const cuFloatComplex* factor;
    i64 element_count;
    f32* inverse_scale;
    f64* scale;
};

__global__ auto cu_or_status(const int* status, int* flag) -> void;
__global__ auto cu_absmax_inverse(AbsmaxInverseArgs args) -> void;
__global__ auto cu_apply_inverse_scale(
    cuFloatComplex* factor, i64 element_count, const f32* inverse_scale
) -> void;

}

#endif
