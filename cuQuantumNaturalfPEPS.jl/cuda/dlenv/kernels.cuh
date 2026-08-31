#ifndef QNPEPS_DLENV_KERNELS_CUH
#define QNPEPS_DLENV_KERNELS_CUH

#include "core/cuda_utils.cuh"

namespace qnpeps::dlenv
{

struct ProductArgs
{
    const cuDoubleComplex* state{};
    const cuFloatComplex* ket{};
    i64 state_left{};
    i64 state_right{};
    i64 down{};
    i64 left{};
    i64 up{};
    i64 right{};
    i64 dim_phys{};
    i64 output_state{};
    cuDoubleComplex* product{};
};

struct PromoteArgs
{
    const cuFloatComplex* input{};
    usize count{};
    cuDoubleComplex* output{};
};

struct PackArgs
{
    const cuDoubleComplex* input{};
    i64 physical{};
    i64 right{};
    i64 left{};
    cuFloatComplex* output{};
};

struct ScaleArgs
{
    const f64* normalization_log{};
    usize count{};
    f64* scales{};
};

__global__ auto cu_product(ProductArgs args) -> void;
__global__ auto cu_promote(PromoteArgs args) -> void;
__global__ auto cu_pack(PackArgs args) -> void;
__global__ auto cu_scales(ScaleArgs args) -> void;

}

#endif
