#ifndef QNPEPS_DLENV_KERNELS_CUH
#define QNPEPS_DLENV_KERNELS_CUH

#include "core/cuda_utils.cuh"
#include "densitymatrix/kernels.cuh"

namespace qnpeps::dlenv
{
using densitymatrix::cu_pack;
using densitymatrix::cu_promote;
using densitymatrix::PackArgs;
using densitymatrix::PromoteArgs;

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

struct ScaleArgs
{
    const f64* normalization_log{};
    usize count{};
    f64* scales{};
};

__global__ auto cu_product(ProductArgs args) -> void;
__global__ auto cu_scales(ScaleArgs args) -> void;

}

#endif
