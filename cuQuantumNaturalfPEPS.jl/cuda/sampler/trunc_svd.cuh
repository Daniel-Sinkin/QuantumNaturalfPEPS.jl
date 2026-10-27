#ifndef QNPEPS_SAMPLER_TRUNC_SVD_CUH
#define QNPEPS_SAMPLER_TRUNC_SVD_CUH

#include "linalg/linalg.cuh"
#include "core/types.cuh"

namespace qnpeps::trunc_svd
{
struct CuRecordGesvdaFailuresArgs
{
    const int* status{};
    int* retry{};
    int dim_batch{};
};

struct CuPadUnconvergedLanesArgs
{
    cuFloatComplex* left{};
    const int* status{};
    i64 left_stride{};
    int rows{};
    int rank{};
    int dim_batch{};
};

__global__ auto cu_record_gesvda_failures(CuRecordGesvdaFailuresArgs args) -> void;
__global__ auto cu_pad_unconverged_lanes(CuPadUnconvergedLanesArgs args) -> void;

auto batched_rangefinder_svd(Linalg& la, const RangefinderArgs& args) -> void;
auto orthonormalize_panel(Linalg& la, CuMatrixCF32 panel) -> void;
}

#endif
