#ifndef QNPEPS_SAMPLER_TRUNC_SVD_CUH
#define QNPEPS_SAMPLER_TRUNC_SVD_CUH

#include "linalg/linalg.cuh"
#include "core/types.cuh"

namespace qnpeps::trunc_svd
{
inline constexpr const char* k_sampler_variable{"QNPEPS_SAMPLER_TRUNC"};
inline constexpr const char* k_dlenv_variable{"QNPEPS_DLENV_TRUNC"};

enum class Route
{
    rangefinder,
    svd,
    invalid
};

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

[[nodiscard]] auto require_sampler_route() -> Route;
[[nodiscard]] auto require_dlenv_route() -> Route;

auto batched_rangefinder_svd(Linalg& la, const RangefinderArgs& args) -> void;
auto orthonormalize_panel(Linalg& la, CuMatrixCF32 panel) -> void;
}

#endif
