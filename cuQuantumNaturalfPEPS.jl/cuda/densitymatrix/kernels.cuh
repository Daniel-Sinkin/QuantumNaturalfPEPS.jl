#ifndef QNPEPS_DENSITYMATRIX_KERNELS_CUH
#define QNPEPS_DENSITYMATRIX_KERNELS_CUH

#include "capi/qnpeps.h"
#include "core/types.cuh"

namespace qnpeps::densitymatrix
{
struct LocalProductArgs
{
    const cuDoubleComplex* state{};
    const cuDoubleComplex* operator_values{};
    i64 state_left{};
    i64 physical_input{};
    i64 state_right{};
    i64 operator_left{};
    i64 physical_output{};
    i64 operator_right{};
    cuDoubleComplex* product{};
};

struct RightBlockArgs
{
    const cuDoubleComplex* product{};
    const cuDoubleComplex* carried{};
    i64 combined_left{};
    i64 combined_right{};
    i64 physical_output{};
    i64 right_rank{};
    cuDoubleComplex* right_block{};
};

struct RightBlockSliceArgs
{
    const cuDoubleComplex* product{};
    const cuDoubleComplex* carried{};
    i64 combined_left{};
    i64 combined_right{};
    i64 physical_output{};
    i64 output_state{};
    i64 right_rank{};
    cuDoubleComplex* right_block{};
};

struct FilterArgs
{
    const f64* solver_values{};
    i64 order{};
    i64 natural_cap{};
    i64 applied_cap{};
    i64 mindim{};
    f64 cutoff{};
    f64* sorted_values{};
    f64* retained_values{};
    i32* sorted_indices{};
    i32* active_rank{};
    QnpepsDensityRankRecord* record{};
};

struct GatherBasisArgs
{
    const cuDoubleComplex* solver_vectors{};
    const i32* sorted_indices{};
    const i32* active_rank{};
    i64 order{};
    i64 upper_bond{};
    cuDoubleComplex* basis{};
};

struct ProjectRightArgs
{
    const cuDoubleComplex* right_block{};
    const cuDoubleComplex* basis{};
    i64 combined_left{};
    i64 order{};
    i64 active_rank{};
    cuDoubleComplex* carried{};
};

struct PackSiteArgs
{
    const cuDoubleComplex* basis{};
    i64 physical_output{};
    i64 right_rank{};
    i64 left_rank{};
    cuDoubleComplex* output{};
};

struct PackFirstArgs
{
    const cuDoubleComplex* right_block{};
    i64 physical_output{};
    i64 right_rank{};
    cuDoubleComplex* output{};
};

struct DimensionsArgs
{
    i64 site{};
    i32 left{};
    i32 physical{};
    i32 right{};
    i32* dimensions{};
};

struct NormalizeArgs
{
    cuDoubleComplex* values{};
    usize count{};
    bool normalize{};
    f64 input_gauge{};
    f64* normalization_log{};
    f64* output_gauge{};
};

__global__ auto cu_local_product(LocalProductArgs args) -> void;
__global__ auto cu_right_block(RightBlockArgs args) -> void;
__global__ auto cu_right_block_slice(RightBlockSliceArgs args) -> void;
__global__ auto cu_filter(FilterArgs args) -> void;
__global__ auto cu_gather_basis(GatherBasisArgs args) -> void;
__global__ auto cu_project_right(ProjectRightArgs args) -> void;
__global__ auto cu_pack_site(PackSiteArgs args) -> void;
__global__ auto cu_pack_first(PackFirstArgs args) -> void;
__global__ auto cu_dimensions(DimensionsArgs args) -> void;
__global__ auto cu_normalize(NormalizeArgs args) -> void;
}

#endif
