#ifndef QNPEPS_DENSITYMATRIX_KERNELS_CUH
#define QNPEPS_DENSITYMATRIX_KERNELS_CUH

#include "capi/qnpeps.h"
#include "core/types.cuh"

namespace qnpeps::densitymatrix
{
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

inline constexpr u32 k_lane_solver_failed{1u};
inline constexpr u32 k_lane_nonfinite_spectrum{2u};
inline constexpr u32 k_lane_nonfinite_norm{4u};

struct OperatorStrides
{
    i64 left{};
    i64 input{};
    i64 output{};
    i64 right{};
};

struct BatchedProductArgs
{
    const cuFloatComplex* state{};
    i64 state_lane_stride{};
    i64 state_left{};
    i64 physical_input{};
    i64 state_right{};
    const cuFloatComplex* operator_values{};
    i64 operator_lane_stride{};
    OperatorStrides operator_strides{};
    i64 operator_left{};
    i64 physical_output{};
    i64 operator_right{};
    cuDoubleComplex* products{};
    i64 product_lane_stride{};
    i64 lanes{};
};

struct BatchedBoundaryArgs
{
    const cuFloatComplex* input{};
    i64 input_lane_stride{};
    OperatorStrides input_strides{};
    i64 left{};
    i64 vertical{};
    i64 right{};
    cuFloatComplex* output{};
    i64 output_lane_stride{};
    f64* log_gauge{};
    u32* lane_flags{};
};

struct BatchedRightBlockArgs
{
    const cuDoubleComplex* products{};
    i64 product_lane_stride{};
    const cuDoubleComplex* carried{};
    i64 carried_lane_stride{};
    i64 combined_left{};
    i64 combined_right{};
    i64 physical_output{};
    i64 right_rank{};
    cuDoubleComplex* right_block{};
    i64 block_lane_stride{};
    i64 lanes{};
};

struct BatchedSelectArgs
{
    const f64* eigenvalues{};
    i64 eigenvalue_lane_stride{};
    i64 order{};
    const i32* solver_information{};
    i32* sorted_indices{};
    i64 index_lane_stride{};
    u32* lane_flags{};
};

struct BatchedBasisArgs
{
    const cuDoubleComplex* vectors{};
    i64 vector_lane_stride{};
    const i32* sorted_indices{};
    i64 index_lane_stride{};
    i64 order{};
    i64 rank{};
    i64 lanes{};
};

struct BatchedPackSiteArgs
{
    BatchedBasisArgs basis{};
    cuFloatComplex* output{};
    i64 output_lane_stride{};
};

struct BatchedProjectArgs
{
    BatchedBasisArgs basis{};
    const cuDoubleComplex* right_block{};
    i64 block_lane_stride{};
    i64 combined_left{};
    cuDoubleComplex* carried{};
    i64 carried_lane_stride{};
};

struct BatchedNormalizeArgs
{
    const cuDoubleComplex* values{};
    i64 value_lane_stride{};
    i64 count{};
    cuFloatComplex* output{};
    i64 output_lane_stride{};
    f64* log_gauge{};
    u32* lane_flags{};
};

__global__ auto cu_promote(PromoteArgs args) -> void;
__global__ auto cu_pack(PackArgs args) -> void;
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
__global__ auto cu_batched_unit(cuDoubleComplex* values, i64 lane_stride, i64 lanes) -> void;
__global__ auto cu_batched_products(BatchedProductArgs args) -> void;
__global__ auto cu_batched_boundary(BatchedBoundaryArgs args) -> void;
__global__ auto cu_batched_right_block(BatchedRightBlockArgs args) -> void;
__global__ auto cu_batched_select(BatchedSelectArgs args) -> void;
__global__ auto cu_batched_pack_site(BatchedPackSiteArgs args) -> void;
__global__ auto cu_batched_project(BatchedProjectArgs args) -> void;
__global__ auto cu_batched_normalize(BatchedNormalizeArgs args) -> void;
}

#endif
