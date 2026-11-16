#ifndef QNPEPS_DENSITYMATRIX_BATCHED_CUH
#define QNPEPS_DENSITYMATRIX_BATCHED_CUH

#include "core/arena_cursor.cuh"
#include "densitymatrix/kernels.cuh"
#include "linalg/linalg.cuh"

#include <span>
#include <vector>

namespace qnpeps::densitymatrix
{
struct BatchedSite
{
    i64 state_left{};
    i64 physical_input{};
    i64 state_right{};
    i64 operator_left{};
    i64 physical_output{};
    i64 operator_right{};
    OperatorStrides operator_strides{};
    const cuFloatComplex* state{};
    const cuFloatComplex* operator_values{};
    cuFloatComplex* output{};
};

struct BatchedGeometry
{
    i64 num_sites{};
    i64 state_bond{};
    i64 operator_bond{};
    i64 output_dimension{};
    i64 upper_bond{};
    i64 lanes{};
};

struct BatchedLayout
{
    i64 combined{};
    i64 order{};
    i64 product_lane{};
    i64 environment_cut{};
    i64 environment_lane{};
    i64 block_lane{};
    i64 density_lane{};
    i64 carried_lane{};
    i64 spectrum_lane{};
};

struct BatchedWorkspace
{
    BatchedGeometry geometry{};
    BatchedLayout layout{};
    cuDoubleComplex* products{};
    cuDoubleComplex* left_environments{};
    cuDoubleComplex* right_block{};
    cuDoubleComplex* environment_product{};
    cuDoubleComplex* density{};
    cuDoubleComplex* carried{};
    f64* eigenvalues{};
    i32* sorted_indices{};
    void* solver_workspace{};
    usize solver_workspace_bytes{};
    i32* solver_information{};
    u32* lane_flags{};
};

struct BatchedRow
{
    std::span<const BatchedSite> sites{};
    i64 state_lane_stride{};
    i64 operator_lane_stride{};
    i64 output_lane_stride{};
    f64* log_gauge{};
    int lanes{};
};

[[nodiscard]] auto batched_layout(const BatchedGeometry& geometry) noexcept -> BatchedLayout;
[[nodiscard]] auto _take_batched_workspace_check(const BatchedGeometry& geometry) noexcept -> bool;
[[nodiscard]] auto take_batched_workspace(
    Linalg& linalg, ArenaCursor& arena, const BatchedGeometry& geometry
) -> BatchedWorkspace;
[[nodiscard]] auto kept_ranks(std::span<const BatchedSite> sites, i64 upper_bond)
    -> std::vector<i64>;
[[nodiscard]] auto _apply_batched_check(
    const BatchedWorkspace& workspace, const BatchedRow& row, bool has_state
) noexcept -> bool;
[[nodiscard]] auto apply_batched(
    Linalg& linalg, const BatchedWorkspace& workspace, const BatchedRow& row
) -> std::vector<i64>;
[[nodiscard]] auto normalize_batched(
    Linalg& linalg, const BatchedWorkspace& workspace, const BatchedRow& row
) -> std::vector<i64>;
auto clear_lane_flags(Linalg& linalg, const BatchedWorkspace& workspace) -> void;
[[nodiscard]] auto read_lane_flags(Linalg& linalg, const BatchedWorkspace& workspace, int lanes)
    -> std::vector<u32>;
}

#endif
