#ifndef QNPEPS_DENSITYMATRIX_TYPES_CUH
#define QNPEPS_DENSITYMATRIX_TYPES_CUH

#include "core/arena_cursor.cuh"
#include "capi/qnpeps.h"
#include "linalg/linalg.cuh"

#include <span>

namespace qnpeps::densitymatrix
{
struct Site
{
    usize site{};
    usize num_sites{};
    i64 state_left{};
    i64 physical_input{};
    i64 state_right{};
    i64 operator_left{};
    i64 physical_output{};
    i64 operator_right{};
    usize state_offset{};
    usize operator_offset{};
    usize output_offset{};
};

struct Workspace
{
    cuDoubleComplex* left_environments{};
    cuDoubleComplex* right_blocks{};
    cuDoubleComplex* temporaries{};
    cuDoubleComplex* density{};
    f64* eigenvalues{};
    f64* sorted_eigenvalues{};
    i32* sort_indices{};
    i32* active_ranks{};
    QnpepsDensityRankRecord* records{};
    void* solver_workspace{};
    usize solver_workspace_bytes{};
    i32* solver_information{};
};

using Product = void (*)(
    Linalg&,
    const Site&,
    i64,
    const cuDoubleComplex*,
    const cuDoubleComplex*,
    cuDoubleComplex*,
    const void*
);

struct Apply
{
    QnpepsDensitySettings settings{};
    i64 upper_bond{};
    bool normalize{};
    f64 input_gauge{};
    std::span<const Site> sites{};
    Workspace workspace{};
    const cuDoubleComplex* state_values{};
    usize state_value_count{};
    const cuDoubleComplex* operator_values{};
    usize operator_value_count{};
    i32* result_dimensions{};
    usize result_dimension_count{};
    cuDoubleComplex* result_values{};
    usize result_value_count{};
    f64* normalization_log{};
    f64* output_gauge{};
    i32* active_ranks_out{};
    QnpepsDensityRankRecord* records_out{};
    Product product{};
    const void* product_context{};
};

struct Geometry
{
    i64 num_sites{};
    i64 input_bond{};
    i64 operator_bond{};
    i64 output_dimension{};
    i64 upper_bond{};
    i64 lanes{1};
};

auto take_workspace(Linalg& linalg, ArenaCursor& arena, const Geometry& geometry) -> Workspace;
auto apply(Linalg& linalg, const Apply& args) -> void;
}

#endif
