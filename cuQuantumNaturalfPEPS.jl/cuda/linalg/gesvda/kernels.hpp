#pragma once

#include "../eo.cuh"

#include <algorithm>
#include <cstdint>

#include "workspace.hpp"

namespace qn_eloc::gesvda
{

struct CuRecordGesvdaInfoArgs
{
    const int* info;
    int* fail_flag;
    int dim_batch;
};

struct CuCopyFactorOutputsArgs
{
    const cf* q_candidate;
    i64 q_candidate_stride;
    const cf* r_candidate;
    i64 r_candidate_stride;
    cf* r_out;
    i64 r_stride;
    cf* q_out;
    i64 q_stride;
    const int* info;
    int rows;
    int cols;
    int k;
    int dim_batch;
};

struct CuAdjointPanelsArgs
{
    const cf* input;
    i64 input_stride;
    cf* output;
    i64 output_stride;
    int rows;
    int cols;
    int dim_batch;
};

struct CuPanelTotalWeightArgs
{
    const cf* panel;
    i64 panel_stride;
    int elements;
    f64* total_weight;
    int dim_batch;
};

struct CuSingularComponentWeightArgs
{
    const f32* singular;
    int k;
    f64* component_weight;
    int dim_batch;
};

struct CuSelectEffectiveRankArgs
{
    const f64* total_weight;
    const f64* component_weight;
    int* effective_rank;
    int k;
    f64 cutoff;
    int dim_batch;
};

struct CuMaskFactorTailsArgs
{
    cf* q;
    i64 q_stride;
    cf* r;
    i64 r_stride;
    const int* effective_rank;
    int rows;
    int cols;
    int k;
    int dim_batch;
};

__global__ auto cu_record_gesvda_info(CuRecordGesvdaInfoArgs args) -> void;

__global__ auto cu_copy_factor_outputs(CuCopyFactorOutputsArgs args) -> void;

__global__ auto cu_adjoint_panels(CuAdjointPanelsArgs args) -> void;

__global__ auto cu_panel_total_weight(CuPanelTotalWeightArgs args) -> void;

__global__ auto cu_singular_component_weight(CuSingularComponentWeightArgs args) -> void;

__global__ auto cu_select_effective_rank(CuSelectEffectiveRankArgs args) -> void;

__global__ auto cu_mask_factor_tails(CuMaskFactorTailsArgs args) -> void;

}
