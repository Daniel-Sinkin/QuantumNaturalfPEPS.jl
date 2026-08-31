#pragma once

#include "../eo.cuh"

#include <algorithm>
#include <cstdint>

#include "workspace.hpp"

namespace qn_eloc::e0191
{

#line 90 "cuda/linalg/experimental_svd.cuh"

struct CuRecordGesvdaInfoArgs
{
    const int* info;
    int* failure_log;
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
    const int* replay_mask;
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

__global__ auto cu_record_gesvda_info(CuRecordGesvdaInfoArgs args) -> void
{
    const auto lane = int{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
    if (lane >= args.dim_batch or args.info[lane] == 0) return;
    const auto encoded = int{args.info[lane] > 0 ? args.info[lane] : 1000000 - args.info[lane]};
    if (args.failure_log) atomicCAS(args.failure_log + lane, 0, encoded);
    if (args.fail_flag) atomicOr(args.fail_flag, 1);
}

__global__ auto cu_copy_factor_outputs(CuCopyFactorOutputsArgs args) -> void
{
    const auto q_count = i64{static_cast<i64>(args.rows) * args.k};
    const auto r_count = i64{static_cast<i64>(args.k) * args.cols};
    const auto lane_count = i64{q_count + r_count};
    const auto total = i64{lane_count * args.dim_batch};
    const auto grid_stride = i64{static_cast<i64>(gridDim.x) * blockDim.x};
    for (auto index = i64{static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x}; index < total;
         index += grid_stride)
    {
        const auto lane = int{static_cast<int>(index / lane_count)};
        if (args.info[lane] != 0) continue;
        if (args.replay_mask and args.replay_mask[lane] == 0
            and args.replay_mask[args.dim_batch + lane] == 0)
            continue;
        const auto element = i64{index % lane_count};
        if (element < q_count)
        {
            args.q_out[static_cast<i64>(lane) * args.q_stride + element] =
                args.q_candidate[static_cast<i64>(lane) * args.q_candidate_stride + element];
            continue;
        }
        const auto r_element = i64{element - q_count};
        args.r_out[static_cast<i64>(lane) * args.r_stride + r_element] =
            args.r_candidate[static_cast<i64>(lane) * args.r_candidate_stride + r_element];
    }
}

__global__ auto cu_adjoint_panels(CuAdjointPanelsArgs args) -> void
{
    const auto lane_count = i64{static_cast<i64>(args.rows) * args.cols};
    const auto total = i64{lane_count * args.dim_batch};
    const auto grid_stride = i64{static_cast<i64>(gridDim.x) * blockDim.x};
    for (auto index = i64{static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x}; index < total;
         index += grid_stride)
    {
        const auto lane = i64{index / lane_count};
        const auto element = i64{index % lane_count};
        const auto row = int{static_cast<int>(element % args.cols)};
        const auto col = int{static_cast<int>(element / args.cols)};
        const auto value =
            cf{args.input[lane * args.input_stride + col + static_cast<i64>(row) * args.rows]};
        args.output[lane * args.output_stride + row + static_cast<i64>(col) * args.cols] =
            cf{value.re, -value.im};
    }
}

__global__ auto cu_panel_total_weight(CuPanelTotalWeightArgs args) -> void
{
    __shared__ qnpeps::CuArray<f64, 256> partial;
    const auto lane = int{static_cast<int>(blockIdx.x)};
    f64 sum{};
    if (lane < args.dim_batch)
    {
        for (auto element = int{static_cast<int>(threadIdx.x)}; element < args.elements;
             element += static_cast<int>(blockDim.x))
        {
            const auto value = cf{args.panel[static_cast<i64>(lane) * args.panel_stride + element]};
            sum += static_cast<f64>(value.re) * value.re + static_cast<f64>(value.im) * value.im;
        }
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (auto offset = int{128}; offset > 0; offset /= 2)
    {
        if (threadIdx.x < offset) partial[threadIdx.x] += partial[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0 and lane < args.dim_batch) args.total_weight[lane] = partial[0];
}

__global__ auto cu_singular_component_weight(CuSingularComponentWeightArgs args) -> void
{
    const auto index = int{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
    const auto total = int{args.k * args.dim_batch};
    if (index >= total) return;
    const auto value = f64{args.singular[index]};
    args.component_weight[index] = value * value;
}

__global__ auto cu_select_effective_rank(CuSelectEffectiveRankArgs args) -> void
{
    const auto lane = int{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
    if (lane >= args.dim_batch) return;
    const auto total = f64{args.total_weight[lane]};
    f64 captured{};
    for (int component{}; component < args.k; ++component)
        captured += args.component_weight[static_cast<i64>(lane) * args.k + component];
    auto discarded = f64{fmax(0.0, total - captured)};
    const auto budget = f64{args.cutoff * total};
    auto retained = int{args.k};
    for (auto component = int{args.k - 1}; component >= 1; --component)
    {
        const auto weight = f64{args.component_weight[static_cast<i64>(lane) * args.k + component]};
        if (discarded + weight > budget) break;
        discarded += weight;
        retained = component;
    }
    args.effective_rank[lane] = retained;
}

__global__ auto cu_mask_factor_tails(CuMaskFactorTailsArgs args) -> void
{
    const auto q_count = i64{static_cast<i64>(args.rows) * args.k};
    const auto r_count = i64{static_cast<i64>(args.k) * args.cols};
    const auto lane_count = i64{q_count + r_count};
    const auto total = i64{lane_count * args.dim_batch};
    const auto grid_stride = i64{static_cast<i64>(gridDim.x) * blockDim.x};
    for (auto index = i64{static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x}; index < total;
         index += grid_stride)
    {
        const auto lane = int{static_cast<int>(index / lane_count)};
        const auto element = i64{index % lane_count};
        if (element < q_count)
        {
            const auto component = int{static_cast<int>(element / args.rows)};
            if (component >= args.effective_rank[lane])
                args.q[static_cast<i64>(lane) * args.q_stride + element] = cf{};
            continue;
        }
        const auto r_element = i64{element - q_count};
        const auto component = int{static_cast<int>(r_element % args.k)};
        if (component >= args.effective_rank[lane])
            args.r[static_cast<i64>(lane) * args.r_stride + r_element] = cf{};
    }
}

}
