#pragma once

#include "../experimental_svd.cuh"
#include "../eo.cuh"

#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda/std/cmath>

#include "core/complex.cuh"
#include "factorization_kernels.hpp"
#include "rangefinder_routes.hpp"

namespace qnpeps
{

#line 1478 "cuda/linalg/eo_routes.cu"

auto rangefinder_run_production_gate(
    Linalg& la,
    MatrixBatch panel,
    int rank,
    const cf* omega,
    MatrixBatch q_out,
    MatrixBatch r_out,
    int dim_batch,
    EoDeviceBuffer sketch,
    EoDeviceBuffer proj,
    EoDeviceBuffer gram,
    cf** gram_ptrs,
    cf** sketch_ptrs,
    int* info,
    int* fail_flag,
    int* failure_log,
    const int* fallback_info,
    void* robust_scratch,
    usize robust_scratch_bytes
) -> bool
{
    batched_rangefinder(
        la,
        panel,
        rank,
        false,
        omega,
        q_out,
        r_out,
        dim_batch,
        sketch,
        proj,
        gram,
        gram_ptrs,
        sketch_ptrs,
        info,
        fail_flag,
        failure_log,
        fallback_info,
        robust_scratch,
        robust_scratch_bytes
    );
    return qn::err_state() == QNPEPS_ELOC_OK;
}

auto rangefinder_experimental_carve(
    Linalg& la,
    ArenaCursor& arena,
    int max_dim,
    int rank,
    int batch,
    char*& scratch,
    usize& scratch_bytes
) -> void
{
    qn_eloc::e0191::carve_rangefinder_workspace(
        la, arena, max_dim, rank, batch, scratch, scratch_bytes
    );
}

__global__ auto cu_fill_first_one(cf* x, i64 stride, int n, int dim_batch) -> void
{
    const auto n_i64 = i64{static_cast<i64>(n)};
    const auto total = i64{n_i64 * dim_batch};
    const auto grid_stride = i64{static_cast<i64>(gridDim.x) * blockDim.x};
    for (auto tid = static_cast<i64>(blockIdx.x * blockDim.x + threadIdx.x); tid < total;
         tid += grid_stride)
    {
        const auto elem = int{static_cast<int>(tid % n)};
        const auto lane = i64{tid / n};
        const auto out_idx = i64{lane * stride + elem};
        x[out_idx] = (elem == 0) ? cf{1.0f, 0.0f} : cf{0.0f, 0.0f};
    }
}
}
