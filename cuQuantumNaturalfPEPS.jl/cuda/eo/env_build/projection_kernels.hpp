#pragma once

#include "common.cuh"
#include "core/arena_cursor.cuh"
#include "core/complex.cuh"
#include "core/defer.cuh"
#include "core/session.cuh"
#include "density.cuh"
#include "dtensor.cuh"
#include "eloc_kernels.cuh"
#include "env_build.cuh"
#include "../linalg/eo.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <new>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "permutation.hpp"

namespace
{

#line 193 "cuda/eo/env_build.cu"

__global__ auto cu_project_slab(
    const cf* site,
    const u8* samples,
    i64 lane_stride,
    int site_idx,
    i64 slab,
    cf* out,
    i64 out_stride,
    int lanes
) -> void
{
    const auto total = i64{slab * lanes};
    const auto grid_stride = i64{static_cast<i64>(gridDim.x) * blockDim.x};
    for (auto tid = static_cast<i64>(blockIdx.x * blockDim.x + threadIdx.x); tid < total;
         tid += grid_stride)
    {
        const auto lane = i64{tid / slab};
        const auto elem = i64{tid % slab};
        const auto spin = i64{static_cast<i64>(samples[lane * lane_stride + site_idx])};
        out[lane * out_stride + elem] = site[spin * slab + elem];
    }
}

__global__ auto cu_project_slab_dual_packed(
    const cf* site,
    const u8* samples,
    i64 lane_stride,
    int site_idx,
    i64 slab,
    const int* destination_indices,
    cf* native_out,
    i64 native_stride,
    cf* packed_out,
    i64 packed_stride,
    int lanes
) -> void
{
    const auto total = i64{slab * lanes};
    const auto grid_stride = i64{static_cast<i64>(gridDim.x) * blockDim.x};
    for (auto tid = static_cast<i64>(blockIdx.x * blockDim.x + threadIdx.x); tid < total;
         tid += grid_stride)
    {
        const auto lane = i64{tid / slab};
        const auto native_elem = int{static_cast<int>(tid % slab)};
        const auto spin = i64{static_cast<i64>(samples[lane * lane_stride + site_idx])};
        const auto value = cf{site[spin * slab + native_elem]};
        native_out[lane * native_stride + native_elem] = value;
        packed_out[lane * packed_stride + destination_indices[native_elem]] = value;
    }
}

__global__ auto cu_scale_last(cf* site, i64 stride, i64 n, const cf* r, i64 rstride, int lanes)
    -> void
{
    const auto total = i64{n * lanes};
    const auto grid_stride = i64{static_cast<i64>(gridDim.x) * blockDim.x};
    for (auto tid = static_cast<i64>(blockIdx.x * blockDim.x + threadIdx.x); tid < total;
         tid += grid_stride)
    {
        const auto lane = i64{tid / n};
        const auto elem = i64{tid % n};
        const auto s = static_cast<cf>(r[lane * rstride]);
        const auto v = static_cast<cf>(site[lane * stride + elem]);
        site[lane * stride + elem] = cf{v.re * s.re - v.im * s.im, v.re * s.im + v.im * s.re};
    }
}

__global__ auto cu_log_finish(
    const cf* acc,
    i64 astride,
    const f64* f_top,
    const f64* f_down,
    const f64* f_ov,
    f64* out,
    int lanes
) -> void
{
    const auto lane = int{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
    if (lane >= lanes) return;
    const auto z = static_cast<cf>(acc[lane * astride]);
    const auto re = f64{static_cast<f64>(z.re)};
    const auto im = f64{static_cast<f64>(z.im)};
    out[2 * lane] = 0.5 * log(norm2(re, im)) + f_top[lane] + f_down[lane] + f_ov[lane];
    out[2 * lane + 1] = atan2(im, re);
}
}
