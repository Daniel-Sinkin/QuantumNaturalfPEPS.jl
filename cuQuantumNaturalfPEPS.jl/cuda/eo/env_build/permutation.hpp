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

#line 25 "cuda/eo/env_build.cu"
using namespace qnpeps;

namespace
{

constexpr int k_rangefinder_recovery_limit{4};

__global__ auto cu_gather(
    cf* out,
    const cf* in,
    const int* gather_indices,
    int n,
    i64 stride_out,
    i64 stride_in,
    int conjugate,
    int dim_batch
) -> void
{
    const auto n_i64 = i64{static_cast<i64>(n)};
    const auto total = i64{n_i64 * dim_batch};
    const auto grid_stride = i64{static_cast<i64>(gridDim.x) * blockDim.x};
    for (auto tid = static_cast<i64>(blockIdx.x * blockDim.x + threadIdx.x); tid < total;
         tid += grid_stride)
    {
        const auto lane = int{static_cast<int>(tid / n)};
        const auto elem = int{static_cast<int>(tid % n)};
        const auto lane_i64 = i64{static_cast<i64>(lane)};
        const auto in_idx = i64{lane_i64 * stride_in + gather_indices[elem]};
        const auto out_idx = i64{lane_i64 * stride_out + elem};
        auto value = static_cast<cf>(in[in_idx]);
        if (conjugate) value.im = -value.im;
        out[out_idx] = value;
    }
}

auto perm_key(const std::vector<int>& dims, const std::vector<int>& perm) -> std::string
{
    auto result = static_cast<std::string>("p");
    for (const int dim : dims)
        result += "_" + std::to_string(dim);
    result += "x";
    for (const int axis : perm)
        result += "_" + std::to_string(axis);
    return result;
}

struct PermutationCache
{
    std::map<std::string, int*> cache;
    auto get(const std::vector<int>& dims, const std::vector<int>& perm) -> int*
    {
        const auto key = perm_key(dims, perm);
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;
        const auto gather_indices = permutation_index_map(dims, perm);
        if (gather_indices.size() > std::numeric_limits<usize>::max() / sizeof(int))
        {
            qn::set_err(QNPEPS_ELOC_ERR_OOM);
            return nullptr;
        }
        const auto bytes = usize{sizeof(int) * gather_indices.size()};
        int* device_ptr{};
        CUDA_CHECK(cudaMalloc(&device_ptr, bytes));
        CUDA_CHECK(cudaMemcpy(device_ptr, gather_indices.data(), bytes, cudaMemcpyHostToDevice));
        cache.emplace(key, device_ptr);
        return device_ptr;
    }
    auto get_inverse(const std::vector<int>& dims, const std::vector<int>& perm) -> int*
    {
        const auto key = "i" + perm_key(dims, perm);
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;
        const auto source_indices = permutation_index_map(dims, perm);
        auto destination_indices = std::vector<int>(source_indices.size());
        for (auto out = usize{0}; out < source_indices.size(); ++out)
            destination_indices[static_cast<usize>(source_indices[out])] = static_cast<int>(out);
        if (destination_indices.size() > std::numeric_limits<usize>::max() / sizeof(int))
        {
            qn::set_err(QNPEPS_ELOC_ERR_OOM);
            return nullptr;
        }
        const auto bytes = usize{sizeof(int) * destination_indices.size()};
        int* device_ptr{};
        CUDA_CHECK(cudaMalloc(&device_ptr, bytes));
        CUDA_CHECK(
            cudaMemcpy(device_ptr, destination_indices.data(), bytes, cudaMemcpyHostToDevice)
        );
        cache.emplace(key, device_ptr);
        return device_ptr;
    }
    auto release() -> void
    {
        for (auto& entry : cache)
            maybe_free_device(entry.second);
        cache.clear();
    }
};

inline auto packed_producers_from_env() -> unsigned
{
    static thread_local const auto mask = []
    {
        auto value{std::getenv("QNPEPS_ELOC_PACKED_PRODUCERS")};
        if (not value or value[0] == '\0') return 0u;
        return static_cast<unsigned>(std::strtoul(value, nullptr, 0)) & 7u;
    }();
    return mask;
}

auto permuted_dims(const std::vector<int>& dims, const std::vector<int>& perm) -> std::vector<int>
{
    std::vector<int> result{};
    result.reserve(perm.size());
    for (const int axis : perm)
        result.push_back(dims[static_cast<usize>(axis)]);
    return result;
}

struct PermuteOp
{
    EoDeviceBuffer dst{};
    EoDeviceBuffer src{};
    std::vector<int> dims_in{};
    std::vector<int> perm{};
    int batch{};
    const char* kind{"perm"};
    int m{};
    int n{};
    int k{};
};

auto device_permute(Linalg& linalg, PermutationCache& lookup_tables, const PermuteOp& op) -> void
{
    auto element_count = i64{1};
    for (const int dim : op.dims_in)
    {
        if (dim < 1 or element_count > std::numeric_limits<int>::max() / dim)
        {
            qn::set_err(QNPEPS_ELOC_ERR_OOM);
            return;
        }
        element_count *= dim;
    }
    if (op.batch < 1 or element_count > std::numeric_limits<i64>::max() / op.batch)
    {
        qn::set_err(QNPEPS_ELOC_ERR_OOM);
        return;
    }
    auto gather_indices = lookup_tables.get(op.dims_in, op.perm);
    if (qn::err_state() != QNPEPS_ELOC_OK or not gather_indices) return;
    const auto threads = int{256};
    const auto blocks = int{
        static_cast<int>(std::min<i64>(4096, (element_count * op.batch + threads - 1) / threads))
    };
    cu_gather<<<blocks, threads, 0, linalg.stream()>>>(
        op.dst.p,
        op.src.p,
        gather_indices,
        static_cast<int>(element_count),
        op.dst.stride,
        op.src.stride,
        0,
        op.batch
    );
    CUDA_CHECK(cudaGetLastError());
}

}
