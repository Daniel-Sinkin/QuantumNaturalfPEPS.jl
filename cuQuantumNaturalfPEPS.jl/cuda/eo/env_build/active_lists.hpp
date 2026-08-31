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
#include "projection_kernels.hpp"
#include "arena.hpp"
#include "worker.hpp"
#include "environment_rows.hpp"
#include "tensor_network.hpp"

namespace qn_eloc::env
{

#line 2301 "cuda/eo/env_build.cu"

enum class Bucket
{
    horizontal,
    fourbody,
    longer_horizontal
};

struct FlipInst
{
    Bucket bucket{};
    int active_slot{-1};
    int j2_group{-1};
    int j2_column_group{-1};
    int n_flips{};
    qnpeps::CuArray<int, 4> site{};
    qnpeps::CuArray<int, 4> value{};
    int mask_a{};
    int mask_b{};
    f64 coeff_re{};
    f64 coeff_im{};
};

struct ActiveLists
{
    int slots{};
    int max_lanes{};
    int* masks{};
    int* counts{};
    int* indices{};
    bool owns_device{};
    std::vector<int> host_counts{};
};

inline auto active_compact_requested_from_env() -> bool
{
    auto value{std::getenv("QNPEPS_ELOC_ACTIVE_COMPACT")};
    return value == nullptr or value[0] == '\0' or value[0] != '0';
}

inline auto active_compact_min_from_env() -> int
{
    auto value{std::getenv("QNPEPS_ELOC_ACTIVE_MIN")};
    return value and value[0] != '\0' ? std::max(1, std::atoi(value)) : 2;
}

auto assign_active_slots(std::vector<FlipInst>& terms, bool enabled) -> int
{
    int slots{};
    if (not enabled) return slots;
    for (FlipInst& term : terms)
        if (term.mask_a >= 0 and term.mask_b >= 0) term.active_slot = slots++;
    return slots;
}

__global__ auto cu_build_active_lists(
    const u8* samples,
    i64 lane_stride,
    const int* mask_a,
    const int* mask_b,
    int slots,
    int lanes,
    int max_lanes,
    int* indices,
    int* counts
) -> void
{
    const auto slot = int{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
    if (slot >= slots) return;
    int count{};
    for (auto lane = int{0}; lane < lanes; ++lane)
    {
        const auto base = i64{static_cast<i64>(lane) * lane_stride};
        if (samples[base + mask_a[slot]] != samples[base + mask_b[slot]])
            indices[static_cast<i64>(slot) * max_lanes + count++] = lane;
    }
    counts[slot] = count;
}

auto active_lists_setup(
    ActiveLists& lists,
    const std::vector<FlipInst>& terms,
    int slots,
    int max_lanes,
    ContextArena& context_arena
) -> void
{
    lists.slots = slots;
    lists.max_lanes = max_lanes;
    lists.host_counts.assign(static_cast<usize>(slots), 0);
    if (slots == 0) return;

    usize mask_count{};
    usize index_count{};
    if (slots < 0 or max_lanes < 0
        or not arena_product({2u, static_cast<std::uint64_t>(slots)}, mask_count)
        or not arena_product(
            {static_cast<std::uint64_t>(slots), static_cast<std::uint64_t>(max_lanes)}, index_count
        ))
        return;
    auto masks = std::vector<int>(mask_count);
    for (const FlipInst& term : terms)
    {
        if (term.active_slot >= 0)
        {
            masks[static_cast<usize>(term.active_slot)] = term.mask_a;
            masks[static_cast<usize>(slots + term.active_slot)] = term.mask_b;
        }
    }
    usize mask_bytes{};
    if (not arena_product({sizeof(int), masks.size()}, mask_bytes)) return;
    lists.masks = context_arena.take<int>(masks.size());
    lists.counts = context_arena.take<int>(static_cast<usize>(slots));
    lists.indices = context_arena.take<int>(index_count);
    lists.owns_device = false;
    if (qn::err_state() != QNPEPS_ELOC_OK) return;
    CUDA_CHECK(cudaMemcpy(lists.masks, masks.data(), mask_bytes, cudaMemcpyHostToDevice));
}

auto active_lists_teardown(ActiveLists& lists) -> void
{
    if (lists.owns_device)
    {
        maybe_free_device(lists.masks);
        maybe_free_device(lists.counts);
        maybe_free_device(lists.indices);
    }
    lists = ActiveLists{};
}

auto active_lists_rearm(ActiveLists& lists, const std::vector<FlipInst>& terms, cudaStream_t stream)
    -> void
{
    if (lists.slots == 0) return;
    usize mask_count{};
    usize mask_bytes{};
    if (not arena_product({2u, static_cast<std::uint64_t>(lists.slots)}, mask_count)
        or not arena_product({sizeof(int), mask_count}, mask_bytes))
        return;
    auto masks = std::vector<int>(mask_count);
    for (const FlipInst& term : terms)
    {
        if (term.active_slot >= 0)
        {
            masks[static_cast<usize>(term.active_slot)] = term.mask_a;
            masks[static_cast<usize>(lists.slots + term.active_slot)] = term.mask_b;
        }
    }
    CUDA_CHECK(
        cudaMemcpyAsync(lists.masks, masks.data(), mask_bytes, cudaMemcpyHostToDevice, stream)
    );
}

auto prepare_active_lists(
    ActiveLists& lists, const u8* samples, i64 lane_stride, int lanes, cudaStream_t stream
) -> void
{
    if (lists.slots == 0) return;
    const auto threads = int{256};
    const auto blocks = int{(lists.slots + threads - 1) / threads};
    cu_build_active_lists<<<blocks, threads, 0, stream>>>(
        samples,
        lane_stride,
        lists.masks,
        lists.masks + lists.slots,
        lists.slots,
        lanes,
        lists.max_lanes,
        lists.indices,
        lists.counts
    );
    CUDA_CHECK(cudaGetLastError());
    usize count_bytes{};
    if (not arena_product({sizeof(int), static_cast<std::uint64_t>(lists.slots)}, count_bytes))
        return;
    CUDA_CHECK(cudaMemcpyAsync(
        lists.host_counts.data(), lists.counts, count_bytes, cudaMemcpyDeviceToHost, stream
    ));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}
