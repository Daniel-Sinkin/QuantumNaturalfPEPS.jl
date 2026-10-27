#include "parts.cuh"

namespace qn_eloc::env
{

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
