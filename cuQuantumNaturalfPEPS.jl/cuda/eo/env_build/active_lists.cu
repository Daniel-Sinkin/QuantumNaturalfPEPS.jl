#include "parts.cuh"

namespace qn_eloc::env
{

__global__ auto cu_build_active_lists(CuActiveListArgs args) -> void
{
    const auto slot = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (slot >= static_cast<usize>(args.slot_count)) return;
    const auto& active_slot = args.slots[slot];
    const auto lanes = static_cast<usize>(args.lanes);
    const auto first_index = slot * static_cast<usize>(args.max_lanes);
    auto count = 0;
    for (auto lane = 0_uz; lane < lanes; ++lane)
    {
        const auto lane_sample = args.samples + lane * static_cast<usize>(args.lane_stride);
        const auto state = local_state_index(active_slot.local_state, lane_sample);
        if (active_slot.active[state] == 0) continue;
        args.indices[first_index + static_cast<usize>(count)] = static_cast<int>(lane);
        ++count;
    }
    args.counts[slot] = count;
}

auto active_lists_setup(
    ActiveLists& lists,
    const std::vector<TermPiece>& pieces,
    int max_lanes,
    qnpeps::ArenaCursor& context_arena
) -> void
{
    auto slots = 0;
    for (const auto& piece : pieces)
        if (piece.active_slot >= 0) ++slots;
    lists.slots = slots;
    lists.max_lanes = max_lanes;
    lists.host_counts.assign(static_cast<usize>(slots), 0);
    if (slots == 0) return;
    usize index_count{};
    const auto slot_count = static_cast<std::uint64_t>(slots);
    if (max_lanes < 0) return;
    arena_product({slot_count, static_cast<std::uint64_t>(max_lanes)}, index_count);
    lists.counts = context_arena.take<int>(static_cast<usize>(slots));
    lists.indices = context_arena.take<int>(index_count);
}

auto upload_active_slots(
    ActiveLists& lists,
    const std::vector<TermPiece>& pieces,
    const u8* device_active,
    qnpeps::ArenaCursor& context_arena
) -> void
{
    if (lists.slots == 0) return;
    auto slots = std::vector<CuActiveSlot>{};
    for (const auto& piece : pieces)
    {
        if (piece.active_slot < 0) continue;
        slots.push_back({piece.local_state, device_active + piece.state_offset});
    }
    usize slot_bytes{};
    arena_product({sizeof(CuActiveSlot), static_cast<std::uint64_t>(slots.size())}, slot_bytes);
    lists.device_slots = context_arena.take<CuActiveSlot>(slots.size());
    check(cudaMemcpy(lists.device_slots, slots.data(), slot_bytes, cudaMemcpyHostToDevice));
}

auto prepare_active_lists(
    ActiveLists& lists, const u8* samples, i64 lane_stride, int lanes, cudaStream_t stream
) -> void
{
    if (lists.slots == 0) return;
    constexpr auto threads = 256;
    const auto blocks = (lists.slots + threads - 1) / threads;
    const auto args = CuActiveListArgs{
        .samples = samples,
        .lane_stride = lane_stride,
        .slots = lists.device_slots,
        .slot_count = lists.slots,
        .lanes = lanes,
        .max_lanes = lists.max_lanes,
        .indices = lists.indices,
        .counts = lists.counts,
    };
    cu_build_active_lists<<<blocks, threads, 0, stream>>>(args);
    check(cudaGetLastError());
    usize count_bytes{};
    arena_product({sizeof(int), static_cast<std::uint64_t>(lists.slots)}, count_bytes);
    check(cudaMemcpyAsync(
        lists.host_counts.data(), lists.counts, count_bytes, cudaMemcpyDeviceToHost, stream
    ));
    check(cudaStreamSynchronize(stream));
}

}
