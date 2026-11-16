#include "core/defer.cuh"
#include "core/error.hpp"
#include "eo/shards.cuh"
#include "core/predicates.cuh"
#include "linalg/linalg.cuh"
#include "linalg/transfer.cuh"
#include "minsr/distributed.cuh"
#include "minsr/solve.cuh"

#include <algorithm>
#include <cuda_runtime.h>
#include <optional>
#include <vector>

namespace qnpeps::minsr
{
namespace
{
constexpr i64 k_peer_tile_bytes_default{static_cast<i64>(1) << 30};

[[nodiscard]] auto peer_chunk_rows(i64 peer_tile_bytes, i64 compact_count, i64 max_count) noexcept
    -> i64
{
    const auto budget = all_positive(peer_tile_bytes) ? peer_tile_bytes : k_peer_tile_bytes_default;
    const auto nonempty_compact_count = std::max(compact_count, i64{1});
    const auto row_bytes = nonempty_compact_count * static_cast<i64>(sizeof(ComplexF32));
    auto rows = budget / row_bytes;
    rows = std::max(rows, i64{1});
    rows = std::min(rows, max_count);
    return rows;
}

template <class T>
[[nodiscard]] auto device_alloc(i64 count) -> T*
{
    const auto allocation_count = static_cast<usize>(std::max(count, i64{1}));
    void* pointer{};
    check_allocation(cudaMalloc(&pointer, sizeof(T) * allocation_count));
    return static_cast<T*>(pointer);
}
}

auto build_gram_blockrow(const DistributedGramArgs& args) -> void
{
    const auto& destination = *args.destination;
    const auto max_count = [&]
    {
        i64 count{};
        for (const auto& lane : args.lanes)
            count = std::max(count, lane.count);
        return count;
    }();
    const auto chunk_rows = peer_chunk_rows(args.peer_tile_bytes, args.compact, max_count);
    ComplexF32* blockrow{};
    ComplexF32* tile{};
    ComplexF32* visit{};
    const Defer release_buffers{[&]
                                {
                                    unchecked(cudaFree(blockrow));
                                    unchecked(cudaFree(tile));
                                    unchecked(cudaFree(visit));
                                }};
    blockrow = device_alloc<ComplexF32>(destination.count * args.n_samples);
    tile = device_alloc<ComplexF32>(destination.count * max_count);
    visit = device_alloc<ComplexF32>(chunk_rows * args.compact);

    for (const auto& source : args.lanes)
    {
        if (all_nonpositive(source.count)) continue;
        if (source.device == destination.device)
        {
            eo::gram_tile({
                .config = args.config,
                .rows_a = destination.rows,
                .samples_a = destination.samples + destination.base * args.sites,
                .count_a = destination.count,
                .rows_b = destination.rows,
                .samples_b = destination.samples + source.base * args.sites,
                .count_b = source.count,
                .tile = tile,
                .stream = destination.linalg->stream(),
            });
            {
                const auto destination_pitch = static_cast<usize>(args.n_samples);
                const auto source_pitch = static_cast<usize>(source.count);
                const auto height = static_cast<usize>(destination.count);
                copy_device_2d_async(
                    *destination.linalg,
                    blockrow + source.base,
                    destination_pitch,
                    tile,
                    source_pitch,
                    source_pitch,
                    height
                );
                check(cudaStreamSynchronize(destination.linalg->stream()));
            }
            continue;
        }

        const auto source_count = static_cast<usize>(source.count);
        const auto chunk_size = static_cast<usize>(chunk_rows);
        for (auto base = 0_uz; base < source_count; base += chunk_size)
        {
            const auto count = std::min(chunk_size, source_count - base);
            const auto base_offset = static_cast<i64>(base);
            const auto count_value = static_cast<i64>(count);
            const auto compact_count = static_cast<usize>(args.compact);
            const auto visit_count = count * compact_count;
            copy_peer_async(
                *destination.linalg,
                visit,
                destination.device,
                source.rows + base * compact_count,
                source.device,
                visit_count
            );
            check(cudaStreamSynchronize(destination.linalg->stream()));
            eo::gram_tile({
                .config = args.config,
                .rows_a = destination.rows,
                .samples_a = destination.samples + destination.base * args.sites,
                .count_a = destination.count,
                .rows_b = visit,
                .samples_b = destination.samples + (source.base + base_offset) * args.sites,
                .count_b = count_value,
                .tile = tile,
                .stream = destination.linalg->stream(),
            });
            {
                const auto destination_pitch = static_cast<usize>(args.n_samples);
                const auto source_pitch = count;
                const auto height = static_cast<usize>(destination.count);
                copy_device_2d_async(
                    *destination.linalg,
                    blockrow + source.base + base_offset,
                    destination_pitch,
                    tile,
                    source_pitch,
                    source_pitch,
                    height
                );
                check(cudaStreamSynchronize(destination.linalg->stream()));
            }
        }
    }

    {
        const auto blockrow_count = static_cast<usize>(destination.count * args.n_samples);
        copy_peer_async(
            *destination.linalg,
            args.gram_device0 + destination.base * args.n_samples,
            0,
            blockrow,
            destination.device,
            blockrow_count
        );
        check(cudaStreamSynchronize(destination.linalg->stream()));
    }
}

auto scatter_ring(const DistributedScatterArgs& args) -> void
{
    auto accumulators = std::vector<cuDoubleComplex*>{args.lanes.size()};
    auto coefficients = std::vector<cuDoubleComplex*>{args.lanes.size()};
    auto slot_sites = std::vector<i32*>{args.lanes.size()};
    const Defer release_buffers{[&]
                                {
                                    for (auto index = 0_uz; index < args.lanes.size(); ++index)
                                    {
                                        unchecked(cudaSetDevice(args.lanes[index].device));
                                        unchecked(cudaFree(accumulators[index]));
                                        unchecked(cudaFree(coefficients[index]));
                                        unchecked(cudaFree(slot_sites[index]));
                                    }
                                    unchecked(cudaSetDevice(0));
                                }};
    std::optional<usize> previous{};
    const auto lane_count = args.lanes.size();
    const auto sample_count = static_cast<usize>(args.n_samples);
    const auto compact_count = static_cast<usize>(args.compact);
    const auto dense_count = static_cast<usize>(args.dense);
    const auto site_count = static_cast<int>(args.sites);

    for (auto index = 0_uz; index < lane_count; ++index)
    {
        const auto& lane = args.lanes[index];
        if (all_nonpositive(lane.count)) continue;
        check(cudaSetDevice(lane.device));
        accumulators[index] = device_alloc<cuDoubleComplex>(args.dense);
        coefficients[index] = device_alloc<cuDoubleComplex>(args.n_samples);
        slot_sites[index] = device_alloc<i32>(args.compact);
        upload_async(*lane.linalg, coefficients[index], args.coefficients.data(), sample_count);
        upload_async(*lane.linalg, slot_sites[index], args.slot_site.data(), compact_count);
        if (not previous)
        {
            zero_async(*lane.linalg, accumulators[index], dense_count);
        }
        else
        {
            copy_peer_async(
                *lane.linalg,
                accumulators[index],
                lane.device,
                accumulators[*previous],
                args.lanes[*previous].device,
                dense_count
            );
            check(cudaDeviceSynchronize());
        }
        launch_scatter(
            *lane.linalg,
            {
                .accumulator = accumulators[index],
                .rows = lane.rows,
                .row_base = lane.base,
                .compact_count = args.compact,
                .coefficients = coefficients[index],
                .samples = lane.samples,
                .slot_sites = slot_sites[index],
                .site_count = site_count,
                .physical_dimension = args.dim_phys,
                .row_begin = static_cast<int>(lane.base),
                .row_end = static_cast<int>(lane.base + lane.count),
            }
        );
        check(cudaDeviceSynchronize());
        previous = index;
    }

    {
        check(cudaSetDevice(0));
        auto* const accumulator0 = accumulators[0];
        if (previous and *previous != 0)
        {
            copy_peer_async(
                *args.lanes[0].linalg,
                accumulator0,
                0,
                accumulators[*previous],
                args.lanes[*previous].device,
                dense_count
            );
            check(cudaDeviceSynchronize());
        }
        launch_cast_accumulator(
            *args.lanes[0].linalg, args.theta_device0, accumulator0, args.dense
        );
        check(cudaStreamSynchronize(args.lanes[0].linalg->stream()));
    }
}
}
