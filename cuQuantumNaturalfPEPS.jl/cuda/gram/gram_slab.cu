#include "core/error.hpp"
#include "gram/gram_cublas.cuh"
#include "gram/gram_slab.cuh"
#include "linalg/linalg.cuh"

#include <algorithm>
#include <climits>

namespace qnpeps::gram_slab
{
namespace
{
constexpr i64 k_workspace_bytes{static_cast<i64>(1) << 30};

struct Slab
{
    int first_site{};
    int next_site{};
    int compact_begin{};
    int compact_count{};
    int dense_width{};
};

auto next_slab(const Geometry& geometry, i64 slab_width, int first_site) -> Slab
{
    Slab slab{};
    slab.first_site = first_site;
    slab.compact_begin = geometry.offsets_host[first_site];
    int compact_end{slab.compact_begin};
    int site{first_site};
    while (site < geometry.sites)
    {
        const int candidate{geometry.offsets_host[site] + geometry.slices_host[site]};
        if (site > first_site and 2ll * (candidate - slab.compact_begin) > slab_width) break;
        compact_end = candidate;
        ++site;
    }
    slab.next_site = site;
    slab.compact_count = compact_end - slab.compact_begin;
    slab.dense_width = 2 * slab.compact_count;
    return slab;
}

auto valid_geometry(const Geometry& geometry) -> bool
{
    return geometry.sites > 0 and geometry.sites <= INT_MAX and geometry.compact > 0
           and geometry.compact <= INT_MAX and geometry.offsets_host and geometry.slices_host
           and geometry.slot_device and geometry.offsets_device and geometry.slices_device;
}
}

auto default_width(i64 compact, int max_slice, i64 samples) -> i64
{
    const i64 bounded_rows{std::max<i64>((samples + k_virtual_shards - 1) / k_virtual_shards, 1)};
    const i64 target{k_workspace_bytes / (bounded_rows * static_cast<i64>(sizeof(cuFloatComplex)))};
    return std::min<i64>(2 * compact, std::max<i64>(2ll * max_slice, target));
}

auto slab_count(const Geometry& geometry, i64 slab_width) -> int
{
    if (not valid_geometry(geometry) or slab_width < 1) return 0;
    int count{};
    for (int first{}; first < geometry.sites;)
    {
        const Slab slab{next_slab(geometry, slab_width, first)};
        if (slab.next_site <= first) return 0;
        ++count;
        first = slab.next_site;
    }
    return count;
}

auto accumulate_block(
    const Workspace& workspace,
    const Geometry& geometry,
    const std::uint8_t* samples,
    const cuFloatComplex* rows_a,
    i64 base_a,
    i64 count_a,
    const cuFloatComplex* rows_b,
    i64 base_b,
    i64 count_b,
    cuFloatComplex* output,
    int output_ld,
    bool diagonal
) -> void
{
    const auto invalid_args =
        not valid_geometry(geometry) or not workspace.dense_a or not workspace.linalg or not samples
        or not rows_a or not output or workspace.slab_width < 1 or count_a < 1 or count_a > INT_MAX
        or count_b < 1 or count_b > INT_MAX or base_a < 0 or base_b < 0 or output_ld < 1;
    require(
        not invalid_args,
        Status::bad_config,
        "Gram block rejects rows {} at {} and {} at {}, slab width {}, leading dimension {}",
        count_a,
        base_a,
        count_b,
        base_b,
        workspace.slab_width,
        output_ld
    );
    if (diagonal)
    {
        const auto same_block = rows_a == rows_b and base_a == base_b and count_a == count_b;
        require(
            same_block,
            Status::bad_config,
            "diagonal Gram block pairs rows {} at {} with rows {} at {}",
            count_a,
            base_a,
            count_b,
            base_b
        );
    }
    else
    {
        const auto has_second_operand = workspace.dense_b != nullptr and rows_b != nullptr;
        require(
            has_second_operand,
            Status::null_argument,
            "off-diagonal Gram block has no second operand"
        );
    }

    auto& linalg{*workspace.linalg};
    const auto stream = linalg.stream();
    for (int first{}; first < geometry.sites;)
    {
        const Slab slab{next_slab(geometry, workspace.slab_width, first)};
        gram_cublas::expand(
            workspace.dense_a,
            rows_a,
            samples,
            count_a,
            geometry.compact,
            base_a,
            static_cast<int>(geometry.sites),
            slab.first_site,
            slab.compact_begin,
            slab.compact_count,
            slab.dense_width,
            geometry.slot_device,
            geometry.offsets_device,
            geometry.slices_device,
            stream
        );
        if (diagonal)
        {
            gram_cublas::accumulate_diagonal_block(
                linalg,
                static_cast<int>(count_a),
                slab.dense_width,
                workspace.dense_a,
                output,
                output_ld
            );
        }
        else
        {
            gram_cublas::expand(
                workspace.dense_b,
                rows_b,
                samples,
                count_b,
                geometry.compact,
                base_b,
                static_cast<int>(geometry.sites),
                slab.first_site,
                slab.compact_begin,
                slab.compact_count,
                slab.dense_width,
                geometry.slot_device,
                geometry.offsets_device,
                geometry.slices_device,
                stream
            );
            gram_cublas::accumulate_offdiagonal_block(
                linalg,
                static_cast<int>(count_a),
                static_cast<int>(count_b),
                slab.dense_width,
                workspace.dense_a,
                workspace.dense_b,
                output,
                output_ld
            );
        }
        first = slab.next_site;
    }
}

auto finalize_block(
    cuFloatComplex* output,
    int output_ld,
    int row_count,
    int global_base,
    int samples,
    cudaStream_t stream
) -> void
{
    const auto valid =
        output and output_ld >= 1 and row_count >= 1 and global_base >= 0 and samples >= 1;
    require(
        valid,
        Status::bad_config,
        "Gram finalize rejects {} rows at {} of {} samples, leading dimension {}",
        row_count,
        global_base,
        samples,
        output_ld
    );
    gram_cublas::finalize(output, output_ld, row_count, global_base, samples, stream);
}
}
