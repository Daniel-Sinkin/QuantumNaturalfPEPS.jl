#include "kernels.cuh"

namespace qnpeps::gram_cublas
{

__global__ auto cu_expand(ExpandArgs args) -> void
{
    const i64 total{args.row_count * args.compact_count};
    for (i64 index{global_lane()}; index < total; index += grid_stride())
    {
        const i64 row{index / args.compact_count};
        const int local{static_cast<int>(index - row * args.compact_count)};
        const int global{args.compact_begin + local};
        const int site{args.slot[global]};
        const int slice{args.slices[site]};
        const int within{global - args.offsets[site]};
        const int spin{args.samples[(args.sample_base + row) * args.sites + site]};
        const int dense_index{
            2 * (args.offsets[site] - args.offsets[args.first_site]) + spin * slice + within
        };
        args.dense[row * args.dense_width + dense_index] = args.rows[row * args.compact + global];
    }
}

__global__ auto cu_finalize_offdiagonal(FinalizeOffdiagonalArgs args) -> void
{
    const i64 total{static_cast<i64>(args.row_count) * args.samples};
    for (i64 index{global_lane()}; index < total; index += grid_stride())
    {
        const int row{static_cast<int>(index / args.samples)};
        const int column{static_cast<int>(index - static_cast<i64>(row) * args.samples)};
        const auto outside_local = column < args.local_base;
        const auto beyond_local = column >= args.local_base + args.row_count;
        if (outside_local or beyond_local)
        {
            auto& value = args.output[static_cast<i64>(row) * args.output_ld + column];
            value.y = -value.y;
        }
    }
}

__global__ auto cu_finalize_diagonal(FinalizeDiagonalArgs args) -> void
{
    const i64 total{static_cast<i64>(args.row_count) * args.row_count};
    for (i64 index{global_lane()}; index < total; index += grid_stride())
    {
        const int lower{static_cast<int>(index / args.row_count)};
        const int upper{static_cast<int>(index - static_cast<i64>(lower) * args.row_count)};
        const auto lower_index = static_cast<i64>(lower) * args.output_ld + args.local_base + upper;
        const auto upper_index = static_cast<i64>(upper) * args.output_ld + args.local_base + lower;
        if (lower > upper)
        {
            const cuFloatComplex value{args.output[upper_index]};
            args.output[lower_index] = value;
            args.output[upper_index] = make_cuFloatComplex(value.x, -value.y);
        }
        else if (lower == upper)
        {
            args.output[lower_index].y = 0.0f;
        }
    }
}

}
