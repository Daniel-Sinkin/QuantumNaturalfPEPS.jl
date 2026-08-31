#include "kernels.cuh"

namespace qnpeps
{

__global__ auto cu_gather(CuGatherArgs args) -> void
{
    const auto total = args.element_count * args.batch_count;
    for (auto thread_index = global_lane(); thread_index < total; thread_index += grid_stride())
    {
        const auto batch_index = thread_index / args.element_count;
        const auto output_element = thread_index % args.element_count;
        const auto input_element = args.gather_indices[output_element];
        const auto input_offset = batch_index * args.input_stride;
        const auto output_offset = batch_index * args.output_stride;
        const auto input_index = input_offset + input_element;
        const auto output_index = output_offset + output_element;
        auto value = args.input[input_index];
        if (args.conjugate) value = cuConjf(value);
        args.output[output_index] = value;
    }
}

__global__ auto cu_permute(CuPermuteArgs args) -> void
{
    const auto flat_index = global_lane();
    if (flat_index >= args.element_count) return;

    i64 input_index{};
    i64 output_stride{1};
    for (int axis{}; axis < args.rank; ++axis)
    {
        const auto output_extent = args.plan.output_extents[axis];
        const auto strided_index = flat_index / output_stride;
        const auto coordinate = strided_index % output_extent;
        const auto input_stride = args.plan.input_strides[axis];
        input_index += coordinate * input_stride;
        output_stride *= output_extent;
    }
    auto value = args.input[input_index];
    if (args.conjugate) value = cuConjf(value);
    args.output[flat_index] = value;
}

}
