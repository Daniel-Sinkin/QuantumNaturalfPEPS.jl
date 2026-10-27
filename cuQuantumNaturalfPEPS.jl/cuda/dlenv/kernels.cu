#include "kernels.cuh"

#include "core/complex.cuh"

#include <cuda/std/cmath>

namespace qnpeps::dlenv
{

__global__ auto cu_product(ProductArgs args) -> void
{
    const auto operator_left = args.left * args.left;
    const auto operator_right = args.right * args.right;
    const auto combined_left = args.state_left * operator_left;
    const auto combined_right = args.state_right * operator_right;
    const auto count = combined_left * combined_right;
    for (auto lane = global_lane(); lane < count; lane += grid_stride())
    {
        const auto combined_left_index = lane % combined_left;
        const auto combined_right_index = lane / combined_left;
        const auto state_left_index = combined_left_index % args.state_left;
        const auto operator_left_index = combined_left_index / args.state_left;
        const auto ket_left = operator_left_index % args.left;
        const auto bra_left = operator_left_index / args.left;
        const auto state_right_index = combined_right_index % args.state_right;
        const auto operator_right_index = combined_right_index / args.state_right;
        const auto ket_right = operator_right_index % args.right;
        const auto bra_right = operator_right_index / args.right;
        const auto ket_up = args.output_state % args.up;
        const auto bra_up = args.output_state / args.up;
        auto value = make_cuDoubleComplex(0.0, 0.0);
        for (i64 bra_down{}; bra_down < args.down; ++bra_down)
        {
            for (i64 ket_down{}; ket_down < args.down; ++ket_down)
            {
                const auto state_index =
                    state_left_index
                    + args.state_left
                          * (ket_down + args.down * (bra_down + args.down * state_right_index));
                for (i64 physical{}; physical < args.dim_phys; ++physical)
                {
                    const auto ket_index =
                        physical
                        + args.dim_phys
                              * (ket_up
                                 + args.up
                                       * (ket_right
                                          + args.right * (ket_down + args.down * ket_left)));
                    const auto bra_index =
                        physical
                        + args.dim_phys
                              * (bra_up
                                 + args.up
                                       * (bra_right
                                          + args.right * (bra_down + args.down * bra_left)));
                    const auto ket = make_cuDoubleComplex(
                        static_cast<f64>(args.ket[ket_index].x),
                        static_cast<f64>(args.ket[ket_index].y)
                    );
                    const auto bra = make_cuDoubleComplex(
                        static_cast<f64>(args.ket[bra_index].x),
                        -static_cast<f64>(args.ket[bra_index].y)
                    );
                    value = cuCadd(value, cuCmul(args.state[state_index], cuCmul(ket, bra)));
                }
            }
        }
        args.product[lane] = value;
    }
}

__global__ auto cu_scales(ScaleArgs args) -> void
{
    for (auto lane = global_lane(); lane < static_cast<i64>(args.count); lane += grid_stride())
        args.scales[lane] = lane == 0 ? cuda::std::exp(*args.normalization_log) : 1.0;
}

}
