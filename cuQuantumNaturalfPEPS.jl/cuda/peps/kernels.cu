#include "kernels.cuh"

#include "linalg/random.cuh"

#include <cuda/std/cmath>

namespace qnpeps::peps
{

__global__ auto cu_fill_complex_normal(FillComplexNormalArgs args) -> void
{
    for (auto index = global_lane(); index < args.count; index += grid_stride())
    {
        DeviceRandomState state{};
        initialize_random(state, args.seed, args.sequence_offset + static_cast<u64>(index), 0);
        const auto draw = random_normal_pair(state);
        args.matrix[index] = make_cuFloatComplex(draw.x, draw.y);
    }
}

__global__ auto cu_extract_r_phases(ExtractPhasesArgs args) -> void
{
    for (auto column = global_lane(); column < args.cols; column += grid_stride())
    {
        const auto diagonal = args.factors[static_cast<i64>(column) * args.rows + column];
        const auto magnitude = cuda::std::hypot(cuCrealf(diagonal), cuCimagf(diagonal));
        if (magnitude <= 0.0f or not cuda::std::isfinite(magnitude))
        {
            args.phases[column] = make_cuFloatComplex(1.0f, 0.0f);
            atomicExch(args.failure, 1);
            continue;
        }
        args.phases[column] =
            make_cuFloatComplex(cuCrealf(diagonal) / magnitude, cuCimagf(diagonal) / magnitude);
    }
}

__global__ auto cu_fill_spectrum(f32* spectrum, int count, f64 half_negative_alpha) -> void
{
    for (auto index = global_lane(); index < count; index += grid_stride())
        spectrum[index] =
            static_cast<f32>(cuda::std::pow(static_cast<f64>(index + 1), half_negative_alpha));
}

__global__ auto cu_pack_site(PackSiteArgs args) -> void
{
    for (auto index = global_lane(); index < args.layout.elements(); index += grid_stride())
    {
        auto remainder = index;
        const auto left = static_cast<int>(remainder % args.layout.bond_left);
        remainder /= args.layout.bond_left;
        const auto down = static_cast<int>(remainder % args.layout.bond_down);
        remainder /= args.layout.bond_down;
        const auto right = static_cast<int>(remainder % args.layout.bond_right);
        remainder /= args.layout.bond_right;
        const auto up = static_cast<int>(remainder % args.layout.bond_up);
        remainder /= args.layout.bond_up;
        const auto physical = static_cast<int>(remainder);
        const auto incoming_row =
            physical + args.layout.dim_phys * (left + args.layout.bond_left * up);
        const auto outgoing_column = right + args.layout.bond_right * down;
        cuFloatComplex value{};
        if (args.layout.transposed())
        {
            const auto source =
                args.isometry
                    [static_cast<i64>(incoming_row) * args.layout.tall_rows + outgoing_column];
            value = cuConjf(cuCmulf(source, args.phases[incoming_row]));
        }
        else
        {
            const auto source =
                args.isometry
                    [static_cast<i64>(outgoing_column) * args.layout.tall_rows + incoming_row];
            value = cuCmulf(source, args.phases[outgoing_column]);
        }
        if (args.spectrum)
        {
            f32 scale{1.0f};
            if (args.layout.bond_left > 1) scale *= args.spectrum[left];
            if (args.layout.bond_down > 1) scale *= args.spectrum[down];
            if (args.layout.bond_right > 1) scale *= args.spectrum[right];
            if (args.layout.bond_up > 1) scale *= args.spectrum[up];
            value = make_cuFloatComplex(cuCrealf(value) * scale, cuCimagf(value) * scale);
        }
        args.output[args.layout.output_offset + index] = value;
    }
}

}
