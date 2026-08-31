#include "core/complex.cuh"
#include "core/cuda_utils.cuh"
#include "core/error.cuh"
#include "core/predicates.cuh"
#include "linalg/linalg.cuh"
#include "minsr/kernels.cuh"

#include <cuda/std/cmath>

namespace qnpeps::minsr
{
namespace
{
inline constexpr u32 k_eigenvector_threads{128u};
inline constexpr u32 k_matrix_block_width{16u};
inline constexpr usize k_physical_states{2};
inline constexpr f64 k_relative_eigenvalue_floor{1.0e-13};
inline constexpr f64 k_inverse_filter_power{6.0};
inline constexpr f64 k_unset_squared_magnitude{-1.0};
}

__global__ auto cu_canonicalize_eigenvectors(cd* matrix, int order) -> void
{
    const auto column = static_cast<usize>(blockIdx.x * blockDim.x + threadIdx.x);
    const auto order_bound = static_cast<usize>(order);
    if (column >= order_bound) return;
    auto* const vector = matrix + column * order_bound;
    usize pivot{};
    f64 maximum{k_unset_squared_magnitude};
    for (auto row = 0_uz; row < order_bound; ++row)
    {
        const cd value{vector[row]};
        const f64 real{cuCreal(value)};
        const f64 imaginary{cuCimag(value)};
        const f64 squared_magnitude{real * real + imaginary * imaginary};
        if (squared_magnitude > maximum)
        {
            maximum = squared_magnitude;
            pivot = row;
        }
    }
    if (not all_positive(maximum)) return;
    const cd anchor{vector[pivot]};
    const f64 magnitude{cuda::std::sqrt(maximum)};
    const f64 phase_real{cuCreal(anchor) / magnitude};
    const f64 phase_imaginary{-cuCimag(anchor) / magnitude};
    for (auto row = 0_uz; row < order_bound; ++row)
    {
        const cd value{vector[row]};
        const f64 real{cuCreal(value)};
        const f64 imaginary{cuCimag(value)};
        vector[row] = make_cuDoubleComplex(
            real * phase_real - imaginary * phase_imaginary,
            real * phase_imaginary + imaginary * phase_real
        );
    }
    vector[pivot] = make_cuDoubleComplex(magnitude, 0.0);
}

__global__ auto cu_beta(BetaArgs args) -> void
{
    const auto column = static_cast<usize>(blockIdx.x * blockDim.x + threadIdx.x);
    const auto sample_count = static_cast<usize>(args.sample_count);
    if (column >= sample_count) return;
    f64 real_sum{0.0};
    f64 imaginary_sum{0.0};
    for (auto row = 0_uz; row < sample_count; ++row)
    {
        const cf value{args.gram[column * sample_count + row]};
        const f64 weight{args.weights[row]};
        real_sum += static_cast<f64>(value.re) * weight;
        imaginary_sum += static_cast<f64>(value.im) * weight;
    }
    const f64 inverse_sample_count{1.0 / static_cast<f64>(args.sample_count)};
    args.output[column] =
        make_cuDoubleComplex(real_sum * inverse_sample_count, imaginary_sum * inverse_sample_count);
}

__global__ auto cu_build_matrix(BuildMatrixArgs args) -> void
{
    const auto column = static_cast<usize>(blockIdx.x * blockDim.x + threadIdx.x);
    const auto row = static_cast<usize>(blockIdx.y * blockDim.y + threadIdx.y);
    const auto sample_count = static_cast<usize>(args.sample_count);
    if (row >= sample_count or column >= sample_count) return;
    const cf gram_value{args.gram[row * sample_count + column]};
    const cd row_mean{args.gram_means[row]};
    const cd column_mean{args.gram_means[column]};
    const f64 weight_scale{cuda::std::sqrt(args.weights[row] * args.weights[column])};
    const f64 real{
        (static_cast<f64>(gram_value.re) - cuCreal(row_mean) - cuCreal(column_mean)
         + cuCreal(args.total_mean))
        * weight_scale
    };
    const f64 imaginary{
        (static_cast<f64>(gram_value.im) - cuCimag(row_mean) + cuCimag(column_mean)
         + cuCimag(args.total_mean))
        * weight_scale
    };
    args.output[row + column * sample_count] = make_cuDoubleComplex(real, -imaginary);
}

__global__ auto cu_apply_inverse(ApplyInverseArgs args) -> void
{
    const auto index = static_cast<usize>(blockIdx.x * blockDim.x + threadIdx.x);
    const auto sample_count = static_cast<usize>(args.sample_count);
    if (index >= sample_count) return;
    const f64 eigenvalue{args.eigenvalues[index]};
    f64 inverse{0.0};
    if (eigenvalue / args.largest_eigenvalue >= k_relative_eigenvalue_floor)
    {
        const auto cutoff = args.largest_eigenvalue * args.relative_cut + args.absolute_cut;
        const f64 softening{
            cuda::std::pow(cutoff / cuda::std::abs(eigenvalue), k_inverse_filter_power)
        };
        inverse = 1.0 / (eigenvalue * (1.0 + softening));
    }
    const cd value{args.values[index]};
    args.values[index] = make_cuDoubleComplex(cuCreal(value) * inverse, cuCimag(value) * inverse);
}

__global__ auto cu_scatter(ScatterArgs args) -> void
{
    const auto compact_index = static_cast<usize>(blockIdx.x * blockDim.x + threadIdx.x);
    const auto compact_count = static_cast<usize>(args.compact_count);
    if (compact_index >= compact_count) return;
    const auto site = static_cast<usize>(args.slot_sites[compact_index]);
    const auto physical_dimension = static_cast<usize>(args.physical_dimension);
    const auto site_count = static_cast<usize>(args.site_count);
    const auto row_begin = static_cast<usize>(args.row_begin);
    const auto row_end = static_cast<usize>(args.row_end);
    const auto row_base = static_cast<usize>(args.row_base);
    qnpeps::CuArray<f64, k_physical_states> real_parts;
    qnpeps::CuArray<f64, k_physical_states> imaginary_parts;
    for (auto spin = 0_uz; spin < physical_dimension; ++spin)
    {
        const cd value{args.accumulator[physical_dimension * compact_index + spin]};
        real_parts[spin] = cuCreal(value);
        imaginary_parts[spin] = cuCimag(value);
    }
    for (auto row_index = row_begin; row_index < row_end; ++row_index)
    {
        const auto spin = static_cast<usize>(args.samples[row_index * site_count + site]);
        const cf row{args.rows[(row_index - row_base) * compact_count + compact_index]};
        const cd coefficient{args.coefficients[row_index]};
        const f64 row_real{static_cast<f64>(row.re)};
        const f64 row_imaginary{-static_cast<f64>(row.im)};
        const f64 coefficient_real{cuCreal(coefficient)};
        const f64 coefficient_imaginary{cuCimag(coefficient)};
        real_parts[spin] += coefficient_real * row_real - coefficient_imaginary * row_imaginary;
        imaginary_parts[spin] +=
            coefficient_real * row_imaginary + coefficient_imaginary * row_real;
    }
    for (auto spin = 0_uz; spin < physical_dimension; ++spin)
    {
        args.accumulator[physical_dimension * compact_index + spin] =
            make_cuDoubleComplex(real_parts[spin], imaginary_parts[spin]);
    }
}

__global__ auto cu_cast_accumulator(cf* output, const cd* accumulator, i64 count) -> void
{
    const auto stride = static_cast<usize>(gridDim.x * blockDim.x);
    const auto count_bound = static_cast<usize>(count);
    usize index{static_cast<usize>(blockIdx.x * blockDim.x + threadIdx.x)};
    for (; index < count_bound; index += stride)
    {
        output[index] = to_cf(accumulator[index]);
    }
}

[[nodiscard]] auto launch_canonicalize_eigenvectors(Linalg& linalg, cd* matrix, int order)
    -> qnpeps_status
{
    const auto order_bound = static_cast<u32>(order);
    const auto blocks = (order_bound + k_eigenvector_threads - 1) / k_eigenvector_threads;
    cu_canonicalize_eigenvectors<<<blocks, k_eigenvector_threads, 0, linalg.stream()>>>(
        matrix, order
    );
    return qnpeps::cuda_status(cudaGetLastError());
}

auto launch_beta(Linalg& linalg, const BetaArgs& args) -> void
{
    cu_beta<<<grid_blocks_exact(args.sample_count), k_threads_per_block, 0, linalg.stream()>>>(
        args
    );
    CUDA_CHECK(cudaGetLastError());
}

auto launch_build_matrix(Linalg& linalg, const BuildMatrixArgs& args) -> void
{
    const dim3 block{k_matrix_block_width, k_matrix_block_width};
    const auto sample_count = static_cast<u32>(args.sample_count);
    const auto block_count = (sample_count + k_matrix_block_width - 1) / k_matrix_block_width;
    const dim3 grid{block_count, block_count};
    cu_build_matrix<<<grid, block, 0, linalg.stream()>>>(args);
    CUDA_CHECK(cudaGetLastError());
}

auto launch_apply_inverse(Linalg& linalg, const ApplyInverseArgs& args) -> void
{
    cu_apply_inverse<<<
        grid_blocks_exact(args.sample_count),
        k_threads_per_block,
        0,
        linalg.stream()>>>(args);
    CUDA_CHECK(cudaGetLastError());
}

auto launch_scatter(Linalg& linalg, const ScatterArgs& args) -> void
{
    cu_scatter<<<grid_blocks_exact(args.compact_count), k_threads_per_block, 0, linalg.stream()>>>(
        args
    );
    CUDA_CHECK(cudaGetLastError());
}

auto launch_cast_accumulator(Linalg& linalg, cf* output, const cd* accumulator, i64 count) -> void
{
    cu_cast_accumulator<<<grid_blocks_exact(count), k_threads_per_block, 0, linalg.stream()>>>(
        output, accumulator, count
    );
    CUDA_CHECK(cudaGetLastError());
}
}
