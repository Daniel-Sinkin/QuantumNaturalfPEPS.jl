#include "sampler/kernels.cuh"

#include <cmath>
#include <curand_kernel.h>

namespace qnpeps
{
namespace
{
[[nodiscard]] __device__ auto magnitude(cuFloatComplex value) -> f32
{
    return sqrtf(value.x * value.x + value.y * value.y);
}
}

__global__ auto cu_slice_ket(CuSliceKetArgs args) -> void
{
    const auto total = args.slice_elems * args.dim_batch;
    for (auto index = global_lane(); index < total; index += grid_stride())
    {
        const auto lane = index / args.slice_elems;
        const auto output_index = index % args.slice_elems;
        const auto block_index = output_index % args.ket_bond_l;
        const auto trailing_index = output_index / args.ket_bond_l;
        const auto spin = args.chosen_spins[lane];
        const auto spin_block = spin + args.dim_phys * trailing_index;
        const auto input_index = block_index + args.ket_bond_l * spin_block;
        const auto output_offset = lane * args.stride_out + output_index;
        const auto input_offset = lane * args.stride_in + input_index;
        args.out[output_offset] = args.ket[input_offset];
    }
}

__global__ auto cu_project_mpo(CuProjectMpoArgs args) -> void
{
    const auto total = static_cast<i64>(args.output_elems) * args.dim_batch;
    for (auto index = global_lane(); index < total; index += grid_stride())
    {
        const auto lane = static_cast<int>(index / args.output_elems);
        const auto output_index = static_cast<int>(index % args.output_elems);
        const auto block_index = output_index % args.spin_block;
        const auto trailing_index = output_index / args.spin_block;
        const auto spin = args.chosen_spins[lane];
        const auto spin_block = spin + args.dim_phys * trailing_index;
        const auto input_index = block_index + args.spin_block * spin_block;
        args.out[static_cast<i64>(lane) * args.stride_out + output_index] = args.mpo[input_index];
    }
}

__global__ auto cu_normalize_log(CuNormalizeLogArgs args) -> void
{
    __shared__ f32 shared_max[k_tree_reduce_threads];
    const auto lane = static_cast<int>(blockIdx.x);
    if (lane >= args.dim_batch) return;

    auto* values = args.x + lane * args.stride;
    f32 local_max{0.0f};
    for (auto index = threadIdx.x; index < args.n; index += blockDim.x)
    {
        const auto component_abs_sum = fabsf(values[index].x) + fabsf(values[index].y);
        if (component_abs_sum > local_max) local_max = component_abs_sum;
    }
    shared_max[threadIdx.x] = local_max;
    __syncthreads();

    for (auto offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset and shared_max[threadIdx.x + offset] > shared_max[threadIdx.x])
        {
            shared_max[threadIdx.x] = shared_max[threadIdx.x + offset];
        }
        __syncthreads();
    }

    const auto scale = shared_max[0] > 0.0f ? shared_max[0] : 1.0f;
    const auto inverse_scale = 1.0f / scale;
    for (auto index = threadIdx.x; index < args.n; index += blockDim.x)
    {
        values[index].x *= inverse_scale;
        values[index].y *= inverse_scale;
    }
    if (threadIdx.x == 0)
    {
        if (args.lognorm_acc) args.lognorm_acc[lane] += log(static_cast<f64>(scale));
        if (args.scale_out) args.scale_out[lane] = scale;
    }
}

__global__ auto cu_draw(CuDrawArgs args) -> void
{
    const auto rho = args.rho;
    const auto dim_phys = args.dim_phys;
    const auto stride_rho = args.stride_rho;
    const auto seed_ptr = args.seed_ptr;
    const auto site_counter = args.site_counter;
    const auto samples_site = args.samples_site;
    const auto sample_stride = args.sample_stride;
    const auto logpc = args.logpc;
    const auto chosen_spins = args.chosen_spins;
    const auto dim_batch = args.dim_batch;

    const auto lane = static_cast<int>(global_lane());
    if (lane >= dim_batch) return;

    const u64 seed{*seed_ptr};
    const auto* rho_lane = rho + lane * stride_rho;

    f32 total_weight{};
    for (auto phys = 0; phys < dim_phys; ++phys)
        total_weight += magnitude(rho_lane[diag_index(phys, dim_phys)]);

    const bool uniform{not(total_weight > 0.0f) or not isfinite(total_weight)};
    if (uniform) total_weight = static_cast<f32>(dim_phys);

    curandStatePhilox4_32_10_t rng_state{};
    curand_init(seed, static_cast<u64>(lane), static_cast<u64>(site_counter), &rng_state);
    const f32 uniform_draw{curand_uniform(&rng_state)};

    f32 cumulative{};
    int chosen{dim_phys - 1};
    for (auto phys = 0; phys < dim_phys; ++phys)
    {
        const f32 weight{uniform ? 1.0f : magnitude(rho_lane[diag_index(phys, dim_phys)])};
        cumulative += weight / total_weight;
        if (uniform_draw < cumulative)
        {
            chosen = phys;
            break;
        }
    }
    chosen_spins[lane] = chosen;

    const f32 chosen_weight{uniform ? 1.0f : magnitude(rho_lane[diag_index(chosen, dim_phys)])};
    const auto chosen_prob = static_cast<f64>(chosen_weight / total_weight);
    logpc[lane] += log(chosen_prob);
    samples_site[lane * sample_stride] = static_cast<u8>(chosen);
}

__global__ auto cu_project(CuProjectArgs args) -> void
{
    const auto sigma_full = args.sigma_full;
    const auto rho = args.rho;
    const auto sigma = args.sigma;
    const auto chosen_spins = args.chosen_spins;
    const auto dim_phys = args.dim_phys;
    const auto sigma_elems = args.sigma_elems;
    const auto stride_full = args.stride_full;
    const auto stride_rho = args.stride_rho;
    const auto stride_out = args.stride_out;
    const auto dim_batch = args.dim_batch;

    const i64 total{static_cast<i64>(sigma_elems) * dim_batch};
    for (i64 tid{global_lane()}; tid < total; tid += grid_stride())
    {
        const auto lane = static_cast<int>(tid / sigma_elems);
        const auto elem = static_cast<int>(tid % sigma_elems);
        const int spin{chosen_spins[lane]};
        const auto lane_i64 = static_cast<i64>(lane);

        const cuFloatComplex rho_diag{rho[lane_i64 * stride_rho + diag_index(spin, dim_phys)]};
        const f32 mag{magnitude(rho_diag)};
        const f32 inv_mag{mag > 0.0f ? 1.0f / mag : 1.0f};

        const i64 full_idx{
            lane_i64 * stride_full + diag_index(spin, dim_phys)
            + static_cast<i64>(dim_phys) * dim_phys * elem
        };
        const cuFloatComplex value{sigma_full[full_idx]};
        sigma[lane_i64 * stride_out + elem] = cuFloatComplex{value.x * inv_mag, value.y * inv_mag};
    }
}

__global__ auto cu_chol_shift(cuFloatComplex* gram, int k, i64 stride, int dim_batch) -> void
{
    const auto lane = static_cast<int>(global_lane());
    if (lane >= dim_batch) return;

    auto* gram_lane = gram + lane * stride;
    f32 trace{};
    for (auto diag = 0; diag < k; ++diag)
        trace += gram_lane[diag_index(diag, k)].x;
    const f32 shift{1.0e-5f * (trace > 0.0f ? trace / k : 1.0f) + 1.0e-30f};
    for (auto diag = 0; diag < k; ++diag)
        gram_lane[diag_index(diag, k)].x += shift;
}

__global__ auto cu_any_chol_failed(const int* info, int n, int* flag) -> void
{
    for (i64 i{global_lane()}; i < n; i += grid_stride())
        if (info[i] != 0) atomicOr(flag, 1);
}

__global__ auto cu_fill_first_one(cuFloatComplex* x, i64 stride, int n, int dim_batch) -> void
{
    const i64 total{static_cast<i64>(n) * dim_batch};
    for (i64 tid{global_lane()}; tid < total; tid += grid_stride())
    {
        const auto elem = static_cast<int>(tid % n);
        const i64 lane{tid / n};
        const i64 out_idx{lane * stride + elem};
        x[out_idx] = cuFloatComplex{elem == 0 ? 1.0f : 0.0f, 0.0f};
    }
}

}
