#include "sampler/kernels.cuh"

#include <cmath>
#include <curand_kernel.h>

namespace qnpeps
{
__global__ auto cu_gather(CuGatherArgs args) -> void
{
    const auto out = args.out;
    const auto in = args.in;
    const auto gather_indices = args.gather_indices;
    const auto n = args.n;
    const auto stride_out = args.stride_out;
    const auto stride_in = args.stride_in;
    const auto conjugate = args.conjugate;
    const auto dim_batch = args.dim_batch;

    const i64 total{static_cast<i64>(n) * dim_batch};
    for (i64 tid{global_lane()}; tid < total; tid += grid_stride())
    {
        const auto lane = static_cast<int>(tid / n);
        const auto elem = static_cast<int>(tid % n);
        const auto lane_i64 = static_cast<i64>(lane);
        const i64 in_idx{lane_i64 * stride_in + gather_indices[elem]};
        const i64 out_idx{lane_i64 * stride_out + elem};
        auto value = in[in_idx];
        if (conjugate) value.im = -value.im;
        out[out_idx] = value;
    }
}

__global__ auto cu_slice_ket(CuSliceKetArgs args) -> void
{
    const auto out = args.out;
    const auto ket = args.ket;
    const auto chosen_spins = args.chosen_spins;
    const auto ket_bond_l = args.ket_bond_l;
    const auto dim_phys = args.dim_phys;
    const auto bond_below = args.bond_below;
    const auto ket_bond_r = args.ket_bond_r;
    const auto stride_out = args.stride_out;
    const auto stride_in = args.stride_in;
    const auto dim_batch = args.dim_batch;

    const int n{ket_bond_l * bond_below * ket_bond_r};
    const i64 total{static_cast<i64>(n) * dim_batch};
    const auto ket_bond_l_i64 = static_cast<i64>(ket_bond_l);
    for (i64 tid{global_lane()}; tid < total; tid += grid_stride())
    {
        const auto lane = static_cast<int>(tid / n);
        auto linear = static_cast<int>(tid % n);
        const int idx_left{linear % ket_bond_l};
        linear /= ket_bond_l;
        const int idx_below{linear % bond_below};
        const int idx_right{linear / bond_below};
        const int spin{chosen_spins[lane]};
        const auto lane_i64 = static_cast<i64>(lane);
        const i64 out_idx{
            lane_i64 * stride_out
            + (idx_left + ket_bond_l_i64 * (idx_below + bond_below * idx_right))
        };
        const i64 in_idx{
            lane_i64 * stride_in
            + (idx_left + ket_bond_l_i64 * (spin + dim_phys * (idx_below + bond_below * idx_right)))
        };
        out[out_idx] = ket[in_idx];
    }
}

__global__ auto cu_normalize_log(CuNormalizeLogArgs args) -> void
{
    const auto x = args.x;
    const auto n = args.n;
    const auto stride = args.stride;
    const auto lognorm_acc = args.lognorm_acc;
    const auto dim_batch = args.dim_batch;
    const auto scale_out = args.scale_out;

    __shared__ f32 smax[256];
    const auto lane = static_cast<int>(blockIdx.x);
    if (lane >= dim_batch) return;
    auto* row_ptr = x + lane * stride;
    f32 max_abs{};
    for (auto elem = static_cast<int>(threadIdx.x); elem < n; elem += blockDim.x)
    {
        const f32 abs_sum{fabsf(row_ptr[elem].re) + fabsf(row_ptr[elem].im)};
        max_abs = abs_sum > max_abs ? abs_sum : max_abs;
    }
    smax[threadIdx.x] = max_abs;
    __syncthreads();
    for (auto step = static_cast<int>(blockDim.x / 2); step > 0; step >>= 1)
    {
        if (threadIdx.x < step)
        {
            smax[threadIdx.x] =
                fmaxf(smax[threadIdx.x], smax[threadIdx.x + static_cast<unsigned>(step)]);
        }
        __syncthreads();
    }
    const f32 scale{smax[0] > 0.0f ? smax[0] : 1.0f};
    const f32 inv_scale{1.0f / scale};
    for (auto elem = static_cast<int>(threadIdx.x); elem < n; elem += blockDim.x)
    {
        row_ptr[elem].re *= inv_scale;
        row_ptr[elem].im *= inv_scale;
    }
    if (threadIdx.x == 0)
    {
        if (lognorm_acc) lognorm_acc[lane] += log(static_cast<f64>(scale));
        if (scale_out) scale_out[lane] = scale;
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
        total_weight += rho_lane[diag_index(phys, dim_phys)].abs();

    const bool uniform{not(total_weight > 0.0f) or not isfinite(total_weight)};
    if (uniform) total_weight = static_cast<f32>(dim_phys);

    curandStatePhilox4_32_10_t rng_state{};
    curand_init(seed, static_cast<u64>(lane), static_cast<u64>(site_counter), &rng_state);
    const f32 uniform_draw{curand_uniform(&rng_state)};

    f32 cumulative{};
    int chosen{dim_phys - 1};
    for (auto phys = 0; phys < dim_phys; ++phys)
    {
        const f32 weight{uniform ? 1.0f : rho_lane[diag_index(phys, dim_phys)].abs()};
        cumulative += weight / total_weight;
        if (uniform_draw < cumulative)
        {
            chosen = phys;
            break;
        }
    }
    chosen_spins[lane] = chosen;

    const f32 chosen_weight{uniform ? 1.0f : rho_lane[diag_index(chosen, dim_phys)].abs()};
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

        const cf rho_diag{rho[lane_i64 * stride_rho + diag_index(spin, dim_phys)]};
        const f32 mag{rho_diag.abs()};
        const f32 inv_mag{mag > 0.0f ? 1.0f / mag : 1.0f};

        const i64 full_idx{
            lane_i64 * stride_full + diag_index(spin, dim_phys)
            + static_cast<i64>(dim_phys) * dim_phys * elem
        };
        const cf value{sigma_full[full_idx]};
        sigma[lane_i64 * stride_out + elem] = cf{value.re * inv_mag, value.im * inv_mag};
    }
}

__global__ auto cu_chol_shift(cf* gram, int k, i64 stride, int dim_batch) -> void
{
    const auto lane = static_cast<int>(global_lane());
    if (lane >= dim_batch) return;

    auto* gram_lane = gram + lane * stride;
    f32 trace{};
    for (auto diag = 0; diag < k; ++diag)
        trace += gram_lane[diag_index(diag, k)].re;
    const f32 shift{1.0e-5f * (trace > 0.0f ? trace / k : 1.0f) + 1.0e-30f};
    for (auto diag = 0; diag < k; ++diag)
        gram_lane[diag_index(diag, k)].re += shift;
}

__global__ auto cu_any_chol_failed(const int* info, int n, int* flag) -> void
{
    for (i64 i{global_lane()}; i < n; i += grid_stride())
        if (info[i] != 0) atomicOr(flag, 1);
}

__global__ auto cu_fill_first_one(cf* x, i64 stride, int n, int dim_batch) -> void
{
    const i64 total{static_cast<i64>(n) * dim_batch};
    for (i64 tid{global_lane()}; tid < total; tid += grid_stride())
    {
        const auto elem = static_cast<int>(tid % n);
        const i64 lane{tid / n};
        const i64 out_idx{lane * stride + elem};
        x[out_idx] = cf{elem == 0 ? 1.0f : 0.0f, 0.0f};
    }
}

auto permute_batched(Linalg& linalg, PermutationCache& permutation_cache, const PermuteOp& op)
    -> void
{
    i64 element_count{1};
    for (const int dim : op.dims_in)
        element_count *= dim;
    const PermuteKey key{op.dims_in.get(), op.perm.get(), op.conj != 0};
    auto* gather_indices = permutation_cache.get(key, permutation_index_map(op.dims_in, op.perm));
    const auto blocks = grid_blocks_capped(element_count * op.batch);
    const CuGatherArgs gather_args{
        .out = op.dst.p,
        .in = op.src.p,
        .gather_indices = gather_indices,
        .n = static_cast<int>(element_count),
        .stride_out = op.dst.stride,
        .stride_in = op.src.stride,
        .conjugate = op.conj,
        .dim_batch = op.batch,
    };
    cu_gather<<<blocks, k_threads_per_block, 0, linalg.stream()>>>(gather_args);
}

namespace
{
[[nodiscard]] auto is_identity(const std::vector<int>& order) -> bool
{
    for (auto i = 0_uz; i < order.size(); ++i)
        if (order[i] != static_cast<int>(i)) return false;
    return true;
}

[[nodiscard]] auto swapped_groups(const std::vector<int>& order, usize first_group_size)
    -> std::vector<int>
{
    std::vector<int> out{};
    out.reserve(order.size());
    for (auto i = first_group_size; i < order.size(); ++i)
        out.push_back(order[i]);
    for (auto i = 0_uz; i < first_group_size; ++i)
        out.push_back(order[i]);
    return out;
}

struct PreparedContract
{
    int m{};
    int k{};
    int n{};
    int b_rows{};
    int b_cols{};
    MatmulConfig op{};
    bool a_permuted{};
    bool b_permuted{};
};

[[nodiscard]] auto prepare_contract(
    Linalg& la,
    PermutationCache& cache,
    const ContractSpec& spec,
    const ContractOperand& a,
    const ContractOperand& b
) -> PreparedContract
{
    const auto plan = contract_plan(spec.dims_a, spec.contracted_a, spec.dims_b, spec.contracted_b);

    const auto a_permuted = not is_identity(plan.perm_a.get());
    if (a_permuted)
    {
        permute_batched(
            la,
            cache,
            {
                .dst = a.scratch,
                .src = a.src,
                .dims_in = spec.dims_a,
                .perm = plan.perm_a,
                .batch = spec.dim_batch,
            }
        );
    }

    MatmulConfig op{};
    bool b_permuted{false};
    if (is_identity(plan.perm_b.get()))
    {
        op.op_b = spec.conj_b ? BlasOp::conj : BlasOp::none;
    }
    else if (is_identity(swapped_groups(plan.perm_b.get(), spec.contracted_b.size())))
    {
        op.op_b = spec.conj_b ? BlasOp::conj_trans : BlasOp::trans;
    }
    else
    {
        b_permuted = true;
        permute_batched(
            la,
            cache,
            {
                .dst = b.scratch,
                .src = b.src,
                .dims_in = spec.dims_b,
                .perm = plan.perm_b,
                .batch = spec.dim_batch,
                .conj = spec.conj_b,
            }
        );
    }

    const bool b_stored_transposed = op.op_b == BlasOp::trans or op.op_b == BlasOp::conj_trans;
    return {
        .m = plan.M,
        .k = plan.K,
        .n = plan.N,
        .b_rows = b_stored_transposed ? plan.N : plan.K,
        .b_cols = b_stored_transposed ? plan.K : plan.N,
        .op = op,
        .a_permuted = a_permuted,
        .b_permuted = b_permuted,
    };
}
}

auto contract_batched(
    Linalg& la,
    PermutationCache& cache,
    const ContractSpec& spec,
    const ContractOperand& a,
    const ContractOperand& b,
    const ContractOut& out
) -> void
{
    const auto c = prepare_contract(la, cache, spec, a, b);
    la.matmul_batched_ptr(
        a.ptrs, c.m, c.k, b.ptrs, c.b_rows, c.b_cols, out.ptrs, c.m, c.n, spec.dim_batch, c.op
    );
}

auto contract_strided_batched(
    Linalg& la,
    PermutationCache& cache,
    const ContractSpec& spec,
    const ContractOperand& a,
    const ContractOperand& b,
    const ContractOut& out
) -> void
{
    const auto c = prepare_contract(la, cache, spec, a, b);
    const CuArrayConst a_view = c.a_permuted ? CuArrayConst{a.scratch} : a.src;
    const CuArrayConst b_view = c.b_permuted ? CuArrayConst{b.scratch} : b.src;
    la.matmul_batched(
        {a_view, c.m, c.k}, {b_view, c.b_rows, c.b_cols}, {out.view, c.m, c.n}, spec.dim_batch, c.op
    );
}
}
