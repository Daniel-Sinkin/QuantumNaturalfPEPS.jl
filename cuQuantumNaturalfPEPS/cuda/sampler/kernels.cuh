#ifndef QNPEPS_SAMPLER_KERNELS_CUH
#define QNPEPS_SAMPLER_KERNELS_CUH

#include "dtensor.cuh"
#include "linalg.cuh"

#include <algorithm>
#include <map>
#include <vector>

namespace qnpeps
{
inline constexpr u32 k_threads_per_block{256};
inline constexpr i64 k_max_blocks{4096};

[[nodiscard]] inline constexpr auto grid_blocks_capped(i64 work_items) -> u32
{
    return static_cast<u32>(std::min(k_max_blocks, ceil_div(work_items, k_threads_per_block)));
}

[[nodiscard]] inline constexpr auto grid_blocks_exact(i64 work_items) -> u32
{
    return static_cast<u32>(ceil_div(work_items, k_threads_per_block));
}

struct CuGatherArgs
{
    cf* out{};
    const cf* in{};
    const int* gather_indices{};
    int n{};
    i64 stride_out{};
    i64 stride_in{};
    int conjugate{};
    int dim_batch{};
};

struct CuSliceKetArgs
{
    cf* out{};
    const cf* ket{};
    const int* chosen_spins{};
    int ket_bond_l{};
    int dim_phys{};
    int bond_below{};
    int ket_bond_r{};
    i64 stride_out{};
    i64 stride_in{};
    int dim_batch{};
};

struct CuNormalizeLogArgs
{
    cf* x{};
    int n{};
    i64 stride{};
    f64* lognorm_acc{};
    int dim_batch{};
    f32* scale_out{};
};

struct CuDrawArgs
{
    const cf* rho{};
    int dim_phys{};
    i64 stride_rho{};
    const u64* seed_ptr{};
    int site_counter{};
    u8* samples_site{};
    int sample_stride{};
    f64* logpc{};
    int* chosen_spins{};
    int dim_batch{};
};

struct CuProjectArgs
{
    const cf* sigma_full{};
    const cf* rho{};
    cf* sigma{};
    const int* chosen_spins{};
    int dim_phys{};
    int sigma_elems{};
    i64 stride_full{};
    i64 stride_rho{};
    i64 stride_out{};
    int dim_batch{};
};

__global__ auto cu_gather(CuGatherArgs args) -> void;
__global__ auto cu_slice_ket(CuSliceKetArgs args) -> void;
__global__ auto cu_normalize_log(CuNormalizeLogArgs args) -> void;
__global__ auto cu_draw(CuDrawArgs args) -> void;
__global__ auto cu_project(CuProjectArgs args) -> void;

__global__ auto cu_fill_first_one(cf* x, i64 stride, int n, int dim_batch) -> void;
__global__ auto cu_chol_shift(cf* gram, int k, i64 stride, int dim_batch) -> void;
__global__ auto cu_any_chol_failed(const int* info, int n, int* flag) -> void;

struct PermuteKey
{
    std::vector<int> dims{};
    std::vector<int> perm{};
    bool conj{};

    [[nodiscard]] auto operator<(const PermuteKey& other) const noexcept -> bool
    {
        if (conj != other.conj) return conj < other.conj;
        if (dims != other.dims) return dims < other.dims;
        return perm < other.perm;
    }
};

struct PermutationCache
{
    std::map<PermuteKey, int*> cache{};
    auto get(const PermuteKey& key, const std::vector<int>& gather_indices) -> int*
    {
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;
        int* device_ptr{};
        CUDA_CHECK(cudaMalloc(&device_ptr, gather_indices.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(
            device_ptr,
            gather_indices.data(),
            gather_indices.size() * sizeof(int),
            cudaMemcpyHostToDevice
        ));
        cache.emplace(key, device_ptr);
        return device_ptr;
    }
    auto release() -> void
    {
        for (auto& entry : cache)
            if (entry.second) CUDA_NOCHECK(cudaFree(entry.second));
        cache.clear();
    }
};

struct PermuteOp
{
    CuArray dst{};
    CuArrayConst src{};
    Shape dims_in{};
    Permutation perm{};
    int batch{};
    int conj{0};
};

auto permute_batched(Linalg& linalg, PermutationCache& permutation_cache, const PermuteOp& op)
    -> void;

struct ContractSpec
{
    Shape dims_a{};
    std::vector<int> contracted_a{};
    Shape dims_b{};
    std::vector<int> contracted_b{};
    int conj_b{};
    int dim_batch{};
};

struct ContractOperand
{
    CuArrayConst src{};
    CuArray scratch{};
    cf* const* ptrs{};
};

struct ContractOut
{
    CuArray view{};
    cf* const* ptrs{};
};

auto contract_batched(
    Linalg& la,
    PermutationCache& cache,
    const ContractSpec& spec,
    const ContractOperand& a,
    const ContractOperand& b,
    const ContractOut& out
) -> void;
auto contract_strided_batched(
    Linalg& la,
    PermutationCache& cache,
    const ContractSpec& spec,
    const ContractOperand& a,
    const ContractOperand& b,
    const ContractOut& out
) -> void;
}

#endif
