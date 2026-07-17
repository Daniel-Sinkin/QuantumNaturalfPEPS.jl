#ifndef QNPEPS_SAMPLER_KERNELS_CUH
#define QNPEPS_SAMPLER_KERNELS_CUH

#include "cuda_utils.cuh"

namespace qnpeps
{
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

__global__ auto cu_slice_ket(CuSliceKetArgs args) -> void;
__global__ auto cu_normalize_log(CuNormalizeLogArgs args) -> void;
__global__ auto cu_draw(CuDrawArgs args) -> void;
__global__ auto cu_project(CuProjectArgs args) -> void;

__global__ auto cu_fill_first_one(cf* x, i64 stride, int n, int dim_batch) -> void;
__global__ auto cu_chol_shift(cf* gram, int k, i64 stride, int dim_batch) -> void;
__global__ auto cu_any_chol_failed(const int* info, int n, int* flag) -> void;

}

#endif
