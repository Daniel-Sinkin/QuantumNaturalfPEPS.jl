#ifndef QNPEPS_PEPS_KERNELS_CUH
#define QNPEPS_PEPS_KERNELS_CUH

#include "core/cuda_utils.cuh"

namespace qnpeps::peps
{

struct SiteLayout
{
    int bond_left{};
    int bond_down{};
    int bond_right{};
    int bond_up{};
    int dim_phys{};
    int incoming{};
    int outgoing{};
    int tall_rows{};
    int thin_cols{};
    i64 output_offset{};

    [[nodiscard]] __host__ __device__ constexpr auto elements() const noexcept -> i64
    {
        return static_cast<i64>(incoming) * outgoing;
    }

    [[nodiscard]] __host__ __device__ constexpr auto transposed() const noexcept -> bool
    {
        return incoming < outgoing;
    }
};

struct FillComplexNormalArgs
{
    cuFloatComplex* matrix;
    i64 count;
    u64 seed;
    u64 sequence_offset;
};

struct ExtractPhasesArgs
{
    const cuFloatComplex* factors;
    int rows;
    int cols;
    cuFloatComplex* phases;
    int* failure;
};

struct PackSiteArgs
{
    const cuFloatComplex* isometry;
    const cuFloatComplex* phases;
    SiteLayout layout;
    const f32* spectrum;
    cuFloatComplex* output;
};

__global__ auto cu_fill_complex_normal(FillComplexNormalArgs args) -> void;
__global__ auto cu_extract_r_phases(ExtractPhasesArgs args) -> void;
__global__ auto cu_fill_spectrum(f32* spectrum, int count, f64 half_negative_alpha) -> void;
__global__ auto cu_pack_site(PackSiteArgs args) -> void;

}

#endif
