#ifndef QNPEPS_GRAM_SLAB_CUH
#define QNPEPS_GRAM_SLAB_CUH

#include "capi/qnpeps.h"
#include "core/types.cuh"

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace qnpeps
{
class Linalg;
}

namespace qnpeps::gram_slab
{
inline constexpr int k_virtual_shards{4};

struct Geometry
{
    i64 sites{};
    i64 compact{};
    const int* offsets_host{};
    const int* slices_host{};
    const int* slot_device{};
    const int* offsets_device{};
    const int* slices_device{};
};

struct Workspace
{
    cuFloatComplex* dense_a{};
    cuFloatComplex* dense_b{};
    i64 slab_width{};
    Linalg* linalg{};
};

auto default_width(i64 compact, int max_slice, i64 samples) -> i64;

auto slab_count(const Geometry& geometry, i64 slab_width) -> int;

auto accumulate_block(
    const Workspace& workspace,
    const Geometry& geometry,
    const std::uint8_t* samples,
    const cuFloatComplex* rows_a,
    i64 base_a,
    i64 count_a,
    const cuFloatComplex* rows_b,
    i64 base_b,
    i64 count_b,
    cuFloatComplex* output,
    int output_ld,
    bool diagonal
) -> qnpeps_status;

auto finalize_block(
    cuFloatComplex* output,
    int output_ld,
    int row_count,
    int global_base,
    int samples,
    cudaStream_t stream
) -> qnpeps_status;
}

#endif
