#ifndef QNPEPS_ELOC_KERNELS_CUH
#define QNPEPS_ELOC_KERNELS_CUH

#include "eloc_fixed.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace qn_eloc
{
using i64 = std::int64_t;
using cf = fx::cf;

void launch_gram_tile(
    cf* out,
    int out_ld,
    int compact_np,
    int n_blocks,
    const cf* rows_a,
    const cf* rows_b,
    const int* spins_a,
    const int* spins_b,
    const int* block_offset,
    const int* block_slice,
    int s0,
    int s_len,
    int u0,
    int u_len,
    cudaStream_t stream
);

}

#endif
