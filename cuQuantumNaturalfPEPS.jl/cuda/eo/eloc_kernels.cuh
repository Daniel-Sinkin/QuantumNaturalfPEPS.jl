#ifndef QNPEPS_ELOC_KERNELS_CUH
#define QNPEPS_ELOC_KERNELS_CUH

#include "eloc_fixed.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace qn_eloc
{
using i64 = std::int64_t;
using cf = fx::cf;

void launch_eloc_chains(
    i64 n_chains,
    int chi,
    const cf* ma,
    const cf* mb,
    const cf* vin,
    const cf* vend,
    cf* out,
    cudaStream_t stream
);

void launch_build_o(
    i64 n,
    int slice_dim,
    const cf* env,
    const cf* slice_in,
    const cf* gscale,
    cf* out,
    cudaStream_t stream
);

void launch_gram(
    int ns,
    int compact_np,
    int n_blocks,
    const cf* compact_rows,
    const int* spins,
    const int* block_offset,
    const int* block_slice,
    cf* out,
    cudaStream_t stream
);

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
