#ifndef QNPEPS_EO_SHARDS_CUH
#define QNPEPS_EO_SHARDS_CUH

#include "capi/qnpeps.h"
#include "core/types.cuh"
#include "dans_qnpeps_eloc.h"

#include <cuda_runtime.h>

namespace qnpeps
{
class Linalg;
}

namespace qnpeps::eo
{
struct ShardArgs
{
    const QnpepsElocConfig* config;
    const ComplexF32* device_peps;
    const u8* samples;
    i64 n_samples;
    const QnpepsElocTermTable* terms;
    f64* logpsi;
    f64* e_loc;
    ComplexF32* o_rows_device;
    ComplexF32* o_rows_host;
    Linalg* linalg;
};

struct GramTileArgs
{
    const QnpepsElocConfig* config;
    const ComplexF32* rows_a;
    const u8* samples_a;
    i64 count_a;
    const ComplexF32* rows_b;
    const u8* samples_b;
    i64 count_b;
    ComplexF32* tile;
    cudaStream_t stream;
};

auto run_shard(const ShardArgs& args) -> void;
auto gram_tile(const GramTileArgs& args) -> void;
}

#endif
