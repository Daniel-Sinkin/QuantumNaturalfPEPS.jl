#ifndef QNPEPS_EO_SHARDS_CUH
#define QNPEPS_EO_SHARDS_CUH

#include "capi/qnpeps.h"
#include "core/types.cuh"
#include "dans_qnpeps_eloc.h"

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

auto run_shard(const ShardArgs& args) -> qnpeps_status;
}

#endif
