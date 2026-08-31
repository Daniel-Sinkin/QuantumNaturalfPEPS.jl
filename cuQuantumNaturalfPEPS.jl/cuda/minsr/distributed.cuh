#ifndef QNPEPS_MINSR_DISTRIBUTED_CUH
#define QNPEPS_MINSR_DISTRIBUTED_CUH

#include "capi/qnpeps.h"
#include "core/types.cuh"
#include "dans_qnpeps_eloc.h"

#include <span>

namespace qnpeps
{
class Linalg;
}

namespace qnpeps::minsr
{
struct DistributedLane
{
    int device;
    i64 base;
    i64 count;
    const u8* samples;
    const ComplexF32* rows;
    Linalg* linalg;
};

struct DistributedGramArgs
{
    const QnpepsElocConfig* config;
    i64 n_samples;
    i64 sites;
    i64 compact;
    i64 peer_tile_bytes;
    const DistributedLane* destination;
    std::span<const DistributedLane> lanes;
    ComplexF32* gram_device0;
};

struct DistributedScatterArgs
{
    i64 n_samples;
    i64 sites;
    i64 compact;
    i64 dense;
    int dim_phys;
    std::span<const DistributedLane> lanes;
    std::span<const cuDoubleComplex> coefficients;
    std::span<const i32> slot_site;
    ComplexF32* theta_device0;
};

auto build_gram_blockrow(const DistributedGramArgs& args) -> qnpeps_status;
auto scatter_ring(const DistributedScatterArgs& args) -> qnpeps_status;
}

#endif
