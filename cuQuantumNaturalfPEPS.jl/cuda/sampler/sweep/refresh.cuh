#ifndef QNPEPS_SAMPLER_SWEEP_REFRESH_CUH
#define QNPEPS_SAMPLER_SWEEP_REFRESH_CUH

#include "sampler/sweep/common.cuh"

namespace qnpeps::sampler
{

auto ctx_sample_refresh(qnpeps_ctx& ctx, const void* device_peps, PepsLayout layout) -> void;

}

#endif
