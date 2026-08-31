#ifndef QNPEPS_SAMPLER_SWEEP_RUN_CUH
#define QNPEPS_SAMPLER_SWEEP_RUN_CUH

#include "sampler/sweep/common.cuh"

namespace qnpeps::sampler
{

auto ctx_sample_run(qnpeps_ctx& ctx, std::span<const int> batch_ids, HostSampleOutput* host_output)
    -> void;

}

#endif
