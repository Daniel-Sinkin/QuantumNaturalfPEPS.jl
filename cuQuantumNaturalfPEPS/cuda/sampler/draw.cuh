#ifndef QNPEPS_SAMPLER_DRAW_CUH
#define QNPEPS_SAMPLER_DRAW_CUH

#include "capi/qnpeps.h"
#include "types.cuh"

#include <cstdint>

namespace qnpeps::sampler
{
struct SampleArgs
{
    const void* device_peps;
    const void* device_dlenv;
    void* scratch;
    uint64_t scratch_bytes;
    u8* output;
    f64* logpc_out;
    f64* lognorm_out;
    uint64_t n_samples;
    uint64_t batch_base;
    uint64_t dim_batch_pin;
    void* stream;
};

struct SampleMultigpuArgs
{
    const void* device_peps;
    const void* device_dlenv;
    int gpus;
    u8* output;
    f64* logpc_out;
    f64* lognorm_out;
    uint64_t n_samples;
};

auto sample(const QnpepsConfig& config, const SampleArgs& args) -> qnpeps_status;

auto sample_multigpu(const QnpepsConfig& config, const SampleMultigpuArgs& args) -> qnpeps_status;

[[nodiscard]] auto sample_arena_bytes(const QnpepsConfig& config, int max_dim_batch) -> int64_t;
}

#endif
