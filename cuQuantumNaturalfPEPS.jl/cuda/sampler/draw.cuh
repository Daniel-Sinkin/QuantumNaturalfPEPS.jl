#ifndef QNPEPS_SAMPLER_DRAW_CUH
#define QNPEPS_SAMPLER_DRAW_CUH

#include "capi/qnpeps.h"
#include "core/types.cuh"

#include <cstdint>
#include <span>

namespace qnpeps
{
class Linalg;
}

namespace qnpeps::sampler
{
enum class SampleOutputLocation
{
    device,
    host
};

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
    uint64_t dim_batch;
    void* stream;
    SampleOutputLocation output_location;
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
    uint64_t batch_base;
    uint64_t dim_batch;
    SampleOutputLocation output_location;
};

auto sample(const QnpepsConfig& config, const SampleArgs& args) -> qnpeps_status;
auto sample(const QnpepsConfig& config, const SampleArgs& args, qnpeps::Linalg& linalg)
    -> qnpeps_status;

auto sample_multigpu(const QnpepsConfig& config, const SampleMultigpuArgs& args) -> qnpeps_status;
auto sample_multigpu(
    const QnpepsConfig& config,
    const SampleMultigpuArgs& args,
    std::span<qnpeps::Linalg* const> linalg
) -> qnpeps_status;

}

#endif
