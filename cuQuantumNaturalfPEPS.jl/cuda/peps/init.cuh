
#ifndef QNPEPS_PEPS_INIT_CUH
#define QNPEPS_PEPS_INIT_CUH

#include "capi/qnpeps.h"
#include "core/types.cuh"

#include <cuda_runtime.h>

namespace qnpeps
{
class Linalg;

namespace peps
{
struct RandomUnitaryArgs
{
    const QnpepsConfig& config;
    cuFloatComplex* output{};
    usize output_bytes{};
    u64 seed{};
    f64 alpha{};
};

auto random_unitary(Linalg& linalg, const RandomUnitaryArgs& args) -> void;
}
}

#endif
