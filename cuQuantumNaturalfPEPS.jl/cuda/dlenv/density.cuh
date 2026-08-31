#ifndef QNPEPS_DLENV_DENSITY_CUH
#define QNPEPS_DLENV_DENSITY_CUH

#include "core/arena_cursor.cuh"
#include "capi/qnpeps.h"
#include "densitymatrix/backend.cuh"
#include "tensor/tensor.cuh"
#include "core/types.cuh"

#include <span>
#include <vector>

namespace qnpeps::dlenv
{
struct DensityRowArgs
{
    std::span<const DeviceTensor> row_ket{};
    std::span<const DeviceTensor> environment{};
    int maxdim{};
    f64 cutoff{};
    f64 input_gauge{};
    f64* device_scales{};
};

auto density_row(
    Linalg& linalg,
    ArenaCursor& known,
    ArenaCursor& arena,
    const DensityRowArgs& args,
    f64& output_gauge
) -> std::vector<DeviceTensor>;
}

#endif
