#ifndef QNPEPS_DLENV_BUILD_EXECUTE_CUH
#define QNPEPS_DLENV_BUILD_EXECUTE_CUH

#include "dlenv/build/arena.cuh"
#include "dlenv/build/rows.cuh"
#include "dlenv/build/sampling_buffers.cuh"

namespace qnpeps::dlenv
{

auto build_dlenv(qnpeps_ctx& ctx, const void* device_peps, f64* cumulative_row_logs) -> int;

}

#endif
