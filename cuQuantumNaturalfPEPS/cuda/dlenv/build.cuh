#ifndef QNPEPS_DLENV_BUILD_CUH
#define QNPEPS_DLENV_BUILD_CUH

#include "arena_cursor.cuh"
#include "capi/qnpeps.h"
#include "dtensor.cuh"
#include "types.cuh"

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>

namespace qnpeps
{
class Linalg;

inline constexpr usize k_dl_bond_left{0};
inline constexpr usize k_dl_ket{1};
inline constexpr usize k_dl_bra{2};
inline constexpr usize k_dl_bond_right{3};
inline constexpr usize k_dl_axis_count{4};
}

namespace qnpeps::dlenv
{
using qnpeps::DeviceTensor;

struct Arenas
{
    ArenaCursor& known;
    ArenaCursor& rolling_r;
    ArenaCursor& scratch;
};

auto build_dlenv_row(
    Linalg& la,
    const QnpepsConfig& cfg,
    int row,
    int maxdim,
    const void* device_peps_row,
    const void* device_env_below,
    void* dlenv_row_out,
    f64* row_log_out
) -> int;
}

#endif
