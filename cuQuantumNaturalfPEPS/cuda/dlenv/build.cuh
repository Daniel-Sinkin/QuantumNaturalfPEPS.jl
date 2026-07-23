#ifndef QNPEPS_DLENV_BUILD_CUH
#define QNPEPS_DLENV_BUILD_CUH

#include "arena_cursor.cuh"
#include "capi/qnpeps.h"
#include "dlenv/state.cuh"
#include "tensor.cuh"
#include "types.cuh"
#include "zipup_mpo_mps.cuh"

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

using Arenas = zipup::Arenas;

[[nodiscard]] auto zipup_peps_row_bytes(const QnpepsConfig& config, int maxdim) -> i64;
auto create_zipup_context(const QnpepsConfig& config, int maxdim, cudaStream_t stream)
    -> qnpeps_zipup_ctx*;
auto destroy_zipup_context(qnpeps_zipup_ctx* context) -> void;
auto begin_zipup_context(qnpeps_zipup_ctx& context) -> qnpeps_status;
auto enqueue_peps_row(
    qnpeps_zipup_ctx& context, const QnpepsZipupPepsRowArgs& args
) -> qnpeps_status;
auto finish_zipup_context(qnpeps_zipup_ctx& context, f64* scales, usize count) -> qnpeps_status;
}

#endif
