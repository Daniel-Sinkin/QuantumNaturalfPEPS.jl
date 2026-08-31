#ifndef QNPEPS_E2E_MINSR_CUH
#define QNPEPS_E2E_MINSR_CUH

#include "../e2e/common.cuh"
#include "core/arena_cursor.cuh"
#include "linalg/linalg.cuh"
#include "minsr.cuh"
#include "minsr/kernels.cuh"

#include <vector>

namespace qnpeps::minsr
{
using cf = qnpeps::ComplexF32;
using cd = cuDoubleComplex;
using qn_e2e::err_state;
using qn_e2e::set_err;

class MinsrSolveContext;

[[nodiscard]] auto prepared_solve_create(i64 samples, Linalg& linalg) -> MinsrSolveContext*;

auto prepared_solve_run(
    MinsrSolveContext& solve,
    const f64* device_log_amplitudes,
    const f64* device_local_energies,
    const f64* device_log_proposals,
    const cf* device_gram,
    f64 relative_cut,
    f64 absolute_cut,
    f64* energy_mean_output,
    f64* energy_variance_output,
    f64* ess_output,
    std::vector<cd>& coefficient_output,
    bool& did_solve_output
) -> void;

[[nodiscard]] auto prepared_solve_transient_arena(const MinsrSolveContext& solve)
    -> TransientArenaCursor;
[[nodiscard]] auto prepared_solve_samples(const MinsrSolveContext& solve) noexcept -> i64;
[[nodiscard]] auto prepared_solve_linalg(const MinsrSolveContext& solve) noexcept -> Linalg&;
auto prepared_solve_destroy(MinsrSolveContext* solve) noexcept -> void;

auto minsr_context_create(
    const QnpepsE2eConfig& config,
    i64 samples,
    i64 host_tile_bytes,
    cudaStream_t stream,
    qnpeps_e2e_minsr_ctx** output
) -> void;

auto minsr_context_create(
    const QnpepsE2eConfig& config,
    i64 samples,
    i64 host_tile_bytes,
    Linalg& linalg,
    qnpeps_e2e_minsr_ctx** output
) -> void;

auto minsr_context_run(
    qnpeps_e2e_minsr_ctx& context,
    const u8* samples,
    const f64* device_log_amplitudes,
    const f64* device_local_energies,
    const f64* device_log_proposals,
    const cf* device_gram,
    const cf* device_rows,
    const cf* host_rows,
    f64 relative_cut,
    f64 absolute_cut,
    cf* theta_output,
    f64* energy_mean_output,
    f64* energy_variance_output,
    f64* ess_output
) -> void;

auto minsr_context_destroy(qnpeps_e2e_minsr_ctx* context) noexcept -> void;
[[nodiscard]] auto minsr_context_stream(const qnpeps_e2e_minsr_ctx& context) noexcept
    -> cudaStream_t;
[[nodiscard]] auto minsr_context_device(const qnpeps_e2e_minsr_ctx& context) noexcept -> int;

auto minsr_solve(
    i64 samples,
    const f64* device_log_amplitudes,
    const f64* device_local_energies,
    const f64* device_log_proposals,
    const cf* device_gram,
    f64 relative_cut,
    f64 absolute_cut,
    f64* energy_mean_output,
    f64* energy_variance_output,
    f64* ess_output,
    std::vector<cd>& coefficient_output,
    bool& did_solve_output,
    Linalg& linalg,
    MinsrSolveContext* cached_solve = nullptr
) -> void;

auto minsr_solve(
    i64 samples,
    const f64* device_log_amplitudes,
    const f64* device_local_energies,
    const f64* device_log_proposals,
    const cf* device_gram,
    f64 relative_cut,
    f64 absolute_cut,
    f64* energy_mean_output,
    f64* energy_variance_output,
    f64* ess_output,
    std::vector<cd>& coefficient_output,
    bool& did_solve_output,
    cudaStream_t stream
) -> void;

auto qn_e2e_minsr_impl(
    const QnpepsE2eConfig& config,
    i64 sample_count,
    const u8* samples,
    const f64* device_log_amplitudes,
    const f64* device_local_energies,
    const f64* device_log_proposals,
    const cf* device_gram,
    const cf* device_rows,
    const cf* host_rows,
    i64 host_tile_bytes,
    f64 relative_cut,
    f64 absolute_cut,
    cf* theta_output,
    f64* energy_mean_output,
    f64* energy_variance_output,
    f64* ess_output,
    cudaStream_t stream
) -> void;

auto qn_e2e_minsr_impl(
    const QnpepsE2eConfig& config,
    i64 sample_count,
    const u8* samples,
    const f64* device_log_amplitudes,
    const f64* device_local_energies,
    const f64* device_log_proposals,
    const cf* device_gram,
    const cf* device_rows,
    const cf* host_rows,
    i64 host_tile_bytes,
    f64 relative_cut,
    f64 absolute_cut,
    cf* theta_output,
    f64* energy_mean_output,
    f64* energy_variance_output,
    f64* ess_output,
    Linalg& linalg,
    MinsrSolveContext* cached_solve = nullptr
) -> void;

[[nodiscard]] auto qn_e2e_minsr_scratch(
    const QnpepsE2eConfig& config, i64 samples, i64 host_tile_bytes
) -> u64;
}

#endif
