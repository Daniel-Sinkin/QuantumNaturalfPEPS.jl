#include "../e2e/layout.cuh"
#include "core/predicates.cuh"
#include "core/session.cuh"
#include "linalg/transfer.cuh"
#include "minsr/solve.cuh"

#include <algorithm>
#include <memory>
#include <new>
#include <vector>

namespace qnpeps::minsr
{
namespace
{
[[nodiscard]] auto host_tile_rows(i64 host_tile_bytes, i64 compact_count, i64 sample_count) noexcept
    -> usize
{
    const auto budget = all_positive(host_tile_bytes) ? host_tile_bytes : k_host_tile_bytes_default;
    const auto nonempty_compact_count = std::max(compact_count, i64{1});
    const auto row_bytes = nonempty_compact_count * static_cast<i64>(sizeof(cf));
    auto tile_rows = budget / row_bytes;
    tile_rows = std::max(tile_rows, i64{1});
    tile_rows = std::min(tile_rows, sample_count);
    return static_cast<usize>(tile_rows);
}
}

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
    MinsrSolveContext* cached_solve
) -> void
{
    did_solve_output = false;
    coefficient_output.assign(static_cast<usize>(samples), make_cuDoubleComplex(0.0, 0.0));
    const std::unique_ptr<MinsrSolveContext, decltype(&prepared_solve_destroy)> owned_solve{
        cached_solve ? nullptr : prepared_solve_create(samples, linalg), prepared_solve_destroy
    };
    auto* const solve = cached_solve ? cached_solve : owned_solve.get();
    if (not solve or err_state() != QNPEPS_E2E_OK) return;
    prepared_solve_run(
        *solve,
        device_log_amplitudes,
        device_local_energies,
        device_log_proposals,
        device_gram,
        relative_cut,
        absolute_cut,
        energy_mean_output,
        energy_variance_output,
        ess_output,
        coefficient_output,
        did_solve_output
    );
}

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
) -> void
{
    const auto session = make_session(stream);
    if (not session) return;
    minsr_solve(
        samples,
        device_log_amplitudes,
        device_local_energies,
        device_log_proposals,
        device_gram,
        relative_cut,
        absolute_cut,
        energy_mean_output,
        energy_variance_output,
        ess_output,
        coefficient_output,
        did_solve_output,
        session->linalg()
    );
}

auto minsr_run_with_solve(
    MinsrSolveContext& solve,
    const QnpepsE2eConfig& config,
    i64 sample_count_input,
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
    f64* ess_output
) -> void
{
    auto& linalg = prepared_solve_linalg(solve);
    const auto sample_count = static_cast<usize>(sample_count_input);
    const int physical_dimension{config.dim_phys};
    const int site_count{config.lx * config.ly};

    i64 compact_count{};
    const auto slot_sites = qn_e2e::build_slot_site(config, compact_count);
    const auto dense_count = static_cast<i64>(physical_dimension) * compact_count;
    const auto compact_size = static_cast<usize>(compact_count);
    const auto dense_size = static_cast<usize>(dense_count);

    bool did_solve{};
    std::vector<cd> host_coefficients{};
    prepared_solve_run(
        solve,
        device_log_amplitudes,
        device_local_energies,
        device_log_proposals,
        device_gram,
        relative_cut,
        absolute_cut,
        energy_mean_output,
        energy_variance_output,
        ess_output,
        host_coefficients,
        did_solve
    );

    if (did_solve)
    {
        auto transient_arena = prepared_solve_transient_arena(solve);
        auto& arena = transient_arena.cursor();
        auto* const device_coefficients = arena.take<cd>(sample_count);
        auto* const device_accumulator = arena.take<cd>(dense_size);
        auto* const device_slot_sites = arena.take<i32>(compact_size);
        if (err_state() == QNPEPS_E2E_OK)
        {
            upload_async(linalg, device_coefficients, host_coefficients.data(), sample_count);
            upload_async(linalg, device_slot_sites, slot_sites.data(), compact_size);
            zero_async(linalg, device_accumulator, dense_size);
            if (device_rows)
            {
                launch_scatter(
                    linalg,
                    {
                        .accumulator = device_accumulator,
                        .rows = device_rows,
                        .row_base = 0,
                        .compact_count = compact_count,
                        .coefficients = device_coefficients,
                        .samples = samples,
                        .slot_sites = device_slot_sites,
                        .site_count = site_count,
                        .physical_dimension = physical_dimension,
                        .row_begin = 0,
                        .row_end = static_cast<int>(sample_count_input),
                    }
                );
            }
            else
            {
                const auto tile_rows =
                    host_tile_rows(host_tile_bytes, compact_count, sample_count_input);
                auto* const device_stage = arena.take<cf>(tile_rows * compact_size);
                if (err_state() == QNPEPS_E2E_OK)
                {
                    for (auto base = 0_uz; base < sample_count; base += tile_rows)
                    {
                        const auto count = std::min(tile_rows, sample_count - base);
                        upload_async(
                            linalg,
                            device_stage,
                            host_rows + base * compact_size,
                            count * compact_size
                        );
                        const auto row_begin = static_cast<int>(base);
                        const auto row_end = static_cast<int>(base + count);
                        launch_scatter(
                            linalg,
                            {
                                .accumulator = device_accumulator,
                                .rows = device_stage,
                                .row_base = static_cast<i64>(base),
                                .compact_count = compact_count,
                                .coefficients = device_coefficients,
                                .samples = samples,
                                .slot_sites = device_slot_sites,
                                .site_count = site_count,
                                .physical_dimension = physical_dimension,
                                .row_begin = row_begin,
                                .row_end = row_end,
                            }
                        );
                        CUDA_CHECK(cudaStreamSynchronize(linalg.stream()));
                    }
                }
            }
            launch_cast_accumulator(linalg, theta_output, device_accumulator, dense_count);
        }
        CUDA_CHECK(cudaStreamSynchronize(linalg.stream()));
        return;
    }

    zero_async(linalg, theta_output, dense_size);
    CUDA_CHECK(cudaStreamSynchronize(linalg.stream()));
}

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
    MinsrSolveContext* cached_solve
) -> void
{
    const std::unique_ptr<MinsrSolveContext, decltype(&prepared_solve_destroy)> owned_solve{
        cached_solve ? nullptr : prepared_solve_create(sample_count, linalg), prepared_solve_destroy
    };
    auto* const solve = cached_solve ? cached_solve : owned_solve.get();
    if (not solve)
    {
        const auto dense_size = static_cast<usize>(qn_e2e::dense_count(config));
        zero_async(linalg, theta_output, dense_size);
        CUDA_CHECK(cudaStreamSynchronize(linalg.stream()));
        return;
    }
    minsr_run_with_solve(
        *solve,
        config,
        sample_count,
        samples,
        device_log_amplitudes,
        device_local_energies,
        device_log_proposals,
        device_gram,
        device_rows,
        host_rows,
        host_tile_bytes,
        relative_cut,
        absolute_cut,
        theta_output,
        energy_mean_output,
        energy_variance_output,
        ess_output
    );
}

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
) -> void
{
    const auto session = make_session(stream);
    if (not session) return;
    qn_e2e_minsr_impl(
        config,
        sample_count,
        samples,
        device_log_amplitudes,
        device_local_energies,
        device_log_proposals,
        device_gram,
        device_rows,
        host_rows,
        host_tile_bytes,
        relative_cut,
        absolute_cut,
        theta_output,
        energy_mean_output,
        energy_variance_output,
        ess_output,
        session->linalg()
    );
}

[[nodiscard]] auto qn_e2e_minsr_scratch(const QnpepsE2eConfig&, i64, i64) -> u64
{
    return static_cast<u64>(arena_reservation_bytes());
}
}

struct qnpeps_e2e_minsr_ctx
{
    QnpepsE2eConfig config{};
    qnpeps::i64 sample_count{};
    qnpeps::i64 host_tile_bytes{};
    int device{};
    std::unique_ptr<qnpeps::Session> session{};
    qnpeps::Linalg* linalg{};
    qnpeps::minsr::MinsrSolveContext* solve{};
};

namespace qnpeps::minsr
{
namespace
{
auto minsr_context_create_impl(
    const QnpepsE2eConfig& config,
    i64 samples,
    i64 host_tile_bytes,
    std::unique_ptr<Session> session,
    Linalg& linalg,
    qnpeps_e2e_minsr_ctx** output
) -> void
{
    if (not output)
    {
        set_err(QNPEPS_E2E_ERR_NULL_ARG);
        return;
    }
    *output = nullptr;

    std::unique_ptr<MinsrSolveContext, decltype(&prepared_solve_destroy)> solve{
        prepared_solve_create(samples, linalg), prepared_solve_destroy
    };
    if (not solve or err_state() != QNPEPS_E2E_OK) return;

    auto* const context = new (std::nothrow) qnpeps_e2e_minsr_ctx;
    if (not context)
    {
        set_err(QNPEPS_E2E_ERR_OOM);
        return;
    }
    context->config = config;
    context->sample_count = samples;
    context->host_tile_bytes = host_tile_bytes;
    context->device = linalg.device();
    context->session = std::move(session);
    context->linalg = &linalg;
    context->solve = solve.release();
    *output = context;
}

[[nodiscard]] auto _minsr_context_run_check(const qnpeps_e2e_minsr_ctx& context) noexcept -> bool
{
    int device{};
    CUDA_CHECK(cudaGetDevice(&device));
    if (err_state() != QNPEPS_E2E_OK) return false;
    if (device != context.device)
    {
        set_err(QNPEPS_E2E_ERR_BAD_CONFIG);
        return false;
    }
    const auto valid_solve =
        context.solve and prepared_solve_samples(*context.solve) == context.sample_count;
    if (not valid_solve) set_err(QNPEPS_E2E_ERR_INTERNAL);
    return valid_solve;
}
}

auto minsr_context_create(
    const QnpepsE2eConfig& config,
    i64 samples,
    i64 host_tile_bytes,
    cudaStream_t stream,
    qnpeps_e2e_minsr_ctx** output
) -> void
{
    auto session = make_session(stream);
    if (not session) return;
    auto& linalg = session->linalg();
    minsr_context_create_impl(config, samples, host_tile_bytes, std::move(session), linalg, output);
}

auto minsr_context_create(
    const QnpepsE2eConfig& config,
    i64 samples,
    i64 host_tile_bytes,
    Linalg& linalg,
    qnpeps_e2e_minsr_ctx** output
) -> void
{
    minsr_context_create_impl(config, samples, host_tile_bytes, nullptr, linalg, output);
}

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
) -> void
{
    if (not _minsr_context_run_check(context)) return;
    minsr_run_with_solve(
        *context.solve,
        context.config,
        context.sample_count,
        samples,
        device_log_amplitudes,
        device_local_energies,
        device_log_proposals,
        device_gram,
        device_rows,
        host_rows,
        context.host_tile_bytes,
        relative_cut,
        absolute_cut,
        theta_output,
        energy_mean_output,
        energy_variance_output,
        ess_output
    );
}

[[nodiscard]] auto minsr_context_stream(const qnpeps_e2e_minsr_ctx& context) noexcept
    -> cudaStream_t
{
    return context.linalg->stream();
}

[[nodiscard]] auto minsr_context_device(const qnpeps_e2e_minsr_ctx& context) noexcept -> int
{
    return context.device;
}

auto minsr_context_destroy(qnpeps_e2e_minsr_ctx* context) noexcept -> void
{
    if (not context) return;
    int caller_device{};
    const auto have_caller = cudaGetDevice(&caller_device) == cudaSuccess;
    const auto switched = have_caller and caller_device != context->device
                          and cudaSetDevice(context->device) == cudaSuccess;
    cudaStreamSynchronize(context->linalg->stream());
    prepared_solve_destroy(context->solve);
    delete context;
    if (switched) cudaSetDevice(caller_device);
}
}
