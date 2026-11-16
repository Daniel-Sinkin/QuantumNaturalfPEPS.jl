#include "capi/qnpeps.h"
#include "dans_qnpeps_e2e.h"
#include "core/error.hpp"
#include "core/session.cuh"
#include "dlenv/build.cuh"
#include "linalg/linalg.cuh"
#include "minsr/solve.cuh"
#include "sampler/draw.cuh"
#include "term_builders.cuh"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <exception>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace
{
using qnpeps::operator""_uz;

constexpr qnpeps::usize k_repeats{3};
constexpr std::uint64_t k_seed{17};
constexpr std::uint64_t k_sample_count{64};
constexpr std::uint64_t k_small_batch{16};
constexpr std::uint64_t k_large_batch{32};
constexpr std::int64_t k_small_solve{48};
constexpr std::int64_t k_large_solve{80};
constexpr std::int64_t k_node_samples{32};
constexpr qnpeps::usize k_node_steps{2};
constexpr double k_node_relative_cut{1.0e-6};
constexpr double k_node_absolute_cut{1.0e-12};
constexpr double k_node_coupling{1.0};
constexpr int k_node_bond{2};
constexpr int k_node_chi{4};
constexpr int k_node_meo{16};
constexpr std::int64_t k_wait_samples{64};
constexpr std::int64_t k_wait_batch{16};
constexpr qnpeps::usize k_wait_settle_steps{2};
constexpr qnpeps::usize k_wait_steps{6};
constexpr double k_wait_learning_rate{1.0e-2};

int failures{};

struct ArenaReading
{
    qnpeps::usize lease_bytes{};
    qnpeps::usize high_water{};
    qnpeps::usize reservations{};
    qnpeps::usize transient_bytes{};

    auto operator==(const ArenaReading&) const -> bool = default;
};

[[nodiscard]] auto read_arena(qnpeps::Linalg& linalg) -> ArenaReading
{
    qnpeps::check(cudaStreamSynchronize(linalg.stream()));
    const auto transient_bytes = [&]
    {
        auto transient = linalg.transient_arena();
        return transient.cursor().remaining();
    }();
    auto& persistent = linalg.persistent_arena();
    return {
        .lease_bytes = persistent.total(),
        .high_water = persistent.high_water(),
        .reservations = persistent.reservation_count(),
        .transient_bytes = transient_bytes,
    };
}

auto report(std::string_view name, const ArenaReading& expected, const ArenaReading& observed)
    -> void
{
    const auto passed = observed == expected;
    std::printf(
        "[arena_release_gate] %s %.*s lease=%zu/%zu high_water=%zu/%zu reservations=%zu/%zu "
        "transient=%zu/%zu\n",
        passed ? "ok" : "FAIL",
        static_cast<int>(name.size()),
        name.data(),
        observed.lease_bytes,
        expected.lease_bytes,
        observed.high_water,
        expected.high_water,
        observed.reservations,
        expected.reservations,
        observed.transient_bytes,
        expected.transient_bytes
    );
    if (not passed) ++failures;
}

auto expect_held(std::string_view name, const ArenaReading& released, const ArenaReading& held)
    -> void
{
    const auto passed = held.reservations > released.reservations
                        and held.transient_bytes < released.transient_bytes;
    std::printf(
        "[arena_release_gate] %s %.*s reservations=%zu>%zu transient=%zu<%zu\n",
        passed ? "ok" : "FAIL",
        static_cast<int>(name.size()),
        name.data(),
        held.reservations,
        released.reservations,
        held.transient_bytes,
        released.transient_bytes
    );
    if (not passed) ++failures;
}

auto expect_repeatable(
    std::string_view name, qnpeps::Linalg& linalg, const std::function<void()>& operation
) -> void
{
    const auto before = read_arena(linalg);
    for (auto repeat = 0_uz; repeat < k_repeats; ++repeat)
    {
        operation();
        report(name, before, read_arena(linalg));
    }
}

auto expect_closed(std::string_view name) -> void
{
    qnpeps::check(cudaDeviceSynchronize());
    const auto probe = qnpeps::shared_device_arena(0)->acquire_lease(nullptr);
    const auto reservations = probe->reservation_count();
    const auto high_water = probe->high_water();
    const auto passed = reservations == 0 and high_water == 0;
    std::printf(
        "[arena_release_gate] %s %.*s reservations_with_quarantined=%zu high_water=%zu\n",
        passed ? "ok" : "FAIL",
        static_cast<int>(name.size()),
        name.data(),
        reservations,
        high_water
    );
    if (not passed) ++failures;
}

auto require_ok(qnpeps_status status, std::string_view operation) -> void
{
    qnpeps::require(
        status == QNPEPS_OK,
        "{} failed with status {}, {}",
        operation,
        static_cast<int>(status),
        qnpeps_last_error()
    );
}

[[nodiscard]] auto gate_config() -> QnpepsConfig
{
    QnpepsConfig config{};
    config.struct_size = sizeof(config);
    config.lx = 4;
    config.ly = 4;
    config.dim_phys = 2;
    config.dim_bond = 3;
    config.chi_s = 6;
    config.chi_dl = 9;
    config.seed = k_seed;
    config.sampling_mode = QNPEPS_SAMPLING_FAST;
    config.chi_c = 6;
    config.dlenv_truncation_route = QNPEPS_TRUNCATION_DEFAULT;
    config.sampler_truncation_route = QNPEPS_TRUNCATION_DEFAULT;
    return config;
}

struct DeviceAllocation
{
    void* pointer{};

    explicit DeviceAllocation(qnpeps::usize bytes) { qnpeps::check(cudaMalloc(&pointer, bytes)); }

    DeviceAllocation(const DeviceAllocation&) = delete;
    auto operator=(const DeviceAllocation&) -> DeviceAllocation& = delete;

    ~DeviceAllocation() { qnpeps::maybe_free_device(pointer); }

    template <typename T>
    [[nodiscard]] auto as() const noexcept -> T*
    {
        return static_cast<T*>(pointer);
    }
};

[[nodiscard]] auto node_config() -> QnpepsE2eConfig
{
    QnpepsE2eConfig config{};
    config.struct_size = sizeof(config);
    config.lx = 4;
    config.ly = 4;
    config.dim_phys = 2;
    config.dim_bond = k_node_bond;
    config.chi_s = k_node_chi;
    config.chi_dl = k_node_chi;
    config.chi_eo = k_node_chi;
    config.meo = k_node_meo;
    config.seed = k_seed;
    return config;
}

[[nodiscard]] auto node_peps_config(const QnpepsE2eConfig& config) -> QnpepsConfig
{
    QnpepsConfig peps_config{};
    peps_config.struct_size = sizeof(peps_config);
    peps_config.lx = config.lx;
    peps_config.ly = config.ly;
    peps_config.dim_phys = config.dim_phys;
    peps_config.dim_bond = config.dim_bond;
    peps_config.chi_s = config.chi_s;
    peps_config.chi_dl = config.chi_dl;
    peps_config.sampling_mode = QNPEPS_SAMPLING_FAST;
    peps_config.chi_c = config.chi_s;
    return peps_config;
}

[[nodiscard]] auto heisenberg_terms(int lx, int ly) -> eo_test::TermTableBuilder
{
    eo_test::TermTableBuilder terms{};
    for (auto row = 0; row < lx; ++row)
    {
        for (auto col = 0; col < ly; ++col)
        {
            const auto site = row * ly + col;
            if (col + 1 < ly) terms.add_heisenberg_bond(site, site + 1, k_node_coupling);
            if (row + 1 < lx) terms.add_heisenberg_bond(site, site + ly, k_node_coupling);
        }
    }
    return terms;
}

auto run_node_close() -> void
{
    const auto config = node_config();
    const auto terms = heisenberg_terms(config.lx, config.ly);
    const auto table = terms.table();
    std::int64_t dense{};
    require_ok(qnpeps_e2e_dense_count(&config, &dense), "dense count");
    const auto dense_count = static_cast<qnpeps::usize>(dense);
    const auto sample_count = static_cast<qnpeps::usize>(k_node_samples);
    const auto site_count = static_cast<qnpeps::usize>(config.lx * config.ly);

    const auto peps_config = node_peps_config(config);
    const auto peps_bytes = dense_count * sizeof(cuFloatComplex);
    qnpeps::require(
        qnpeps_peps_bytes(&peps_config) == static_cast<std::int64_t>(peps_bytes),
        "node PEPS of {} dense elements does not match the sampler PEPS size",
        dense
    );
    const DeviceAllocation device_peps{peps_bytes};
    const DeviceAllocation device_theta{peps_bytes};
    const DeviceAllocation device_samples{sample_count * site_count};
    const DeviceAllocation device_log_proposals{sample_count * sizeof(qnpeps::f64)};
    const DeviceAllocation device_log_gauges{sample_count * sizeof(qnpeps::f64)};
    const DeviceAllocation device_log_amplitudes{2 * sample_count * sizeof(qnpeps::f64)};
    const DeviceAllocation device_local_energies{2 * sample_count * sizeof(qnpeps::f64)};
    require_ok(
        qnpeps_random_unitary_peps(
            &peps_config, device_peps.as<qnpeps_device_peps>(), peps_bytes, k_seed, 0.0, nullptr
        ),
        "random node PEPS"
    );

    qnpeps_e2e_node* node{};
    require_ok(
        qnpeps_e2e_node_create(&config, 1, k_node_samples, 0, k_node_samples, 0, &table, &node),
        "node create"
    );
    auto energy_mean = std::vector<double>(2);
    double energy_variance{};
    double ess{};
    auto epochs = std::vector<std::int64_t>(sample_count);
    const auto run_step = [&]
    {
        const auto submit_status = qnpeps_e2e_node_submit_theta(node, device_peps.pointer);
        if (submit_status != QNPEPS_OK) return submit_status;
        return qnpeps_e2e_node_step(
            node,
            k_node_samples,
            k_node_relative_cut,
            k_node_absolute_cut,
            device_theta.as<qnpeps_e2e_cbuf>(),
            energy_mean.data(),
            &energy_variance,
            &ess,
            device_samples.as<std::uint8_t>(),
            device_log_proposals.as<double>(),
            device_log_gauges.as<double>(),
            device_log_amplitudes.as<double>(),
            device_local_energies.as<double>(),
            nullptr,
            epochs.data()
        );
    };
    auto step_status = QNPEPS_OK;
    for (auto step = 0_uz; step < k_node_steps and step_status == QNPEPS_OK; ++step)
        step_status = run_step();
    const auto destroy_status = qnpeps_e2e_node_destroy(node);
    require_ok(step_status, "node step");
    require_ok(destroy_status, "node destroy");
}

[[nodiscard]] auto read_device_arenas(int gpus) -> std::vector<QnpepsArenaReading>
{
    auto readings = std::vector<QnpepsArenaReading>(static_cast<qnpeps::usize>(gpus));
    for (auto device = 0_uz; device < readings.size(); ++device)
    {
        auto& reading = readings[device];
        reading.struct_size = sizeof(reading);
        require_ok(
            qnpeps_device_arena_reading(static_cast<std::int32_t>(device), &reading),
            "arena reading"
        );
    }
    return readings;
}

[[nodiscard]] auto is_same_reading(
    const QnpepsArenaReading& first, const QnpepsArenaReading& second
) -> bool
{
    return first.device == second.device and first.capacity_bytes == second.capacity_bytes
           and first.in_use_bytes == second.in_use_bytes and first.top_bytes == second.top_bytes
           and first.largest_free_bytes == second.largest_free_bytes
           and first.reservation_count == second.reservation_count
           and first.quarantined_count == second.quarantined_count;
}

[[nodiscard]] auto is_same_readings(
    const std::vector<QnpepsArenaReading>& first, const std::vector<QnpepsArenaReading>& second
) -> bool
{
    if (first.size() != second.size()) return false;
    for (auto device = 0_uz; device < first.size(); ++device)
    {
        if (not is_same_reading(first[device], second[device])) return false;
    }
    return true;
}

auto report_devices(
    std::string_view name,
    qnpeps::usize step,
    const std::vector<QnpepsArenaReading>& expected,
    const std::vector<QnpepsArenaReading>& observed
) -> void
{
    for (auto device = 0_uz; device < observed.size(); ++device)
    {
        const auto& seen = observed[device];
        const auto& want = expected[device];
        const auto passed = is_same_reading(want, seen);
        std::printf(
            "[arena_release_gate] %s %.*s step=%zu device=%d in_use=%" PRIu64 "/%" PRIu64
            " top=%" PRIu64 "/%" PRIu64 " largest_free=%" PRIu64 "/%" PRIu64
            " reservations=%" PRIu64 "/%" PRIu64 " quarantined=%" PRIu64 "/%" PRIu64 "\n",
            passed ? "ok" : "FAIL",
            static_cast<int>(name.size()),
            name.data(),
            step,
            seen.device,
            seen.in_use_bytes,
            want.in_use_bytes,
            seen.top_bytes,
            want.top_bytes,
            seen.largest_free_bytes,
            want.largest_free_bytes,
            seen.reservation_count,
            want.reservation_count,
            seen.quarantined_count,
            want.quarantined_count
        );
        if (not passed) ++failures;
    }
}

auto expect_status(std::string_view name, qnpeps_status expected, qnpeps_status observed) -> void
{
    const auto passed = observed == expected;
    std::printf(
        "[arena_release_gate] %s %.*s status=%d/%d\n",
        passed ? "ok" : "FAIL",
        static_cast<int>(name.size()),
        name.data(),
        static_cast<int>(observed),
        static_cast<int>(expected)
    );
    if (not passed) ++failures;
}

auto run_node_wait() -> void
{
    const auto config = node_config();
    const auto terms = heisenberg_terms(config.lx, config.ly);
    const auto table = terms.table();
    std::int64_t dense{};
    require_ok(qnpeps_e2e_dense_count(&config, &dense), "dense count");
    const auto peps_bytes = static_cast<qnpeps::usize>(dense) * sizeof(cuFloatComplex);
    const auto gpus = []
    {
        int count{};
        qnpeps::check(cudaGetDeviceCount(&count));
        return count;
    }();

    const auto peps_config = node_peps_config(config);
    const DeviceAllocation device_peps{peps_bytes};
    require_ok(
        qnpeps_random_unitary_peps(
            &peps_config, device_peps.as<qnpeps_device_peps>(), peps_bytes, k_seed, 0.0, nullptr
        ),
        "random node PEPS"
    );

    expect_status("node_wait_null", QNPEPS_ERR_NULL_ARG, qnpeps_e2e_node_wait_environment(nullptr));

    qnpeps_e2e_node* node{};
    require_ok(
        qnpeps_e2e_node_create(&config, gpus, k_wait_samples, 0, k_wait_batch, 0, &table, &node),
        "node create"
    );
    expect_status("node_wait_unsubmitted", QNPEPS_OK, qnpeps_e2e_node_wait_environment(node));

    auto energy_mean = std::vector<double>(2);
    double energy_variance{};
    double ess{};
    QnpepsE2eEulerStepArgs args{};
    args.struct_size = sizeof(args);
    args.precision = QNPEPS_E2E_UPDATE_F32;
    args.n_samples = k_wait_samples;
    args.relative_cut = k_node_relative_cut;
    args.absolute_cut = k_node_absolute_cut;
    args.learning_rate = k_wait_learning_rate;
    args.peps_f32_io = device_peps.as<qnpeps_e2e_cbuf>();
    args.peps_f32_bytes = peps_bytes;
    args.energy_mean_output = energy_mean.data();
    args.energy_variance_output = &energy_variance;
    args.ess_output = &ess;

    const auto run_steps = [&]
    {
        require_ok(qnpeps_e2e_node_submit_theta(node, device_peps.pointer), "node submit");
        auto previous = std::vector<QnpepsArenaReading>{};
        auto in_flight_reads = 0_uz;
        for (auto step = 0_uz; step < k_wait_steps; ++step)
        {
            require_ok(qnpeps_e2e_node_step_euler(node, &args), "node Euler step");
            const auto returned = read_device_arenas(gpus);
            require_ok(qnpeps_e2e_node_wait_environment(node), "environment wait");
            const auto waited = read_device_arenas(gpus);
            if (not is_same_readings(returned, waited)) ++in_flight_reads;
            if (step > k_wait_settle_steps)
                report_devices("node_wait_step", step, previous, waited);
            previous = waited;
        }
        require_ok(qnpeps_e2e_node_wait_environment(node), "repeated environment wait");
        report_devices("node_wait_repeated", k_wait_steps, previous, read_device_arenas(gpus));
        std::printf(
            "[arena_release_gate] info node_wait gpus=%d in_flight_reads=%zu of %zu\n",
            gpus,
            in_flight_reads,
            k_wait_steps
        );
    };
    const auto run_status = qnpeps::guarded(run_steps);
    const auto run_failure = std::string{qnpeps::last_error_message()};
    const auto destroy_status = qnpeps_e2e_node_destroy(node);
    qnpeps::require(run_status == QNPEPS_OK, "node wait steps failed, {}", run_failure);
    require_ok(destroy_status, "node destroy");
}

auto run_session_gate() -> void
{
    const auto config = gate_config();
    const auto peps_bytes = static_cast<qnpeps::usize>(qnpeps_peps_bytes(&config));
    const auto dlenv_bytes = static_cast<qnpeps::usize>(qnpeps_dlenv_bytes(&config));
    const auto site_count = static_cast<qnpeps::usize>(config.lx * config.ly);
    const DeviceAllocation device_peps{peps_bytes};
    const DeviceAllocation device_dlenv{dlenv_bytes};
    const DeviceAllocation device_samples{k_sample_count * site_count};
    const DeviceAllocation device_log_proposals{k_sample_count * sizeof(qnpeps::f64)};
    const DeviceAllocation device_log_gauges{k_sample_count * sizeof(qnpeps::f64)};

    const auto peps_status = qnpeps_random_unitary_peps(
        &config, device_peps.as<qnpeps_device_peps>(), peps_bytes, k_seed, 0.0, nullptr
    );
    qnpeps::require(
        peps_status == QNPEPS_OK, "random PEPS failed with status {}", static_cast<int>(peps_status)
    );

    auto session = qnpeps::make_session(nullptr);
    auto& linalg = session->linalg();

    {
        const auto released = read_arena(linalg);
        auto* const solve = qnpeps::minsr::prepared_solve_create(k_small_solve, linalg);
        expect_held("minsr_solve_held", released, read_arena(linalg));
        qnpeps::minsr::prepared_solve_destroy(solve);
        report("minsr_solve_released", released, read_arena(linalg));
    }

    expect_repeatable(
        "minsr_solve_resized",
        linalg,
        [&]
        {
            auto* const small = qnpeps::minsr::prepared_solve_create(k_small_solve, linalg);
            qnpeps::minsr::prepared_solve_destroy(small);
            auto* const large = qnpeps::minsr::prepared_solve_create(k_large_solve, linalg);
            qnpeps::minsr::prepared_solve_destroy(large);
        }
    );

    expect_repeatable(
        "dlenv_build",
        linalg,
        [&]
        {
            qnpeps::dlenv::build_dlenv_packed(
                config, device_peps.pointer, device_dlenv.pointer, nullptr, linalg
            );
        }
    );

    const auto draw = [&](std::uint64_t dim_batch)
    {
        qnpeps::sampler::sample(
            config,
            {
                .device_peps = device_peps.pointer,
                .device_dlenv = device_dlenv.pointer,
                .scratch = nullptr,
                .scratch_bytes = 0,
                .output = device_samples.as<qnpeps::u8>(),
                .logpc_out = device_log_proposals.as<qnpeps::f64>(),
                .lognorm_out = device_log_gauges.as<qnpeps::f64>(),
                .n_samples = k_sample_count,
                .batch_base = 0,
                .dim_batch = dim_batch,
                .stream = linalg.stream(),
                .output_location = qnpeps::sampler::SampleOutputLocation::device,
            },
            linalg
        );
    };
    expect_repeatable("sampler_draw", linalg, [&] { draw(k_small_batch); });
    expect_repeatable(
        "sampler_draw_growing_batch",
        linalg,
        [&]
        {
            draw(k_small_batch);
            draw(k_large_batch);
        }
    );
}

auto run_gate() -> void
{
    run_session_gate();
    expect_closed("session_closed");
    run_node_close();
    expect_closed("node_closed");
    run_node_wait();
    expect_closed("node_wait_closed");
}
}

auto main() -> int
{
    try
    {
        qnpeps::check(cudaSetDevice(0));
        run_gate();
    }
    catch (const std::exception& error)
    {
        std::printf("[arena_release_gate] FAIL exception %s\n", error.what());
        return 1;
    }
    std::printf("[arena_release_gate] %s failures=%d\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
