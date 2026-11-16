#include "capi/qnpeps.h"
#include "dans_qnpeps_e2e.h"
#include "dans_qnpeps_eloc.h"
#include "term_builders.cuh"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <string>
#include <string_view>
#include <vector>

namespace
{
int failures{};

auto expect_failure(
    std::string_view name, qnpeps_status status, qnpeps_status expected, std::string_view cause
) -> void
{
    const std::string message{qnpeps_last_error()};
    const auto passed =
        status == expected and not message.empty() and message.find(cause) != std::string::npos;
    std::printf(
        "[error_channel_gate] %s %.*s status=%d expected=%d message=%s\n",
        passed ? "ok" : "FAIL",
        static_cast<int>(name.size()),
        name.data(),
        static_cast<int>(status),
        static_cast<int>(expected),
        message.c_str()
    );
    if (not passed) ++failures;
}

auto expect_success(std::string_view name, qnpeps_status status) -> void
{
    const auto passed = status == QNPEPS_OK;
    std::printf(
        "[error_channel_gate] %s %.*s status=%d message=%s\n",
        passed ? "ok" : "FAIL",
        static_cast<int>(name.size()),
        name.data(),
        static_cast<int>(status),
        qnpeps_last_error()
    );
    if (not passed) ++failures;
}

template <typename T>
auto device_buffer(std::size_t count) -> T*
{
    void* pointer{};
    if (cudaMalloc(&pointer, sizeof(T) * (count < 1 ? 1 : count)) != cudaSuccess) return nullptr;
    return static_cast<T*>(pointer);
}

auto sampler_config(int lx, int ly) -> QnpepsConfig
{
    QnpepsConfig config{};
    config.struct_size = sizeof(QnpepsConfig);
    config.lx = lx;
    config.ly = ly;
    config.dim_phys = 2;
    config.dim_bond = 2;
    config.chi_s = 4;
    config.chi_dl = 4;
    config.seed = 7;
    config.sampling_mode = QNPEPS_SAMPLING_FAST;
    return config;
}

auto random_peps(const QnpepsConfig& config) -> void*
{
    const auto bytes = qnpeps_peps_bytes(&config);
    if (bytes < 0) return nullptr;
    auto* peps = device_buffer<std::uint8_t>(static_cast<std::size_t>(bytes));
    const auto status = qnpeps_random_unitary_peps(
        &config,
        reinterpret_cast<qnpeps_device_peps*>(peps),
        static_cast<std::uint64_t>(bytes),
        11,
        0.0,
        nullptr
    );
    cudaDeviceSynchronize();
    return status == QNPEPS_OK ? peps : nullptr;
}

auto rejected_configuration() -> void
{
    auto config = sampler_config(1, 3);
    qnpeps_ctx* context{};
    expect_failure(
        "rejected_configuration",
        qnpeps_ctx_create(&config, nullptr, &context),
        QNPEPS_ERR_BAD_CONFIG,
        "config rejects lattice=1x3"
    );
    QnpepsE2eConfig node_config{};
    node_config.struct_size = sizeof(QnpepsE2eConfig);
    node_config.lx = 3;
    node_config.ly = 3;
    node_config.dim_phys = 3;
    node_config.dim_bond = 2;
    std::int64_t count{};
    expect_failure(
        "rejected_node_configuration",
        qnpeps_e2e_dense_count(&node_config, &count),
        QNPEPS_ERR_BAD_CONFIG,
        "dim_phys=3"
    );
    const std::string cause{qnpeps_last_error()};
    static_cast<void>(qnpeps_peps_bytes(nullptr));
    qnpeps_ctx_destroy(nullptr);
    const auto kept = not cause.empty() and std::string_view{qnpeps_last_error()} == cause;
    std::printf(
        "[error_channel_gate] %s message_survives_query_and_destroy\n", kept ? "ok" : "FAIL"
    );
    if (not kept) ++failures;
}

auto arena_overflow() -> void
{
    constexpr int rows{16};
    constexpr int cols{16};
    constexpr int rank{4};
    constexpr int batch{2};
    auto* input = device_buffer<float>(2 * rows * cols * batch);
    auto* q_out = device_buffer<float>(2 * rows * rank * batch);
    auto* r_out = device_buffer<float>(2 * rank * cols * batch);
    auto* scratch = device_buffer<std::uint8_t>(256);
    expect_failure(
        "arena_overflow",
        qnpeps_batched_rangefinder(
            input,
            rows,
            cols,
            rank,
            batch,
            rows * cols,
            3,
            q_out,
            rows * rank,
            r_out,
            rank * cols,
            scratch,
            256,
            nullptr
        ),
        QNPEPS_ERR_OOM,
        "arena needs"
    );
    cudaFree(scratch);
    cudaFree(r_out);
    cudaFree(q_out);
    cudaFree(input);
}

auto invalid_cuda_call() -> void
{
    int device_count{};
    cudaGetDeviceCount(&device_count);
    const auto config = sampler_config(3, 3);
    auto* peps = device_buffer<std::uint8_t>(64);
    auto* dlenv = device_buffer<std::uint8_t>(64);
    auto* samples = device_buffer<std::uint8_t>(64);
    QnpepsSampleArgs args{};
    args.struct_size = sizeof(QnpepsSampleArgs);
    args.peps = reinterpret_cast<const qnpeps_device_peps*>(peps);
    args.dlenv = reinterpret_cast<const qnpeps_device_dlenv*>(dlenv);
    args.gpus = device_count + 1;
    args.samples_out = samples;
    args.n_samples = 4;
    args.dim_batch = 4;
    expect_failure(
        "invalid_cuda_call",
        qnpeps_sample(&config, &args),
        QNPEPS_ERR_CUDA,
        "cudaErrorInvalidDevice"
    );
    int current_device{-1};
    cudaGetDevice(&current_device);
    std::printf(
        "[error_channel_gate] %s device_restored device=%d\n",
        current_device == 0 ? "ok" : "FAIL",
        current_device
    );
    if (current_device != 0) ++failures;
    cudaFree(samples);
    cudaFree(dlenv);
    cudaFree(peps);
}

auto sample_once(qnpeps_ctx* context, std::uint8_t* samples, double* logpc, double* gauge)
    -> qnpeps_status
{
    QnpepsCtxSampleArgs args{};
    args.struct_size = sizeof(QnpepsCtxSampleArgs);
    args.samples_out = samples;
    args.log_prob_config = logpc;
    args.log_gauge = gauge;
    args.n_samples = 8;
    args.dim_batch = 4;
    return qnpeps_ctx_sample(context, &args);
}

auto graph_capture_failure() -> void
{
    const auto config = sampler_config(3, 3);
    auto* peps = random_peps(config);
    qnpeps_ctx* context{};
    expect_success("capture_context_create", qnpeps_ctx_create(&config, nullptr, &context));
    expect_success(
        "capture_environment_build",
        qnpeps_ctx_build_dlenv(context, reinterpret_cast<const qnpeps_device_peps*>(peps), nullptr)
    );
    auto* samples = device_buffer<std::uint8_t>(8 * 9);
    auto* logpc = device_buffer<double>(8);
    auto* gauge = device_buffer<double>(8);
    setenv("QNPEPS_INJECT_CAPTURE_FAILURE", "1", 1);
    expect_failure(
        "graph_capture_failure",
        sample_once(context, samples, logpc, gauge),
        QNPEPS_ERR_INTERNAL,
        "graph capture"
    );
    unsetenv("QNPEPS_INJECT_CAPTURE_FAILURE");
    expect_success("sampling_after_capture_failure", sample_once(context, samples, logpc, gauge));
    expect_success(
        "second_sampling_after_capture_failure", sample_once(context, samples, logpc, gauge)
    );
    qnpeps_ctx_destroy(context);
    cudaFree(gauge);
    cudaFree(logpc);
    cudaFree(samples);
    cudaFree(peps);
}

auto node_config() -> QnpepsE2eConfig
{
    QnpepsE2eConfig config{};
    config.struct_size = sizeof(QnpepsE2eConfig);
    config.lx = 3;
    config.ly = 3;
    config.dim_phys = 2;
    config.dim_bond = 2;
    config.chi_s = 4;
    config.chi_dl = 4;
    config.chi_eo = 4;
    config.meo = 4;
    config.seed = 5;
    config.sampling_mode = QNPEPS_SAMPLING_FAST;
    return config;
}

auto node_step(qnpeps_e2e_node* node, std::int64_t samples, std::int64_t dense) -> qnpeps_status
{
    auto* theta = device_buffer<float>(static_cast<std::size_t>(2 * dense));
    std::array<double, 2> energy_mean{};
    double energy_variance{};
    double ess{};
    const auto status = qnpeps_e2e_node_step(
        node,
        samples,
        1.0e-8,
        1.0e-12,
        reinterpret_cast<qnpeps_e2e_cbuf*>(theta),
        energy_mean.data(),
        &energy_variance,
        &ess,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr
    );
    cudaFree(theta);
    return status;
}

auto worker_thread_failure() -> void
{
    int device_count{};
    cudaGetDeviceCount(&device_count);
    if (device_count < 2)
    {
        std::printf(
            "[error_channel_gate] skip worker_thread_failure needs 2 GPUs, found %d\n", device_count
        );
        return;
    }
    const auto config = node_config();
    eo_test::TermTableBuilder terms{};
    for (int row{}; row < config.lx; ++row)
    {
        for (int column{}; column < config.ly; ++column)
        {
            const auto site = row * config.ly + column;
            if (column + 1 < config.ly) terms.add_zz(site, site + 1, 1.0);
            if (row + 1 < config.lx) terms.add_zz(site, site + config.ly, 1.0);
        }
    }
    const auto table = terms.table();
    std::int64_t dense{};
    qnpeps_e2e_dense_count(&config, &dense);
    const auto peps_config = [&]
    {
        auto sampler = sampler_config(config.lx, config.ly);
        sampler.seed = config.seed;
        return sampler;
    }();
    auto* peps = random_peps(peps_config);

    setenv("QNPEPS_E2E_INJECT_OOM_GPU", "1", 1);
    qnpeps_e2e_node* node{};
    const auto created = qnpeps_e2e_node_create(&config, 2, 24, 0, 4, 0, &table, &node);
    unsetenv("QNPEPS_E2E_INJECT_OOM_GPU");
    expect_success("worker_node_create", created);
    if (created != QNPEPS_OK)
    {
        cudaFree(peps);
        return;
    }
    expect_success("worker_submit_theta", qnpeps_e2e_node_submit_theta(node, peps));
    const auto cause = std::string_view{"[GPU 1]"};
    expect_failure("worker_thread_failure", node_step(node, 8, dense), QNPEPS_ERR_OOM, cause);
    const std::string first_message{qnpeps_last_error()};
    const auto injected =
        first_message.find("injected out-of-memory failure on GPU 1") != std::string::npos;
    std::printf("[error_channel_gate] %s worker_cause_in_message\n", injected ? "ok" : "FAIL");
    if (not injected) ++failures;
    expect_failure("worker_node_stays_failed", node_step(node, 8, dense), QNPEPS_ERR_OOM, cause);
    expect_success("worker_node_destroy", qnpeps_e2e_node_destroy(node));
    cudaSetDevice(0);
    cudaFree(peps);
}

auto zipup_sweep_ends_on_failure() -> void
{
    const auto config = sampler_config(3, 3);
    qnpeps_zipup_ctx* context{};
    expect_success("zipup_context_create", qnpeps_zipup_ctx_create(&config, 4, nullptr, &context));
    if (context == nullptr) return;
    expect_success("zipup_sweep_begin", qnpeps_zipup_ctx_begin(context));
    auto scale = 0.0;
    expect_failure(
        "zipup_finish_rejects_count",
        qnpeps_zipup_ctx_finish(context, &scale, 1),
        QNPEPS_ERR_BAD_CONFIG,
        "zipup finish asks for 1 scales"
    );
    expect_success("zipup_begin_after_failed_finish", qnpeps_zipup_ctx_begin(context));
    qnpeps_zipup_ctx_destroy(context);
}

auto process_continues() -> void
{
    const auto config = sampler_config(3, 3);
    qnpeps_ctx* context{};
    expect_success("process_continues", qnpeps_ctx_create(&config, nullptr, &context));
    const auto cleared = std::string_view{qnpeps_last_error()}.empty();
    std::printf("[error_channel_gate] %s message_cleared_after_success\n", cleared ? "ok" : "FAIL");
    if (not cleared) ++failures;
    qnpeps_ctx_destroy(context);
}
}

auto main() -> int
{
    cudaSetDevice(0);
    rejected_configuration();
    arena_overflow();
    invalid_cuda_call();
    graph_capture_failure();
    worker_thread_failure();
    zipup_sweep_ends_on_failure();
    process_continues();
    std::printf("[error_channel_gate] %d failures\n", failures);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
