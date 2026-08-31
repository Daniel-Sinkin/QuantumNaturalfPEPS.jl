#include "capi/qnpeps.h"
#include "sample_batch.h"
#include "test_oracle.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <vector>

namespace
{
[[nodiscard]] auto now_seconds() -> double
{
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

[[nodiscard]] auto check_cuda(cudaError_t rc, const char* what) -> bool
{
    if (rc != cudaSuccess)
    {
        std::fprintf(stderr, "[nsys_driver] cuda error at %s: %s\n", what, cudaGetErrorString(rc));
        return false;
    }
    return true;
}

[[nodiscard]] auto check_status(qnpeps_status rc, const char* what) -> bool
{
    if (rc != QNPEPS_OK)
    {
        std::fprintf(stderr, "[nsys_driver] qnpeps error at %s: %s\n", what, qnpeps_strerror(rc));
        return false;
    }
    return true;
}

[[nodiscard]] auto gen_peps(
    int (*selfgen)(const OracleConfig*, float*), const OracleConfig& oracle, size_t peps_float_count
) -> void*
{
    std::vector<float> host{};
    host.resize(peps_float_count);
    selfgen(&oracle, host.data());
    void* d{};
    if (not check_cuda(cudaMalloc(&d, peps_float_count * sizeof(float)), "malloc peps"))
        return nullptr;
    const auto copied = check_cuda(
        cudaMemcpy(d, host.data(), peps_float_count * sizeof(float), cudaMemcpyHostToDevice),
        "memcpy peps"
    );
    if (not copied) return nullptr;
    return d;
}

[[nodiscard]] auto run_oneshot(
    const QnpepsConfig& config,
    const void* d_peps_a,
    const void* d_peps_b,
    uint64_t count,
    int32_t iters,
    double& build_sample_ms,
    double& sample_only_ms
) -> bool
{
    void* d_dlenv{};
    const auto dlenv_bytes = static_cast<size_t>(qnpeps_dlenv_bytes(&config));
    if (not check_cuda(cudaMalloc(&d_dlenv, dlenv_bytes), "malloc dlenv")) return false;
    const auto dim_batch = test_sample_batch_size(count);
    const auto scratch_size = qnpeps_sample_scratch_bytes(&config, dim_batch);
    const auto scratch_bytes = static_cast<uint64_t>(scratch_size);
    void* d_scratch{};
    if (not check_cuda(cudaMalloc(&d_scratch, scratch_bytes), "malloc scratch")) return false;
    const auto sample_bytes = static_cast<size_t>(qnpeps_sample_bytes(&config, count));
    uint8_t* d_samples{};
    double* d_logpc{};
    if (not check_cuda(cudaMalloc(&d_samples, sample_bytes), "malloc samples")) return false;
    const auto logpc_bytes = static_cast<size_t>(count) * sizeof(double);
    if (not check_cuda(cudaMalloc(&d_logpc, logpc_bytes), "malloc logpc")) return false;

    const auto* peps_a = static_cast<const qnpeps_device_peps*>(d_peps_a);
    const auto* peps_b = static_cast<const qnpeps_device_peps*>(d_peps_b);
    auto* dlenv = static_cast<qnpeps_device_dlenv*>(d_dlenv);

    auto build = [&](const qnpeps_device_peps* peps) -> bool
    {
        return check_status(
            qnpeps_build_dlenv(&config, peps, dlenv, nullptr, nullptr), "oneshot build_dlenv"
        );
    };
    auto sample = [&](const qnpeps_device_peps* peps) -> bool
    {
        const QnpepsSampleArgs sample_args{
            .struct_size = sizeof(QnpepsSampleArgs),
            .peps = peps,
            .dlenv = dlenv,
            .gpus = 1,
            .scratch = d_scratch,
            .scratch_bytes = scratch_bytes,
            .samples_out = d_samples,
            .log_prob_config = d_logpc,
            .log_gauge = nullptr,
            .n_samples = count,
            .batch_base = 0,
            .dim_batch = dim_batch,
            .stream = nullptr,
        };
        return check_status(qnpeps_sample(&config, &sample_args), "oneshot sample");
    };
    auto build_and_sample = [&](const qnpeps_device_peps* peps) -> bool
    { return build(peps) and sample(peps); };

    if (not build_and_sample(peps_a)) return false;
    if (not build_and_sample(peps_b)) return false;

    if (not check_cuda(cudaDeviceSynchronize(), "sync warmup")) return false;
    if (not check_cuda(cudaProfilerStart(), "profiler start")) return false;

    if (not check_cuda(cudaDeviceSynchronize(), "sync pre build_sample")) return false;
    const auto t0 = now_seconds();
    for (auto i = 0; i < iters; ++i)
    {
        if (not build_and_sample((i % 2 == 0) ? peps_a : peps_b)) return false;
    }
    if (not check_cuda(cudaDeviceSynchronize(), "sync post build_sample")) return false;
    const auto t1 = now_seconds();

    if (not build(peps_a)) return false;

    if (not check_cuda(cudaDeviceSynchronize(), "sync pre sample_only")) return false;
    const auto t2 = now_seconds();
    for (auto i = 0; i < iters; ++i)
    {
        if (not sample(peps_a)) return false;
    }
    if (not check_cuda(cudaDeviceSynchronize(), "sync post sample_only")) return false;
    const auto t3 = now_seconds();

    if (not check_cuda(cudaProfilerStop(), "profiler stop")) return false;

    build_sample_ms = (t1 - t0) * 1000.0 / static_cast<double>(iters);
    sample_only_ms = (t3 - t2) * 1000.0 / static_cast<double>(iters);

    cudaFree(d_dlenv);
    cudaFree(d_scratch);
    cudaFree(d_samples);
    cudaFree(d_logpc);
    return true;
}

[[nodiscard]] auto run_ctx(
    const QnpepsConfig& config,
    const void* d_peps_a,
    const void* d_peps_b,
    uint64_t count,
    int32_t iters,
    double& build_sample_ms,
    double& sample_only_ms
) -> bool
{
    qnpeps_ctx* ctx{};
    if (not check_status(qnpeps_ctx_create(&config, nullptr, &ctx), "ctx_create") or not ctx)
        return false;

    const auto sample_bytes = static_cast<size_t>(qnpeps_sample_bytes(&config, count));
    uint8_t* d_samples{};
    double* d_logpc{};
    if (not check_cuda(cudaMalloc(&d_samples, sample_bytes), "malloc samples")) return false;
    const auto logpc_bytes = static_cast<size_t>(count) * sizeof(double);
    if (not check_cuda(cudaMalloc(&d_logpc, logpc_bytes), "malloc logpc")) return false;

    const auto* peps_a = static_cast<const qnpeps_device_peps*>(d_peps_a);
    const auto* peps_b = static_cast<const qnpeps_device_peps*>(d_peps_b);

    auto build = [&](const qnpeps_device_peps* peps) -> bool
    { return check_status(qnpeps_ctx_build_dlenv(ctx, peps, nullptr), "ctx_build_dlenv"); };
    auto sample = [&]() -> bool
    {
        const QnpepsCtxSampleArgs ctx_args{
            .struct_size = sizeof(QnpepsCtxSampleArgs),
            .samples_out = d_samples,
            .log_prob_config = d_logpc,
            .log_gauge = nullptr,
            .n_samples = count,
            .batch_base = 0,
            .dim_batch = std::min<uint64_t>(count, 2048),
        };
        return check_status(qnpeps_ctx_sample(ctx, &ctx_args), "ctx_sample");
    };

    for (auto i = 0; i < 4; ++i)
    {
        if (not build((i % 2 == 0) ? peps_a : peps_b)) return false;
        if (not sample()) return false;
    }
    if (not sample()) return false;

    if (not check_cuda(cudaDeviceSynchronize(), "sync warmup")) return false;
    if (not check_cuda(cudaProfilerStart(), "profiler start")) return false;

    if (not check_cuda(cudaDeviceSynchronize(), "sync pre build_sample")) return false;
    const auto t0 = now_seconds();
    for (auto i = 0; i < iters; ++i)
    {
        if (not build((i % 2 == 0) ? peps_a : peps_b)) return false;
        if (not sample()) return false;
    }
    if (not check_cuda(cudaDeviceSynchronize(), "sync post build_sample")) return false;
    const auto t1 = now_seconds();

    if (not check_cuda(cudaDeviceSynchronize(), "sync pre sample_only")) return false;
    const auto t2 = now_seconds();
    for (auto i = 0; i < iters; ++i)
    {
        if (not sample()) return false;
    }
    if (not check_cuda(cudaDeviceSynchronize(), "sync post sample_only")) return false;
    const auto t3 = now_seconds();

    if (not check_cuda(cudaProfilerStop(), "profiler stop")) return false;

    build_sample_ms = (t1 - t0) * 1000.0 / static_cast<double>(iters);
    sample_only_ms = (t3 - t2) * 1000.0 / static_cast<double>(iters);

    qnpeps_ctx_destroy(ctx);
    cudaFree(d_samples);
    cudaFree(d_logpc);
    return true;
}
}

int main(int argc, char** argv)
{
    if (argc < 7)
    {
        std::fprintf(
            stderr, "[nsys_driver] usage: nsys_driver <mode> <L> <D> <chi> <count> <iters>\n"
        );
        return 1;
    }
    auto mode = static_cast<const char*>(argv[1]);
    const auto lattice = std::atoi(argv[2]);
    const auto dim_bond = std::atoi(argv[3]);
    const auto chi_s = std::atoi(argv[4]);
    const auto count = static_cast<uint64_t>(std::strtoull(argv[5], nullptr, 10));
    const auto iters = std::atoi(argv[6]);

    const auto is_ctx = std::strcmp(mode, "ctx") == 0;
    const auto is_oneshot = std::strcmp(mode, "oneshot") == 0;
    if (not is_ctx and not is_oneshot)
    {
        std::fprintf(stderr, "[nsys_driver] mode must be 'oneshot' or 'ctx'\n");
        return 1;
    }
    if (iters < 1)
    {
        std::fprintf(stderr, "[nsys_driver] iters must be >= 1\n");
        return 1;
    }

    const auto oracle_handle = dlopen(oracle_so_path(), RTLD_NOW | RTLD_LOCAL);
    if (not oracle_handle)
    {
        std::fprintf(stderr, "[nsys_driver] dlopen oracle failed: %s\n", dlerror());
        return 1;
    }
    const auto selfgen = oracle_symbol<OracleGeneratePeps>(oracle_handle, "peps_export_selfgen");
    const auto paramcnt = oracle_symbol<OracleCount>(oracle_handle, "peps_sample_param_count");
    if (not selfgen or not paramcnt)
    {
        std::fprintf(stderr, "[nsys_driver] dlsym failed\n");
        return 1;
    }

    const auto dim_batch = static_cast<int32_t>(count);
    OracleConfig oracle_a{lattice, lattice, 2, dim_bond, chi_s, 1, dim_batch, 1, 1, 0};
    OracleConfig oracle_b{lattice, lattice, 2, dim_bond, chi_s, 2, dim_batch, 1, 1, 0};

    QnpepsConfig config{};
    config.struct_size = sizeof(QnpepsConfig);
    config.lx = lattice;
    config.ly = lattice;
    config.dim_phys = 2;
    config.dim_bond = dim_bond;
    config.chi_s = chi_s;
    config.chi_dl = dim_bond;
    config.seed = 1;

    const auto nparam = paramcnt(&oracle_a);
    const auto peps_float_count = static_cast<size_t>(2 * nparam);
    auto d_peps_a = gen_peps(selfgen, oracle_a, peps_float_count);
    auto d_peps_b = gen_peps(selfgen, oracle_b, peps_float_count);
    if (not d_peps_a or not d_peps_b) return 1;

    double build_sample_ms{};
    double sample_only_ms{};
    auto ok = false;
    if (is_ctx)
    {
        ok = run_ctx(config, d_peps_a, d_peps_b, count, iters, build_sample_ms, sample_only_ms);
    }
    else
    {
        ok = run_oneshot(config, d_peps_a, d_peps_b, count, iters, build_sample_ms, sample_only_ms);
    }

    qnpeps_sampler_pool_release();
    cudaFree(d_peps_a);
    cudaFree(d_peps_b);

    if (not ok)
    {
        std::fprintf(stderr, "[nsys_driver] FAILED\n");
        return 2;
    }

    std::printf(
        "[nsys_driver] mode=%s L=%d D=%d chi=%d count=%llu iters=%d "
        "build_sample_ms_per_iter=%.3f sample_only_ms_per_iter=%.3f\n",
        mode,
        lattice,
        dim_bond,
        chi_s,
        static_cast<unsigned long long>(count),
        iters,
        build_sample_ms,
        sample_only_ms
    );
    std::printf("[nsys_driver] DONE\n");
    return 0;
}
