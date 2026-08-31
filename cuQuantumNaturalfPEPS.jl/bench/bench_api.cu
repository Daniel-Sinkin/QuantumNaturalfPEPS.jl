#include "capi/qnpeps.h"
#include "core/defer.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cuda_runtime.h>
#include <random>
#include <utility>
#include <vector>

namespace
{
using usize = std::size_t;
inline constexpr uint64_t k_max_batch_size{2048};

[[nodiscard]] auto sample_batch_size(uint64_t count) -> uint64_t
{
    return count < k_max_batch_size ? count : k_max_batch_size;
}

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
        std::fprintf(stderr, "cuda error at %s: %s\n", what, cudaGetErrorString(rc));
        return false;
    }
    return true;
}

[[nodiscard]] auto check_status(qnpeps_status rc, const char* what) -> bool
{
    if (rc != QNPEPS_OK)
    {
        std::fprintf(stderr, "qnpeps error at %s: %s\n", what, qnpeps_strerror(rc));
        return false;
    }
    return true;
}

[[nodiscard]] auto mean_sd(const std::vector<double>& samples) -> std::pair<double, double>
{
    const auto count = static_cast<double>(samples.size());
    auto sum = 0.0;
    for (const auto value : samples)
        sum += value;
    const auto mean = sum / count;
    auto squared_error = 0.0;
    for (const auto value : samples)
        squared_error += (value - mean) * (value - mean);
    const auto sd = std::sqrt(squared_error / (count - 1.0));
    return {mean, sd};
}

[[nodiscard]] auto
bench(int32_t lattice, int32_t dim_bond, int32_t chi_s, uint64_t count, int32_t iters) -> bool
{
    QnpepsConfig config{};
    config.struct_size = sizeof(QnpepsConfig);
    config.lx = lattice;
    config.ly = lattice;
    config.dim_phys = 2;
    config.dim_bond = dim_bond;
    config.chi_s = chi_s;
    config.chi_dl = dim_bond;
    config.seed = 1;

    const auto peps_bytes = qnpeps_peps_bytes(&config);
    if (peps_bytes < 0)
    {
        std::fprintf(stderr, "bad config\n");
        return false;
    }
    const auto peps_bytes_uz = static_cast<usize>(peps_bytes);
    const auto count_uz = static_cast<usize>(count);
    const auto iters_uz = static_cast<usize>(iters);

    std::vector<float> host_peps{};
    host_peps.resize(peps_bytes_uz / sizeof(float));
    std::mt19937 rng{1234567u};
    std::uniform_real_distribution<float> dist{-1.0f, 1.0f};
    for (auto& value : host_peps)
        value = dist(rng);

    void* d_peps{};
    if (not check_cuda(cudaMalloc(&d_peps, peps_bytes_uz), "malloc peps")) return false;
    DEFER([&] { cudaFree(d_peps); });
    if (not check_cuda(
            cudaMemcpy(d_peps, host_peps.data(), peps_bytes_uz, cudaMemcpyHostToDevice),
            "memcpy peps"
        ))
    {
        return false;
    }

    const auto dlenv_bytes = static_cast<usize>(qnpeps_dlenv_bytes(&config));
    void* d_dlenv_warmup{};
    if (not check_cuda(cudaMalloc(&d_dlenv_warmup, dlenv_bytes), "malloc dlenv warmup"))
        return false;
    DEFER([&] { cudaFree(d_dlenv_warmup); });

    std::vector<void*> d_dlenv_warm{};
    d_dlenv_warm.resize(iters_uz);
    DEFER(
        [&]
        {
            for (auto* warm : d_dlenv_warm)
                cudaFree(warm);
        }
    );
    for (auto& warm : d_dlenv_warm)
    {
        if (not check_cuda(cudaMalloc(&warm, dlenv_bytes), "malloc dlenv warm")) return false;
    }

    const auto dim_batch = sample_batch_size(count);
    const auto scratch_bytes =
        static_cast<uint64_t>(qnpeps_sample_scratch_bytes(&config, dim_batch));
    void* d_scratch{};
    if (not check_cuda(cudaMalloc(&d_scratch, scratch_bytes), "malloc scratch")) return false;
    DEFER([&] { cudaFree(d_scratch); });

    const auto sample_bytes = static_cast<usize>(qnpeps_sample_bytes(&config, count));
    uint8_t* d_samples{};
    if (not check_cuda(cudaMalloc(&d_samples, sample_bytes), "malloc samples")) return false;
    DEFER([&] { cudaFree(d_samples); });

    double* d_logpc{};
    if (not check_cuda(cudaMalloc(&d_logpc, count_uz * sizeof(double)), "malloc logpc"))
    {
        return false;
    }
    DEFER([&] { cudaFree(d_logpc); });

    const auto* peps = static_cast<const qnpeps_device_peps*>(d_peps);
    const auto* dlenv = static_cast<const qnpeps_device_dlenv*>(d_dlenv_warmup);

    auto build = [&](void* dlenv_out) -> bool
    {
        return check_status(
            qnpeps_build_dlenv(
                &config, peps, static_cast<qnpeps_device_dlenv*>(dlenv_out), nullptr, nullptr
            ),
            "build_dlenv"
        );
    };
    auto sample = [&]() -> bool
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
        return check_status(qnpeps_sample(&config, &sample_args), "sample");
    };

    if (not check_cuda(cudaDeviceSynchronize(), "sync pre build_warmup")) return false;
    if (not build(d_dlenv_warmup)) return false;
    if (not check_cuda(cudaDeviceSynchronize(), "sync post build_warmup")) return false;

    std::vector<double> build_samples{};
    build_samples.reserve(iters_uz);
    for (auto* warm : d_dlenv_warm)
    {
        const auto iter_start = now_seconds();
        if (not build(warm)) return false;
        if (not check_cuda(cudaDeviceSynchronize(), "sync build iter")) return false;
        build_samples.push_back((now_seconds() - iter_start) * 1000.0);
    }

    for (auto i = 0; i < 2; ++i)
    {
        if (not sample()) return false;
    }
    if (not check_cuda(cudaDeviceSynchronize(), "sync warmup sample")) return false;

    std::vector<double> sample_samples{};
    sample_samples.reserve(iters_uz);
    for (auto i = 0; i < iters; ++i)
    {
        const auto iter_start = now_seconds();
        if (not sample()) return false;
        if (not check_cuda(cudaDeviceSynchronize(), "sync sample iter")) return false;
        sample_samples.push_back((now_seconds() - iter_start) * 1000.0);
    }

    const auto [build_ms, build_sd] = mean_sd(build_samples);
    const auto [sample_ms, sample_sd] = mean_sd(sample_samples);

    std::printf(
        "api=c L=%d D=%d chi=%d count=%llu iters=%d "
        "build_ms=%.3f build_sd=%.3f sample_ms=%.3f sample_sd=%.3f\n",
        lattice,
        dim_bond,
        chi_s,
        static_cast<unsigned long long>(count),
        iters,
        build_ms,
        build_sd,
        sample_ms,
        sample_sd
    );
    return true;
}

[[nodiscard]] auto report_bytes(int32_t lattice, int32_t dim_bond, int32_t chi_s, uint64_t count)
    -> bool
{
    QnpepsConfig config{};
    config.struct_size = sizeof(QnpepsConfig);
    config.lx = lattice;
    config.ly = lattice;
    config.dim_phys = 2;
    config.dim_bond = dim_bond;
    config.chi_s = chi_s;
    config.chi_dl = dim_bond;
    config.seed = 1;

    const auto peps_b = qnpeps_peps_bytes(&config);
    if (peps_b < 0)
    {
        std::fprintf(stderr, "bad config\n");
        return false;
    }
    const auto dlenv_b = qnpeps_dlenv_bytes(&config);
    const auto scratch_b = qnpeps_sample_scratch_bytes(&config, sample_batch_size(count));
    const auto sample_b = qnpeps_sample_bytes(&config, count);
    constexpr double gib = 1024.0 * 1024.0 * 1024.0;
    std::printf(
        "bytes L=%d D=%d chi=%d count=%llu peps_GiB=%.4f dlenv_GiB=%.4f "
        "scratch_GiB=%.4f samples_GiB=%.4f logpc_GiB=%.4f\n",
        lattice,
        dim_bond,
        chi_s,
        static_cast<unsigned long long>(count),
        static_cast<double>(peps_b) / gib,
        static_cast<double>(dlenv_b) / gib,
        static_cast<double>(scratch_b) / gib,
        static_cast<double>(sample_b) / gib,
        static_cast<double>(count) * sizeof(double) / gib
    );
    return true;
}
}

int main(int argc, char** argv)
{
    DEFER([] { qnpeps_sampler_pool_release(); });
    bool ok{false};
    if (argc == 2)
    {
        const int bench_case = std::atoi(argv[1]);
        if (bench_case == 1)
            ok = bench(8, 4, 4, 1024, 100);
        else if (bench_case == 2)
            ok = bench(16, 7, 7, 512, 40);
        else
            return 1;
    }
    else if (argc == 6 && std::strcmp(argv[1], "bytes") == 0)
    {
        ok = report_bytes(
            std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]),
            std::strtoull(argv[5], nullptr, 10)
        );
    }
    else if (argc == 6)
    {
        ok = bench(
            std::atoi(argv[1]),
            std::atoi(argv[2]),
            std::atoi(argv[3]),
            std::strtoull(argv[4], nullptr, 10),
            std::atoi(argv[5])
        );
    }
    else
    {
        std::fprintf(
            stderr,
            "usage: %s <case 1|2> | <L D chi count iters> | bytes <L D chi count>\n",
            argv[0]
        );
        return 1;
    }
    if (not ok) return 1;
    return 0;
}
