#include "capi/qnpeps.h"
#include "sample_batch.h"
#include "test_oracle.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <vector>

namespace
{
struct SampleResult
{
    std::vector<uint8_t> samples;
    std::vector<double> logpc;
    bool status_ok;
};

[[nodiscard]] auto bytes_equal(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) -> bool
{
    if (a.size() != b.size()) return false;
    if (a.empty()) return true;
    return std::memcmp(a.data(), b.data(), a.size()) == 0;
}

[[nodiscard]] auto logs_bit_exact(const std::vector<double>& a, const std::vector<double>& b)
    -> bool
{
    if (a.size() != b.size()) return false;
    if (a.empty()) return true;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

auto report_match(const char* name, const SampleResult& ref, const SampleResult& got, int& fails)
    -> void
{
    const auto samples_ok = bytes_equal(ref.samples, got.samples);
    const auto logpc_ok = logs_bit_exact(ref.logpc, got.logpc);
    const auto ok = samples_ok and logpc_ok and got.status_ok;
    std::printf(
        "[stage4_gate] %-28s samples %s  logpc %s  %s\n",
        name,
        samples_ok ? "EQ" : "NE",
        logpc_ok ? "EQ" : "NE",
        ok ? "BIT-EXACT" : "*** DIFFER ***"
    );
    if (not ok) ++fails;
}

auto gen_peps(
    int (*selfgen)(const OracleConfig*, float*), const OracleConfig& oracle, size_t peps_float_count
) -> void*
{
    std::vector<float> host{};
    host.resize(peps_float_count);
    selfgen(&oracle, host.data());
    void* d{};
    cudaMalloc(&d, peps_float_count * sizeof(float));
    cudaMemcpy(d, host.data(), peps_float_count * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}

[[nodiscard]] auto oneshot(const QnpepsConfig& config, const void* d_peps, uint64_t count)
    -> SampleResult
{
    void* d_dlenv{};
    cudaMalloc(&d_dlenv, static_cast<size_t>(qnpeps_dlenv_bytes(&config)));
    const auto build_status = qnpeps_build_dlenv(
        &config,
        static_cast<const qnpeps_device_peps*>(d_peps),
        static_cast<qnpeps_device_dlenv*>(d_dlenv),
        nullptr,
        nullptr
    );

    const auto dim_batch = test_sample_batch_size(count);
    const auto scratch_bytes = qnpeps_sample_scratch_bytes(&config, dim_batch);
    void* d_scratch{};
    cudaMalloc(&d_scratch, static_cast<size_t>(scratch_bytes));

    const auto sample_bytes = static_cast<size_t>(qnpeps_sample_bytes(&config, count));
    uint8_t* d_samples{};
    double* d_logpc{};
    cudaMalloc(&d_samples, sample_bytes);
    cudaMalloc(&d_logpc, static_cast<size_t>(count) * sizeof(double));
    const QnpepsSampleArgs sample_args{
        .struct_size = sizeof(QnpepsSampleArgs),
        .peps = static_cast<const qnpeps_device_peps*>(d_peps),
        .dlenv = static_cast<const qnpeps_device_dlenv*>(d_dlenv),
        .gpus = 1,
        .scratch = d_scratch,
        .scratch_bytes = static_cast<uint64_t>(scratch_bytes),
        .samples_out = d_samples,
        .log_prob_config = d_logpc,
        .log_gauge = nullptr,
        .n_samples = count,
        .batch_base = 0,
        .dim_batch = std::min<uint64_t>(count, 2048),
        .dim_batch = dim_batch,
        .stream = nullptr,
    };
    const auto sample_status = qnpeps_sample(&config, &sample_args);

    SampleResult r{};
    r.status_ok = build_status == QNPEPS_OK and sample_status == QNPEPS_OK;
    r.samples.resize(sample_bytes);
    r.logpc.resize(static_cast<size_t>(count));
    cudaMemcpy(r.samples.data(), d_samples, sample_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(
        r.logpc.data(), d_logpc, static_cast<size_t>(count) * sizeof(double), cudaMemcpyDeviceToHost
    );
    cudaFree(d_dlenv);
    cudaFree(d_scratch);
    cudaFree(d_samples);
    cudaFree(d_logpc);
    return r;
}

[[nodiscard]] auto run_ctx(
    const QnpepsConfig& config, qnpeps_ctx* ctx, const void* d_peps, uint64_t count
) -> SampleResult
{
    const auto* peps = static_cast<const qnpeps_device_peps*>(d_peps);
    const auto build_status = qnpeps_ctx_build_dlenv(ctx, peps, nullptr);

    const auto sample_bytes = static_cast<size_t>(qnpeps_sample_bytes(&config, count));
    uint8_t* d_samples{};
    double* d_logpc{};
    cudaMalloc(&d_samples, sample_bytes);
    cudaMalloc(&d_logpc, static_cast<size_t>(count) * sizeof(double));
    const QnpepsCtxSampleArgs ctx_args{
        .struct_size = sizeof(QnpepsCtxSampleArgs),
        .samples_out = d_samples,
        .log_prob_config = d_logpc,
        .log_gauge = nullptr,
        .n_samples = count,
        .batch_base = 0,
        .dim_batch = std::min<uint64_t>(count, 2048),
    };
    const auto sample_status = qnpeps_ctx_sample(ctx, &ctx_args);

    SampleResult r{};
    r.status_ok = build_status == QNPEPS_OK and sample_status == QNPEPS_OK;
    r.samples.resize(sample_bytes);
    r.logpc.resize(static_cast<size_t>(count));
    cudaMemcpy(r.samples.data(), d_samples, sample_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(
        r.logpc.data(), d_logpc, static_cast<size_t>(count) * sizeof(double), cudaMemcpyDeviceToHost
    );
    cudaFree(d_samples);
    cudaFree(d_logpc);
    return r;
}
}

int main(int argc, char** argv)
{
    const auto oracle_handle = dlopen(oracle_so_path(), RTLD_NOW | RTLD_LOCAL);
    if (not oracle_handle)
    {
        std::printf("[stage4_gate] dlopen oracle failed: %s\n", dlerror());
        return 1;
    }
    const auto selfgen = oracle_symbol<OracleGeneratePeps>(oracle_handle, "peps_export_selfgen");
    const auto paramcnt = oracle_symbol<OracleCount>(oracle_handle, "peps_sample_param_count");
    if (not selfgen or not paramcnt)
    {
        std::printf("[stage4_gate] dlsym failed\n");
        return 1;
    }

    const auto lattice = argc > 1 ? std::atoi(argv[1]) : 4;
    const auto dim_bond = argc > 2 ? std::atoi(argv[2]) : 2;
    const auto chi_s = argc > 3 ? std::atoi(argv[3]) : 2;
    const auto seed = 1;
    const auto count = static_cast<uint64_t>(argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 64);
    std::printf(
        "[stage4_gate] lattice=%d dim_bond=%d chi_s=%d count=%llu\n",
        lattice,
        dim_bond,
        chi_s,
        static_cast<unsigned long long>(count)
    );

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
    config.seed = static_cast<uint64_t>(seed);

    const auto nparam = paramcnt(&oracle_a);
    const auto peps_float_count = static_cast<size_t>(2 * nparam);
    auto d_peps_a = gen_peps(selfgen, oracle_a, peps_float_count);
    auto d_peps_b = gen_peps(selfgen, oracle_b, peps_float_count);

    int fails{};

    const auto ref_a = oneshot(config, d_peps_a, count);
    const auto ref_b = oneshot(config, d_peps_b, count);
    if (not ref_a.status_ok or not ref_b.status_ok)
    {
        std::printf("[stage4_gate] one-shot reference failed\n");
        return 1;
    }

    if (bytes_equal(ref_a.samples, ref_b.samples))
    {
        std::printf(
            "[stage4_gate] INCONCLUSIVE: peps_A and peps_B yield identical samples; "
            "the differential flip test would be vacuous\n"
        );
        ++fails;
    }
    else
    {
        std::printf("[stage4_gate] precondition: peps_A samples != peps_B samples (distinct)\n");
    }

    qnpeps_ctx* ctx{};
    if (qnpeps_ctx_create(&config, nullptr, &ctx) != QNPEPS_OK or not ctx)
    {
        std::printf("[stage4_gate] ctx_create failed\n");
        return 1;
    }

    const auto c1 = run_ctx(config, ctx, d_peps_a, count);
    report_match("cyc1 build A [warm,buf0]", ref_a, c1, fails);

    const auto c2 = run_ctx(config, ctx, d_peps_b, count);
    report_match("cyc2 build B [capture,buf1]", ref_b, c2, fails);

    const auto c3 = run_ctx(config, ctx, d_peps_a, count);
    report_match("cyc3 build A [replay,flip>buf0]", ref_a, c3, fails);

    const auto c4 = run_ctx(config, ctx, d_peps_b, count);
    report_match("cyc4 build B [replay,flip>buf1]", ref_b, c4, fails);

    qnpeps_ctx_destroy(ctx);
    qnpeps_sampler_pool_release();
    cudaFree(d_peps_a);
    cudaFree(d_peps_b);

    std::printf(
        fails == 0
            ? "\n[stage4_gate] PASS: double-buffer flip drives replayed sample_graph bit-exact\n"
            : "\n[stage4_gate] FAIL: %d check(s) diverged\n",
        fails
    );
    return fails == 0 ? 0 : 2;
}
