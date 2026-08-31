#include "capi/qnpeps.h"
#include "sample_batch.h"
#include "test_oracle.h"

#include <algorithm>
#include <cmath>
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
    std::vector<double> rowlogs;
    std::vector<double> lognorm;
    bool status_ok;
};

[[nodiscard]] auto max_abs_diff(const std::vector<double>& a, const std::vector<double>& b)
    -> double
{
    const auto n = a.size() < b.size() ? a.size() : b.size();
    double m{0.0};
    for (size_t i{0}; i < n; ++i)
    {
        const auto d = std::abs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

[[nodiscard]] auto logs_bit_exact(const std::vector<double>& a, const std::vector<double>& b)
    -> bool
{
    if (a.size() != b.size()) return false;
    if (a.empty()) return true;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

[[nodiscard]] auto bytes_equal(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) -> bool
{
    if (a.size() != b.size()) return false;
    if (a.empty()) return true;
    return std::memcmp(a.data(), b.data(), a.size()) == 0;
}

auto report_match(const char* name, const SampleResult& ref, const SampleResult& other, int& fails)
    -> void
{
    const auto samples_ok = bytes_equal(ref.samples, other.samples);
    const auto logpc_ok = logs_bit_exact(ref.logpc, other.logpc);
    const auto rowlog_ok = logs_bit_exact(ref.rowlogs, other.rowlogs);
    const auto lognorm_ok = logs_bit_exact(ref.lognorm, other.lognorm);
    const auto logpc_diff = max_abs_diff(ref.logpc, other.logpc);
    const auto rowlog_diff = max_abs_diff(ref.rowlogs, other.rowlogs);
    const auto ok = samples_ok and logpc_ok and rowlog_ok and lognorm_ok;
    std::printf(
        "[ctx_gate] %-24s logpc max|Δ| = %g  rowlog max|Δ| = %g  samples %s  lognorm %s  %s\n",
        name,
        logpc_diff,
        rowlog_diff,
        samples_ok ? "EQ" : "NE",
        lognorm_ok ? "EQ" : "NE",
        ok ? "BIT-EXACT" : "*** DIFFER ***"
    );
    if (not ok) ++fails;
}

[[nodiscard]] auto run_ctx(
    const QnpepsConfig& config,
    qnpeps_ctx* ctx,
    const void* d_peps,
    uint64_t count,
    size_t rowlog_count
) -> SampleResult
{
    double* d_rowlogs{};
    cudaMalloc(&d_rowlogs, rowlog_count * sizeof(double));
    const auto* peps = static_cast<const qnpeps_device_peps*>(d_peps);
    const auto build_status = qnpeps_ctx_build_dlenv(ctx, peps, d_rowlogs);

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
    r.rowlogs.resize(rowlog_count);
    cudaMemcpy(r.samples.data(), d_samples, sample_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(
        r.logpc.data(), d_logpc, static_cast<size_t>(count) * sizeof(double), cudaMemcpyDeviceToHost
    );
    cudaMemcpy(r.rowlogs.data(), d_rowlogs, rowlog_count * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_rowlogs);
    cudaFree(d_samples);
    cudaFree(d_logpc);
    return r;
}

[[nodiscard]] auto oneshot_sample(
    const QnpepsConfig& config,
    const void* d_peps,
    const void* d_dlenv,
    void* d_scratch,
    int64_t scratch_bytes,
    uint64_t count,
    uint64_t batch_base,
    uint64_t dim_batch,
    bool want_lognorm
) -> SampleResult
{
    const auto sample_bytes = static_cast<size_t>(qnpeps_sample_bytes(&config, count));
    uint8_t* d_samples{};
    double* d_logpc{};
    double* d_lognorm{};
    cudaMalloc(&d_samples, sample_bytes);
    cudaMalloc(&d_logpc, static_cast<size_t>(count) * sizeof(double));
    if (want_lognorm) cudaMalloc(&d_lognorm, static_cast<size_t>(count) * sizeof(double));
    const QnpepsSampleArgs sample_args{
        .struct_size = sizeof(QnpepsSampleArgs),
        .peps = static_cast<const qnpeps_device_peps*>(d_peps),
        .dlenv = static_cast<const qnpeps_device_dlenv*>(d_dlenv),
        .gpus = 1,
        .scratch = d_scratch,
        .scratch_bytes = static_cast<uint64_t>(scratch_bytes),
        .samples_out = d_samples,
        .log_prob_config = d_logpc,
        .log_gauge = want_lognorm ? d_lognorm : nullptr,
        .n_samples = count,
        .batch_base = batch_base,
        .dim_batch = std::min<uint64_t>(count, 2048),
        .dim_batch = dim_batch,
        .stream = nullptr,
    };
    const auto sample_status = qnpeps_sample(&config, &sample_args);
    SampleResult r{};
    r.status_ok = sample_status == QNPEPS_OK;
    r.samples.resize(sample_bytes);
    r.logpc.resize(static_cast<size_t>(count));
    cudaMemcpy(r.samples.data(), d_samples, sample_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(
        r.logpc.data(), d_logpc, static_cast<size_t>(count) * sizeof(double), cudaMemcpyDeviceToHost
    );
    if (want_lognorm)
    {
        r.lognorm.resize(static_cast<size_t>(count));
        cudaMemcpy(
            r.lognorm.data(),
            d_lognorm,
            static_cast<size_t>(count) * sizeof(double),
            cudaMemcpyDeviceToHost
        );
        cudaFree(d_lognorm);
    }
    cudaFree(d_samples);
    cudaFree(d_logpc);
    return r;
}

[[nodiscard]] auto ctx_sample_only(
    const QnpepsConfig& config,
    qnpeps_ctx* ctx,
    uint64_t count,
    uint64_t batch_base,
    bool want_lognorm
) -> SampleResult
{
    const auto sample_bytes = static_cast<size_t>(qnpeps_sample_bytes(&config, count));
    uint8_t* d_samples{};
    double* d_logpc{};
    double* d_lognorm{};
    cudaMalloc(&d_samples, sample_bytes);
    cudaMalloc(&d_logpc, static_cast<size_t>(count) * sizeof(double));
    if (want_lognorm) cudaMalloc(&d_lognorm, static_cast<size_t>(count) * sizeof(double));
    const QnpepsCtxSampleArgs ctx_args{
        .struct_size = sizeof(QnpepsCtxSampleArgs),
        .samples_out = d_samples,
        .log_prob_config = d_logpc,
        .log_gauge = want_lognorm ? d_lognorm : nullptr,
        .n_samples = count,
        .batch_base = batch_base,
        .dim_batch = std::min<uint64_t>(count, 2048),
    };
    const auto sample_status = qnpeps_ctx_sample(ctx, &ctx_args);
    SampleResult r{};
    r.status_ok = sample_status == QNPEPS_OK;
    r.samples.resize(sample_bytes);
    r.logpc.resize(static_cast<size_t>(count));
    cudaMemcpy(r.samples.data(), d_samples, sample_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(
        r.logpc.data(), d_logpc, static_cast<size_t>(count) * sizeof(double), cudaMemcpyDeviceToHost
    );
    if (want_lognorm)
    {
        r.lognorm.resize(static_cast<size_t>(count));
        cudaMemcpy(
            r.lognorm.data(),
            d_lognorm,
            static_cast<size_t>(count) * sizeof(double),
            cudaMemcpyDeviceToHost
        );
        cudaFree(d_lognorm);
    }
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
        std::printf("[ctx_gate] dlopen oracle failed: %s\n", dlerror());
        return 1;
    }
    const auto selfgen = oracle_symbol<OracleGeneratePeps>(oracle_handle, "peps_export_selfgen");
    const auto paramcnt = oracle_symbol<OracleCount>(oracle_handle, "peps_sample_param_count");
    if (not selfgen or not paramcnt)
    {
        std::printf("[ctx_gate] dlsym failed\n");
        return 1;
    }

    const auto lattice = argc > 1 ? std::atoi(argv[1]) : 4;
    const auto dim_bond = argc > 2 ? std::atoi(argv[2]) : 2;
    const auto chi_s = argc > 3 ? std::atoi(argv[3]) : 2;
    const auto seed = 1;
    const uint64_t count = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 64;
    const bool full_mode{std::getenv("QNPEPS_GATE_FULL") != nullptr};
    std::printf(
        "[ctx_gate] lattice=%d dim_bond=%d chi_s=%d count=%llu mode=%s contract=%d\n",
        lattice,
        dim_bond,
        chi_s,
        static_cast<unsigned long long>(count),
        full_mode ? "full" : "fast",
        full_mode ? 3 * dim_bond : 0
    );

    const auto dim_batch = static_cast<int32_t>(count);
    OracleConfig oracle{lattice, lattice, 2, dim_bond, chi_s, seed, dim_batch, 1, 1, 0};
    QnpepsConfig config{};
    config.struct_size = sizeof(QnpepsConfig);
    config.lx = lattice;
    config.ly = lattice;
    config.dim_phys = 2;
    config.dim_bond = dim_bond;
    config.chi_s = chi_s;
    config.chi_dl = dim_bond;
    config.seed = static_cast<uint64_t>(seed);
    config.sampling_mode = full_mode ? QNPEPS_SAMPLING_FULL : QNPEPS_SAMPLING_FAST;
    config.chi_c = full_mode ? 3 * dim_bond : 0;

    int fails{};

    const auto nparam = paramcnt(&oracle);
    const auto peps_float_count = static_cast<size_t>(2 * nparam);
    std::vector<float> host_peps{};
    host_peps.resize(peps_float_count);
    selfgen(&oracle, host_peps.data());

    void* d_peps{};
    cudaMalloc(&d_peps, peps_float_count * sizeof(float));
    cudaMemcpy(d_peps, host_peps.data(), peps_float_count * sizeof(float), cudaMemcpyHostToDevice);

    const auto rowlog_count = static_cast<size_t>(lattice - 1);

    void* d_dlenv{};
    cudaMalloc(&d_dlenv, qnpeps_dlenv_bytes(&config));
    double* d_rowlogs_clean{};
    cudaMalloc(&d_rowlogs_clean, rowlog_count * sizeof(double));
    if (qnpeps_build_dlenv(
            &config,
            static_cast<const qnpeps_device_peps*>(d_peps),
            static_cast<qnpeps_device_dlenv*>(d_dlenv),
            d_rowlogs_clean,
            nullptr
        )
        != QNPEPS_OK)
    {
        std::printf("[ctx_gate] one-shot build_dlenv failed\n");
        return 1;
    }

    const auto sample_dim_batch = test_sample_batch_size(count);
    const auto scratch_bytes = qnpeps_sample_scratch_bytes(&config, k_test_max_batch_size);
    void* d_scratch{};
    cudaMalloc(&d_scratch, static_cast<size_t>(scratch_bytes));

    const auto sample_bytes = static_cast<size_t>(qnpeps_sample_bytes(&config, count));
    uint8_t* d_samples_clean{};
    double* d_logpc_clean{};
    cudaMalloc(&d_samples_clean, sample_bytes);
    cudaMalloc(&d_logpc_clean, static_cast<size_t>(count) * sizeof(double));
    const QnpepsSampleArgs sample_args{
        .struct_size = sizeof(QnpepsSampleArgs),
        .peps = static_cast<const qnpeps_device_peps*>(d_peps),
        .dlenv = static_cast<const qnpeps_device_dlenv*>(d_dlenv),
        .gpus = 1,
        .scratch = d_scratch,
        .scratch_bytes = static_cast<uint64_t>(scratch_bytes),
        .samples_out = d_samples_clean,
        .log_prob_config = d_logpc_clean,
        .log_gauge = nullptr,
        .n_samples = count,
        .batch_base = 0,
        .dim_batch = sample_dim_batch,
        .stream = nullptr,
    };
    if (qnpeps_sample(&config, &sample_args) != QNPEPS_OK)
    {
        std::printf("[ctx_gate] one-shot sample failed\n");
        return 1;
    }

    SampleResult clean{};
    clean.status_ok = true;
    clean.samples.resize(sample_bytes);
    clean.logpc.resize(static_cast<size_t>(count));
    clean.rowlogs.resize(rowlog_count);
    cudaMemcpy(clean.samples.data(), d_samples_clean, sample_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(
        clean.logpc.data(),
        d_logpc_clean,
        static_cast<size_t>(count) * sizeof(double),
        cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        clean.rowlogs.data(), d_rowlogs_clean, rowlog_count * sizeof(double), cudaMemcpyDeviceToHost
    );

    qnpeps_ctx* ctx{};
    if (qnpeps_ctx_create(&config, nullptr, &ctx) != QNPEPS_OK or not ctx)
    {
        std::printf("[ctx_gate] ctx_create failed\n");
        return 1;
    }

    const auto ctx1 = run_ctx(config, ctx, d_peps, count, rowlog_count);
    if (not ctx1.status_ok)
    {
        std::printf("[ctx_gate] ctx build/sample #1 returned non-OK status\n");
        ++fails;
    }
    report_match("ctx==one-shot", clean, ctx1, fails);

    const auto ctx2 = run_ctx(config, ctx, d_peps, count, rowlog_count);
    if (not ctx2.status_ok)
    {
        std::printf("[ctx_gate] ctx build/sample #2 returned non-OK status\n");
        ++fails;
    }
    report_match("persist #2==#1", ctx1, ctx2, fails);

    const auto ctx3 = run_ctx(config, ctx, d_peps, count, rowlog_count);
    if (not ctx3.status_ok)
    {
        std::printf("[ctx_gate] ctx build/sample #3 returned non-OK status\n");
        ++fails;
    }
    report_match("persist #3==#1", ctx1, ctx3, fails);

    const uint64_t base_probe{12345};
    const auto os_base = oneshot_sample(
        config, d_peps, d_dlenv, d_scratch, scratch_bytes, count, base_probe, sample_dim_batch, true
    );
    const auto ctx_base = ctx_sample_only(config, ctx, count, base_probe, true);
    if (not os_base.status_ok or not ctx_base.status_ok)
    {
        std::printf("[ctx_gate] base-parity build/sample returned non-OK status\n");
        ++fails;
    }
    report_match("ctx base==oneshot", os_base, ctx_base, fails);

    const bool fresh{not bytes_equal(ctx_base.samples, ctx1.samples)};
    std::printf(
        "[ctx_gate] %-24s base=%llu vs base=0  %s\n",
        "freshness",
        static_cast<unsigned long long>(base_probe),
        fresh ? "PASS" : "*** FAIL (param ignored?) ***"
    );
    if (not fresh) ++fails;

    if (lattice == 4)
    {
        const uint64_t count_mb{3000};
        const auto os_mb = oneshot_sample(
            config,
            d_peps,
            d_dlenv,
            d_scratch,
            scratch_bytes,
            count_mb,
            0,
            test_sample_batch_size(count_mb),
            false
        );
        const auto ctx_mb = ctx_sample_only(config, ctx, count_mb, 0, false);
        if (not os_mb.status_ok or not ctx_mb.status_ok)
        {
            std::printf("[ctx_gate] multi-batch build/sample returned non-OK status\n");
            ++fails;
        }
        report_match("ctx multibatch==oneshot", os_mb, ctx_mb, fails);

        const auto ctx_back = run_ctx(config, ctx, d_peps, count, rowlog_count);
        report_match("ctx post-mb==persist#1", ctx1, ctx_back, fails);
    }

    qnpeps_ctx_destroy(ctx);
    qnpeps_sampler_pool_release();
    cudaFree(d_peps);
    cudaFree(d_dlenv);
    cudaFree(d_rowlogs_clean);
    cudaFree(d_scratch);
    cudaFree(d_samples_clean);
    cudaFree(d_logpc_clean);

    std::printf(
        fails == 0 ? "\n[ctx_gate] PASS: ctx path matches one-shot and persists bit-exact\n"
                   : "\n[ctx_gate] FAIL: %d check(s) diverged\n",
        fails
    );
    int exit_code{};
    if (fails == 0)
    {
        exit_code = 0;
    }
    else
    {
        exit_code = 2;
    }
    return exit_code;
}
