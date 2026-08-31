#include "capi/qnpeps.h"
#include "sample_batch.h"
#include "test_oracle.h"

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
struct DrawResult
{
    std::vector<uint8_t> samples;
    std::vector<double> logpc;
    std::vector<double> loggauge;
    qnpeps_status status;
};

[[nodiscard]] auto draw_base(
    const QnpepsConfig& config,
    const qnpeps_device_peps* peps,
    const qnpeps_device_dlenv* dlenv,
    void* scratch,
    uint64_t scratch_bytes,
    uint64_t n_samples,
    uint64_t batch_base,
    uint64_t dim_batch
) -> DrawResult
{
    const auto sample_bytes = qnpeps_sample_bytes(&config, n_samples);
    uint8_t* d_samples{};
    double* d_logpc{};
    double* d_loggauge{};
    cudaMalloc(&d_samples, static_cast<size_t>(sample_bytes));
    cudaMalloc(&d_logpc, static_cast<size_t>(n_samples) * sizeof(double));
    cudaMalloc(&d_loggauge, static_cast<size_t>(n_samples) * sizeof(double));
    const QnpepsSampleArgs sample_args{
        .struct_size = sizeof(QnpepsSampleArgs),
        .peps = peps,
        .dlenv = dlenv,
        .gpus = 1,
        .scratch = scratch,
        .scratch_bytes = scratch_bytes,
        .samples_out = d_samples,
        .log_prob_config = d_logpc,
        .log_gauge = d_loggauge,
        .n_samples = n_samples,
        .batch_base = batch_base,
        .dim_batch = dim_batch,
        .stream = nullptr,
    };
    const auto status = qnpeps_sample(&config, &sample_args);
    DrawResult r{};
    r.status = status;
    r.samples.resize(static_cast<size_t>(sample_bytes));
    r.logpc.resize(static_cast<size_t>(n_samples));
    r.loggauge.resize(static_cast<size_t>(n_samples));
    cudaMemcpy(
        r.samples.data(), d_samples, static_cast<size_t>(sample_bytes), cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        r.logpc.data(),
        d_logpc,
        static_cast<size_t>(n_samples) * sizeof(double),
        cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        r.loggauge.data(),
        d_loggauge,
        static_cast<size_t>(n_samples) * sizeof(double),
        cudaMemcpyDeviceToHost
    );
    cudaFree(d_samples);
    cudaFree(d_logpc);
    cudaFree(d_loggauge);
    return r;
}

[[nodiscard]] auto draw_plain(
    const QnpepsConfig& config,
    const qnpeps_device_peps* peps,
    const qnpeps_device_dlenv* dlenv,
    void* scratch,
    uint64_t scratch_bytes,
    uint64_t n_samples
) -> DrawResult
{
    const auto sample_bytes = qnpeps_sample_bytes(&config, n_samples);
    uint8_t* d_samples{};
    double* d_logpc{};
    double* d_loggauge{};
    cudaMalloc(&d_samples, static_cast<size_t>(sample_bytes));
    cudaMalloc(&d_logpc, static_cast<size_t>(n_samples) * sizeof(double));
    cudaMalloc(&d_loggauge, static_cast<size_t>(n_samples) * sizeof(double));
    const QnpepsSampleArgs sample_args{
        .struct_size = sizeof(QnpepsSampleArgs),
        .peps = peps,
        .dlenv = dlenv,
        .gpus = 1,
        .scratch = scratch,
        .scratch_bytes = scratch_bytes,
        .samples_out = d_samples,
        .log_prob_config = d_logpc,
        .log_gauge = d_loggauge,
        .n_samples = n_samples,
        .batch_base = 0,
        .dim_batch = test_sample_batch_size(n_samples),
        .stream = nullptr,
    };
    const auto status = qnpeps_sample(&config, &sample_args);
    DrawResult r{};
    r.status = status;
    r.samples.resize(static_cast<size_t>(sample_bytes));
    r.logpc.resize(static_cast<size_t>(n_samples));
    r.loggauge.resize(static_cast<size_t>(n_samples));
    cudaMemcpy(
        r.samples.data(), d_samples, static_cast<size_t>(sample_bytes), cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        r.logpc.data(),
        d_logpc,
        static_cast<size_t>(n_samples) * sizeof(double),
        cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        r.loggauge.data(),
        d_loggauge,
        static_cast<size_t>(n_samples) * sizeof(double),
        cudaMemcpyDeviceToHost
    );
    cudaFree(d_samples);
    cudaFree(d_logpc);
    cudaFree(d_loggauge);
    return r;
}

[[nodiscard]] auto bitwise_equal(const DrawResult& a, const DrawResult& b) -> bool
{
    if (a.samples.size() != b.samples.size() or a.logpc.size() != b.logpc.size()
        or a.loggauge.size() != b.loggauge.size())
    {
        return false;
    }
    for (size_t i{0}; i < a.samples.size(); ++i)
        if (a.samples[i] != b.samples[i]) return false;
    for (size_t i{0}; i < a.logpc.size(); ++i)
    {
        if (std::memcmp(&a.logpc[i], &b.logpc[i], sizeof(double)) != 0) return false;
        if (std::memcmp(&a.loggauge[i], &b.loggauge[i], sizeof(double)) != 0) return false;
    }
    return true;
}

[[nodiscard]] auto concat(const std::vector<DrawResult>& parts) -> DrawResult
{
    DrawResult r{};
    r.status = QNPEPS_OK;
    for (const auto& p : parts)
    {
        if (p.status != QNPEPS_OK) r.status = p.status;
        r.samples.insert(r.samples.end(), p.samples.begin(), p.samples.end());
        r.logpc.insert(r.logpc.end(), p.logpc.begin(), p.logpc.end());
        r.loggauge.insert(r.loggauge.end(), p.loggauge.begin(), p.loggauge.end());
    }
    return r;
}

auto report(const char* name, bool ok, int& fails) -> void
{
    std::printf("[gate_base] %-28s %s\n", name, ok ? "BIT-EXACT" : "*** DIFFER ***");
    if (not ok) ++fails;
}
}

int main(int argc, char** argv)
{
    const auto oracle_handle = dlopen(oracle_so_path(), RTLD_NOW | RTLD_LOCAL);
    if (not oracle_handle)
    {
        std::printf("[gate_base] dlopen oracle failed: %s\n", dlerror());
        return 1;
    }
    const auto selfgen = oracle_symbol<OracleGeneratePeps>(oracle_handle, "peps_export_selfgen");
    const auto paramcnt = oracle_symbol<OracleCount>(oracle_handle, "peps_sample_param_count");
    if (not selfgen or not paramcnt)
    {
        std::printf("[gate_base] dlsym failed\n");
        return 1;
    }

    const auto lattice = argc > 1 ? std::atoi(argv[1]) : 4;
    const auto dim_bond = argc > 2 ? std::atoi(argv[2]) : 2;
    const auto chi_s = argc > 3 ? std::atoi(argv[3]) : 2;
    const auto seed = 1;
    const auto count = static_cast<uint64_t>(argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 128);
    std::printf(
        "[gate_base] lattice=%d dim_bond=%d chi_s=%d count=%llu\n",
        lattice,
        dim_bond,
        chi_s,
        static_cast<unsigned long long>(count)
    );
    if (count < 4 or count % 4 != 0)
    {
        std::printf("[gate_base] count must be a multiple of 4 and >= 4 for slice tests\n");
        return 1;
    }

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

    const auto nparam = paramcnt(&oracle);
    const auto peps_float_count = static_cast<size_t>(2 * nparam);
    std::vector<float> host_peps{};
    host_peps.resize(peps_float_count);
    selfgen(&oracle, host_peps.data());

    void* d_peps{};
    cudaMalloc(&d_peps, peps_float_count * sizeof(float));
    cudaMemcpy(d_peps, host_peps.data(), peps_float_count * sizeof(float), cudaMemcpyHostToDevice);
    const auto* peps_dev = static_cast<const qnpeps_device_peps*>(d_peps);
    void* d_dlenv{};
    cudaMalloc(&d_dlenv, qnpeps_dlenv_bytes(&config));
    auto* dlenv_dev = static_cast<qnpeps_device_dlenv*>(d_dlenv);
    if (qnpeps_build_dlenv(&config, peps_dev, dlenv_dev, nullptr, nullptr) != QNPEPS_OK)
    {
        std::printf("[gate_base] build_dlenv failed\n");
        return 1;
    }

    const auto scratch_bytes = qnpeps_sample_scratch_bytes(&config, k_test_max_batch_size);
    void* d_scratch{};
    cudaMalloc(&d_scratch, static_cast<size_t>(scratch_bytes));
    const auto sb = static_cast<uint64_t>(scratch_bytes);

    int fails{};

    const auto draw = [&](uint64_t sample_count, uint64_t batch_base, uint64_t dim_batch)
    {
        return draw_base(
            config, peps_dev, dlenv_dev, d_scratch, sb, sample_count, batch_base, dim_batch
        );
    };
    const auto default_dim_batch = test_sample_batch_size(count);
    const auto plain = draw_plain(config, peps_dev, dlenv_dev, d_scratch, sb, count);
    const auto base00 = draw(count, 0, default_dim_batch);
    report("explicit batch==helper batch", bitwise_equal(plain, base00), fails);

    const uint64_t d{count / 4};
    const auto single = draw(count, 0, d);

    const auto half_lo = draw(count / 2, 0, d);
    const auto half_hi = draw(count / 2, (count / 2) / d, d);
    report("slice 2-way==single", bitwise_equal(concat({half_lo, half_hi}), single), fails);

    const auto third0 = draw(d, 0, d);
    const auto third1 = draw(2 * d, 1, d);
    const auto third2 = draw(d, 3, d);
    report("slice 3-way==single", bitwise_equal(concat({third0, third1, third2}), single), fails);

    const auto v0 = draw(count, 0, 0);
    const auto v1 = draw(count, 0, 1);
    const auto vmax = draw(count, 0, k_test_max_batch_size);
    const auto vover = draw(count, 0, k_test_max_batch_size + 1);
    const bool validation_ok{
        v0.status == QNPEPS_ERR_BAD_CONFIG and v1.status == QNPEPS_OK and vmax.status == QNPEPS_OK
        and vover.status == QNPEPS_ERR_BAD_CONFIG
    };
    std::printf(
        "[gate_base] dim_batch validation: 0=%d 1=%d cap=%d cap+1=%d  %s\n",
        v0.status,
        v1.status,
        vmax.status,
        vover.status,
        validation_ok ? "OK" : "*** WRONG ***"
    );
    if (not validation_ok) ++fails;
    const auto* error_file = qnpeps_last_error_file();
    const auto error_line = qnpeps_last_error_line();
    const bool error_location_ok{
        error_file and std::strstr(error_file, "capi/capi.cu") and error_line > 0
    };
    std::printf(
        "[gate_base] error location: %s:%d  %s\n",
        error_file ? error_file : "<none>",
        error_line,
        error_location_ok ? "OK" : "*** WRONG ***"
    );
    if (not error_location_ok) ++fails;

    qnpeps_sampler_pool_release();
    cudaFree(d_peps);
    cudaFree(d_dlenv);
    cudaFree(d_scratch);

    std::printf(
        fails == 0 ? "\n[gate_base] PASS: base endpoint slice-equivalent and validated\n"
                   : "\n[gate_base] FAIL: %d check(s) diverged\n",
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
