#include "capi/qnpeps.h"
#include "sample_batch.h"
#include "test_oracle.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <vector>

int main(int argc, char** argv)
{
    int device_count{};
    cudaGetDeviceCount(&device_count);
    const auto lattice = argc > 1 ? std::atoi(argv[1]) : 8;
    const auto dim_bond = argc > 2 ? std::atoi(argv[2]) : 4;
    const auto chi_s = argc > 3 ? std::atoi(argv[3]) : 4;
    const auto count = static_cast<uint64_t>(argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 1024);
    const auto gpus = argc > 5 ? std::atoi(argv[5]) : device_count;
    const auto seed = 1;
    const auto count_display = static_cast<unsigned long long>(count);
    std::printf(
        "[mg_gate] devices=%d testing gpus=%d  lattice=%d dim_bond=%d chi_s=%d count=%llu\n",
        device_count,
        gpus,
        lattice,
        dim_bond,
        chi_s,
        count_display
    );
    if (gpus < 2)
    {
        std::printf("[mg_gate] fewer than 2 GPUs (got %d), running single-vs-self\n", gpus);
    }

    auto oracle_handle{dlopen(oracle_so_path(), RTLD_NOW | RTLD_LOCAL)};
    if (not oracle_handle)
    {
        std::printf("[mg_gate] dlopen oracle failed: %s\n", dlerror());
        return 1;
    }
    const auto selfgen = oracle_symbol<OracleGeneratePeps>(oracle_handle, "peps_export_selfgen");
    const auto paramcnt = oracle_symbol<OracleCount>(oracle_handle, "peps_sample_param_count");
    if (not selfgen or not paramcnt)
    {
        std::printf("[mg_gate] dlsym failed\n");
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
        std::printf("[mg_gate] build_dlenv failed\n");
        return 1;
    }

    const auto sample_bytes = qnpeps_sample_bytes(&config, count);
    const auto sample_byte_count = static_cast<size_t>(sample_bytes);

    const auto sample_dim_batch = test_sample_batch_size(count);
    const auto scratch_bytes = qnpeps_sample_scratch_bytes(&config, sample_dim_batch);
    void* d_scratch{};
    cudaMalloc(&d_scratch, static_cast<size_t>(scratch_bytes));
    uint8_t* d_samples_single{};
    double* d_logpc_single{};
    cudaMalloc(&d_samples_single, sample_byte_count);
    cudaMalloc(&d_logpc_single, count * sizeof(double));
    const QnpepsSampleArgs single_args{
        .struct_size = sizeof(QnpepsSampleArgs),
        .peps = peps_dev,
        .dlenv = dlenv_dev,
        .gpus = 1,
        .scratch = d_scratch,
        .scratch_bytes = static_cast<uint64_t>(scratch_bytes),
        .samples_out = d_samples_single,
        .log_prob_config = d_logpc_single,
        .log_gauge = nullptr,
        .n_samples = count,
        .batch_base = 0,
        .dim_batch = sample_dim_batch,
        .stream = nullptr,
    };
    const auto status_single = qnpeps_sample(&config, &single_args);

    uint8_t* d_samples_multi{};
    double* d_logpc_multi{};
    cudaMalloc(&d_samples_multi, sample_byte_count);
    cudaMalloc(&d_logpc_multi, count * sizeof(double));
    const bool use_mg{gpus > 1};
    const QnpepsSampleArgs multi_args{
        .struct_size = sizeof(QnpepsSampleArgs),
        .peps = peps_dev,
        .dlenv = dlenv_dev,
        .gpus = gpus,
        .scratch = use_mg ? nullptr : d_scratch,
        .scratch_bytes = use_mg ? static_cast<uint64_t>(0) : static_cast<uint64_t>(scratch_bytes),
        .samples_out = d_samples_multi,
        .log_prob_config = d_logpc_multi,
        .log_gauge = nullptr,
        .n_samples = count,
        .batch_base = 0,
        .dim_batch = sample_dim_batch,
        .stream = nullptr,
    };
    const auto status_multi = qnpeps_sample(&config, &multi_args);
    std::printf(
        "[mg_gate] status: single=%d (%s)  mg=%d (%s)\n",
        status_single,
        qnpeps_strerror(status_single),
        status_multi,
        qnpeps_strerror(status_multi)
    );
    if (status_single != QNPEPS_OK or status_multi != QNPEPS_OK)
    {
        std::printf("[mg_gate] FAIL (status)\n");
        return 1;
    }

    std::vector<uint8_t> samples_single{};
    samples_single.resize(sample_byte_count);
    std::vector<uint8_t> samples_multi{};
    samples_multi.resize(sample_byte_count);
    std::vector<double> logpc_single{};
    logpc_single.resize(count);
    std::vector<double> logpc_multi{};
    logpc_multi.resize(count);
    cudaMemcpy(samples_single.data(), d_samples_single, sample_byte_count, cudaMemcpyDeviceToHost);
    cudaMemcpy(samples_multi.data(), d_samples_multi, sample_byte_count, cudaMemcpyDeviceToHost);
    cudaMemcpy(logpc_single.data(), d_logpc_single, count * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(logpc_multi.data(), d_logpc_multi, count * sizeof(double), cudaMemcpyDeviceToHost);

    int64_t byte_diff{};
    for (size_t i{0}; i < sample_byte_count; ++i)
    {
        if (samples_single[i] != samples_multi[i])
        {
            ++byte_diff;
        }
    }
    double max_logpc_diff{};
    for (uint64_t i{0}; i < count; ++i)
    {
        const auto logpc_diff = std::abs(logpc_single[i] - logpc_multi[i]);
        if (logpc_diff > max_logpc_diff) max_logpc_diff = logpc_diff;
    }
    std::printf(
        "[mg_gate] SAMPLES bytes: %ld/%ld differ  %s\n",
        byte_diff,
        sample_bytes,
        byte_diff == 0 ? "BIT-EXACT" : "*** DIFFER ***"
    );
    std::printf(
        "[mg_gate] SAMPLES logpc max|single-mg| = %g  %s\n",
        max_logpc_diff,
        max_logpc_diff == 0.0 ? "BIT-EXACT" : "*** DIFFER ***"
    );

    const auto pass = (byte_diff == 0 and max_logpc_diff == 0.0);
    std::printf(
        "[mg_gate] %s: gpus=%d %s single-GPU\n", pass ? "PASS" : "FAIL", gpus, pass ? "==" : "!="
    );
    cudaFree(d_peps);
    cudaFree(d_dlenv);
    cudaFree(d_scratch);
    cudaFree(d_samples_single);
    cudaFree(d_logpc_single);
    cudaFree(d_samples_multi);
    cudaFree(d_logpc_multi);
    int exit_code{};
    if (pass)
    {
        exit_code = 0;
    }
    else
    {
        exit_code = 1;
    }
    return exit_code;
}
