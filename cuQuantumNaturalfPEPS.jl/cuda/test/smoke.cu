#include "capi/qnpeps.h"
#include "sample_batch.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

int main()
{
    QnpepsConfig config{};
    config.struct_size = sizeof(QnpepsConfig);
    config.lx = 4;
    config.ly = 4;
    config.dim_phys = 2;
    config.dim_bond = 2;
    config.chi_s = 2;
    config.chi_dl = config.dim_bond;
    config.seed = 1;
    std::printf(
        "[smoke] config lx=%d ly=%d dim_phys=%d dim_bond=%d chi_s=%d  capi=%s\n",
        config.lx,
        config.ly,
        config.dim_phys,
        config.dim_bond,
        config.chi_s,
        qnpeps_capi_version()
    );

    const auto peps_bytes = qnpeps_peps_bytes(&config);
    const auto float_bytes = static_cast<int64_t>(sizeof(float));
    const auto peps_float_count = peps_bytes / float_bytes;
    const auto host_peps_count = static_cast<size_t>(peps_float_count);
    std::vector<float> host_peps{};
    host_peps.resize(host_peps_count);
    for (int64_t i{0}; i < peps_float_count; ++i)
    {
        const auto idx = static_cast<size_t>(i);
        host_peps[idx] = 0.3f * std::sin(0.1f * i);
    }
    void* d_peps{};
    cudaMalloc(&d_peps, peps_bytes);
    cudaMemcpy(d_peps, host_peps.data(), peps_bytes, cudaMemcpyHostToDevice);
    const auto* peps_dev = static_cast<const qnpeps_device_peps*>(d_peps);
    std::printf("[smoke] PEPS uploaded: %ld bytes\n", peps_bytes);

    const auto dlenv_bytes = qnpeps_dlenv_bytes(&config);
    void* d_dlenv{};
    cudaMalloc(&d_dlenv, dlenv_bytes);
    auto* dlenv_dev = static_cast<qnpeps_device_dlenv*>(d_dlenv);
    auto status = qnpeps_build_dlenv(&config, peps_dev, dlenv_dev, nullptr, nullptr);
    std::printf(
        "[smoke] build_dlenv: %d (%s), buffer=%ld bytes\n",
        status,
        qnpeps_strerror(status),
        dlenv_bytes
    );
    if (status != QNPEPS_OK) return 1;

    std::vector<int32_t> header{};
    header.resize(12);
    cudaMemcpy(header.data(), d_dlenv, 12 * sizeof(int32_t), cudaMemcpyDeviceToHost);
    std::printf("[smoke] dl-env site dims [bond_left, ket, bra, bond_right]: ");
    for (auto site = 0; site < 3; ++site)
    {
        const auto site_base = 4 * site;
        std::printf(
            "(%d,%d,%d,%d) ",
            header[site_base],
            header[site_base + 1],
            header[site_base + 2],
            header[site_base + 3]
        );
    }
    std::printf("\n");

    const uint64_t count{64};
    const auto count_display = static_cast<unsigned long long>(count);
    const auto sample_bytes = qnpeps_sample_bytes(&config, count);
    const auto dim_batch = test_sample_batch_size(count);
    const auto scratch_bytes = qnpeps_sample_scratch_bytes(&config, dim_batch);
    const auto scratch_size = static_cast<size_t>(scratch_bytes);
    void* d_scratch{};
    cudaMalloc(&d_scratch, scratch_size);
    std::printf("[smoke] scratch: %ld bytes\n", scratch_bytes);
    uint8_t* d_samples{};
    double* d_logpc{};
    cudaMalloc(&d_samples, sample_bytes);
    cudaMalloc(&d_logpc, count * sizeof(double));
    const auto scratch_bytes_arg = static_cast<uint64_t>(scratch_bytes);
    const QnpepsSampleArgs sample_args{
        .struct_size = sizeof(QnpepsSampleArgs),
        .peps = peps_dev,
        .dlenv = dlenv_dev,
        .gpus = 1,
        .scratch = d_scratch,
        .scratch_bytes = scratch_bytes_arg,
        .samples_out = d_samples,
        .log_prob_config = d_logpc,
        .log_gauge = nullptr,
        .n_samples = count,
        .batch_base = 0,
        .dim_batch = dim_batch,
        .stream = nullptr,
    };
    status = qnpeps_sample(&config, &sample_args);
    std::printf(
        "[smoke] sample(count=%llu): %d (%s)\n", count_display, status, qnpeps_strerror(status)
    );
    if (status != QNPEPS_OK) return 1;

    const auto samples_count = static_cast<size_t>(sample_bytes);
    std::vector<uint8_t> samples{};
    samples.resize(samples_count);
    std::vector<double> logpc{};
    logpc.resize(count);
    cudaMemcpy(samples.data(), d_samples, sample_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(logpc.data(), d_logpc, count * sizeof(double), cudaMemcpyDeviceToHost);
    const auto site0 = static_cast<uint32_t>(samples[0]);
    const auto site1 = static_cast<uint32_t>(samples[1]);
    const auto site2 = static_cast<uint32_t>(samples[2]);
    const auto site3 = static_cast<uint32_t>(samples[3]);
    std::printf("[smoke] config0 sites[0..3]: %u %u %u %u\n", site0, site1, site2, site3);
    std::printf("[smoke] logpc[0..3]: %g %g %g %g\n", logpc[0], logpc[1], logpc[2], logpc[3]);
    int finite{};
    for (const auto value : logpc)
        if (std::isfinite(value) and value <= 0.0) ++finite;
    std::printf("[smoke] valid logpc (finite, <=0): %d/%llu\n", finite, count_display);

    qnpeps_sampler_pool_release();
    cudaFree(d_peps);
    cudaFree(d_dlenv);
    cudaFree(d_scratch);
    cudaFree(d_samples);
    cudaFree(d_logpc);
    const auto count_as_int = static_cast<int>(count);
    const bool all_valid{finite == count_as_int};
    int exit_code{};
    if (all_valid)
    {
        std::printf("[smoke] PASS\n");
        exit_code = 0;
    }
    else
    {
        std::printf("[smoke] SUSPECT (logpc not all valid)\n");
        exit_code = 2;
    }
    return exit_code;
}
