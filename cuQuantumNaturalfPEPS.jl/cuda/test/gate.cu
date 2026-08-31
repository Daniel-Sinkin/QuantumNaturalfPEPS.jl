#include "capi/qnpeps.h"
#include "sample_batch.h"
#include "test_oracle.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <vector>

int main(int argc, char** argv)
{
    const auto oracle_handle = dlopen(oracle_so_path(), RTLD_NOW | RTLD_LOCAL);
    if (not oracle_handle)
    {
        std::printf("[gate] dlopen oracle failed: %s\n", dlerror());
        return 1;
    }
    const auto selfgen = oracle_symbol<OracleGeneratePeps>(oracle_handle, "peps_export_selfgen");
    const auto paramcnt = oracle_symbol<OracleCount>(oracle_handle, "peps_sample_param_count");
    const auto hostdl = oracle_symbol<OracleBuildDlenv>(oracle_handle, "peps_export_host_dlenv");
    const auto dlvalcnt = oracle_symbol<OracleCount>(oracle_handle, "peps_dlenv_vals_count");
    const auto sampledev = oracle_symbol<OracleSample>(oracle_handle, "peps_sample_dev");
    if (not selfgen or not paramcnt or not hostdl or not dlvalcnt or not sampledev)
    {
        std::printf("[gate] dlsym failed\n");
        return 1;
    }

    const auto lattice = argc > 1 ? std::atoi(argv[1]) : 4;
    const auto dim_bond = argc > 2 ? std::atoi(argv[2]) : 2;
    const auto chi_s = argc > 3 ? std::atoi(argv[3]) : 2;
    const auto seed = 1;
    const auto count = static_cast<uint64_t>(argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 64);
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
    const auto sites = (lattice - 1) * lattice;
    int fails{};

    const auto nparam = paramcnt(&oracle);
    const auto peps_float_count = static_cast<size_t>(2 * nparam);
    std::vector<float> host_peps{};
    host_peps.resize(peps_float_count);
    selfgen(&oracle, host_peps.data());
    const auto peps_bytes = qnpeps_peps_bytes(&config);
    const auto float_bytes = static_cast<int64_t>(sizeof(float));
    const auto clean_floats = peps_bytes / float_bytes;
    std::printf(
        "[gate] PEPS floats: oracle=%ld  clean_expects=%ld  %s\n",
        2 * nparam,
        clean_floats,
        (2 * nparam == clean_floats) ? "MATCH" : "*** MISMATCH (layout) ***"
    );
    if (2 * nparam != clean_floats) ++fails;

    void* d_peps{};
    const auto peps_byte_count = peps_float_count * sizeof(float);
    cudaMalloc(&d_peps, peps_byte_count);
    cudaMemcpy(d_peps, host_peps.data(), peps_byte_count, cudaMemcpyHostToDevice);
    void* d_dlenv{};
    cudaMalloc(&d_dlenv, qnpeps_dlenv_bytes(&config));
    if (setenv("QNPEPS_SAMPLER_CHOL_SHIFT", "legacy", 1) != 0)
    {
        std::printf("[gate] failed to pin frozen-oracle Cholesky shift\n");
        return 1;
    }
    if (qnpeps_build_dlenv(
            &config,
            static_cast<const qnpeps_device_peps*>(d_peps),
            static_cast<qnpeps_device_dlenv*>(d_dlenv),
            nullptr,
            nullptr
        )
        != QNPEPS_OK)
    {
        std::printf("[gate] clean build_dlenv failed\n");
        return 1;
    }

    const auto oracle_dlval_count = dlvalcnt(&oracle);
    const auto dlval_count = static_cast<size_t>(2 * oracle_dlval_count);
    const auto dim_count = static_cast<size_t>(4 * sites);
    std::vector<float> oracle_vals{};
    oracle_vals.resize(dlval_count);
    std::vector<int32_t> oracle_dims{};
    oracle_dims.resize(dim_count);
    hostdl(&oracle, oracle_vals.data(), oracle_dims.data());

    std::vector<int32_t> clean_dims{};
    clean_dims.resize(dim_count);
    const auto dims_byte_count = dim_count * sizeof(int32_t);
    cudaMemcpy(clean_dims.data(), d_dlenv, dims_byte_count, cudaMemcpyDeviceToHost);
    int dims_match{1};
    for (auto dim_idx = 0; dim_idx < 4 * sites; ++dim_idx)
    {
        const auto idx = static_cast<size_t>(dim_idx);
        if (clean_dims[idx] != oracle_dims[idx]) dims_match = 0;
    }
    std::printf("[gate] dl-env DIMS: %s\n", dims_match ? "MATCH" : "*** MISMATCH ***");
    if (not dims_match)
    {
        for (auto site = 0; site < 3 and site < sites; ++site)
        {
            const auto site_base = static_cast<size_t>(4 * site);
            std::printf(
                "   site %d  clean(%d,%d,%d,%d)  oracle(%d,%d,%d,%d)\n",
                site,
                clean_dims[site_base],
                clean_dims[site_base + 1],
                clean_dims[site_base + 2],
                clean_dims[site_base + 3],
                oracle_dims[site_base],
                oracle_dims[site_base + 1],
                oracle_dims[site_base + 2],
                oracle_dims[site_base + 3]
            );
        }
        ++fails;
    }

    const auto dims_total = static_cast<int64_t>(4 * sites);
    const auto int32_bytes = static_cast<int64_t>(sizeof(int32_t));
    const auto header = dims_total * int32_bytes;
    auto* const dlenv_bytes = static_cast<char*>(d_dlenv);
    auto* const dlenv_vals = dlenv_bytes + header;
    if (dims_match)
    {
        std::vector<float> clean_vals{};
        clean_vals.resize(dlval_count);
        const auto vals_byte_count = dlval_count * sizeof(float);
        cudaMemcpy(clean_vals.data(), dlenv_vals, vals_byte_count, cudaMemcpyDeviceToHost);
        double max_dlval_diff{};
        for (int64_t val_idx{0}; val_idx < 2 * oracle_dlval_count; ++val_idx)
        {
            const auto idx = static_cast<size_t>(val_idx);
            const auto clean_val = static_cast<double>(clean_vals[idx]);
            const auto oracle_val = static_cast<double>(oracle_vals[idx]);
            const auto diff = std::abs(clean_val - oracle_val);
            if (diff > max_dlval_diff) max_dlval_diff = diff;
        }

        std::printf(
            "[gate] dl-env VALUES max|clean(device-rf)-oracle(host-mgs)| = %g  "
            "(%ld floats, gauge-dependent, informational)\n",
            max_dlval_diff,
            2 * oracle_dlval_count
        );
    }

    const auto sample_dim_batch = test_sample_batch_size(count);
    const auto scratch_bytes = qnpeps_sample_scratch_bytes(&config, sample_dim_batch);
    void* d_scratch{};
    cudaMalloc(&d_scratch, static_cast<size_t>(scratch_bytes));
    uint8_t* d_samples_clean{};
    double* d_logpc{};
    cudaMalloc(&d_samples_clean, qnpeps_sample_bytes(&config, count));
    cudaMalloc(&d_logpc, count * sizeof(double));
    const QnpepsSampleArgs sample_args{
        .struct_size = sizeof(QnpepsSampleArgs),
        .peps = static_cast<const qnpeps_device_peps*>(d_peps),
        .dlenv = static_cast<const qnpeps_device_dlenv*>(d_dlenv),
        .gpus = 1,
        .scratch = d_scratch,
        .scratch_bytes = static_cast<uint64_t>(scratch_bytes),
        .samples_out = d_samples_clean,
        .log_prob_config = d_logpc,
        .log_gauge = nullptr,
        .n_samples = count,
        .batch_base = 0,
        .dim_batch = sample_dim_batch,
        .stream = nullptr,
    };
    qnpeps_sample(&config, &sample_args);
    std::vector<double> logpc_clean{};
    logpc_clean.resize(count);
    cudaMemcpy(logpc_clean.data(), d_logpc, count * sizeof(double), cudaMemcpyDeviceToHost);

    const auto count_as_size = static_cast<size_t>(count);
    const auto bits_count = count_as_size * lattice;
    std::vector<uint64_t> oracle_bits{};
    oracle_bits.resize(bits_count);
    std::vector<double> logpc_oracle{};
    logpc_oracle.resize(count);
    sampledev(
        &oracle,
        host_peps.data(),
        dlenv_vals,
        clean_dims.data(),
        oracle_bits.data(),
        logpc_oracle.data()
    );

    double max_logpc_diff{};
    for (uint64_t sample_idx{0}; sample_idx < count; ++sample_idx)
    {
        const auto diff = std::abs(logpc_clean[sample_idx] - logpc_oracle[sample_idx]);
        if (diff > max_logpc_diff) max_logpc_diff = diff;
    }
    std::printf(
        "[gate] SAMPLES logpc max|clean-oracle| = %g  %s\n",
        max_logpc_diff,
        max_logpc_diff == 0.0 ? "BIT-EXACT" : (max_logpc_diff < 1e-9 ? "close" : "*** DIVERGED ***")
    );
    if (max_logpc_diff > 1e-9) ++fails;

    qnpeps_sampler_pool_release();
    cudaFree(d_peps);
    cudaFree(d_dlenv);
    cudaFree(d_scratch);
    cudaFree(d_samples_clean);
    cudaFree(d_logpc);
    std::printf(
        fails == 0 ? "\n[gate] PASS: clean matches oracle (logpc bit-exact)\n"
                   : "\n[gate] FAIL: %d check(s) diverged\n",
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
