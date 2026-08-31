#include "dans_qnpeps_e2e.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>

namespace
{

struct Cf
{
    float re{};
    float im{};
};

static_assert(sizeof(Cf) == 2 * sizeof(float));

auto cuda_ok(cudaError_t status, const char* operation) -> bool
{
    if (status == cudaSuccess) return true;
    std::fprintf(
        stderr, "[gate_gram] CUDA failure in %s with %s\n", operation, cudaGetErrorString(status)
    );
    return false;
}

template <class T>
auto device_allocate(std::size_t count) -> T*
{
    T* pointer{};
    if (not cuda_ok(
            cudaMalloc(&pointer, std::max<std::size_t>(count, 1) * sizeof(T)), "allocation"
        ))
        return nullptr;
    return pointer;
}

auto site_slices(const QnpepsE2eConfig& config) -> std::vector<int>
{
    auto bond = [](int length, int position, int dimension)
    { return position <= 0 or position >= length ? 1 : dimension; };
    std::vector<int> slices{};
    for (int row{}; row < config.lx; ++row)
    {
        for (int column{}; column < config.ly; ++column)
        {
            slices.push_back(
                bond(config.ly, column, config.dim_bond) * bond(config.lx, row + 1, config.dim_bond)
                * bond(config.ly, column + 1, config.dim_bond)
                * bond(config.lx, row, config.dim_bond)
            );
        }
    }
    return slices;
}

auto host_reference(
    const QnpepsE2eConfig& config,
    int samples,
    const std::vector<std::uint8_t>& spins,
    const std::vector<Cf>& rows
) -> std::vector<Cf>
{
    const std::vector<int> slices{site_slices(config)};
    int compact{};
    for (int slice : slices)
        compact += slice;
    auto gram = std::vector<Cf>(static_cast<std::size_t>(samples) * samples);
    for (int left{}; left < samples; ++left)
    {
        for (int right{}; right < samples; ++right)
        {
            double real{};
            double imag{};
            int offset{};
            for (int site{}; site < config.lx * config.ly; ++site)
            {
                if (spins[left * config.lx * config.ly + site]
                    == spins[right * config.lx * config.ly + site])
                {
                    for (int local{}; local < slices[site]; ++local)
                    {
                        const Cf a{rows[static_cast<std::size_t>(left) * compact + offset + local]};
                        const Cf b{
                            rows[static_cast<std::size_t>(right) * compact + offset + local]
                        };
                        real += static_cast<double>(a.re) * b.re + static_cast<double>(a.im) * b.im;
                        imag += static_cast<double>(a.re) * b.im - static_cast<double>(a.im) * b.re;
                    }
                }
                offset += slices[site];
            }
            gram[static_cast<std::size_t>(left) * samples + right] =
                Cf{static_cast<float>(real), static_cast<float>(imag)};
        }
    }
    return gram;
}

auto relative_error(const std::vector<Cf>& observed, const std::vector<Cf>& reference) -> double
{
    double numerator{};
    double denominator{};
    for (std::size_t index{}; index < observed.size(); ++index)
    {
        const double dr{static_cast<double>(observed[index].re) - reference[index].re};
        const double di{static_cast<double>(observed[index].im) - reference[index].im};
        numerator += dr * dr + di * di;
        denominator += static_cast<double>(reference[index].re) * reference[index].re
                       + static_cast<double>(reference[index].im) * reference[index].im;
    }
    return std::sqrt(numerator / denominator);
}

struct Recorder
{
    std::ofstream file{};
    bool pass{true};

    explicit Recorder(const char* path)
    {
        if (path)
        {
            file.open(path);
            if (file)
                file << "mode,check,expected,observed,status\n";
            else
                pass = false;
        }
    }

    void add(
        const char* mode, const char* check, const char* expected, const char* observed, bool ok
    )
    {
        std::printf(
            "[gate_gram] mode=%s check=%s expected=%s observed=%s status=%s\n",
            mode,
            check,
            expected,
            observed,
            ok ? "PASS" : "FAIL"
        );
        if (file)
        {
            file << mode << ',' << check << ',' << expected << ',' << observed << ','
                 << (ok ? "PASS" : "FAIL") << '\n';
        }
        pass = pass and ok;
    }
};

struct Fixture
{
    QnpepsE2eConfig config{};
    int samples{8};
    int sites{};
    std::int64_t compact{};
    std::int64_t dense{};
    std::vector<std::uint8_t> spins{};
    std::vector<Cf> rows{};
    std::vector<double> logpsi{};
    std::vector<double> energy{};
    std::vector<double> logq{};
};

auto make_fixture() -> Fixture
{
    Fixture fixture{};
    fixture.config.struct_size = sizeof(QnpepsE2eConfig);
    fixture.config.lx = 2;
    fixture.config.ly = 3;
    fixture.config.dim_phys = 2;
    fixture.config.dim_bond = 2;
    fixture.config.chi_s = 2;
    fixture.config.chi_dl = 2;
    fixture.config.chi_eo = 8;
    fixture.config.meo = 4;
    fixture.sites = fixture.config.lx * fixture.config.ly;
    qnpeps_e2e_compact_count(&fixture.config, &fixture.compact);
    qnpeps_e2e_dense_count(&fixture.config, &fixture.dense);
    fixture.spins.resize(static_cast<std::size_t>(fixture.samples) * fixture.sites);
    fixture.rows.resize(static_cast<std::size_t>(fixture.samples) * fixture.compact);
    fixture.logpsi.resize(static_cast<std::size_t>(2 * fixture.samples));
    fixture.energy.resize(static_cast<std::size_t>(2 * fixture.samples));
    fixture.logq.resize(static_cast<std::size_t>(fixture.samples));
    for (int sample{}; sample < fixture.samples; ++sample)
    {
        for (int site{}; site < fixture.sites; ++site)
        {
            fixture.spins[sample * fixture.sites + site] =
                static_cast<std::uint8_t>(((sample * 5 + site * 3 + sample / 2) >> (site % 3)) & 1);
        }
        fixture.logpsi[2 * sample] = -0.13 * sample + 0.01 * sample * sample;
        fixture.logpsi[2 * sample + 1] = 0.07 * sample;
        fixture.energy[2 * sample] = -1.2 + 0.11 * sample;
        fixture.energy[2 * sample + 1] = 0.03 * (sample - 3);
        fixture.logq[sample] = -0.2 * sample - 0.5;
        for (std::int64_t local{}; local < fixture.compact; ++local)
        {
            const double coordinate{static_cast<double>(1 + sample * fixture.compact + local)};
            fixture.rows[static_cast<std::size_t>(sample) * fixture.compact + local] =
                Cf{static_cast<float>(0.17 * std::sin(0.019 * coordinate) + 0.003 * sample),
                   static_cast<float>(0.11 * std::cos(0.023 * coordinate) - 0.002 * local)};
        }
    }
    return fixture;
}

auto run_mode(const Fixture& fixture, Recorder& recorder) -> bool
{
    constexpr auto mode = "pedantic";
    setenv("QNPEPS_MINSR_GRAM_CONSUMER", "cublas_slab", 1);
    setenv("QNPEPS_MINSR_HERMITIAN_CHECK", "0", 1);

    auto* samples_device{device_allocate<std::uint8_t>(fixture.spins.size())};
    auto* rows_device{device_allocate<Cf>(fixture.rows.size())};
    auto* gram_device{
        device_allocate<Cf>(static_cast<std::size_t>(fixture.samples) * fixture.samples)
    };
    if (not samples_device or not rows_device or not gram_device) return false;
    cudaMemcpy(samples_device, fixture.spins.data(), fixture.spins.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(
        rows_device, fixture.rows.data(), fixture.rows.size() * sizeof(Cf), cudaMemcpyHostToDevice
    );

    qnpeps_e2e_gram_ctx* gram_context{};
    const qnpeps_e2e_status create_status{
        qnpeps_e2e_gram_ctx_create(&fixture.config, fixture.samples, nullptr, &gram_context)
    };
    recorder.add(
        mode,
        "standalone_create",
        "status_0",
        create_status == QNPEPS_E2E_OK ? "status_0" : "error",
        create_status == QNPEPS_E2E_OK
    );

    QnpepsE2eGramFootprint footprint{};
    footprint.struct_size = sizeof(footprint);
    const qnpeps_e2e_status footprint_status{
        qnpeps_e2e_gram_ctx_footprint(gram_context, &footprint)
    };
    const bool footprint_ok{
        footprint_status == QNPEPS_E2E_OK and footprint.context_device_bytes > 0
        and footprint.caller_samples_bytes == fixture.spins.size()
        and footprint.caller_rows_bytes == fixture.rows.size() * sizeof(Cf)
        and footprint.caller_gram_bytes
                == static_cast<std::uint64_t>(fixture.samples) * fixture.samples * sizeof(Cf)
    };
    recorder.add(
        mode,
        "footprint",
        "exact_nonzero",
        footprint_ok ? "exact_nonzero" : "different",
        footprint_ok
    );

    QnpepsE2eGramTimings timings{};
    timings.struct_size = sizeof(timings);
    const qnpeps_e2e_status run_status{qnpeps_e2e_gram_ctx_run(
        gram_context,
        samples_device,
        reinterpret_cast<const qnpeps_e2e_cbuf*>(rows_device),
        reinterpret_cast<qnpeps_e2e_cbuf*>(gram_device),
        &timings
    )};
    auto standalone = std::vector<Cf>(static_cast<std::size_t>(fixture.samples) * fixture.samples);
    cudaMemcpy(
        standalone.data(), gram_device, standalone.size() * sizeof(Cf), cudaMemcpyDeviceToHost
    );
    recorder.add(
        mode,
        "standalone_run",
        "status_0",
        run_status == QNPEPS_E2E_OK ? "status_0" : "error",
        run_status == QNPEPS_E2E_OK
    );
    const bool timing_ok{
        timings.slabs >= 1 and timings.virtual_shards == 4 and timings.block_calls >= 16
        and timings.slab_width > 0 and timings.complete_s >= 0.0
    };
    recorder.add(mode, "timings", "populated", timing_ok ? "populated" : "invalid", timing_ok);

    const qnpeps_e2e_status repeat_status{qnpeps_e2e_gram_ctx_run(
        gram_context,
        samples_device,
        reinterpret_cast<const qnpeps_e2e_cbuf*>(rows_device),
        reinterpret_cast<qnpeps_e2e_cbuf*>(gram_device),
        nullptr
    )};
    auto repeated = std::vector<Cf>(standalone.size());
    cudaMemcpy(repeated.data(), gram_device, repeated.size() * sizeof(Cf), cudaMemcpyDeviceToHost);
    const bool repeat_exact{
        repeat_status == QNPEPS_E2E_OK
        and std::memcmp(standalone.data(), repeated.data(), repeated.size() * sizeof(Cf)) == 0
    };
    recorder.add(
        mode, "repeat", "byte_exact", repeat_exact ? "byte_exact" : "different", repeat_exact
    );

    const std::vector<Cf> reference{
        host_reference(fixture.config, fixture.samples, fixture.spins, fixture.rows)
    };
    const double error{relative_error(standalone, reference)};
    constexpr double threshold{5.0e-5};
    char observed_error[64]{};
    std::snprintf(observed_error, sizeof(observed_error), "%.9g", error);
    recorder.add(
        mode,
        "independent_reference",
        "relative_le_5e-5",
        observed_error,
        std::isfinite(error) and error <= threshold
    );

    qnpeps_e2e_gram_ctx_destroy(gram_context);
    cudaFree(gram_device);
    cudaFree(rows_device);
    cudaFree(samples_device);
    return create_status == QNPEPS_E2E_OK and run_status == QNPEPS_E2E_OK and repeat_exact
           and footprint_ok and timing_ok and std::isfinite(error) and error <= threshold;
}

}

auto main(int argc, char** argv) -> int
{
    if (argc > 2)
    {
        std::fprintf(stderr, "[gate_gram] usage gate_gram [OUTPUT_CSV]\n");
        return 2;
    }
    Recorder recorder{argc == 2 ? argv[1] : nullptr};
    const Fixture fixture{make_fixture()};
    const bool pedantic{run_mode(fixture, recorder)};
    unsetenv("QNPEPS_MINSR_GRAM_CONSUMER");
    unsetenv("QNPEPS_MINSR_HERMITIAN_CHECK");
    const bool pass{recorder.pass and pedantic};
    std::printf("[gate_gram] RESULT=%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
