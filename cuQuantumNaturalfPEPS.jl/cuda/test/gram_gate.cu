#include "capi/qnpeps.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <limits>
#include <vector>

namespace
{
using cfl = std::complex<float>;
using cdb = std::complex<double>;

auto bond(int length, int position, int dimension) -> int
{
    return position <= 0 or position >= length ? 1 : dimension;
}

auto cuda_ok(cudaError_t status, const char* operation) -> bool
{
    if (status == cudaSuccess) return true;
    std::fprintf(
        stderr, "[gram_gate] CUDA failure in %s with %s\n", operation, cudaGetErrorString(status)
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

struct Fixture
{
    QnpepsGramDesc descriptor{};
    int samples{8};
    int sites{};
    std::int64_t compact{};
    std::vector<int> slices{};
    std::vector<std::uint8_t> spins{};
    std::vector<cfl> rows{};
    std::vector<cfl> transposed{};
};

auto site_slices(const QnpepsGramDesc& descriptor) -> std::vector<int>
{
    std::vector<int> slices{};
    for (int row{}; row < descriptor.lx; ++row)
    {
        for (int column{}; column < descriptor.ly; ++column)
        {
            slices.push_back(
                bond(descriptor.ly, column, descriptor.dim_bond)
                * bond(descriptor.lx, row + 1, descriptor.dim_bond)
                * bond(descriptor.ly, column + 1, descriptor.dim_bond)
                * bond(descriptor.lx, row, descriptor.dim_bond)
            );
        }
    }
    return slices;
}

auto make_fixture() -> Fixture
{
    Fixture fixture{};
    fixture.descriptor.struct_size = sizeof(QnpepsGramDesc);
    fixture.descriptor.lx = 2;
    fixture.descriptor.ly = 3;
    fixture.descriptor.dim_phys = 2;
    fixture.descriptor.dim_bond = 2;
    fixture.descriptor.consumer = QNPEPS_GRAM_CONSUMER_SLAB;
    fixture.descriptor.reserved = 0;
    fixture.descriptor.n_samples = fixture.samples;
    fixture.sites = fixture.descriptor.lx * fixture.descriptor.ly;
    fixture.slices = site_slices(fixture.descriptor);
    fixture.compact = 0;
    for (const int slice : fixture.slices)
        fixture.compact += slice;

    fixture.spins.resize(static_cast<std::size_t>(fixture.samples) * fixture.sites);
    fixture.rows.resize(static_cast<std::size_t>(fixture.samples) * fixture.compact);
    for (int sample{}; sample < fixture.samples; ++sample)
    {
        for (int site{}; site < fixture.sites; ++site)
        {
            fixture.spins[static_cast<std::size_t>(sample * fixture.sites + site)] =
                static_cast<std::uint8_t>(((sample * 5 + site * 3 + sample / 2) >> (site % 3)) & 1);
        }
        for (std::int64_t local{}; local < fixture.compact; ++local)
        {
            const double coordinate{static_cast<double>(1 + sample * fixture.compact + local)};
            fixture.rows[static_cast<std::size_t>(sample * fixture.compact + local)] =
                cfl{static_cast<float>(0.17 * std::sin(0.019 * coordinate) + 0.003 * sample),
                    static_cast<float>(0.11 * std::cos(0.023 * coordinate) - 0.002 * local)};
        }
    }

    fixture.transposed = fixture.rows;
    const std::int64_t block{std::min<std::int64_t>(fixture.samples, fixture.compact)};
    for (std::int64_t left{}; left < block; ++left)
    {
        for (std::int64_t right{}; right < block; ++right)
        {
            fixture.transposed[static_cast<std::size_t>(left * fixture.compact + right)] =
                fixture.rows[static_cast<std::size_t>(right * fixture.compact + left)];
        }
    }
    return fixture;
}

auto host_reference(const Fixture& fixture, const std::vector<cfl>& rows) -> std::vector<cfl>
{
    const int samples{fixture.samples};
    auto gram = std::vector<cfl>(static_cast<std::size_t>(samples) * samples);
    for (int left{}; left < samples; ++left)
    {
        for (int right{}; right < samples; ++right)
        {
            double real{};
            double imag{};
            int offset{};
            for (int site{}; site < fixture.sites; ++site)
            {
                if (fixture.spins[static_cast<std::size_t>(left * fixture.sites + site)]
                    == fixture.spins[static_cast<std::size_t>(right * fixture.sites + site)])
                {
                    for (int local{}; local < fixture.slices[static_cast<std::size_t>(site)];
                         ++local)
                    {
                        const cfl a{
                            rows[static_cast<std::size_t>(left * fixture.compact + offset + local)]
                        };
                        const cfl b{
                            rows[static_cast<std::size_t>(right * fixture.compact + offset + local)]
                        };
                        real += static_cast<double>(a.real()) * static_cast<double>(b.real())
                                + static_cast<double>(a.imag()) * static_cast<double>(b.imag());
                        imag += static_cast<double>(a.real()) * static_cast<double>(b.imag())
                                - static_cast<double>(a.imag()) * static_cast<double>(b.real());
                    }
                }
                offset += fixture.slices[static_cast<std::size_t>(site)];
            }
            gram[static_cast<std::size_t>(left * samples + right)] =
                cfl{static_cast<float>(real), static_cast<float>(imag)};
        }
    }
    return gram;
}

auto relative_error(const std::vector<cfl>& observed, const std::vector<cfl>& reference) -> double
{
    double numerator{};
    double denominator{};
    for (std::size_t index{}; index < observed.size(); ++index)
    {
        const double dr{
            static_cast<double>(observed[index].real())
            - static_cast<double>(reference[index].real())
        };
        const double di{
            static_cast<double>(observed[index].imag())
            - static_cast<double>(reference[index].imag())
        };
        numerator += dr * dr + di * di;
        denominator += static_cast<double>(reference[index].real())
                           * static_cast<double>(reference[index].real())
                       + static_cast<double>(reference[index].imag())
                             * static_cast<double>(reference[index].imag());
    }
    return std::sqrt(numerator / denominator);
}

struct Recorder
{
    bool pass{true};

    auto add(
        const char* mode, const char* check, const char* expected, const char* observed, bool ok
    ) -> void
    {
        std::printf(
            "[gram_gate] mode=%s check=%s expected=%s observed=%s status=%s\n",
            mode,
            check,
            expected,
            observed,
            ok ? "PASS" : "FAIL"
        );
        pass = pass and ok;
    }

    auto advisory(
        const char* mode, const char* check, const char* expected, const char* observed, bool ok
    ) -> void
    {
        std::printf(
            "[gram_gate] mode=%s check=%s expected=%s observed=%s status=%s\n",
            mode,
            check,
            expected,
            observed,
            ok ? "ADVISORY_PASS" : "ADVISORY_FAIL"
        );
    }
};

auto jacobi_eigenvalues(const std::vector<cfl>& values, int order) -> std::vector<double>
{
    const auto n = static_cast<std::size_t>(order);
    auto matrix = std::vector<cdb>(n * n);
    for (std::size_t row{}; row < n; ++row)
    {
        for (std::size_t column{}; column < n; ++column)
        {
            const cfl value{values[row * n + column]};
            matrix[row + column * n] = cdb{value.real(), value.imag()};
        }
    }
    for (std::size_t column{}; column < n; ++column)
    {
        matrix[column + column * n] = cdb{matrix[column + column * n].real(), 0.0};
        for (std::size_t row{column + 1}; row < n; ++row)
        {
            const cdb average{
                0.5 * (matrix[row + column * n] + std::conj(matrix[column + row * n]))
            };
            matrix[row + column * n] = average;
            matrix[column + row * n] = std::conj(average);
        }
    }
    for (int sweep{}; sweep < 100; ++sweep)
    {
        double off{};
        double diagonal{};
        for (std::size_t column{}; column < n; ++column)
        {
            for (std::size_t row{}; row < n; ++row)
            {
                if (row == column)
                    diagonal += std::norm(matrix[row + column * n]);
                else
                    off += std::norm(matrix[row + column * n]);
            }
        }
        if (off <= 1.0e-28 * std::max(diagonal, 1.0e-300)) break;
        for (std::size_t p{}; p + 1 < n; ++p)
        {
            for (std::size_t q{p + 1}; q < n; ++q)
            {
                const cdb apq{matrix[p + q * n]};
                const double magnitude{std::abs(apq)};
                if (magnitude <= 1.0e-300) continue;
                const cdb phase{apq / magnitude};
                const double app{matrix[p + p * n].real()};
                const double aqq{matrix[q + q * n].real()};
                const double tau{(aqq - app) / (2.0 * magnitude)};
                const double tangent{
                    tau == 0.0
                        ? 1.0
                        : (tau > 0.0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau * tau))
                };
                const double cosine{1.0 / std::sqrt(1.0 + tangent * tangent)};
                const double sine{tangent * cosine};
                const cdb u11{cosine, 0.0};
                const cdb u12{sine, 0.0};
                const cdb u21{-sine * std::conj(phase)};
                const cdb u22{cosine * std::conj(phase)};
                for (std::size_t row{}; row < n; ++row)
                {
                    const cdb left{matrix[row + p * n]};
                    const cdb right{matrix[row + q * n]};
                    matrix[row + p * n] = left * u11 + right * u21;
                    matrix[row + q * n] = left * u12 + right * u22;
                }
                for (std::size_t column{}; column < n; ++column)
                {
                    const cdb top{matrix[p + column * n]};
                    const cdb bottom{matrix[q + column * n]};
                    matrix[p + column * n] = std::conj(u11) * top + std::conj(u21) * bottom;
                    matrix[q + column * n] = std::conj(u12) * top + std::conj(u22) * bottom;
                }
            }
        }
    }
    auto eigenvalues = std::vector<double>(n);
    for (std::size_t index{}; index < n; ++index)
        eigenvalues[index] = matrix[index + index * n].real();
    std::sort(eigenvalues.begin(), eigenvalues.end());
    return eigenvalues;
}

struct GramRun
{
    qnpeps_status status{QNPEPS_ERR_INTERNAL};
    std::vector<cfl> values{};
};

auto gram_once(const Fixture& fixture) -> GramRun
{
    GramRun result{};
    QnpepsGramDesc descriptor{fixture.descriptor};
    const std::size_t gram_elements{static_cast<std::size_t>(fixture.samples) * fixture.samples};
    auto* samples_device{device_allocate<std::uint8_t>(fixture.spins.size())};
    auto* rows_device{device_allocate<cuFloatComplex>(fixture.rows.size())};
    auto* gram_device{device_allocate<cuFloatComplex>(gram_elements)};
    if (not samples_device or not rows_device or not gram_device) return result;
    cudaMemcpy(samples_device, fixture.spins.data(), fixture.spins.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(
        rows_device,
        fixture.rows.data(),
        fixture.rows.size() * sizeof(cuFloatComplex),
        cudaMemcpyHostToDevice
    );
    qnpeps_gram_ctx* context{};
    result.status = qnpeps_gram_ctx_create(&descriptor, nullptr, &context);
    if (result.status == QNPEPS_OK)
    {
        QnpepsGramArgs args{};
        args.struct_size = sizeof(QnpepsGramArgs);
        args.samples = samples_device;
        args.samples_bytes = fixture.spins.size();
        args.o_rows = rows_device;
        args.o_rows_bytes = fixture.rows.size() * sizeof(cuFloatComplex);
        args.gram_out = gram_device;
        args.gram_out_bytes = gram_elements * sizeof(cuFloatComplex);
        result.status = qnpeps_gram_ctx_run(context, &args);
        if (result.status == QNPEPS_OK)
        {
            result.values.resize(gram_elements);
            cudaMemcpy(
                result.values.data(),
                gram_device,
                gram_elements * sizeof(cuFloatComplex),
                cudaMemcpyDeviceToHost
            );
        }
    }
    qnpeps_gram_ctx_destroy(context);
    cudaFree(gram_device);
    cudaFree(rows_device);
    cudaFree(samples_device);
    return result;
}

auto hermitian_relative(const std::vector<cfl>& values, int order) -> double
{
    double maximum{};
    double residual{};
    for (int row{}; row < order; ++row)
    {
        for (int column{}; column < order; ++column)
        {
            const cdb value{
                values[static_cast<std::size_t>(row * order + column)].real(),
                values[static_cast<std::size_t>(row * order + column)].imag()
            };
            const cdb mirror{
                values[static_cast<std::size_t>(column * order + row)].real(),
                values[static_cast<std::size_t>(column * order + row)].imag()
            };
            maximum = std::max(maximum, std::abs(value));
            residual = std::max(residual, std::abs(value - std::conj(mirror)));
        }
    }
    return residual / std::max(maximum, 1.0e-300);
}

auto run_gram_assumptions(const Fixture& fixture, Recorder& recorder) -> bool
{
    const GramRun pedantic_run{gram_once(fixture)};
    if (pedantic_run.status != QNPEPS_OK)
    {
        recorder.add("analytic", "analytic_case8_runs", "status_0", "error", false);
        return false;
    }
    const double asymmetry{hermitian_relative(pedantic_run.values, fixture.samples)};
    const bool hermitian_ok{std::isfinite(asymmetry) and asymmetry <= 1.0e-5};
    char asymmetry_text[64]{};
    std::snprintf(asymmetry_text, sizeof(asymmetry_text), "%.9g", asymmetry);
    recorder.add(
        "analytic", "analytic_case8_hermitian", "relative_le_1e-5", asymmetry_text, hermitian_ok
    );

    const std::vector<double> eigenvalues{jacobi_eigenvalues(pedantic_run.values, fixture.samples)};
    const double lambda_min{eigenvalues.front()};
    const double lambda_max{eigenvalues.back()};
    const double psd_bound{-5.0e-5 * std::max(lambda_max, 1.0)};
    const bool psd_ok{
        std::isfinite(lambda_min) and std::isfinite(lambda_max) and lambda_max >= 0.0
        and lambda_min >= psd_bound
    };
    char eigen_text[96]{};
    std::snprintf(eigen_text, sizeof(eigen_text), "min=%.9g;max=%.9g", lambda_min, lambda_max);
    recorder.add(
        "analytic", "analytic_case8_psd", "eigmin_ge_minus_5e-5_lammax", eigen_text, psd_ok
    );

    std::vector<cfl> nonhermitian{pedantic_run.values};
    nonhermitian[1] += cfl{0.25f, 0.5f};
    const double wrong_asymmetry{hermitian_relative(nonhermitian, fixture.samples)};
    const bool hermitian_control{wrong_asymmetry > 1.0e-5};
    char wrong_hermitian_text[64]{};
    std::snprintf(wrong_hermitian_text, sizeof(wrong_hermitian_text), "%.9g", wrong_asymmetry);
    recorder.add(
        "analytic",
        "analytic_case8_nonhermitian_control",
        "rejected",
        wrong_hermitian_text,
        hermitian_control
    );

    std::vector<cfl> indefinite{pedantic_run.values};
    indefinite[0] -= cfl{static_cast<float>(2.0 * std::max(lambda_max, 1.0)), 0.0f};
    const std::vector<double> wrong_eigenvalues{jacobi_eigenvalues(indefinite, fixture.samples)};
    const double wrong_min{wrong_eigenvalues.front()};
    const bool psd_control{wrong_min < psd_bound};
    char wrong_psd_text[64]{};
    std::snprintf(wrong_psd_text, sizeof(wrong_psd_text), "%.9g", wrong_min);
    recorder.add(
        "analytic", "analytic_case8_indefinite_control", "rejected", wrong_psd_text, psd_control
    );
    return hermitian_ok and psd_ok and hermitian_control and psd_control;
}

auto run_mode(const Fixture& fixture, Recorder& recorder) -> bool
{
    constexpr auto mode = "pedantic";
    QnpepsGramDesc descriptor{fixture.descriptor};

    const std::size_t gram_elements{
        static_cast<std::size_t>(fixture.samples) * static_cast<std::size_t>(fixture.samples)
    };
    auto* samples_device{device_allocate<std::uint8_t>(fixture.spins.size())};
    auto* rows_device{device_allocate<cuFloatComplex>(fixture.rows.size())};
    auto* transposed_device{device_allocate<cuFloatComplex>(fixture.transposed.size())};
    auto* gram_device{device_allocate<cuFloatComplex>(gram_elements)};
    if (not samples_device or not rows_device or not transposed_device or not gram_device)
        return false;
    cudaMemcpy(samples_device, fixture.spins.data(), fixture.spins.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(
        rows_device,
        fixture.rows.data(),
        fixture.rows.size() * sizeof(cuFloatComplex),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        transposed_device,
        fixture.transposed.data(),
        fixture.transposed.size() * sizeof(cuFloatComplex),
        cudaMemcpyHostToDevice
    );

    qnpeps_gram_ctx* context{};
    const qnpeps_status create_status{qnpeps_gram_ctx_create(&descriptor, nullptr, &context)};
    recorder.add(
        mode,
        "create",
        "status_0",
        create_status == QNPEPS_OK ? "status_0" : "error",
        create_status == QNPEPS_OK
    );
    if (create_status != QNPEPS_OK) return false;

    QnpepsGramFootprint footprint{};
    footprint.struct_size = sizeof(footprint);
    const qnpeps_status footprint_status{qnpeps_gram_ctx_footprint(context, &footprint)};
    const std::uint64_t expected_samples_bytes{
        static_cast<std::uint64_t>(fixture.samples) * static_cast<std::uint64_t>(fixture.sites)
    };
    const std::uint64_t expected_rows_bytes{
        static_cast<std::uint64_t>(fixture.samples) * static_cast<std::uint64_t>(fixture.compact)
        * sizeof(cuFloatComplex)
    };
    const std::uint64_t expected_gram_bytes{
        static_cast<std::uint64_t>(gram_elements) * sizeof(cuFloatComplex)
    };
    const std::uint64_t expected_geometry_bytes{
        static_cast<std::uint64_t>(fixture.sites) * 2 * sizeof(std::int32_t)
        + static_cast<std::uint64_t>(fixture.compact) * sizeof(std::int32_t)
    };
    const bool footprint_ok{
        footprint_status == QNPEPS_OK and footprint.reserved == 0
        and footprint.caller_samples_bytes == expected_samples_bytes
        and footprint.caller_rows_bytes == expected_rows_bytes
        and footprint.caller_gram_bytes == expected_gram_bytes
        and footprint.geometry_device_bytes == expected_geometry_bytes
        and footprint.dense_a_device_bytes > 0 and footprint.dense_b_device_bytes > 0
        and footprint.context_device_bytes >= footprint.geometry_device_bytes
                                                  + footprint.dense_a_device_bytes
                                                  + footprint.dense_b_device_bytes
    };
    recorder.add(mode, "footprint", "exact", footprint_ok ? "exact" : "different", footprint_ok);

    QnpepsGramArgs args{};
    args.struct_size = sizeof(QnpepsGramArgs);
    args.reserved = 0;
    args.samples = samples_device;
    args.samples_bytes = expected_samples_bytes;
    args.o_rows = rows_device;
    args.o_rows_bytes = expected_rows_bytes;
    args.gram_out = gram_device;
    args.gram_out_bytes = expected_gram_bytes;
    args.stream = nullptr;

    const qnpeps_status run_status{qnpeps_gram_ctx_run(context, &args)};
    auto observed = std::vector<cfl>(gram_elements);
    cudaMemcpy(
        observed.data(),
        gram_device,
        observed.size() * sizeof(cuFloatComplex),
        cudaMemcpyDeviceToHost
    );
    recorder.add(
        mode,
        "run",
        "status_0",
        run_status == QNPEPS_OK ? "status_0" : "error",
        run_status == QNPEPS_OK
    );

    const qnpeps_status repeat_status{qnpeps_gram_ctx_run(context, &args)};
    auto repeated = std::vector<cfl>(gram_elements);
    cudaMemcpy(
        repeated.data(),
        gram_device,
        repeated.size() * sizeof(cuFloatComplex),
        cudaMemcpyDeviceToHost
    );
    const bool repeat_exact{
        repeat_status == QNPEPS_OK
        and std::memcmp(observed.data(), repeated.data(), repeated.size() * sizeof(cuFloatComplex))
                == 0
    };
    recorder.add(
        mode, "repeat", "byte_exact", repeat_exact ? "byte_exact" : "different", repeat_exact
    );

    const std::vector<cfl> reference{host_reference(fixture, fixture.rows)};
    const double error{relative_error(observed, reference)};
    constexpr double threshold{5.0e-5};
    char observed_error[64]{};
    std::snprintf(observed_error, sizeof(observed_error), "%.9g", error);
    const bool reference_ok{std::isfinite(error) and error <= threshold};
    recorder.add(mode, "independent_reference", "relative_le_5e-5", observed_error, reference_ok);

    QnpepsGramArgs transposed_args{args};
    transposed_args.o_rows = transposed_device;
    const qnpeps_status transposed_status{qnpeps_gram_ctx_run(context, &transposed_args)};
    auto transposed_gram = std::vector<cfl>(gram_elements);
    cudaMemcpy(
        transposed_gram.data(),
        gram_device,
        transposed_gram.size() * sizeof(cuFloatComplex),
        cudaMemcpyDeviceToHost
    );
    const double transposed_error{relative_error(transposed_gram, reference)};
    char observed_transposed[64]{};
    std::snprintf(observed_transposed, sizeof(observed_transposed), "%.9g", transposed_error);
    const bool transposed_rejected{
        transposed_status == QNPEPS_OK
        and (not std::isfinite(transposed_error) or transposed_error > threshold)
    };
    recorder.add(
        mode,
        "wrong_path_transposed_rows",
        "reference_mismatch",
        observed_transposed,
        transposed_rejected
    );

    QnpepsGramArgs short_args{args};
    short_args.gram_out_bytes -= 1;
    const bool short_rejected{qnpeps_gram_ctx_run(context, &short_args) == QNPEPS_ERR_BAD_CONFIG};
    QnpepsGramArgs version_args{args};
    version_args.struct_size = 0;
    const bool version_rejected{
        qnpeps_gram_ctx_run(context, &version_args) == QNPEPS_ERR_BAD_VERSION
    };
    QnpepsGramArgs reserved_args{args};
    reserved_args.reserved = 1;
    const bool reserved_rejected{
        qnpeps_gram_ctx_run(context, &reserved_args) == QNPEPS_ERR_BAD_CONFIG
    };
    QnpepsGramArgs null_args{args};
    null_args.o_rows = nullptr;
    const bool null_rejected{qnpeps_gram_ctx_run(context, &null_args) == QNPEPS_ERR_NULL_ARG};
    const bool negatives_ok{
        short_rejected and version_rejected and reserved_rejected and null_rejected
    };
    recorder.add(
        mode, "negative_args", "rejected", negatives_ok ? "rejected" : "accepted", negatives_ok
    );

    qnpeps_gram_ctx_destroy(context);
    cudaFree(gram_device);
    cudaFree(transposed_device);
    cudaFree(rows_device);
    cudaFree(samples_device);
    return create_status == QNPEPS_OK and run_status == QNPEPS_OK and footprint_ok and repeat_exact
           and reference_ok and transposed_rejected and negatives_ok;
}

auto descriptor_negatives(const Fixture& fixture, Recorder& recorder) -> bool
{
    qnpeps_gram_ctx* context{};
    QnpepsGramDesc version{fixture.descriptor};
    version.struct_size = 0;
    const bool version_rejected{
        qnpeps_gram_ctx_create(&version, nullptr, &context) == QNPEPS_ERR_BAD_VERSION
        and context == nullptr
    };

    QnpepsGramDesc reserved{fixture.descriptor};
    reserved.reserved = 1;
    const bool reserved_rejected{
        qnpeps_gram_ctx_create(&reserved, nullptr, &context) == QNPEPS_ERR_BAD_CONFIG
    };

    QnpepsGramDesc consumer{fixture.descriptor};
    consumer.consumer = QNPEPS_GRAM_CONSUMER_CUSTOM;
    const bool consumer_rejected{
        qnpeps_gram_ctx_create(&consumer, nullptr, &context) == QNPEPS_ERR_BAD_CONFIG
    };

    QnpepsGramDesc lattice{fixture.descriptor};
    lattice.lx = 1;
    const bool lattice_rejected{
        qnpeps_gram_ctx_create(&lattice, nullptr, &context) == QNPEPS_ERR_BAD_CONFIG
    };

    const bool null_rejected{
        qnpeps_gram_ctx_create(&fixture.descriptor, nullptr, nullptr) == QNPEPS_ERR_NULL_ARG
    };

    const bool ok{
        version_rejected and reserved_rejected and consumer_rejected and lattice_rejected
        and null_rejected
    };
    recorder.add("common", "negative_descriptor", "rejected", ok ? "rejected" : "accepted", ok);
    return ok;
}
}

auto main() -> int
{
    const Fixture fixture{make_fixture()};
    Recorder recorder{};
    const bool assumptions{run_gram_assumptions(fixture, recorder)};
    const bool descriptors{descriptor_negatives(fixture, recorder)};
    const bool pedantic{run_mode(fixture, recorder)};
    const bool pass{recorder.pass and assumptions and descriptors and pedantic};
    std::printf("[gram_gate] RESULT=%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
