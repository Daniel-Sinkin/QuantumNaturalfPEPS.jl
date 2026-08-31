#include "capi/qnpeps.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <utility>
#include <vector>

namespace
{
auto close(cuFloatComplex left, cuFloatComplex right, float tolerance) -> bool
{
    return std::hypot(left.x - right.x, left.y - right.y) <= tolerance;
}

auto multisite_smoke() -> bool
{
    const std::vector<int32_t> mpo_dims{
        1,
        2,
        2,
        2,
        2,
        3,
        2,
        2,
        2,
        2,
        3,
        1,
    };
    const std::vector<int32_t> mps_dims{
        1,
        2,
        2,
        2,
        3,
        2,
        2,
        2,
        1,
    };
    const QnpepsZipupMpoMpsDesc descriptor{
        .struct_size = sizeof(QnpepsZipupMpoMpsDesc),
        .num_sites = 3,
        .maxdim = 32,
        .reserved = 0,
        .mpo_dims = mpo_dims.data(),
        .mps_dims = mps_dims.data(),
    };
    const auto output_bytes = qnpeps_zipup_mpo_mps_bytes(&descriptor);
    if (output_bytes != 32 * static_cast<int64_t>(sizeof(cuFloatComplex))) return false;

    auto host_mpo = std::vector<cuFloatComplex>(44);
    auto host_mps = std::vector<cuFloatComplex>(20);
    for (size_t index{}; index < host_mpo.size(); ++index)
    {
        host_mpo[index] = make_cuFloatComplex(
            0.01f * static_cast<float>(index + 1), -0.005f * static_cast<float>((index % 7) + 1)
        );
    }
    for (size_t index{}; index < host_mps.size(); ++index)
    {
        host_mps[index] = make_cuFloatComplex(
            -0.02f * static_cast<float>(index + 1), 0.003f * static_cast<float>((index % 5) + 1)
        );
    }

    cuFloatComplex* device_mpo{};
    cuFloatComplex* device_mps{};
    cuFloatComplex* device_output{};
    if (cudaMalloc(&device_mpo, host_mpo.size() * sizeof(cuFloatComplex)) != cudaSuccess)
        return false;
    if (cudaMalloc(&device_mps, host_mps.size() * sizeof(cuFloatComplex)) != cudaSuccess)
        return false;
    if (cudaMalloc(&device_output, static_cast<size_t>(output_bytes)) != cudaSuccess) return false;
    cudaMemcpy(
        device_mpo,
        host_mpo.data(),
        host_mpo.size() * sizeof(cuFloatComplex),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        device_mps,
        host_mps.data(),
        host_mps.size() * sizeof(cuFloatComplex),
        cudaMemcpyHostToDevice
    );

    double log_gauge{};
    const QnpepsZipupMpoMpsArgs args{
        .struct_size = sizeof(QnpepsZipupMpoMpsArgs),
        .reserved = 0,
        .mpo = device_mpo,
        .mpo_bytes = host_mpo.size() * sizeof(cuFloatComplex),
        .mps = device_mps,
        .mps_bytes = host_mps.size() * sizeof(cuFloatComplex),
        .output = device_output,
        .output_bytes = static_cast<uint64_t>(output_bytes),
        .log_gauge = &log_gauge,
        .stream = nullptr,
    };
    const auto status = qnpeps_zipup_mpo_mps(&descriptor, &args);
    auto output = std::vector<cuFloatComplex>(32);
    cudaMemcpy(
        output.data(), device_output, output.size() * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost
    );
    if (status != QNPEPS_OK or not std::isfinite(log_gauge)) return false;
    for (const auto value : output)
        if (not std::isfinite(value.x) or not std::isfinite(value.y)) return false;

    const std::vector<int> output_dims{
        1,
        2,
        2,
        2,
        2,
        4,
        4,
        3,
        1,
    };
    const auto mps_amplitude = [](const std::vector<cuFloatComplex>& values,
                                  const auto& dimensions,
                                  const std::vector<int>& configuration)
    {
        std::vector<std::complex<double>> carried{{1.0, 0.0}};
        size_t offset{};
        for (size_t site{}; site < configuration.size(); ++site)
        {
            const auto left = dimensions[3 * site];
            const auto physical = dimensions[3 * site + 1];
            const auto right = dimensions[3 * site + 2];
            auto next = std::vector<std::complex<double>>(static_cast<size_t>(right));
            for (auto left_index = 0; left_index < left; ++left_index)
            {
                for (auto right_index = 0; right_index < right; ++right_index)
                {
                    const auto index =
                        offset
                        + static_cast<size_t>(
                            left_index + left * (configuration[site] + physical * right_index)
                        );
                    const auto value = values[index];
                    next[static_cast<size_t>(right_index)] +=
                        carried[static_cast<size_t>(left_index)]
                        * std::complex<double>{value.x, value.y};
                }
            }
            offset += static_cast<size_t>(left * physical * right);
            carried = std::move(next);
        }
        return carried.front();
    };
    const auto mpo_element = [&](const std::vector<int>& input, const std::vector<int>& out)
    {
        std::vector<std::complex<double>> carried{{1.0, 0.0}};
        size_t offset{};
        for (size_t site{}; site < input.size(); ++site)
        {
            const auto left = mpo_dims[4 * site];
            const auto physical_in = mpo_dims[4 * site + 1];
            const auto physical_out = mpo_dims[4 * site + 2];
            const auto right = mpo_dims[4 * site + 3];
            auto next = std::vector<std::complex<double>>(static_cast<size_t>(right));
            for (auto left_index = 0; left_index < left; ++left_index)
            {
                for (auto right_index = 0; right_index < right; ++right_index)
                {
                    const auto index =
                        offset
                        + static_cast<size_t>(
                            left_index
                            + left
                                  * (input[site]
                                     + physical_in * (out[site] + physical_out * right_index))
                        );
                    const auto value = host_mpo[index];
                    next[static_cast<size_t>(right_index)] +=
                        carried[static_cast<size_t>(left_index)]
                        * std::complex<double>{value.x, value.y};
                }
            }
            offset += static_cast<size_t>(left * physical_in * physical_out * right);
            carried = std::move(next);
        }
        return carried.front();
    };

    auto max_error = 0.0;
    auto max_reference = 0.0;
    const auto gauge = std::exp(log_gauge);
    for (auto output_0 = 0; output_0 < 2; ++output_0)
    {
        for (auto output_1 = 0; output_1 < 2; ++output_1)
        {
            for (auto output_2 = 0; output_2 < 3; ++output_2)
            {
                const std::vector<int> output_configuration{output_0, output_1, output_2};
                std::complex<double> reference{};
                for (auto input_0 = 0; input_0 < 2; ++input_0)
                {
                    for (auto input_1 = 0; input_1 < 3; ++input_1)
                    {
                        for (auto input_2 = 0; input_2 < 2; ++input_2)
                        {
                            const std::vector<int> input_configuration{input_0, input_1, input_2};
                            reference += mps_amplitude(host_mps, mps_dims, input_configuration)
                                         * mpo_element(input_configuration, output_configuration);
                        }
                    }
                }
                const auto actual =
                    gauge * mps_amplitude(output, output_dims, output_configuration);
                max_error = std::max(max_error, std::abs(actual - reference));
                max_reference = std::max(max_reference, std::abs(reference));
            }
        }
    }

    cudaFree(device_output);
    cudaFree(device_mps);
    cudaFree(device_mpo);
    return max_error / max_reference < 5.0e-3;
}
}

auto main() -> int
{
    std::vector<int32_t> mpo_dims{1, 2, 3, 1};
    std::vector<int32_t> mps_dims{1, 2, 1};
    QnpepsZipupMpoMpsDesc descriptor{
        .struct_size = sizeof(QnpepsZipupMpoMpsDesc),
        .num_sites = 1,
        .maxdim = 8,
        .reserved = 0,
        .mpo_dims = mpo_dims.data(),
        .mps_dims = mps_dims.data(),
    };
    const auto output_bytes = qnpeps_zipup_mpo_mps_bytes(&descriptor);
    if (output_bytes != 3 * static_cast<int64_t>(sizeof(cuFloatComplex))) return 1;

    auto wrong_version = descriptor;
    wrong_version.struct_size = 0;
    if (qnpeps_zipup_mpo_mps_bytes(&wrong_version) != -1) return 1;
    auto broken_dims = mpo_dims;
    broken_dims[3] = 2;
    auto broken_descriptor = descriptor;
    broken_descriptor.mpo_dims = broken_dims.data();
    if (qnpeps_zipup_mpo_mps_bytes(&broken_descriptor) != -1) return 1;

    const std::vector<cuFloatComplex> host_mps{{0.75f, -0.25f}, {-0.5f, 0.125f}};
    const std::vector<cuFloatComplex> host_mpo{
        {1.0f, 0.0f},
        {0.5f, 0.25f},
        {-0.25f, 0.5f},
        {0.75f, -0.125f},
        {0.4f, 0.2f},
        {-0.3f, 0.6f},
    };
    auto reference = std::vector<cuFloatComplex>(3);
    for (auto output = 0; output < 3; ++output)
    {
        auto value = make_cuFloatComplex(0.0f, 0.0f);
        for (auto input = 0; input < 2; ++input)
        {
            value = cuCaddf(
                value,
                cuCmulf(
                    host_mps[static_cast<size_t>(input)],
                    host_mpo[static_cast<size_t>(input + 2 * output)]
                )
            );
        }
        reference[static_cast<size_t>(output)] = value;
    }

    cuFloatComplex* device_mpo{};
    cuFloatComplex* device_mps{};
    cuFloatComplex* device_output{};
    cudaStream_t stream{};
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return 1;
    if (cudaMalloc(&device_mpo, host_mpo.size() * sizeof(cuFloatComplex)) != cudaSuccess) return 1;
    if (cudaMalloc(&device_mps, host_mps.size() * sizeof(cuFloatComplex)) != cudaSuccess) return 1;
    if (cudaMalloc(&device_output, static_cast<size_t>(output_bytes)) != cudaSuccess) return 1;
    cudaMemcpy(
        device_mpo,
        host_mpo.data(),
        host_mpo.size() * sizeof(cuFloatComplex),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        device_mps,
        host_mps.data(),
        host_mps.size() * sizeof(cuFloatComplex),
        cudaMemcpyHostToDevice
    );

    double log_gauge{};
    QnpepsZipupMpoMpsArgs args{
        .struct_size = sizeof(QnpepsZipupMpoMpsArgs),
        .reserved = 0,
        .mpo = device_mpo,
        .mpo_bytes = host_mpo.size() * sizeof(cuFloatComplex),
        .mps = device_mps,
        .mps_bytes = host_mps.size() * sizeof(cuFloatComplex),
        .output = device_output,
        .output_bytes = static_cast<uint64_t>(output_bytes),
        .log_gauge = &log_gauge,
        .stream = stream,
    };
    auto short_args = args;
    short_args.output_bytes -= 1;
    if (qnpeps_zipup_mpo_mps(&descriptor, &short_args) != QNPEPS_ERR_BAD_CONFIG) return 1;
    if (qnpeps_zipup_mpo_mps(&descriptor, &args) != QNPEPS_OK) return 1;

    auto output = std::vector<cuFloatComplex>(3);
    cudaMemcpy(
        output.data(), device_output, output.size() * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost
    );
    const auto gauge = static_cast<float>(std::exp(log_gauge));
    auto passed = std::isfinite(log_gauge);
    for (size_t element{}; element < output.size(); ++element)
    {
        output[element].x *= gauge;
        output[element].y *= gauge;
        passed = passed and close(output[element], reference[element], 2.0e-3f);
    }

    cudaFree(device_output);
    cudaFree(device_mps);
    cudaFree(device_mpo);
    cudaStreamDestroy(stream);
    passed = passed and multisite_smoke();
    std::printf("zipup_gate,%s,log_gauge=%.9g\n", passed ? "PASS" : "FAIL", log_gauge);
    return passed ? 0 : 1;
}
