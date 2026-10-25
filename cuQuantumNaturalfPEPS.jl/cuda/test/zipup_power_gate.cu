#include "capi/qnpeps.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace
{
struct Result
{
    qnpeps_status status{QNPEPS_ERR_INTERNAL};
    double log_gauge{};
    std::array<cuFloatComplex, 3> output{};
    std::string message{};
};

auto execute(const std::array<cuFloatComplex, 2>& mps, const std::array<cuFloatComplex, 6>& mpo)
    -> Result
{
    const std::array<int32_t, 4> mpo_dims{1, 2, 3, 1};
    const std::array<int32_t, 3> mps_dims{1, 2, 1};
    const QnpepsZipupMpoMpsDesc descriptor{
        .struct_size = sizeof(QnpepsZipupMpoMpsDesc),
        .num_sites = 1,
        .maxdim = 1,
        .reserved = 0,
        .mpo_dims = mpo_dims.data(),
        .mps_dims = mps_dims.data(),
    };
    const auto output_bytes = qnpeps_zipup_mpo_mps_bytes(&descriptor);
    auto result = Result{};
    if (output_bytes != static_cast<int64_t>(result.output.size() * sizeof(cuFloatComplex)))
        return result;

    cuFloatComplex* device_mps{};
    cuFloatComplex* device_mpo{};
    cuFloatComplex* device_output{};
    if (cudaMalloc(&device_mps, sizeof(mps)) != cudaSuccess) return result;
    if (cudaMalloc(&device_mpo, sizeof(mpo)) != cudaSuccess)
    {
        cudaFree(device_mps);
        return result;
    }
    if (cudaMalloc(&device_output, static_cast<size_t>(output_bytes)) != cudaSuccess)
    {
        cudaFree(device_mpo);
        cudaFree(device_mps);
        return result;
    }
    cudaMemcpy(device_mps, mps.data(), sizeof(mps), cudaMemcpyHostToDevice);
    cudaMemcpy(device_mpo, mpo.data(), sizeof(mpo), cudaMemcpyHostToDevice);

    const QnpepsZipupMpoMpsArgs args{
        .struct_size = sizeof(QnpepsZipupMpoMpsArgs),
        .reserved = 0,
        .mpo = device_mpo,
        .mpo_bytes = sizeof(mpo),
        .mps = device_mps,
        .mps_bytes = sizeof(mps),
        .output = device_output,
        .output_bytes = static_cast<uint64_t>(output_bytes),
        .log_gauge = &result.log_gauge,
        .stream = nullptr,
    };
    result.status = qnpeps_zipup_mpo_mps(&descriptor, &args);
    const auto* message = qnpeps_last_error_message();
    if (message) result.message = message;
    if (result.status == QNPEPS_OK)
        cudaMemcpy(
            result.output.data(),
            device_output,
            result.output.size() * sizeof(cuFloatComplex),
            cudaMemcpyDeviceToHost
        );
    cudaFree(device_output);
    cudaFree(device_mpo);
    cudaFree(device_mps);
    return result;
}

auto matches(const Result& result, const std::array<cuFloatComplex, 3>& expected, double tolerance)
    -> bool
{
    if (result.status != QNPEPS_OK or not std::isfinite(result.log_gauge)) return false;
    const auto gauge = std::exp(result.log_gauge);
    for (size_t index{}; index < result.output.size(); ++index)
    {
        const auto actual_real = gauge * static_cast<double>(result.output[index].x);
        const auto actual_imaginary = gauge * static_cast<double>(result.output[index].y);
        const auto expected_real = static_cast<double>(expected[index].x);
        const auto expected_imaginary = static_cast<double>(expected[index].y);
        const auto error =
            std::hypot(actual_real - expected_real, actual_imaginary - expected_imaginary);
        const auto scale = std::max(1.0, std::hypot(expected_real, expected_imaginary));
        if (not std::isfinite(error) or error > tolerance * scale) return false;
    }
    return true;
}

}

auto main() -> int
{
    const std::array<cuFloatComplex, 2> unit_mps{{{1.0f, 0.0f}, {0.0f, 0.0f}}};
    const std::array<cuFloatComplex, 6> ordinary_mpo{{
        {1.0f, 0.25f},
        {0.0f, 0.0f},
        {-0.5f, 0.125f},
        {0.0f, 0.0f},
        {0.25f, -0.75f},
        {0.0f, 0.0f},
    }};
    const std::array<cuFloatComplex, 3> ordinary_expected{
        {ordinary_mpo[0], ordinary_mpo[2], ordinary_mpo[4]}
    };
    const auto ordinary = execute(unit_mps, ordinary_mpo);
    const auto ordinary_ok = matches(ordinary, ordinary_expected, 2.0e-5);

    const auto large = std::ldexp(1.0f, 36);
    const std::array<cuFloatComplex, 6> scaled_mpo{{
        {large, 0.0f},
        {0.0f, 0.0f},
        {-0.5f * large, 0.25f * large},
        {0.0f, 0.0f},
        {0.125f * large, -0.25f * large},
        {0.0f, 0.0f},
    }};
    const std::array<cuFloatComplex, 3> scaled_expected{
        {scaled_mpo[0], scaled_mpo[2], scaled_mpo[4]}
    };
    const auto scaled = execute(unit_mps, scaled_mpo);
    const auto scaled_ok = matches(scaled, scaled_expected, 2.0e-5);

    const auto overflow_component = std::ldexp(1.0f, 100);
    const std::array<cuFloatComplex, 2> overflow_mps{{
        {overflow_component, 0.0f},
        {0.0f, 0.0f},
    }};
    const std::array<cuFloatComplex, 6> overflow_mpo{{
        {overflow_component, 0.0f},
        {0.0f, 0.0f},
        {overflow_component, 0.0f},
        {0.0f, 0.0f},
        {overflow_component, 0.0f},
        {0.0f, 0.0f},
    }};
    const auto overflow = execute(overflow_mps, overflow_mpo);
    const auto overflow_ok = overflow.status == QNPEPS_ERR_INTERNAL
                             and overflow.message == "nonfinite zipup power intermediate";

    std::printf(
        "case,status,passed,log_gauge,message\n"
        "ordinary,%d,%d,%.17g,%s\n"
        "scaled,%d,%d,%.17g,%s\n"
        "finite_input_overflow,%d,%d,%.17g,%s\n",
        ordinary.status,
        ordinary_ok ? 1 : 0,
        ordinary.log_gauge,
        ordinary.message.c_str(),
        scaled.status,
        scaled_ok ? 1 : 0,
        scaled.log_gauge,
        scaled.message.c_str(),
        overflow.status,
        overflow_ok ? 1 : 0,
        overflow.log_gauge,
        overflow.message.c_str()
    );
    return ordinary_ok and scaled_ok and overflow_ok ? 0 : 1;
}
