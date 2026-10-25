#include "capi/qnpeps.h"
#include "core/qnpeps_ctx.cuh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace
{
auto make_peps(const QnpepsConfig& config, float scale) -> qnpeps_device_peps*
{
    const auto bytes = qnpeps_peps_bytes(&config);
    if (bytes <= 0 or bytes % static_cast<int64_t>(sizeof(cuFloatComplex)) != 0) return nullptr;
    auto values = std::vector<cuFloatComplex>(
        static_cast<size_t>(bytes / static_cast<int64_t>(sizeof(cuFloatComplex)))
    );
    for (size_t index{}; index < values.size(); index += 2)
        values[index] = make_cuFloatComplex(scale, 0.0f);
    qnpeps_device_peps* device{};
    if (cudaMalloc(&device, static_cast<size_t>(bytes)) != cudaSuccess) return nullptr;
    if (cudaMemcpy(device, values.data(), static_cast<size_t>(bytes), cudaMemcpyHostToDevice)
        != cudaSuccess)
    {
        cudaFree(device);
        return nullptr;
    }
    return device;
}

}

auto main() -> int
{
    const QnpepsConfig config{
        .struct_size = sizeof(QnpepsConfig),
        .lx = 2,
        .ly = 2,
        .dim_phys = 2,
        .dim_bond = 1,
        .chi_s = 1,
        .chi_dl = 1,
        .seed = 17,
        .sampling_mode = QNPEPS_SAMPLING_FAST,
        .chi_c = 1,
        .dlenv_truncation_route = QNPEPS_TRUNCATION_DEFAULT,
        .sampler_truncation_route = QNPEPS_TRUNCATION_DEFAULT,
        .dlenv_density_cutoff = 1.0e-13,
        .sampler_density_cutoff = 1.0e-3,
        .projected_density_cutoff = 1.0e-4,
    };
    auto* normal = make_peps(config, 1.0f);
    auto* scaled = make_peps(config, std::ldexp(1.0f, 36));
    auto* overflow = make_peps(config, std::ldexp(1.0f, 100));
    if (not normal or not scaled or not overflow) return 1;

    qnpeps_ctx* context{};
    if (qnpeps_ctx_create(&config, nullptr, &context) != QNPEPS_OK or not context) return 1;
    const auto first = qnpeps_ctx_build_dlenv(context, normal, nullptr);
    const auto second = qnpeps_ctx_build_dlenv(context, normal, nullptr);
    const auto lane_one_captured = context->dlenv.lanes[1].graph != nullptr;
    const auto third = qnpeps_ctx_build_dlenv(context, normal, nullptr);
    const auto lane_zero_captured = context->dlenv.lanes[0].graph != nullptr;
    const auto build_count_before_failure = context->dlenv.build_count;
    const auto failure = qnpeps_ctx_build_dlenv(context, overflow, nullptr);
    const auto failure_message = std::string{qnpeps_last_error_message()};
    const auto build_count_after_failure = context->dlenv.build_count;
    const auto recovery = qnpeps_ctx_build_dlenv(context, normal, nullptr);
    const auto recovery_lane = context->dlenv.active_lane;
    const auto large_scale = qnpeps_ctx_build_dlenv(context, scaled, nullptr);

    const auto passed = first == QNPEPS_OK and second == QNPEPS_OK and third == QNPEPS_OK
                        and lane_one_captured and lane_zero_captured
                        and build_count_before_failure == 3 and failure == QNPEPS_ERR_INTERNAL
                        and failure_message == "nonfinite zipup power intermediate"
                        and build_count_after_failure == build_count_before_failure
                        and recovery == QNPEPS_OK and recovery_lane == 1
                        and context->dlenv.build_count == 5 and large_scale == QNPEPS_OK;
    std::printf(
        "case,status,passed,build_count,lane_zero_graph,lane_one_graph,message\n"
        "graph_failure_recovery,%d,%d,%d,%d,%d,%s\n",
        failure,
        passed ? 1 : 0,
        context->dlenv.build_count,
        lane_zero_captured ? 1 : 0,
        lane_one_captured ? 1 : 0,
        failure_message.c_str()
    );

    qnpeps_ctx_destroy(context);
    cudaFree(overflow);
    cudaFree(scaled);
    cudaFree(normal);
    return passed ? 0 : 1;
}
