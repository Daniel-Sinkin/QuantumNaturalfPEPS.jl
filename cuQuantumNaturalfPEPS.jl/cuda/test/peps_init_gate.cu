#include "capi/qnpeps.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

namespace
{
struct Cf
{
    float re{};
    float im{};
};

struct SiteDims
{
    int left{};
    int down{};
    int right{};
    int up{};
    int physical{};

    [[nodiscard]] auto elements() const -> std::size_t
    {
        return static_cast<std::size_t>(left) * down * right * up * physical;
    }
};

auto bond_dim(int extent, int position, int dimension) -> int
{
    return position <= 0 or position >= extent ? 1 : dimension;
}

auto site_dims(const QnpepsConfig& config, int row, int col) -> SiteDims
{
    return {
        .left = bond_dim(config.ly, col, config.dim_bond),
        .down = bond_dim(config.lx, row + 1, config.dim_bond),
        .right = bond_dim(config.ly, col + 1, config.dim_bond),
        .up = bond_dim(config.lx, row, config.dim_bond),
        .physical = config.dim_phys,
    };
}

auto value_at(
    const std::vector<Cf>& peps,
    std::size_t offset,
    const SiteDims& dims,
    int left,
    int down,
    int right,
    int up,
    int physical
) -> std::complex<double>
{
    const auto index =
        offset
        + static_cast<std::size_t>(
            left + dims.left * (down + dims.down * (right + dims.right * (up + dims.up * physical)))
        );
    return {peps[index].re, peps[index].im};
}

auto isometry_residual(const std::vector<Cf>& peps, const QnpepsConfig& config) -> double
{
    std::size_t offset{};
    double maximum{};
    for (auto row = 0; row < config.lx; ++row)
    {
        for (auto col = 0; col < config.ly; ++col)
        {
            const auto dims = site_dims(config, row, col);
            const int incoming{dims.physical * dims.left * dims.up};
            const int outgoing{dims.right * dims.down};
            const int identity_dim{std::min(incoming, outgoing)};
            for (auto first = 0; first < identity_dim; ++first)
            {
                for (auto second = 0; second < identity_dim; ++second)
                {
                    std::complex<double> inner{};
                    if (incoming >= outgoing)
                    {
                        const int first_right{first % dims.right};
                        const int first_down{first / dims.right};
                        const int second_right{second % dims.right};
                        const int second_down{second / dims.right};
                        for (auto index = 0; index < incoming; ++index)
                        {
                            const int physical{index % dims.physical};
                            const int bond{index / dims.physical};
                            const int left{bond % dims.left};
                            const int up{bond / dims.left};
                            const auto first_value = value_at(
                                peps, offset, dims, left, first_down, first_right, up, physical
                            );
                            const auto second_value = value_at(
                                peps, offset, dims, left, second_down, second_right, up, physical
                            );
                            inner += std::conj(first_value) * second_value;
                        }
                    }
                    else
                    {
                        const int first_physical{first % dims.physical};
                        const int first_bond{first / dims.physical};
                        const int first_left{first_bond % dims.left};
                        const int first_up{first_bond / dims.left};
                        const int second_physical{second % dims.physical};
                        const int second_bond{second / dims.physical};
                        const int second_left{second_bond % dims.left};
                        const int second_up{second_bond / dims.left};
                        for (auto index = 0; index < outgoing; ++index)
                        {
                            const int right{index % dims.right};
                            const int down{index / dims.right};
                            const auto first_value = value_at(
                                peps,
                                offset,
                                dims,
                                first_left,
                                down,
                                right,
                                first_up,
                                first_physical
                            );
                            const auto second_value = value_at(
                                peps,
                                offset,
                                dims,
                                second_left,
                                down,
                                right,
                                second_up,
                                second_physical
                            );
                            inner += first_value * std::conj(second_value);
                        }
                    }
                    const std::complex<double> expected{first == second ? 1.0 : 0.0, 0.0};
                    maximum = std::max(maximum, std::abs(inner - expected));
                }
            }
            offset += dims.elements();
        }
    }
    return offset == peps.size() ? maximum : INFINITY;
}

auto spectrum_residual(
    const std::vector<Cf>& pure,
    const std::vector<Cf>& weighted,
    const QnpepsConfig& config,
    double alpha
) -> double
{
    std::size_t offset{};
    double maximum{};
    for (auto row = 0; row < config.lx; ++row)
    {
        for (auto col = 0; col < config.ly; ++col)
        {
            const auto dims = site_dims(config, row, col);
            for (auto physical = 0; physical < dims.physical; ++physical)
            {
                for (auto up = 0; up < dims.up; ++up)
                {
                    for (auto right = 0; right < dims.right; ++right)
                    {
                        for (auto down = 0; down < dims.down; ++down)
                        {
                            for (auto left = 0; left < dims.left; ++left)
                            {
                                float scale{1.0f};
                                const double half_negative_alpha{-0.5 * alpha};
                                if (dims.left > 1)
                                {
                                    scale *= static_cast<float>(
                                        std::pow(static_cast<double>(left + 1), half_negative_alpha)
                                    );
                                }
                                if (dims.down > 1)
                                {
                                    scale *= static_cast<float>(
                                        std::pow(static_cast<double>(down + 1), half_negative_alpha)
                                    );
                                }
                                if (dims.right > 1)
                                {
                                    scale *= static_cast<float>(std::pow(
                                        static_cast<double>(right + 1), half_negative_alpha
                                    ));
                                }
                                if (dims.up > 1)
                                {
                                    scale *= static_cast<float>(
                                        std::pow(static_cast<double>(up + 1), half_negative_alpha)
                                    );
                                }
                                const auto pure_value =
                                    value_at(pure, offset, dims, left, down, right, up, physical);
                                const auto weighted_value = value_at(
                                    weighted, offset, dims, left, down, right, up, physical
                                );
                                maximum = std::max(
                                    maximum,
                                    std::abs(
                                        weighted_value - pure_value * static_cast<double>(scale)
                                    )
                                );
                            }
                        }
                    }
                }
            }
            offset += dims.elements();
        }
    }
    return offset == pure.size() and pure.size() == weighted.size() ? maximum : INFINITY;
}

auto cuda_ok(cudaError_t status, const char* operation) -> bool
{
    if (status == cudaSuccess) return true;
    std::fprintf(
        stderr, "[peps_init_gate] %s failed with %s\n", operation, cudaGetErrorString(status)
    );
    return false;
}
}

auto main() -> int
{
    QnpepsConfig config{};
    config.struct_size = sizeof(config);
    config.lx = 3;
    config.ly = 2;
    config.dim_phys = 2;
    config.dim_bond = 3;
    config.chi_s = 3;
    config.chi_dl = 3;
    config.sampling_mode = QNPEPS_SAMPLING_FAST;
    config.chi_c = 3;

    const auto bytes = qnpeps_peps_bytes(&config);
    if (bytes <= 0 or bytes % static_cast<std::int64_t>(sizeof(Cf)) != 0) return 1;
    Cf* first{};
    Cf* repeat{};
    Cf* weighted{};
    if (not cuda_ok(cudaMalloc(reinterpret_cast<void**>(&first), bytes), "allocate first")
        or not cuda_ok(cudaMalloc(reinterpret_cast<void**>(&repeat), bytes), "allocate repeat")
        or not cuda_ok(cudaMalloc(reinterpret_cast<void**>(&weighted), bytes), "allocate weighted"))
    {
        cudaFree(weighted);
        cudaFree(repeat);
        cudaFree(first);
        return 1;
    }

    constexpr std::uint64_t seed{11};
    constexpr double alpha{2.0};
    const auto first_status = qnpeps_random_unitary_peps(
        &config,
        reinterpret_cast<qnpeps_device_peps*>(first),
        static_cast<std::uint64_t>(bytes),
        seed,
        0.0,
        nullptr
    );
    const auto repeat_status = qnpeps_random_unitary_peps(
        &config,
        reinterpret_cast<qnpeps_device_peps*>(repeat),
        static_cast<std::uint64_t>(bytes),
        seed,
        0.0,
        nullptr
    );
    const auto weighted_status = qnpeps_random_unitary_peps(
        &config,
        reinterpret_cast<qnpeps_device_peps*>(weighted),
        static_cast<std::uint64_t>(bytes),
        seed,
        alpha,
        nullptr
    );
    const auto short_status = qnpeps_random_unitary_peps(
        &config,
        reinterpret_cast<qnpeps_device_peps*>(weighted),
        static_cast<std::uint64_t>(bytes - 1),
        seed,
        alpha,
        nullptr
    );
    const auto alpha_status = qnpeps_random_unitary_peps(
        &config,
        reinterpret_cast<qnpeps_device_peps*>(weighted),
        static_cast<std::uint64_t>(bytes),
        seed,
        -1.0,
        nullptr
    );

    const auto elements = static_cast<std::size_t>(bytes) / sizeof(Cf);
    auto host_first = std::vector<Cf>(elements);
    auto host_repeat = std::vector<Cf>(elements);
    auto host_weighted = std::vector<Cf>(elements);
    const auto copies_ok =
        cuda_ok(
            cudaMemcpy(host_first.data(), first, bytes, cudaMemcpyDeviceToHost), "download first"
        )
        and cuda_ok(
            cudaMemcpy(host_repeat.data(), repeat, bytes, cudaMemcpyDeviceToHost), "download repeat"
        )
        and cuda_ok(
            cudaMemcpy(host_weighted.data(), weighted, bytes, cudaMemcpyDeviceToHost),
            "download weighted"
        );
    cudaFree(weighted);
    cudaFree(repeat);
    cudaFree(first);
    if (not copies_ok) return 1;

    const auto finite = std::all_of(
        host_first.begin(),
        host_first.end(),
        [](const Cf& value) { return std::isfinite(value.re) and std::isfinite(value.im); }
    );
    const auto exact_repeat =
        std::memcmp(host_first.data(), host_repeat.data(), static_cast<std::size_t>(bytes)) == 0;
    const double iso_residual{isometry_residual(host_first, config)};
    const double alpha_residual{spectrum_residual(host_first, host_weighted, config, alpha)};
    const auto passed = first_status == QNPEPS_OK and repeat_status == QNPEPS_OK
                        and weighted_status == QNPEPS_OK and short_status == QNPEPS_ERR_BAD_CONFIG
                        and alpha_status == QNPEPS_ERR_BAD_CONFIG and finite and exact_repeat
                        and iso_residual <= 1.0e-5 and alpha_residual <= 1.0e-7;
    std::printf(
        "[peps_init_gate] exact=%d isometry_residual=%.9g alpha_residual=%.9g status=%s\n",
        exact_repeat ? 1 : 0,
        iso_residual,
        alpha_residual,
        passed ? "PASS" : "FAIL"
    );
    return passed ? 0 : 1;
}
