#include "dans_qnpeps_e2e.h"
#include "update.cuh"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace
{

struct Cf
{
    float re;
    float im;
};

struct Zd
{
    double re;
    double im;
};

struct Site
{
    std::int64_t offset;
    std::int64_t count;
    int dw;
    int ds;
    int de;
    int dn;
    int dp;
};

enum class LayoutControl
{
    correct,
    raw_zip,
    axis_swap,
    site_reverse,
    physical_slow
};

static_assert(sizeof(Cf) == 2 * sizeof(float));
static_assert(sizeof(Zd) == 2 * sizeof(double));

auto bond_dim(int length, int position, int bond) -> int
{
    return position <= 0 or position >= length ? 1 : bond;
}

auto build_sites(const QnpepsE2eConfig& cfg) -> std::vector<Site>
{
    std::vector<Site> sites{};
    std::int64_t offset{};
    for (int row{}; row < cfg.lx; ++row)
    {
        for (int column{}; column < cfg.ly; ++column)
        {
            const int dw{bond_dim(cfg.ly, column, cfg.dim_bond)};
            const int ds{bond_dim(cfg.lx, row + 1, cfg.dim_bond)};
            const int de{bond_dim(cfg.ly, column + 1, cfg.dim_bond)};
            const int dn{bond_dim(cfg.lx, row, cfg.dim_bond)};
            const int dp{cfg.dim_phys};
            const std::int64_t count{static_cast<std::int64_t>(dw) * ds * de * dn * dp};
            sites.push_back(Site{offset, count, dw, ds, de, dn, dp});
            offset += count;
        }
    }
    return sites;
}

auto dense_count(const std::vector<Site>& sites) -> std::int64_t
{
    return sites.empty() ? 0 : sites.back().offset + sites.back().count;
}

auto theta_local(const Site& site, int physical, int west, int north, int south, int east)
    -> std::int64_t
{
    return physical
           + static_cast<std::int64_t>(site.dp)
                 * (west
                    + static_cast<std::int64_t>(site.dw)
                          * (north
                             + static_cast<std::int64_t>(site.dn)
                                   * (south + static_cast<std::int64_t>(site.ds) * east)));
}

auto physical_slow_local(const Site& site, int physical, int west, int north, int south, int east)
    -> std::int64_t
{
    return west
           + static_cast<std::int64_t>(site.dw)
                 * (north
                    + static_cast<std::int64_t>(site.dn)
                          * (south
                             + static_cast<std::int64_t>(site.ds)
                                   * (east + static_cast<std::int64_t>(site.de) * physical)));
}

auto fixture_local(const Site& site, int physical, int west, int north, int south, int east)
    -> std::int64_t
{
    return west
           + static_cast<std::int64_t>(site.dw)
                 * (south
                    + static_cast<std::int64_t>(site.ds)
                          * (east
                             + static_cast<std::int64_t>(site.de)
                                   * (north + static_cast<std::int64_t>(site.dn) * physical)));
}

auto axis_swap_local(const Site& site, int physical, int west, int north, int south, int east)
    -> std::int64_t
{
    return west
           + static_cast<std::int64_t>(site.dw)
                 * (north
                    + static_cast<std::int64_t>(site.dn)
                          * (east
                             + static_cast<std::int64_t>(site.de)
                                   * (south + static_cast<std::int64_t>(site.ds) * physical)));
}

auto separate_f32(float value, float rate, float direction) -> float
{
    volatile float product{rate * direction};
    volatile float result{value + product};
    return result;
}

auto separate_f64(double value, double rate, float direction) -> double
{
    volatile double product{rate * static_cast<double>(direction)};
    volatile double result{value + product};
    return result;
}

auto oracle_f32(
    const std::vector<Site>& sites,
    const std::vector<Cf>& initial,
    const std::vector<Cf>& theta,
    double learning_rate
) -> std::vector<Cf>
{
    std::vector<Cf> output{initial};
    const float rate{static_cast<float>(learning_rate)};
    for (const Site& site : sites)
    {
        for (int east{}; east < site.de; ++east)
        {
            for (int south{}; south < site.ds; ++south)
            {
                for (int north{}; north < site.dn; ++north)
                {
                    for (int west{}; west < site.dw; ++west)
                    {
                        for (int physical{}; physical < site.dp; ++physical)
                        {
                            const std::int64_t source{
                                site.offset + theta_local(site, physical, west, north, south, east)
                            };
                            const std::int64_t destination{
                                site.offset
                                + fixture_local(site, physical, west, north, south, east)
                            };
                            output[destination].re =
                                separate_f32(output[destination].re, rate, theta[source].re);
                            output[destination].im =
                                separate_f32(output[destination].im, rate, theta[source].im);
                        }
                    }
                }
            }
        }
    }
    return output;
}

auto oracle_f64(
    const std::vector<Site>& sites,
    const std::vector<Zd>& initial,
    const std::vector<Cf>& theta,
    double learning_rate,
    LayoutControl control
) -> std::vector<Zd>
{
    std::vector<Zd> output{initial};
    for (std::size_t site_index{}; site_index < sites.size(); ++site_index)
    {
        const auto& site{sites[site_index]};
        for (int east{}; east < site.de; ++east)
        {
            for (int south{}; south < site.ds; ++south)
            {
                for (int north{}; north < site.dn; ++north)
                {
                    for (int west{}; west < site.dw; ++west)
                    {
                        for (int physical{}; physical < site.dp; ++physical)
                        {
                            std::int64_t source_local{
                                theta_local(site, physical, west, north, south, east)
                            };
                            if (control == LayoutControl::physical_slow)
                            {
                                source_local =
                                    physical_slow_local(site, physical, west, north, south, east);
                            }

                            std::int64_t destination_local{
                                fixture_local(site, physical, west, north, south, east)
                            };
                            if (control == LayoutControl::raw_zip)
                            {
                                destination_local =
                                    theta_local(site, physical, west, north, south, east);
                            }
                            if (control == LayoutControl::axis_swap)
                            {
                                destination_local =
                                    axis_swap_local(site, physical, west, north, south, east);
                            }

                            std::int64_t destination_offset{site.offset};
                            if (control == LayoutControl::site_reverse)
                                destination_offset = sites[sites.size() - 1 - site_index].offset;

                            const std::int64_t source{site.offset + source_local};
                            const std::int64_t destination{destination_offset + destination_local};
                            output[destination].re = separate_f64(
                                output[destination].re, learning_rate, theta[source].re
                            );
                            output[destination].im = separate_f64(
                                output[destination].im, learning_rate, theta[source].im
                            );
                        }
                    }
                }
            }
        }
    }
    return output;
}

auto quantized(const std::vector<Zd>& input) -> std::vector<Cf>
{
    auto output = std::vector<Cf>(input.size());
    for (std::size_t index{}; index < input.size(); ++index)
        output[index] =
            Cf{static_cast<float>(input[index].re), static_cast<float>(input[index].im)};
    return output;
}

template <class T>
auto bit_equal(const std::vector<T>& lhs, const std::vector<T>& rhs) -> bool
{
    return lhs.size() == rhs.size()
           and std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(T)) == 0;
}

auto cuda_ok(cudaError_t status, const char* operation) -> bool
{
    if (status == cudaSuccess) return true;
    std::fprintf(
        stderr, "[gate_update] CUDA failure in %s with %s\n", operation, cudaGetErrorString(status)
    );
    return false;
}

struct Recorder
{
    std::ofstream file{};
    bool all{true};

    explicit Recorder(const char* path)
    {
        if (path)
        {
            file.open(path);
            if (file)
                file << "check,precision,expected,observed,status\n";
            else
                all = false;
        }
    }

    void add(
        const char* check,
        const char* precision,
        const char* expected,
        const char* observed,
        bool pass
    )
    {
        std::printf(
            "[gate_update] %s precision=%s expected=%s observed=%s status=%s\n",
            check,
            precision,
            expected,
            observed,
            pass ? "PASS" : "FAIL"
        );
        if (file)
        {
            file << check << ',' << precision << ',' << expected << ',' << observed << ','
                 << (pass ? "PASS" : "FAIL") << '\n';
        }
        all = all and pass;
    }
};

}

auto main(int argc, char** argv) -> int
{
    if (argc > 2)
    {
        std::fprintf(stderr, "[gate_update] usage gate_update [OUTPUT_CSV]\n");
        return 2;
    }
    Recorder record{argc == 2 ? argv[1] : nullptr};

    QnpepsE2eConfig cfg{};
    cfg.struct_size = sizeof(cfg);
    cfg.lx = 2;
    cfg.ly = 3;
    cfg.dim_phys = 2;
    cfg.dim_bond = 2;
    cfg.chi_s = 2;
    cfg.chi_dl = 2;
    cfg.chi_eo = 8;
    cfg.meo = 4;

    const std::vector<Site> sites{build_sites(cfg)};
    const std::int64_t dense{dense_count(sites)};
    const std::uint64_t f32_bytes{static_cast<std::uint64_t>(dense) * sizeof(Cf)};
    const std::uint64_t f64_bytes{static_cast<std::uint64_t>(dense) * sizeof(Zd)};
    const double learning_rate{0.0312500074505806};

    auto initial_f64 = std::vector<Zd>(static_cast<std::size_t>(dense));
    auto theta = std::vector<Cf>(static_cast<std::size_t>(dense));
    for (std::int64_t index{}; index < dense; ++index)
    {
        const auto ordinal{static_cast<std::uint64_t>(index + 1)};
        initial_f64[index].re = std::bit_cast<double>(
            UINT64_C(0x3ff0000000000000)
            | ((ordinal * UINT64_C(0x0000000100000021)) & UINT64_C(0x000fffffffffffff))
        );
        initial_f64[index].im = std::bit_cast<double>(
            UINT64_C(0xbfe0000000000000)
            | ((ordinal * UINT64_C(0x0000000200000043)) & UINT64_C(0x000fffffffffffff))
        );
        theta[index].re =
            std::bit_cast<float>(UINT32_C(0x3d000001) + static_cast<std::uint32_t>(index) * 257u);
        theta[index].im =
            std::bit_cast<float>(UINT32_C(0xbd800001) + static_cast<std::uint32_t>(index) * 263u);
    }

    const std::vector<Cf> initial_f32{quantized(initial_f64)};
    const std::vector<Cf> expected_f32{oracle_f32(sites, initial_f32, theta, learning_rate)};
    const std::vector<Zd> expected_f64{
        oracle_f64(sites, initial_f64, theta, learning_rate, LayoutControl::correct)
    };
    const std::vector<Cf> expected_from_f64{quantized(expected_f64)};

    Zd* state_device{};
    Cf* peps_device{};
    Cf* theta_device{};
    cudaStream_t stream{};
    if (not cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create")
        or not cuda_ok(cudaMalloc(&state_device, f64_bytes), "state allocation")
        or not cuda_ok(cudaMalloc(&peps_device, f32_bytes), "peps allocation")
        or not cuda_ok(cudaMalloc(&theta_device, f32_bytes), "theta allocation")
        or not cuda_ok(
            cudaMemcpy(theta_device, theta.data(), f32_bytes, cudaMemcpyHostToDevice),
            "theta upload"
        ))
    {
        cudaFree(theta_device);
        cudaFree(peps_device);
        cudaFree(state_device);
        if (stream) cudaStreamDestroy(stream);
        return 1;
    }

    const std::vector<qn_e2e::UpdateSite> update_sites{qn_e2e::build_update_sites(cfg)};
    bool metadata_ok{update_sites.size() == sites.size()};
    for (std::size_t index{}; metadata_ok and index < sites.size(); ++index)
    {
        const auto& expected{sites[index]};
        const auto& observed{update_sites[index]};
        metadata_ok = observed.offset == expected.offset and observed.count == expected.count
                      and observed.dw == expected.dw and observed.ds == expected.ds
                      and observed.de == expected.de and observed.dn == expected.dn
                      and observed.dp == expected.dp;
    }
    record.add(
        "site_metadata",
        "layout",
        "independent_match",
        metadata_ok ? "independent_match" : "different",
        metadata_ok
    );

    qn_e2e::UpdateSite* update_sites_device{};
    if (not cuda_ok(
            cudaMalloc(&update_sites_device, update_sites.size() * sizeof(qn_e2e::UpdateSite)),
            "site metadata allocation"
        )
        or not cuda_ok(
            cudaMemcpy(
                update_sites_device,
                update_sites.data(),
                update_sites.size() * sizeof(qn_e2e::UpdateSite),
                cudaMemcpyHostToDevice
            ),
            "site metadata upload"
        ))
    {
        cudaFree(update_sites_device);
        cudaFree(theta_device);
        cudaFree(peps_device);
        cudaFree(state_device);
        cudaStreamDestroy(stream);
        return 1;
    }

    auto observed_f32 = std::vector<Cf>(static_cast<std::size_t>(dense));
    auto observed_f64 = std::vector<Zd>(static_cast<std::size_t>(dense));

    cudaMemcpyAsync(peps_device, initial_f32.data(), f32_bytes, cudaMemcpyHostToDevice, stream);
    const qnpeps_e2e_status f32_status{qn_e2e::launch_update_f32(
        update_sites_device,
        static_cast<int>(update_sites.size()),
        reinterpret_cast<qn_e2e::cf*>(peps_device),
        reinterpret_cast<const qn_e2e::cf*>(theta_device),
        learning_rate,
        stream
    )};
    cudaStreamSynchronize(stream);
    cudaMemcpy(observed_f32.data(), peps_device, f32_bytes, cudaMemcpyDeviceToHost);
    record.add(
        "update_kernel",
        "fp32",
        "byte_exact",
        f32_status == QNPEPS_E2E_OK and bit_equal(observed_f32, expected_f32) ? "byte_exact"
                                                                              : "different",
        f32_status == QNPEPS_E2E_OK and bit_equal(observed_f32, expected_f32)
    );

    cudaMemcpyAsync(state_device, initial_f64.data(), f64_bytes, cudaMemcpyHostToDevice, stream);
    const qnpeps_e2e_status f64_status{qn_e2e::launch_update_f64(
        update_sites_device,
        static_cast<int>(update_sites.size()),
        reinterpret_cast<qn_e2e::zd*>(state_device),
        reinterpret_cast<const qn_e2e::cf*>(theta_device),
        reinterpret_cast<qn_e2e::cf*>(peps_device),
        learning_rate,
        stream
    )};
    cudaStreamSynchronize(stream);
    cudaMemcpy(observed_f64.data(), state_device, f64_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(observed_f32.data(), peps_device, f32_bytes, cudaMemcpyDeviceToHost);
    record.add(
        "update_kernel_master",
        "fp64",
        "byte_exact",
        f64_status == QNPEPS_E2E_OK and bit_equal(observed_f64, expected_f64) ? "byte_exact"
                                                                              : "different",
        f64_status == QNPEPS_E2E_OK and bit_equal(observed_f64, expected_f64)
    );
    record.add(
        "update_kernel_emit",
        "fp64_to_fp32",
        "byte_exact",
        f64_status == QNPEPS_E2E_OK and bit_equal(observed_f32, expected_from_f64) ? "byte_exact"
                                                                                   : "different",
        f64_status == QNPEPS_E2E_OK and bit_equal(observed_f32, expected_from_f64)
    );

    const qnpeps_e2e_status bad_rate_f32{qn_e2e::launch_update_f32(
        update_sites_device,
        static_cast<int>(update_sites.size()),
        reinterpret_cast<qn_e2e::cf*>(peps_device),
        reinterpret_cast<const qn_e2e::cf*>(theta_device),
        std::numeric_limits<double>::quiet_NaN(),
        stream
    )};
    record.add(
        "nonfinite_rate",
        "fp32",
        "rejected",
        bad_rate_f32 == QNPEPS_E2E_ERR_BAD_CONFIG ? "rejected" : "accepted",
        bad_rate_f32 == QNPEPS_E2E_ERR_BAD_CONFIG
    );

    const qnpeps_e2e_status bad_rate_f64{qn_e2e::launch_update_f64(
        update_sites_device,
        static_cast<int>(update_sites.size()),
        reinterpret_cast<qn_e2e::zd*>(state_device),
        reinterpret_cast<const qn_e2e::cf*>(theta_device),
        reinterpret_cast<qn_e2e::cf*>(peps_device),
        std::numeric_limits<double>::infinity(),
        stream
    )};
    record.add(
        "nonfinite_rate",
        "fp64",
        "rejected",
        bad_rate_f64 == QNPEPS_E2E_ERR_BAD_CONFIG ? "rejected" : "accepted",
        bad_rate_f64 == QNPEPS_E2E_ERR_BAD_CONFIG
    );

    const struct
    {
        const char* name;
        LayoutControl control;
    } controls[]{
        {"raw_zip_control", LayoutControl::raw_zip},
        {"axis_swap_control", LayoutControl::axis_swap},
        {"site_reverse_control", LayoutControl::site_reverse},
        {"physical_slow_control", LayoutControl::physical_slow}
    };
    for (const auto& control : controls)
    {
        const std::vector<Zd> wrong{
            oracle_f64(sites, initial_f64, theta, learning_rate, control.control)
        };
        record.add(
            control.name,
            "layout",
            "different",
            bit_equal(wrong, expected_f64) ? "same" : "different",
            not bit_equal(wrong, expected_f64)
        );
    }

    cudaFree(update_sites_device);
    cudaFree(theta_device);
    cudaFree(peps_device);
    cudaFree(state_device);
    cudaStreamDestroy(stream);

    std::printf("[gate_update] RESULT=%s\n", record.all ? "PASS" : "FAIL");
    return record.all ? 0 : 1;
}
