#include "boundary.hpp"
#include "dans_qnpeps_eloc.h"
#include "gate_fixture.cuh"
#include "mps.hpp"
#include "tensor.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>
#include <stdexcept>
#include <vector>

namespace
{

using namespace gate;

auto device_logpsi(
    const GenPeps& g,
    int lx,
    int ly,
    int dim_phys,
    int dim_bond,
    int chi,
    int meo,
    const std::vector<std::uint8_t>& samples,
    int n,
    std::vector<double>& raw_out
) -> qnpeps_eloc_status
{
    QnpepsElocConfig cfg{};
    cfg.struct_size = sizeof(QnpepsElocConfig);
    cfg.lx = lx;
    cfg.ly = ly;
    cfg.dim_phys = dim_phys;
    cfg.dim_bond = dim_bond;
    cfg.chi_eo = chi;
    cfg.meo = meo;

    const std::size_t total{g.flat.size()};
    auto hpeps = std::vector<float>(2 * total);
    for (std::size_t k{0}; k < total; ++k)
    {
        hpeps[2 * k] = static_cast<float>(g.flat[k]);
        hpeps[2 * k + 1] = 0.0f;
    }

    void* d_peps{};
    cudaMalloc(&d_peps, hpeps.size() * sizeof(float));
    cudaMemcpy(d_peps, hpeps.data(), hpeps.size() * sizeof(float), cudaMemcpyHostToDevice);
    std::uint8_t* d_samp{};
    cudaMalloc(reinterpret_cast<void**>(&d_samp), samples.size());
    cudaMemcpy(d_samp, samples.data(), samples.size(), cudaMemcpyHostToDevice);
    double* d_lp{};
    cudaMalloc(reinterpret_cast<void**>(&d_lp), static_cast<std::size_t>(2 * n) * sizeof(double));

    const qnpeps_eloc_status st{qnpeps_eloc_logpsi(
        &cfg, static_cast<const qnpeps_eloc_cbuf*>(d_peps), d_samp, n, d_lp, nullptr
    )};

    raw_out.assign(static_cast<std::size_t>(2 * n), 0.0);
    cudaMemcpy(
        raw_out.data(),
        d_lp,
        static_cast<std::size_t>(2 * n) * sizeof(double),
        cudaMemcpyDeviceToHost
    );
    cudaFree(d_peps);
    cudaFree(d_samp);
    cudaFree(d_lp);
    return st;
}

auto run_case(
    int lx, int ly, int dim_bond, int chi, int n, int meo, int mode, double noise, double tol
) -> int
{
    const int dim_phys{2};
    const std::uint64_t peps_seed{
        0xC0FFEEull ^ (static_cast<std::uint64_t>(lx) * 131 + ly * 17 + dim_bond * 7)
    };
    const GenPeps g{make_peps(lx, ly, dim_bond, dim_phys, chi, mode, noise, peps_seed)};

    int fails{0};
    double max_d_re{0.0};
    double max_r_re{0.0};
    double max_d_im{0.0};

    const std::vector<std::uint8_t> sa{
        gen_samples(lx, ly, dim_phys, n, 0xA11CEull ^ peps_seed, false)
    };
    std::vector<double> dev_a{};
    const qnpeps_eloc_status sta{
        device_logpsi(g, lx, ly, dim_phys, dim_bond, chi, meo, sa, n, dev_a)
    };
    int fail_a{0};
    if (sta != QNPEPS_ELOC_OK)
    {
        std::printf("[gate1]   A device error: %s\n", qnpeps_eloc_strerror(sta));
        fail_a = 1;
    }
    for (int lane{0}; lane < n; ++lane)
    {
        std::complex<double> h{0.0, 0.0};
        bool threw{false};
        try
        {
            h = host_logpsi(g.fx, to_ints(sa, lane, lx, ly));
        }
        catch (const std::exception& e)
        {
            std::printf("[gate1]   A lane %d host threw: %s\n", lane, e.what());
            threw = true;
            fail_a = 1;
        }
        const double d_re{dev_a[static_cast<std::size_t>(2 * lane)]};
        const double d_im{dev_a[static_cast<std::size_t>(2 * lane + 1)]};
        const double abs_re{std::abs(d_re - h.real())};
        const double denom{std::max(1.0, std::abs(h.real()))};
        const double rel_re{abs_re / denom};
        const double dim_ang{ang_dist(d_im, h.imag())};
        if (not threw)
        {
            max_d_re = std::max(max_d_re, abs_re);
            max_r_re = std::max(max_r_re, rel_re);
            max_d_im = std::max(max_d_im, dim_ang);
            if (bad_num(d_re) or bad_num(d_im) or rel_re > tol or dim_ang > tol) fail_a = 1;
        }
        std::printf(
            "[gate1]   A lane %d  host=(% .8e,% .4f)  dev=(% .8e,% .4f)  dRe=%.3e rRe=%.3e "
            "dIm=%.3e\n",
            lane,
            h.real(),
            h.imag(),
            d_re,
            d_im,
            abs_re,
            rel_re,
            dim_ang
        );
    }

    const std::vector<std::uint8_t> sb{
        gen_samples(lx, ly, dim_phys, n, 0xB0B0ull ^ peps_seed, true)
    };
    std::vector<double> dev_b{};
    const qnpeps_eloc_status stb{
        device_logpsi(g, lx, ly, dim_phys, dim_bond, chi, meo, sb, n, dev_b)
    };
    int fail_b{0};
    if (stb != QNPEPS_ELOC_OK)
    {
        std::printf("[gate1]   B device error: %s\n", qnpeps_eloc_strerror(stb));
        fail_b = 1;
    }
    bool lanes_bit_equal{true};
    for (int lane{1}; lane < n; ++lane)
    {
        const bool eq{
            dev_b[static_cast<std::size_t>(2 * lane)] == dev_b[0]
            and dev_b[static_cast<std::size_t>(2 * lane + 1)] == dev_b[1]
        };
        if (not eq) lanes_bit_equal = false;
    }
    if (not lanes_bit_equal) fail_b = 1;
    {
        std::complex<double> h0{0.0, 0.0};
        try
        {
            h0 = host_logpsi(g.fx, to_ints(sb, 0, lx, ly));
        }
        catch (const std::exception& e)
        {
            std::printf("[gate1]   B lane0 host threw: %s\n", e.what());
            fail_b = 1;
        }
        const double rel0{std::abs(dev_b[0] - h0.real()) / std::max(1.0, std::abs(h0.real()))};
        const double ang0{ang_dist(dev_b[1], h0.imag())};
        if (bad_num(dev_b[0]) or bad_num(dev_b[1]) or rel0 > tol or ang0 > tol) fail_b = 1;
        std::printf(
            "[gate1]   B identical: lane0 rRe=%.3e dIm=%.3e  cross-lane-bit-equal=%s\n",
            rel0,
            ang0,
            lanes_bit_equal ? "yes" : "no"
        );
    }

    std::vector<double> dev_c4{};
    std::vector<double> dev_c8{};
    const qnpeps_eloc_status stc4{
        device_logpsi(g, lx, ly, dim_phys, dim_bond, chi, 4, sa, n, dev_c4)
    };
    const qnpeps_eloc_status stc8{
        device_logpsi(g, lx, ly, dim_phys, dim_bond, chi, 8, sa, n, dev_c8)
    };
    int fail_c{0};
    bool wave_bit_equal{true};
    double wave_max_rel{0.0};
    if (stc4 != QNPEPS_ELOC_OK or stc8 != QNPEPS_ELOC_OK) fail_c = 1;
    for (int lane{0}; lane < n; ++lane)
    {
        const double re4{dev_c4[static_cast<std::size_t>(2 * lane)]};
        const double re8{dev_c8[static_cast<std::size_t>(2 * lane)]};
        const double im4{dev_c4[static_cast<std::size_t>(2 * lane + 1)]};
        const double im8{dev_c8[static_cast<std::size_t>(2 * lane + 1)]};
        if (re4 != re8 or im4 != im8) wave_bit_equal = false;
        const double rel{std::abs(re4 - re8) / std::max(1.0, std::abs(re8))};
        const double ang{ang_dist(im4, im8)};
        wave_max_rel = std::max(wave_max_rel, std::max(rel, ang));
    }
    if (not wave_bit_equal and wave_max_rel > 1.0e-6) fail_c = 1;
    auto wave_verdict{wave_bit_equal ? "BITEQ" : (fail_c == 0 ? "REL<=1e-6" : "DIVERGED")};
    std::printf(
        "[gate1]   C wave meo4-vs-meo8: %s  wave_max_rel=%.3e\n", wave_verdict, wave_max_rel
    );

    if (fail_a) ++fails;
    if (fail_b) ++fails;
    if (fail_c) ++fails;

    std::printf(
        "[gate1] CASE lx=%d ly=%d D=%d chi=%d n=%d meo=%d mode=%d tol=%.1e A=%s B=%s C=%s "
        "maxdRe=%.3e maxrRe=%.3e maxdIm=%.3e wave=%s => %s\n",
        lx,
        ly,
        dim_bond,
        chi,
        n,
        meo,
        mode,
        tol,
        fail_a ? "FAIL" : "PASS",
        fail_b ? "FAIL" : "PASS",
        fail_c ? "FAIL" : "PASS",
        max_d_re,
        max_r_re,
        max_d_im,
        wave_verdict,
        fails == 0 ? "PASS" : "FAIL"
    );
    return fails == 0 ? 0 : 1;
}

}

auto main(int argc, char** argv) -> int
{
    const int lx{argc > 1 ? std::atoi(argv[1]) : 2};
    const int ly{argc > 2 ? std::atoi(argv[2]) : 2};
    const int dim_bond{argc > 3 ? std::atoi(argv[3]) : 2};
    const int chi{argc > 4 ? std::atoi(argv[4]) : 4};
    const int n{argc > 5 ? std::atoi(argv[5]) : 8};
    const int meo{argc > 6 ? std::atoi(argv[6]) : 4};
    const int mode{argc > 7 ? std::atoi(argv[7]) : 2};
    const double noise{argc > 8 ? std::atof(argv[8]) : (mode == 0 ? 1.0 : 0.002)};
    const double tol{argc > 9 ? std::atof(argv[9]) : 1.0e-4};
    return run_case(lx, ly, dim_bond, chi, n, meo, mode, noise, tol);
}
