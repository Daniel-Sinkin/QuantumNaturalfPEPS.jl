#include "../../core/types.cuh"
#include "dans_qnpeps_eloc.h"
#include "qnpeps.h"
#include "term_builders.cuh"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <random>
#include <vector>

namespace
{

auto bd(int axis_len, int pos, int dim_bond) -> int
{
    if (pos <= 0 or pos >= axis_len) return 1;
    return dim_bond;
}

auto make_peps(int lx, int ly, int dim_bond, int dim_phys, double noise, std::uint64_t seed)
    -> std::vector<float>
{
    std::vector<float> flat{};
    auto rng = std::mt19937_64(seed);
    auto gauss = std::normal_distribution<double>(0.0, 1.0);
    for (int i{1}; i <= lx; ++i)
    {
        for (int j{1}; j <= ly; ++j)
        {
            const int row0{i - 1};
            const int col0{j - 1};
            const int wdim{bd(ly, col0, dim_bond)};
            const int sdim{bd(lx, row0 + 1, dim_bond)};
            const int edim{bd(ly, col0 + 1, dim_bond)};
            const int ndim{bd(lx, row0, dim_bond)};
            const std::size_t count{static_cast<std::size_t>(wdim) * sdim * edim * ndim * dim_phys};
            for (std::size_t k{0}; k < count; ++k)
            {
                const int wi{static_cast<int>(k % wdim)};
                const int si{static_cast<int>((k / wdim) % sdim)};
                const int ei{
                    static_cast<int>((k / (static_cast<std::size_t>(wdim) * sdim)) % edim)
                };
                const int ni{
                    static_cast<int>((k / (static_cast<std::size_t>(wdim) * sdim * edim)) % ndim)
                };
                const int pi{
                    static_cast<int>(k / (static_cast<std::size_t>(wdim) * sdim * edim * ndim))
                };
                double v{noise * gauss(rng)};
                const qnpeps::CuArray<double, 4> a_arr{1.0, 0.7, 1.3, 0.85};
                const double uw{1.0 + 0.35 * wi / std::max(1, wdim - 1)};
                const double vs{1.0 - 0.25 * si / std::max(1, sdim - 1)};
                const double xe{1.0 + 0.15 * ei / std::max(1, edim - 1)};
                const double yn{1.0 - 0.45 * ni / std::max(1, ndim - 1)};
                v += a_arr[pi % 4] * uw * vs * xe * yn;
                flat.push_back(static_cast<float>(v));
                flat.push_back(0.0f);
            }
        }
    }
    return flat;
}

auto site0(int i, int j, int ly) -> int
{
    return (i - 1) * ly + (j - 1);
}

auto push_bond(eo_test::TermTableBuilder& t, int ai, int aj, int bi, int bj, int ly, double j)
    -> void
{
    t.add_heisenberg_bond(site0(ai, aj, ly), site0(bi, bj, ly), j);
}

auto build_terms(int lx, int ly, double j1, double j2) -> eo_test::TermTableBuilder
{
    eo_test::TermTableBuilder t{};
    for (int i{1}; i <= lx; ++i)
    {
        for (int j{1}; j < ly; ++j)
            push_bond(t, i, j, i, j + 1, ly, j1);
    }
    for (int i{1}; i < lx; ++i)
    {
        for (int j{1}; j <= ly; ++j)
            push_bond(t, i, j, i + 1, j, ly, j1);
    }
    if (j2 != 0.0)
    {
        for (int i{1}; i < lx; ++i)
        {
            for (int j{1}; j < ly; ++j)
            {
                push_bond(t, i, j, i + 1, j + 1, ly, j2);
                push_bond(t, i + 1, j, i, j + 1, ly, j2);
            }
        }
    }
    return t;
}

auto run_case(
    const QnpepsElocConfig& cfg,
    const void* d_peps,
    const std::uint8_t* d_samp,
    int n,
    const QnpepsElocTermTable* table,
    std::vector<double>& lp,
    std::vector<double>& el
) -> qnpeps_status
{
    double* d_lp{};
    double* d_el{};
    cudaMalloc(reinterpret_cast<void**>(&d_lp), static_cast<std::size_t>(2 * n) * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_el), static_cast<std::size_t>(2 * n) * sizeof(double));
    const qnpeps_status st{qnpeps_eloc_run(
        &cfg,
        static_cast<const qnpeps_eloc_cbuf*>(d_peps),
        d_samp,
        n,
        table,
        d_lp,
        d_el,
        nullptr,
        nullptr,
        nullptr,
        0.0,
        nullptr
    )};
    cudaDeviceSynchronize();
    lp.assign(static_cast<std::size_t>(2 * n), 0.0);
    el.assign(static_cast<std::size_t>(2 * n), 0.0);
    cudaMemcpy(lp.data(), d_lp, lp.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(el.data(), d_el, el.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_lp);
    cudaFree(d_el);
    return st;
}

auto route_checks(
    int route,
    const char* route_name,
    double lp_tol,
    double el_tol,
    int L,
    int D,
    int chi,
    int n,
    double j2,
    int dim_phys,
    const void* d_peps,
    const std::uint8_t* d_samp,
    const QnpepsElocTermTable& table
) -> int
{
    int fails{0};
    std::vector<double> lp_a{};
    std::vector<double> el_a{};
    std::vector<double> lp_b{};
    std::vector<double> el_b{};

    QnpepsElocConfig cfg{};
    cfg.struct_size = sizeof(QnpepsElocConfig);
    cfg.lx = L;
    cfg.ly = L;
    cfg.dim_phys = dim_phys;
    cfg.dim_bond = D;
    cfg.chi_eo = chi;
    cfg.truncation_route = route;

    cfg.meo = 2;
    const qnpeps_status st_a{run_case(cfg, d_peps, d_samp, n, &table, lp_a, el_a)};
    std::vector<double> lp_r{};
    std::vector<double> el_r{};
    const qnpeps_status st_r{run_case(cfg, d_peps, d_samp, n, &table, lp_r, el_r)};
    cfg.meo = 3;
    const qnpeps_status st_b{run_case(cfg, d_peps, d_samp, n, &table, lp_b, el_b)};

    if (st_a != QNPEPS_OK or st_b != QNPEPS_OK or st_r != QNPEPS_OK)
    {
        std::printf("[gate_l16] %s status a=%d r=%d b=%d\n", route_name, st_a, st_r, st_b);
        ++fails;
    }
    int nf{0};
    for (int s{0}; s < 2 * n; ++s)
        if (not std::isfinite(lp_a[s]) or not std::isfinite(el_a[s])) ++nf;
    if (nf != 0)
    {
        std::printf("[gate_l16] %s NONFINITE entries: %d of %d\n", route_name, nf, 2 * n);
        ++fails;
    }
    int rep_mism{0};
    for (int s{0}; s < 2 * n; ++s)
    {
        if (std::memcmp(&lp_a[s], &lp_r[s], sizeof(double)) != 0
            or std::memcmp(&el_a[s], &el_r[s], sizeof(double)) != 0)
            ++rep_mism;
    }
    if (rep_mism != 0)
    {
        std::printf(
            "[gate_l16] %s repeat mismatch (same meo): %d of %d\n", route_name, rep_mism, 2 * n
        );
        ++fails;
    }
    double lp_rel{0.0};
    double el_rel{0.0};
    for (int s{0}; s < n; ++s)
    {
        const double dlp{std::abs(lp_a[2 * s] - lp_b[2 * s])};
        lp_rel = std::max(lp_rel, dlp / std::max(1.0, std::abs(lp_a[2 * s])));
        const double ea{std::hypot(el_a[2 * s], el_a[2 * s + 1])};
        const double de{std::hypot(el_a[2 * s] - el_b[2 * s], el_a[2 * s + 1] - el_b[2 * s + 1])};
        el_rel = std::max(el_rel, de / std::max(1.0, ea));
    }
    if (lp_rel > lp_tol or el_rel > el_tol)
    {
        std::printf(
            "[gate_l16] %s cross-meo drift: logpsi_rel=%.3e (tol %.0e) eloc_rel=%.3e (tol %.0e)\n",
            route_name,
            lp_rel,
            lp_tol,
            el_rel,
            el_tol
        );
        ++fails;
    }
    std::printf(
        "[gate_l16] %s L=%d D=%d chi=%d n=%d j2=%.2f logpsi[0]=%.6f eloc[0]=%.6f finite=%s "
        "repeat_bitwise=%s crossmeo(lp=%.2e,el=%.2e) => %s\n",
        route_name,
        L,
        D,
        chi,
        n,
        j2,
        lp_a[0],
        el_a[0],
        nf == 0 ? "yes" : "NO",
        rep_mism == 0 ? "yes" : "NO",
        lp_rel,
        el_rel,
        fails == 0 ? "PASS" : "FAIL"
    );
    return fails;
}

}

auto main(int argc, char** argv) -> int
{
    const int L{argc > 1 ? std::atoi(argv[1]) : 16};
    const int D{argc > 2 ? std::atoi(argv[2]) : 7};
    const int chi{argc > 3 ? std::atoi(argv[3]) : 16};
    const int n{argc > 4 ? std::atoi(argv[4]) : 6};
    const double j2{argc > 5 ? std::atof(argv[5]) : 0.5};
    const int dim_phys{2};

    const std::uint64_t pseed{0xC0FFEEull ^ (static_cast<std::uint64_t>(L) * 131 + L * 17 + D * 7)};
    const std::vector<float> hpeps{make_peps(L, L, D, dim_phys, 0.002, pseed)};
    auto hs = std::vector<std::uint8_t>(static_cast<std::size_t>(n) * L * L);
    auto rng = std::mt19937_64(0xA11CEull ^ pseed);
    for (auto& v : hs)
        v = static_cast<std::uint8_t>(rng() & 1);

    void* d_peps{};
    std::uint8_t* d_samp{};
    cudaMalloc(&d_peps, hpeps.size() * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_samp), hs.size());
    cudaMemcpy(d_peps, hpeps.data(), hpeps.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_samp, hs.data(), hs.size(), cudaMemcpyHostToDevice);

    const auto terms{build_terms(L, L, 1.0, j2)};
    const auto table{terms.table()};

    constexpr double k_rangefinder_logpsi_tolerance{1e-7};
    constexpr double k_rangefinder_energy_tolerance{1e-3};
    constexpr double k_default_logpsi_tolerance{1e-5};
    constexpr double k_default_energy_tolerance{1e-5};
    const int rangefinder_fails{route_checks(
        QNPEPS_TRUNCATION_RANGEFINDER,
        "rangefinder",
        k_rangefinder_logpsi_tolerance,
        k_rangefinder_energy_tolerance,
        L,
        D,
        chi,
        n,
        j2,
        dim_phys,
        d_peps,
        d_samp,
        table
    )};
    const int default_fails{route_checks(
        QNPEPS_TRUNCATION_DEFAULT,
        "default",
        k_default_logpsi_tolerance,
        k_default_energy_tolerance,
        L,
        D,
        chi,
        n,
        j2,
        dim_phys,
        d_peps,
        d_samp,
        table
    )};
    return rangefinder_fails + default_fails == 0 ? 0 : 1;
}
