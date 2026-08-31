#include "../../core/types.cuh"
#include "dans_qnpeps_eloc.h"

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

struct Terms
{
    std::vector<QnpepsElocDiagBond> diag{};
    std::vector<QnpepsElocFlipTerm> flip{};
    QnpepsElocTermTable table{};
};

auto push_bond(Terms& t, int ai, int aj, int bi, int bj, int ly, double j) -> void
{
    t.diag.push_back({site0(ai, aj, ly), site0(bi, bj, ly), j});
    QnpepsElocFlipTerm f{};
    f.n_flips = 2;
    f.flip_site[0] = site0(ai, aj, ly);
    f.flip_site[1] = site0(bi, bj, ly);
    f.flip_value[0] = -1;
    f.flip_value[1] = -1;
    f.mask_a = f.flip_site[0];
    f.mask_b = f.flip_site[1];
    f.coeff_re = 2.0 * j;
    f.coeff_im = 0.0;
    t.flip.push_back(f);
}

auto build_terms(int lx, int ly, double j1, double j2) -> Terms
{
    Terms t{};
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
    t.table.n_diag = static_cast<int32_t>(t.diag.size());
    t.table.diag = t.diag.data();
    t.table.n_flip = static_cast<int32_t>(t.flip.size());
    t.table.flip = t.flip.data();
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
) -> qnpeps_eloc_status
{
    double* d_lp{};
    double* d_el{};
    cudaMalloc(reinterpret_cast<void**>(&d_lp), static_cast<std::size_t>(2 * n) * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_el), static_cast<std::size_t>(2 * n) * sizeof(double));
    const qnpeps_eloc_status st{qnpeps_eloc_run(
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

    Terms terms{build_terms(L, L, 1.0, j2)};

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

    cfg.meo = 2;
    const qnpeps_eloc_status st_a{run_case(cfg, d_peps, d_samp, n, &terms.table, lp_a, el_a)};
    std::vector<double> lp_r{};
    std::vector<double> el_r{};
    const qnpeps_eloc_status st_r{run_case(cfg, d_peps, d_samp, n, &terms.table, lp_r, el_r)};
    cfg.meo = 3;
    const qnpeps_eloc_status st_b{run_case(cfg, d_peps, d_samp, n, &terms.table, lp_b, el_b)};

    if (st_a != QNPEPS_ELOC_OK or st_b != QNPEPS_ELOC_OK or st_r != QNPEPS_ELOC_OK)
    {
        std::printf("[gate_l16] status a=%d r=%d b=%d\n", st_a, st_r, st_b);
        ++fails;
    }
    int nf{0};
    for (int s{0}; s < 2 * n; ++s)
        if (not std::isfinite(lp_a[s]) or not std::isfinite(el_a[s])) ++nf;
    if (nf != 0)
    {
        std::printf("[gate_l16] NONFINITE entries: %d of %d\n", nf, 2 * n);
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
        std::printf("[gate_l16] repeat mismatch (same meo): %d of %d\n", rep_mism, 2 * n);
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
    const double lp_tol{1e-7};
    const double el_tol{1e-3};
    if (lp_rel > lp_tol or el_rel > el_tol)
    {
        std::printf(
            "[gate_l16] cross-meo drift: logpsi_rel=%.3e (tol %.0e) eloc_rel=%.3e (tol %.0e)\n",
            lp_rel,
            lp_tol,
            el_rel,
            el_tol
        );
        ++fails;
    }
    std::printf(
        "[gate_l16] L=%d D=%d chi=%d n=%d j2=%.2f logpsi[0]=%.6f eloc[0]=%.6f finite=%s "
        "repeat_bitwise=%s crossmeo(lp=%.2e,el=%.2e) => %s\n",
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
    return fails == 0 ? 0 : 1;
}
