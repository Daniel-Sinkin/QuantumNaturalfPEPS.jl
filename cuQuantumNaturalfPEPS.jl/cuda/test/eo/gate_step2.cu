#include "boundary.hpp"
#include "dans_qnpeps_eloc.h"
#include "eloc.hpp"
#include "gate_fixture.cuh"
#include "hamiltonian.hpp"
#include "mps.hpp"
#include "tensor.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using namespace gate;
using peps::ElocResult;
using peps::ElocTerm;
using peps::Flip;
using peps::SampleData;

struct Bond
{
    int ai{};
    int aj{};
    int bi{};
    int bj{};
    double j{};
};

auto bonds_horizontal(int lx, int ly, double j1) -> std::vector<Bond>
{
    std::vector<Bond> out{};
    for (auto i = 1; i <= lx; ++i)
    {
        for (auto j = 1; j < ly; ++j)
            out.push_back({i, j, i, j + 1, j1});
    }
    return out;
}

auto bonds_vertical(int lx, int ly, double j1) -> std::vector<Bond>
{
    std::vector<Bond> out{};
    for (auto i = 1; i < lx; ++i)
    {
        for (auto j = 1; j <= ly; ++j)
            out.push_back({i, j, i + 1, j, j1});
    }
    return out;
}

auto bonds_diagonal(int lx, int ly, double j2) -> std::vector<Bond>
{
    std::vector<Bond> out{};
    for (auto i = 1; i < lx; ++i)
    {
        for (auto j = 1; j < ly; ++j)
        {
            out.push_back({i, j, i + 1, j + 1, j2});
            out.push_back({i + 1, j, i, j + 1, j2});
        }
    }
    return out;
}

auto concat(const std::vector<Bond>& a, const std::vector<Bond>& b) -> std::vector<Bond>
{
    std::vector<Bond> out{a};
    out.insert(out.end(), b.begin(), b.end());
    return out;
}

auto site0(int i, int j, int ly) -> int
{
    return (i - 1) * ly + (j - 1);
}

auto flip_term(const Bond& b, int ly, double coeff, bool masked) -> QnpepsElocFlipTerm
{
    QnpepsElocFlipTerm t{};
    t.n_flips = 2;
    t.flip_site[0] = site0(b.ai, b.aj, ly);
    t.flip_site[1] = site0(b.bi, b.bj, ly);
    t.flip_value[0] = -1;
    t.flip_value[1] = -1;
    t.mask_a = masked ? t.flip_site[0] : -1;
    t.mask_b = masked ? t.flip_site[1] : -1;
    t.coeff_re = coeff;
    t.coeff_im = 0.0;
    return t;
}

auto diag_bonds_of(const std::vector<Bond>& bonds, int ly) -> std::vector<QnpepsElocDiagBond>
{
    std::vector<QnpepsElocDiagBond> out{};
    for (const Bond& b : bonds)
        out.push_back({site0(b.ai, b.aj, ly), site0(b.bi, b.bj, ly), b.j});
    return out;
}

auto masked_flips_of(const std::vector<Bond>& bonds, int ly) -> std::vector<QnpepsElocFlipTerm>
{
    std::vector<QnpepsElocFlipTerm> out{};
    for (const Bond& b : bonds)
        out.push_back(flip_term(b, ly, 2.0 * b.j, true));
    return out;
}

auto oracle_terms(const std::vector<Bond>& bonds, const std::vector<int>& sample, int ly)
    -> std::vector<ElocTerm>
{
    peps::cdouble diagonal{0.0};
    std::vector<ElocTerm> off{};
    int next_id{1};
    for (const Bond& b : bonds)
    {
        const int sa{sample[static_cast<std::size_t>(site0(b.ai, b.aj, ly))]};
        const int sb{sample[static_cast<std::size_t>(site0(b.bi, b.bj, ly))]};
        diagonal += peps::cdouble{b.j * ((sa == sb) ? 1.0 : -1.0)};
        if (sa != sb)
        {
            ElocTerm t{};
            t.term_id = next_id++;
            t.coefficient = peps::cdouble{2.0 * b.j};
            t.flips = {Flip{b.ai, b.aj, 1 - sa}, Flip{b.bi, b.bj, 1 - sb}};
            off.push_back(std::move(t));
        }
    }
    std::vector<ElocTerm> terms{};
    ElocTerm d{};
    d.term_id = 0;
    d.coefficient = diagonal;
    terms.push_back(std::move(d));
    for (ElocTerm& t : off)
        terms.push_back(std::move(t));
    return terms;
}

struct Dev
{
    QnpepsElocConfig cfg{};
    void* d_peps{};
    std::uint8_t* d_samp_rand{};
    std::uint8_t* d_samp_ident{};
    double* d_lp{};
    double* d_el{};
    int n{};
};

auto dev_setup(
    Dev& d,
    const GenPeps& g,
    int lx,
    int ly,
    int dim_phys,
    int dim_bond,
    int chi,
    int meo,
    const std::vector<std::uint8_t>& samp_rand,
    const std::vector<std::uint8_t>& samp_ident,
    int n
) -> void
{
    d.cfg.struct_size = sizeof(QnpepsElocConfig);
    d.cfg.lx = lx;
    d.cfg.ly = ly;
    d.cfg.dim_phys = dim_phys;
    d.cfg.dim_bond = dim_bond;
    d.cfg.chi_eo = chi;
    d.cfg.meo = meo;
    d.n = n;

    const std::size_t total{g.flat.size()};
    auto hpeps = std::vector<float>(2 * total);
    for (std::size_t k{0}; k < total; ++k)
    {
        hpeps[2 * k] = static_cast<float>(g.flat[k]);
        hpeps[2 * k + 1] = 0.0f;
    }
    cudaMalloc(&d.d_peps, hpeps.size() * sizeof(float));
    cudaMemcpy(d.d_peps, hpeps.data(), hpeps.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(reinterpret_cast<void**>(&d.d_samp_rand), samp_rand.size());
    cudaMemcpy(d.d_samp_rand, samp_rand.data(), samp_rand.size(), cudaMemcpyHostToDevice);
    cudaMalloc(reinterpret_cast<void**>(&d.d_samp_ident), samp_ident.size());
    cudaMemcpy(d.d_samp_ident, samp_ident.data(), samp_ident.size(), cudaMemcpyHostToDevice);
    cudaMalloc(reinterpret_cast<void**>(&d.d_lp), static_cast<std::size_t>(2 * n) * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d.d_el), static_cast<std::size_t>(2 * n) * sizeof(double));
}

auto dev_teardown(Dev& d) -> void
{
    cudaFree(d.d_peps);
    cudaFree(d.d_samp_rand);
    cudaFree(d.d_samp_ident);
    cudaFree(d.d_lp);
    cudaFree(d.d_el);
}

auto dev_run(
    Dev& d,
    const std::vector<QnpepsElocDiagBond>& diag,
    const std::vector<QnpepsElocFlipTerm>& flips,
    bool ident,
    int meo,
    std::vector<double>& lp,
    std::vector<double>& el
) -> qnpeps_eloc_status
{
    QnpepsElocConfig cfg{d.cfg};
    cfg.meo = meo;
    QnpepsElocTermTable tt{};
    tt.n_diag = static_cast<int32_t>(diag.size());
    tt.diag = diag.empty() ? nullptr : diag.data();
    tt.n_flip = static_cast<int32_t>(flips.size());
    tt.flip = flips.empty() ? nullptr : flips.data();
    const qnpeps_eloc_status st{qnpeps_eloc_run(
        &cfg,
        static_cast<const qnpeps_eloc_cbuf*>(d.d_peps),
        ident ? d.d_samp_ident : d.d_samp_rand,
        d.n,
        &tt,
        d.d_lp,
        d.d_el,
        nullptr,
        nullptr,
        nullptr,
        0.0,
        nullptr
    )};
    lp.assign(static_cast<std::size_t>(2 * d.n), 0.0);
    el.assign(static_cast<std::size_t>(2 * d.n), 0.0);
    cudaMemcpy(
        lp.data(),
        d.d_lp,
        static_cast<std::size_t>(2 * d.n) * sizeof(double),
        cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        el.data(),
        d.d_el,
        static_cast<std::size_t>(2 * d.n) * sizeof(double),
        cudaMemcpyDeviceToHost
    );
    return st;
}

auto build_oracle_lane(const Fixture& fx, const std::vector<int>& sample) -> SampleData
{
    SampleData sd{};
    sd.sample_row_major = sample;
    sd.env_top = peps::build_env_top(fx, sample);
    sd.env_down = peps::build_env_down(fx, sample);
    sd.logpsi = peps::compute_logpsi(sd.env_top, sd.env_down, fx.lx);
    const peps::HorizontalEnvs h{peps::build_horizontal_envs(fx, sample, sd.env_top, sd.env_down)};
    sd.h_env_r = h.h_r;
    sd.h_env_l = h.h_l;
    const peps::FourBodyEnvs f4{peps::build_fourbody_envs(fx, sample, sd.env_top, sd.env_down)};
    sd.fourb_env_r = f4.fb_r;
    sd.fourb_env_l = f4.fb_l;
    return sd;
}

struct SetResult
{
    std::string name{};
    int fail{};
    double max_rel{};
    std::string worst{};
};

auto set_line(const SetResult& r) -> void
{
    std::printf(
        "[gate2] SET %-6s maxRel=%.3e worst=%s => %s\n",
        r.name.c_str(),
        r.max_rel,
        r.worst.empty() ? "-" : r.worst.c_str(),
        r.fail ? "FAIL" : "PASS"
    );
}

auto perterm_set(
    Dev& d,
    const Fixture& fx,
    const std::vector<SampleData>& lanes,
    const std::vector<std::uint8_t>& samp,
    const std::vector<Bond>& bonds,
    const std::string& name,
    double tol
) -> SetResult
{
    SetResult r{};
    r.name = name;
    const int ly{fx.ly};
    const int n{d.n};
    const std::size_t sites{static_cast<std::size_t>(fx.lx) * ly};
    for (const Bond& b : bonds)
    {
        std::vector<QnpepsElocFlipTerm> one{flip_term(b, ly, 1.0, false)};
        std::vector<double> lp{};
        std::vector<double> el{};
        const qnpeps_eloc_status st{dev_run(d, {}, one, false, d.cfg.meo, lp, el)};
        if (st != QNPEPS_ELOC_OK)
        {
            std::printf(
                "[gate2]   %s device error bond=(%d,%d)-(%d,%d): %s\n",
                name.c_str(),
                b.ai,
                b.aj,
                b.bi,
                b.bj,
                qnpeps_eloc_strerror(st)
            );
            r.fail = 1;
            continue;
        }
        for (int lane{0}; lane < n; ++lane)
        {
            const int sa{samp[static_cast<std::size_t>(lane) * sites + site0(b.ai, b.aj, ly)]};
            const int sb{samp[static_cast<std::size_t>(lane) * sites + site0(b.bi, b.bj, ly)]};
            const std::vector<Flip> flips{Flip{b.ai, b.aj, 1 - sa}, Flip{b.bi, b.bj, 1 - sb}};
            double lf_host{0.0};
            bool threw{false};
            try
            {
                lf_host = static_cast<double>(peps::compute_logpsi_flipped(fx, lanes[lane], flips));
            }
            catch (const std::exception& e)
            {
                std::printf(
                    "[gate2]   %s ORACLE THREW bond=(%d,%d)-(%d,%d) lane=%d: %s\n",
                    name.c_str(),
                    b.ai,
                    b.aj,
                    b.bi,
                    b.bj,
                    lane,
                    e.what()
                );
                threw = true;
                r.fail = 1;
            }
            const double rr{el[static_cast<std::size_t>(2 * lane)]};
            const double ri{el[static_cast<std::size_t>(2 * lane + 1)]};
            const double lpf_re{0.5 * std::log(rr * rr + ri * ri) + lp[2 * lane]};
            const double lpf_im{std::atan2(ri, rr) + lp[2 * lane + 1]};
            if (threw) continue;
            if (bad_num(lf_host))
            {
                std::printf(
                    "[gate2]   %s ORACLE NAN bond=(%d,%d)-(%d,%d) lane=%d\n",
                    name.c_str(),
                    b.ai,
                    b.aj,
                    b.bi,
                    b.bj,
                    lane
                );
                r.fail = 1;
                continue;
            }
            const double rel{std::abs(lpf_re - lf_host) / std::max(1.0, std::abs(lf_host))};
            const double dim{ang_dist(lpf_im, 0.0)};
            if (rel > r.max_rel)
            {
                r.max_rel = rel;
                char w[64]{};
                std::snprintf(w, sizeof(w), "(%d,%d)-(%d,%d)@lane%d", b.ai, b.aj, b.bi, b.bj, lane);
                r.worst = w;
            }
            if (bad_num(lpf_re) or bad_num(lpf_im) or rel > tol or dim > tol)
            {
                r.fail = 1;
                std::printf(
                    "[gate2]   %s MISMATCH flips=(%d,%d)+(%d,%d) lane=%d host_lpf=% .8e "
                    "dev_lpf=% .8e rRe=%.3e dIm=%.3e\n",
                    name.c_str(),
                    b.ai,
                    b.aj,
                    b.bi,
                    b.bj,
                    lane,
                    lf_host,
                    lpf_re,
                    rel,
                    dim
                );
            }
        }
    }
    return r;
}

auto eloc_set(
    Dev& d,
    const Fixture& fx,
    const std::vector<SampleData>& lanes,
    const std::vector<Bond>& bonds,
    const std::string& name,
    double tol,
    bool use_expand,
    std::vector<ElocTerm> (*expand)(const Fixture&, const std::vector<int>&)
) -> SetResult
{
    SetResult r{};
    r.name = name;
    const int ly{fx.ly};
    const std::vector<QnpepsElocDiagBond> diag{diag_bonds_of(bonds, ly)};
    const std::vector<QnpepsElocFlipTerm> flips{masked_flips_of(bonds, ly)};
    std::vector<double> lp{};
    std::vector<double> el{};
    const qnpeps_eloc_status st{dev_run(d, diag, flips, false, d.cfg.meo, lp, el)};
    if (st != QNPEPS_ELOC_OK)
    {
        std::printf("[gate2]   %s device error: %s\n", name.c_str(), qnpeps_eloc_strerror(st));
        r.fail = 1;
        return r;
    }
    for (int lane{0}; lane < d.n; ++lane)
    {
        SampleData sd{lanes[static_cast<std::size_t>(lane)]};
        sd.terms = use_expand ? expand(fx, sd.sample_row_major)
                              : oracle_terms(bonds, sd.sample_row_major, ly);
        double e_host{0.0};
        bool threw{false};
        try
        {
            const ElocResult res{peps::compute_eloc(fx, sd)};
            e_host = static_cast<double>(res.e_loc);
        }
        catch (const std::exception& e)
        {
            std::printf("[gate2]   %s ORACLE THREW lane=%d: %s\n", name.c_str(), lane, e.what());
            threw = true;
            r.fail = 1;
        }
        if (threw) continue;
        const double er{el[static_cast<std::size_t>(2 * lane)]};
        const double ei{el[static_cast<std::size_t>(2 * lane + 1)]};
        const double scale{std::max(1.0, std::abs(e_host))};
        const double rel{std::abs(er - e_host) / scale};
        const double aim{std::abs(ei) / scale};
        if (rel > r.max_rel)
        {
            r.max_rel = rel;
            char w[32]{};
            std::snprintf(w, sizeof(w), "lane%d", lane);
            r.worst = w;
        }
        if (bad_num(er) or bad_num(ei) or bad_num(e_host) or rel > tol or aim > 1.0e-5)
        {
            r.fail = 1;
            std::printf(
                "[gate2]   %s MISMATCH lane=%d host=% .8e dev=(% .8e,% .3e) rRe=%.3e "
                "relIm=%.3e\n",
                name.c_str(),
                lane,
                e_host,
                er,
                ei,
                rel,
                aim
            );
        }
    }
    return r;
}

auto mask_set(
    Dev& d,
    const Fixture& fx,
    const std::vector<SampleData>& lanes,
    const std::vector<std::uint8_t>& samp,
    const std::vector<Bond>& bonds,
    const std::string& name,
    double tol,
    double j1
) -> SetResult
{
    SetResult r{};
    r.name = name;
    const int ly{fx.ly};
    const std::size_t sites{static_cast<std::size_t>(fx.lx) * ly};
    const Bond* pick{};
    int n_aligned{0};
    int n_anti{0};
    for (const Bond& b : bonds)
    {
        int al{0};
        int an{0};
        for (int lane{0}; lane < d.n; ++lane)
        {
            const int sa{samp[static_cast<std::size_t>(lane) * sites + site0(b.ai, b.aj, ly)]};
            const int sb{samp[static_cast<std::size_t>(lane) * sites + site0(b.bi, b.bj, ly)]};
            if (sa == sb)
                ++al;
            else
                ++an;
        }
        if (al > 0 and an > 0)
        {
            pick = &b;
            n_aligned = al;
            n_anti = an;
            break;
        }
    }
    if (not pick)
    {
        pick = &bonds.front();
        std::printf("[gate2]   %s no mixed-alignment bond in sample set\n", name.c_str());
    }
    const Bond b{*pick};
    std::printf(
        "[gate2]   %s bond=(%d,%d)-(%d,%d) aligned=%d anti=%d\n",
        name.c_str(),
        b.ai,
        b.aj,
        b.bi,
        b.bj,
        n_aligned,
        n_anti
    );
    std::vector<QnpepsElocFlipTerm> one{flip_term(b, ly, 2.0 * j1, true)};
    std::vector<double> lp{};
    std::vector<double> el{};
    const qnpeps_eloc_status st{dev_run(d, {}, one, false, d.cfg.meo, lp, el)};
    if (st != QNPEPS_ELOC_OK)
    {
        std::printf("[gate2]   %s device error: %s\n", name.c_str(), qnpeps_eloc_strerror(st));
        r.fail = 1;
        return r;
    }
    for (int lane{0}; lane < d.n; ++lane)
    {
        const int sa{samp[static_cast<std::size_t>(lane) * sites + site0(b.ai, b.aj, ly)]};
        const int sb{samp[static_cast<std::size_t>(lane) * sites + site0(b.bi, b.bj, ly)]};
        const double er{el[static_cast<std::size_t>(2 * lane)]};
        const double ei{el[static_cast<std::size_t>(2 * lane + 1)]};
        if (sa == sb)
        {
            if (er != 0.0 or ei != 0.0)
            {
                r.fail = 1;
                r.max_rel = std::max(r.max_rel, std::abs(er) + std::abs(ei));
                std::printf(
                    "[gate2]   %s ALIGNED NONZERO lane=%d dev=(% .8e,% .3e)\n",
                    name.c_str(),
                    lane,
                    er,
                    ei
                );
            }
        }
        else
        {
            const std::vector<Flip> flips{Flip{b.ai, b.aj, 1 - sa}, Flip{b.bi, b.bj, 1 - sb}};
            double lf_host{0.0};
            try
            {
                lf_host = static_cast<double>(peps::compute_logpsi_flipped(fx, lanes[lane], flips));
            }
            catch (const std::exception& e)
            {
                std::printf(
                    "[gate2]   %s ORACLE THREW lane=%d: %s\n", name.c_str(), lane, e.what()
                );
                r.fail = 1;
                continue;
            }
            const double lpd{lanes[static_cast<std::size_t>(lane)].logpsi};
            const double e_host{2.0 * j1 * std::exp(lf_host - lpd)};
            const double rel{std::abs(er - e_host) / std::max(1.0, std::abs(e_host))};
            r.max_rel = std::max(r.max_rel, rel);
            if (bad_num(er) or bad_num(ei) or rel > tol)
            {
                r.fail = 1;
                std::printf(
                    "[gate2]   %s ANTI MISMATCH lane=%d host=% .8e dev=% .8e rRe=%.3e\n",
                    name.c_str(),
                    lane,
                    e_host,
                    er,
                    rel
                );
            }
        }
    }
    return r;
}

auto run_case(
    int lx,
    int ly,
    int dim_bond,
    int chi,
    int n,
    int meo,
    int mode,
    double noise,
    double tol_term,
    double tol_eloc,
    double j2
) -> int
{
    const int dim_phys{2};
    const double j1{1.0};
    const std::uint64_t peps_seed{
        0xC0FFEEull ^ (static_cast<std::uint64_t>(lx) * 131 + ly * 17 + dim_bond * 7)
    };
    GenPeps g{make_peps(lx, ly, dim_bond, dim_phys, chi, mode, noise, peps_seed)};
    g.fx.j1 = j1;
    g.fx.j2 = j2;

    double flat_min{g.flat.empty() ? 0.0 : g.flat[0]};
    double flat_max{flat_min};
    for (const double v : g.flat)
    {
        flat_min = std::min(flat_min, v);
        flat_max = std::max(flat_max, v);
    }
    std::printf(
        "[gate2] fixture mode=%d noise=%g real=yes flat_min=%.4f flat_max=%.4f\n",
        mode,
        noise,
        flat_min,
        flat_max
    );

    const std::vector<std::uint8_t> sa{
        gen_samples(lx, ly, dim_phys, n, 0xA11CEull ^ peps_seed, false)
    };
    const std::vector<std::uint8_t> sb{
        gen_samples(lx, ly, dim_phys, n, 0xB0B0ull ^ peps_seed, true)
    };

    Dev d{};
    dev_setup(d, g, lx, ly, dim_phys, dim_bond, chi, meo, sa, sb, n);

    std::vector<SampleData> lanes{};
    bool oracle_ok{true};
    for (int lane{0}; lane < n; ++lane)
    {
        try
        {
            lanes.push_back(build_oracle_lane(g.fx, to_ints(sa, lane, lx, ly)));
        }
        catch (const std::exception& e)
        {
            std::printf("[gate2] oracle lane %d env build threw: %s\n", lane, e.what());
            oracle_ok = false;
            lanes.push_back(SampleData{});
        }
        if (oracle_ok and bad_num(static_cast<double>(lanes.back().logpsi)))
        {
            std::printf("[gate2] oracle lane %d logpsi non-finite\n", lane);
            oracle_ok = false;
        }
    }

    const std::vector<Bond> bh{bonds_horizontal(lx, ly, j1)};
    const std::vector<Bond> bv{bonds_vertical(lx, ly, j1)};
    const std::vector<Bond> bd2{bonds_diagonal(lx, ly, j2)};
    const std::vector<Bond> bfull{concat(bh, bv)};
    const std::vector<Bond> bj1j2{concat(bfull, bd2)};

    const int lh_row{(lx + 1) / 2};
    const std::vector<Bond> blh{{lh_row, 1, lh_row, 3, j1}};

    std::vector<SetResult> results{};
    if (not oracle_ok)
    {
        std::printf(
            "[gate2] CASE lx=%d ly=%d D=%d chi=%d => FAIL (oracle env build)\n",
            lx,
            ly,
            dim_bond,
            chi
        );
        dev_teardown(d);
        return 1;
    }

    results.push_back(perterm_set(d, g.fx, lanes, sa, bh, "A_h", tol_term));
    results.push_back(perterm_set(d, g.fx, lanes, sa, bv, "A_v", tol_term));
    results.push_back(perterm_set(d, g.fx, lanes, sa, bd2, "A_d", tol_term));
    results.push_back(perterm_set(d, g.fx, lanes, sa, blh, "A_lh", tol_term));

    results.push_back(eloc_set(d, g.fx, lanes, bh, "B_h", tol_eloc, false, nullptr));
    results.push_back(eloc_set(d, g.fx, lanes, bv, "B_v", tol_eloc, false, nullptr));
    results.push_back(eloc_set(d, g.fx, lanes, bfull, "B_full", tol_eloc, false, nullptr));
    results.push_back(
        eloc_set(d, g.fx, lanes, bj1j2, "B_j1j2", tol_eloc, true, &peps::expand_j1j2)
    );

    results.push_back(mask_set(d, g.fx, lanes, sa, bh, "C_h", tol_term, j1));
    results.push_back(mask_set(d, g.fx, lanes, sa, bv, "C_v", tol_term, j1));

    {
        SetResult r{};
        r.name = "D_batch";
        const std::vector<QnpepsElocDiagBond> diag{diag_bonds_of(bj1j2, ly)};
        const std::vector<QnpepsElocFlipTerm> flips{masked_flips_of(bj1j2, ly)};
        std::vector<double> lp{};
        std::vector<double> el{};
        const qnpeps_eloc_status st{dev_run(d, diag, flips, true, meo, lp, el)};
        if (st != QNPEPS_ELOC_OK)
        {
            std::printf("[gate2]   D_batch device error: %s\n", qnpeps_eloc_strerror(st));
            r.fail = 1;
        }
        for (int lane{1}; lane < n and not r.fail; ++lane)
        {
            const bool eq{
                el[static_cast<std::size_t>(2 * lane)] == el[0]
                and el[static_cast<std::size_t>(2 * lane + 1)] == el[1]
                and lp[static_cast<std::size_t>(2 * lane)] == lp[0]
                and lp[static_cast<std::size_t>(2 * lane + 1)] == lp[1]
            };
            if (not eq)
            {
                r.fail = 1;
                char w[32]{};
                std::snprintf(w, sizeof(w), "lane%d", lane);
                r.worst = w;
            }
        }
        results.push_back(r);
    }

    {
        SetResult r{};
        r.name = "E_wave";
        const std::vector<QnpepsElocDiagBond> diag{diag_bonds_of(bj1j2, ly)};
        const std::vector<QnpepsElocFlipTerm> flips{masked_flips_of(bj1j2, ly)};
        std::vector<double> lp4{};
        std::vector<double> el4{};
        std::vector<double> lp8{};
        std::vector<double> el8{};
        const qnpeps_eloc_status st4{dev_run(d, diag, flips, false, 4, lp4, el4)};
        const qnpeps_eloc_status st8{dev_run(d, diag, flips, false, 8, lp8, el8)};
        bool bit_equal{true};
        if (st4 != QNPEPS_ELOC_OK or st8 != QNPEPS_ELOC_OK)
        {
            std::printf(
                "[gate2]   E_wave device error: %s / %s\n",
                qnpeps_eloc_strerror(st4),
                qnpeps_eloc_strerror(st8)
            );
            r.fail = 1;
        }
        for (int lane{0}; lane < n; ++lane)
        {
            const double r4{el4[static_cast<std::size_t>(2 * lane)]};
            const double r8{el8[static_cast<std::size_t>(2 * lane)]};
            const double i4{el4[static_cast<std::size_t>(2 * lane + 1)]};
            const double i8{el8[static_cast<std::size_t>(2 * lane + 1)]};
            if (r4 != r8 or i4 != i8) bit_equal = false;
            const double rel{(std::abs(r4 - r8) + std::abs(i4 - i8)) / std::max(1.0, std::abs(r8))};
            r.max_rel = std::max(r.max_rel, rel);
        }
        if (not bit_equal and r.max_rel > 1.0e-6) r.fail = 1;
        r.worst = bit_equal ? "BITEQ" : (r.fail ? "DIVERGED" : "REL<=1e-6");
        results.push_back(r);
    }

    dev_teardown(d);

    int fails{0};
    double max_term{0.0};
    double max_eloc{0.0};
    for (const SetResult& r : results)
    {
        set_line(r);
        if (r.fail) ++fails;
        if (r.name[0] == 'A' or r.name[0] == 'C') max_term = std::max(max_term, r.max_rel);
        if (r.name[0] == 'B') max_eloc = std::max(max_eloc, r.max_rel);
    }

    std::printf(
        "[gate2] CASE lx=%d ly=%d D=%d chi=%d n=%d meo=%d mode=%d j2=%.2f tolT=%.1e tolE=%.1e "
        "sets_failed=%d maxTerm=%.3e maxEloc=%.3e => %s\n",
        lx,
        ly,
        dim_bond,
        chi,
        n,
        meo,
        mode,
        j2,
        tol_term,
        tol_eloc,
        fails,
        max_term,
        max_eloc,
        fails == 0 ? "PASS" : "FAIL"
    );
    return fails == 0 ? 0 : 1;
}

}

auto main(int argc, char** argv) -> int
{
    const int lx{argc > 1 ? std::atoi(argv[1]) : 4};
    const int ly{argc > 2 ? std::atoi(argv[2]) : 4};
    const int dim_bond{argc > 3 ? std::atoi(argv[3]) : 2};
    const int chi{argc > 4 ? std::atoi(argv[4]) : 8};
    const int n{argc > 5 ? std::atoi(argv[5]) : 8};
    const int meo{argc > 6 ? std::atoi(argv[6]) : 4};
    const int mode{argc > 7 ? std::atoi(argv[7]) : 2};
    const double noise{argc > 8 ? std::atof(argv[8]) : (mode == 0 ? 1.0 : 0.002)};
    const double tol_term{argc > 9 ? std::atof(argv[9]) : 1.0e-4};
    const double tol_eloc{argc > 10 ? std::atof(argv[10]) : 5.0e-4};
    const double j2{argc > 11 ? std::atof(argv[11]) : 0.5};
    return run_case(lx, ly, dim_bond, chi, n, meo, mode, noise, tol_term, tol_eloc, j2);
}
