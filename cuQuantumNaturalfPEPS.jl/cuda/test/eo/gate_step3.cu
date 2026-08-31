#include "boundary.hpp"
#include "dans_qnpeps_eloc.h"
#include "gate_fixture.cuh"
#include "ok.hpp"
#include "tensor.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace
{

using namespace gate;
using peps::OkLayout;
using peps::OkResult;
using peps::SampleData;
using peps::SiteBlock;

struct cf32
{
    float re{};
    float im{};
};

auto strip_boundary(GenPeps& g) -> void
{
    for (auto i = 1; i <= g.fx.lx; ++i)
    {
        for (auto j = 1; j <= g.fx.ly; ++j)
        {
            auto& s = g.fx.peps[static_cast<std::size_t>(i - 1)][static_cast<std::size_t>(j - 1)];
            const std::vector<peps::Index> in{s.tensor.inds};
            std::vector<peps::Index> keep{};
            for (auto k = 0; k < 4; ++k)
            {
                if (in[static_cast<std::size_t>(k)].dim > 1)
                    keep.push_back(in[static_cast<std::size_t>(k)]);
            }
            keep.push_back(s.phys);
            s.tensor.inds = keep;
            std::vector<peps::Index> links{};
            for (int k : {0, 3, 1, 2})
            {
                if (in[static_cast<std::size_t>(k)].dim > 1)
                    links.push_back(in[static_cast<std::size_t>(k)]);
            }
            s.links = links;
        }
    }
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
    return sd;
}

struct Dev
{
    QnpepsElocConfig cfg{};
    void* d_peps{};
    std::uint8_t* d_samp_rand{};
    std::uint8_t* d_samp_ident{};
    double* d_lp{};
    double* d_el{};
    cf32* d_rows{};
    cf32* d_T{};
    cf32* h_rows{};
    int n{};
    std::int64_t compact{};
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
    qnpeps_eloc_compact_count(&d.cfg, &d.compact);

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
    cudaMalloc(
        reinterpret_cast<void**>(&d.d_rows), static_cast<std::size_t>(n) * d.compact * sizeof(cf32)
    );
    cudaMalloc(reinterpret_cast<void**>(&d.d_T), static_cast<std::size_t>(n) * n * sizeof(cf32));
    d.h_rows =
        static_cast<cf32*>(std::malloc(static_cast<std::size_t>(n) * d.compact * sizeof(cf32)));
}

auto dev_teardown(Dev& d) -> void
{
    cudaFree(d.d_peps);
    cudaFree(d.d_samp_rand);
    cudaFree(d.d_samp_ident);
    cudaFree(d.d_lp);
    cudaFree(d.d_el);
    cudaFree(d.d_rows);
    cudaFree(d.d_T);
    std::free(d.h_rows);
}

struct DevOut
{
    std::vector<double> lp{};
    std::vector<double> el{};
    std::vector<cf32> rows{};
    std::vector<cf32> hrows{};
    std::vector<cf32> T{};
    qnpeps_eloc_status st{};
};

auto dev_run(
    Dev& d,
    const std::vector<QnpepsElocDiagBond>& diag,
    const std::vector<QnpepsElocFlipTerm>& flips,
    bool ident,
    int meo,
    bool want_o,
    bool want_host,
    bool want_gram,
    double lambda
) -> DevOut
{
    DevOut o{};
    QnpepsElocConfig cfg{d.cfg};
    cfg.meo = meo;
    QnpepsElocTermTable tt{};
    tt.n_diag = static_cast<int32_t>(diag.size());
    tt.diag = diag.empty() ? nullptr : diag.data();
    tt.n_flip = static_cast<int32_t>(flips.size());
    tt.flip = flips.empty() ? nullptr : flips.data();
    if (want_o) cudaMemset(d.d_rows, 0, static_cast<std::size_t>(d.n) * d.compact * sizeof(cf32));
    if (want_gram) cudaMemset(d.d_T, 0, static_cast<std::size_t>(d.n) * d.n * sizeof(cf32));
    if (want_host)
        std::memset(d.h_rows, 0, static_cast<std::size_t>(d.n) * d.compact * sizeof(cf32));
    o.st = qnpeps_eloc_run(
        &cfg,
        static_cast<const qnpeps_eloc_cbuf*>(d.d_peps),
        ident ? d.d_samp_ident : d.d_samp_rand,
        d.n,
        &tt,
        d.d_lp,
        d.d_el,
        want_o ? reinterpret_cast<qnpeps_eloc_cbuf*>(d.d_rows) : nullptr,
        want_host ? reinterpret_cast<qnpeps_eloc_cbuf*>(d.h_rows) : nullptr,
        want_gram ? reinterpret_cast<qnpeps_eloc_cbuf*>(d.d_T) : nullptr,
        lambda,
        nullptr
    );
    o.lp.assign(static_cast<std::size_t>(2 * d.n), 0.0);
    o.el.assign(static_cast<std::size_t>(2 * d.n), 0.0);
    cudaMemcpy(o.lp.data(), d.d_lp, o.lp.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(o.el.data(), d.d_el, o.el.size() * sizeof(double), cudaMemcpyDeviceToHost);
    if (want_o)
    {
        o.rows.assign(static_cast<std::size_t>(d.n) * d.compact, cf32{});
        cudaMemcpy(o.rows.data(), d.d_rows, o.rows.size() * sizeof(cf32), cudaMemcpyDeviceToHost);
    }
    if (want_host) o.hrows.assign(d.h_rows, d.h_rows + static_cast<std::size_t>(d.n) * d.compact);
    if (want_gram)
    {
        o.T.assign(static_cast<std::size_t>(d.n) * d.n, cf32{});
        cudaMemcpy(o.T.data(), d.d_T, o.T.size() * sizeof(cf32), cudaMemcpyDeviceToHost);
    }
    return o;
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
        "[gate3] SET %-10s maxRel=%.3e worst=%s => %s\n",
        r.name.c_str(),
        r.max_rel,
        r.worst.empty() ? "-" : r.worst.c_str(),
        r.fail ? "FAIL" : "PASS"
    );
}

auto rel_of(double a, double b) -> double
{
    return std::abs(a - b) / std::max(1.0, std::abs(b));
}

auto host_gram(
    const std::vector<cf32>& rows,
    std::int64_t compact,
    int n,
    const OkLayout& layout,
    const std::vector<std::vector<int>>& spins,
    double lambda,
    bool mask
) -> std::vector<std::complex<double>>
{
    auto t = std::vector<std::complex<double>>(static_cast<std::size_t>(n) * n, {0.0, 0.0});
    for (auto s = 0; s < n; ++s)
    {
        for (auto u = 0; u < n; ++u)
        {
            std::complex<double> acc{0.0, 0.0};
            for (auto b = static_cast<std::size_t>(0); b < layout.blocks.size(); ++b)
            {
                if (mask
                    and spins[static_cast<std::size_t>(s)][b]
                            != spins[static_cast<std::size_t>(u)][b])
                    continue;
                const auto& blk = layout.blocks[b];
                for (auto k = static_cast<std::size_t>(0); k < blk.slice_size; ++k)
                {
                    const cf32 rs{
                        rows[static_cast<std::size_t>(s) * compact + blk.compact_offset + k]
                    };
                    const cf32 ru{
                        rows[static_cast<std::size_t>(u) * compact + blk.compact_offset + k]
                    };
                    acc += std::conj(std::complex<double>(rs.re, rs.im))
                           * std::complex<double>(ru.re, ru.im);
                }
            }
            if (s == u) acc += std::complex<double>(lambda, 0.0);
            t[static_cast<std::size_t>(s) * n + u] = acc;
        }
    }
    return t;
}

auto run_case(
    int lx,
    int ly,
    int dim_bond,
    int chi,
    int n,
    int meo,
    double noise,
    double tol_row,
    double tol_gram,
    double lambda_pos
) -> int
{
    const int dim_phys{2};
    const std::uint64_t peps_seed{
        0xC0FFEEull ^ (static_cast<std::uint64_t>(lx) * 131 + ly * 17 + dim_bond * 7)
    };
    GenPeps g{make_peps(lx, ly, dim_bond, dim_phys, chi, 2, noise, peps_seed)};
    strip_boundary(g);

    const std::vector<std::uint8_t> sa{
        gen_samples(lx, ly, dim_phys, n, 0xA11CEull ^ peps_seed, false)
    };
    const std::vector<std::uint8_t> sb{
        gen_samples(lx, ly, dim_phys, n, 0xB0B0ull ^ peps_seed, true)
    };

    Dev d{};
    dev_setup(d, g, lx, ly, dim_phys, dim_bond, chi, meo, sa, sb, n);
    std::printf(
        "[gate3] fixture lx=%d ly=%d D=%d chi=%d n=%d meo=%d compact_count=%lld\n",
        lx,
        ly,
        dim_bond,
        chi,
        n,
        meo,
        static_cast<long long>(d.compact)
    );

    const OkLayout layout{peps::make_ok_layout(g.fx)};
    std::int64_t layout_compact{static_cast<std::int64_t>(layout.compact_count)};

    std::vector<OkResult> ok_ref{};
    std::vector<std::vector<int>> spins{};
    bool oracle_ok{true};
    for (auto lane = 0; lane < n; ++lane)
    {
        const std::vector<int> sample{to_ints(sa, lane, lx, ly)};
        spins.push_back(sample);
        try
        {
            const SampleData sd{build_oracle_lane(g.fx, sample)};
            ok_ref.push_back(peps::compute_ok(g.fx, sd, layout));
        }
        catch (const std::exception& e)
        {
            std::printf("[gate3] oracle lane %d threw: %s\n", lane, e.what());
            oracle_ok = false;
            ok_ref.push_back(OkResult{});
        }
    }
    if (not oracle_ok or layout_compact != d.compact)
    {
        std::printf(
            "[gate3] CASE lx=%d ly=%d D=%d chi=%d => FAIL (oracle=%d compact dev=%lld ref=%lld)\n",
            lx,
            ly,
            dim_bond,
            chi,
            oracle_ok ? 1 : 0,
            static_cast<long long>(d.compact),
            static_cast<long long>(layout_compact)
        );
        dev_teardown(d);
        return 1;
    }

    const DevOut base{dev_run(d, {}, {}, false, meo, true, true, true, 0.0)};
    std::vector<SetResult> results{};

    if (base.st != QNPEPS_ELOC_OK)
    {
        std::printf("[gate3] device error: %s\n", qnpeps_eloc_strerror(base.st));
        dev_teardown(d);
        return 1;
    }
    {
        SetResult c0{};
        c0.name = "C_rows_c0";
        SetResult cg{};
        cg.name = "C_rows_cgt0";
        for (auto b = static_cast<std::size_t>(0); b < layout.blocks.size(); ++b)
        {
            const auto& blk = layout.blocks[b];
            const int col{blk.j - 1};
            auto& tgt{(col == 0) ? c0 : cg};
            for (auto lane = 0; lane < n; ++lane)
            {
                for (auto k = static_cast<std::size_t>(0); k < blk.slice_size; ++k)
                {
                    const std::complex<double> ref{
                        ok_ref[static_cast<std::size_t>(lane)].compact[blk.compact_offset + k]
                    };
                    const cf32 dev{
                        base.rows
                            [static_cast<std::size_t>(lane) * d.compact + blk.compact_offset + k]
                    };
                    const double rr{
                        std::sqrt(
                            (dev.re - ref.real()) * (dev.re - ref.real())
                            + (dev.im - ref.imag()) * (dev.im - ref.imag())
                        )
                        / std::max(1.0, std::abs(ref))
                    };
                    if (rr > tgt.max_rel)
                    {
                        tgt.max_rel = rr;
                        char w[48]{};
                        std::snprintf(w, sizeof(w), "site(%d,%d)k%zu@l%d", blk.i, blk.j, k, lane);
                        tgt.worst = w;
                    }
                    if (bad_num(dev.re) or bad_num(dev.im) or rr > tol_row) tgt.fail = 1;
                }
            }
        }
        results.push_back(c0);
        results.push_back(cg);
    }

    for (double lam : {0.0, lambda_pos})
    {
        const DevOut go{dev_run(d, {}, {}, false, meo, true, false, true, lam)};
        const std::vector<peps::cdouble> tref{peps::direct_sector_gram(
            [&]
            {
                auto cr = std::vector<std::vector<peps::cdouble>>(static_cast<std::size_t>(n));
                for (auto s = 0; s < n; ++s)
                {
                    cr[static_cast<std::size_t>(s)].assign(
                        static_cast<std::size_t>(layout.compact_count), peps::cdouble{0.0}
                    );
                    for (auto kk = static_cast<std::size_t>(0); kk < layout.compact_count; ++kk)
                    {
                        cr[static_cast<std::size_t>(s)][kk] =
                            ok_ref[static_cast<std::size_t>(s)].compact[kk];
                    }
                }
                return cr;
            }(),
            spins,
            layout,
            lam
        )};
        SetResult r{};
        r.name = (lam == 0.0) ? "D_gram_l0" : "D_gram_lpos";
        double max_im{0.0};
        for (auto s = 0; s < n; ++s)
        {
            for (auto u = 0; u < n; ++u)
            {
                const cf32 dv{go.T[static_cast<std::size_t>(s) * n + u]};
                const double ref{static_cast<double>(tref[static_cast<std::size_t>(s) * n + u])};
                const double rr{rel_of(dv.re, ref)};
                max_im = std::max(max_im, std::abs(static_cast<double>(dv.im)));
                if (rr > r.max_rel)
                {
                    r.max_rel = rr;
                    char w[32]{};
                    std::snprintf(w, sizeof(w), "T(%d,%d)", s, u);
                    r.worst = w;
                }
                if (bad_num(dv.re) or bad_num(dv.im) or rr > tol_gram) r.fail = 1;
            }
        }
        if (max_im > tol_gram) r.fail = 1;
        std::printf("[gate3]   %s max|Im|=%.3e\n", r.name.c_str(), max_im);
        results.push_back(r);
    }

    {
        const std::int64_t sites{static_cast<std::int64_t>(lx) * ly};
        const DevOut tref{dev_run(d, {}, {}, false, meo, true, false, true, 0.0)};
        cf32* d_tile{};
        cudaMalloc(
            reinterpret_cast<void**>(&d_tile), static_cast<std::size_t>(n) * n * sizeof(cf32)
        );

        const auto call_tile = [&](std::int64_t ra0,
                                   std::int64_t la,
                                   std::int64_t rb0,
                                   std::int64_t lb,
                                   std::vector<cf32>& out) -> qnpeps_eloc_status
        {
            const qnpeps_eloc_status st{qnpeps_eloc_gram_tile(
                &d.cfg,
                reinterpret_cast<const qnpeps_eloc_cbuf*>(d.d_rows + ra0 * d.compact),
                d.d_samp_rand + ra0 * sites,
                la,
                reinterpret_cast<const qnpeps_eloc_cbuf*>(d.d_rows + rb0 * d.compact),
                d.d_samp_rand + rb0 * sites,
                lb,
                reinterpret_cast<qnpeps_eloc_cbuf*>(d_tile),
                nullptr
            )};
            out.assign(static_cast<std::size_t>(la) * lb, cf32{});
            cudaMemcpy(out.data(), d_tile, out.size() * sizeof(cf32), cudaMemcpyDeviceToHost);
            return st;
        };

        std::vector<std::pair<std::string, std::vector<std::int64_t>>> schemes{};
        schemes.push_back({"Tile_whole", {n}});
        schemes.push_back({"Tile_halves", {n / 2, n - n / 2}});
        if (n >= 3)
        {
            const std::int64_t a{std::max<std::int64_t>(1, n / 5)};
            const std::int64_t b{std::max<std::int64_t>(1, n / 3)};
            const std::int64_t c{n - a - b};
            if (c >= 1) schemes.push_back({"Tile_thirds", {a, b, c}});
        }
        for (const auto& sch : schemes)
        {
            SetResult r{};
            r.name = sch.first;
            auto asm_t = std::vector<cf32>(static_cast<std::size_t>(n) * n, cf32{});
            std::int64_t r0{0};
            for (std::int64_t len : sch.second)
            {
                std::vector<cf32> tl{};
                const qnpeps_eloc_status st{call_tile(r0, len, 0, n, tl)};
                if (st != QNPEPS_ELOC_OK)
                {
                    r.fail = 1;
                    r.worst = qnpeps_eloc_strerror(st);
                }
                for (auto i = static_cast<std::int64_t>(0); i < len; ++i)
                {
                    for (auto u = static_cast<std::int64_t>(0); u < n; ++u)
                    {
                        asm_t[static_cast<std::size_t>(r0 + i) * n + u] =
                            tl[static_cast<std::size_t>(i) * n + u];
                    }
                }
                r0 += len;
            }
            for (auto k = static_cast<std::size_t>(0); k < asm_t.size() and not r.fail; ++k)
            {
                if (asm_t[k].re != tref.T[k].re or asm_t[k].im != tref.T[k].im)
                {
                    r.fail = 1;
                    r.max_rel = 1.0;
                    char w[32]{};
                    std::snprintf(w, sizeof(w), "idx%zu", k);
                    r.worst = w;
                }
            }
            if (not r.fail) r.worst = "BITEQ";
            results.push_back(r);
        }

        {
            SetResult r{};
            r.name = "Tile_diag";
            const std::int64_t r0{n / 2};
            const std::int64_t len{n - r0};
            std::vector<cf32> tl{};
            const qnpeps_eloc_status st{call_tile(r0, len, r0, len, tl)};
            if (st != QNPEPS_ELOC_OK)
            {
                r.fail = 1;
                r.worst = qnpeps_eloc_strerror(st);
            }
            for (auto i = static_cast<std::int64_t>(0); i < len and not r.fail; ++i)
            {
                for (auto j = static_cast<std::int64_t>(0); j < len; ++j)
                {
                    const cf32 dv{tl[static_cast<std::size_t>(i) * len + j]};
                    const cf32 rf{tref.T[static_cast<std::size_t>(r0 + i) * n + (r0 + j)]};
                    if (dv.re != rf.re or dv.im != rf.im)
                    {
                        r.fail = 1;
                        r.max_rel = 1.0;
                        r.worst = "DIVERGED";
                        break;
                    }
                }
            }
            if (not r.fail) r.worst = "BITEQ";
            results.push_back(r);
        }

        {
            SetResult r{};
            r.name = "Tile_rect";
            const std::int64_t la{std::max<std::int64_t>(1, n / 3)};
            const std::int64_t lb{n - la};
            std::vector<cf32> tl{};
            const qnpeps_eloc_status st{call_tile(0, la, la, lb, tl)};
            if (st != QNPEPS_ELOC_OK)
            {
                r.fail = 1;
                r.worst = qnpeps_eloc_strerror(st);
            }
            if (la == lb)
            {
                r.fail = 1;
                r.worst = "NOT-RECTANGULAR";
            }
            for (auto i = static_cast<std::int64_t>(0); i < la and not r.fail; ++i)
            {
                for (auto j = static_cast<std::int64_t>(0); j < lb; ++j)
                {
                    const cf32 dv{tl[static_cast<std::size_t>(i) * lb + j]};
                    const cf32 rf{tref.T[static_cast<std::size_t>(i) * n + (la + j)]};
                    if (dv.re != rf.re or dv.im != rf.im)
                    {
                        r.fail = 1;
                        r.max_rel = 1.0;
                        r.worst = "DIVERGED";
                        break;
                    }
                }
            }
            if (not r.fail) r.worst = "BITEQ";
            results.push_back(r);
        }

        cudaFree(d_tile);
    }

    {
        SetResult r{};
        r.name = "S_stream";
        for (auto k = static_cast<std::size_t>(0); k < base.rows.size(); ++k)
        {
            if (base.rows[k].re != base.hrows[k].re or base.rows[k].im != base.hrows[k].im)
            {
                r.fail = 1;
                r.max_rel = 1.0;
                char w[32]{};
                std::snprintf(w, sizeof(w), "idx%zu", k);
                r.worst = w;
                break;
            }
        }
        results.push_back(r);
    }

    {
        SetResult r{};
        r.name = "R_energy";
        const DevOut en{dev_run(d, {}, {}, false, meo, false, false, false, 0.0)};
        for (auto k = 0; k < 2 * n; ++k)
        {
            if (en.lp[static_cast<std::size_t>(k)] != base.lp[static_cast<std::size_t>(k)]
                or en.el[static_cast<std::size_t>(k)] != base.el[static_cast<std::size_t>(k)])
            {
                r.fail = 1;
                r.max_rel = 1.0;
                char w[24]{};
                std::snprintf(w, sizeof(w), "k%d", k);
                r.worst = w;
                break;
            }
        }
        for (auto lane = 0; lane < n and not r.fail; ++lane)
        {
            const double refre{static_cast<double>(peps::compute_logpsi(
                peps::build_env_top(g.fx, to_ints(sa, lane, lx, ly)),
                peps::build_env_down(g.fx, to_ints(sa, lane, lx, ly)),
                lx
            ))};
            const double rr{rel_of(en.lp[static_cast<std::size_t>(2 * lane)], refre)};
            r.max_rel = std::max(r.max_rel, rr);
            if (rr > 3.0e-4) r.fail = 1;
        }
        results.push_back(r);
    }

    {
        SetResult r{};
        r.name = "Det";
        const DevOut a{dev_run(d, {}, {}, false, meo, true, false, true, 0.0)};
        const DevOut b{dev_run(d, {}, {}, false, meo, true, false, true, 0.0)};
        for (auto k = static_cast<std::size_t>(0); k < a.rows.size(); ++k)
            if (a.rows[k].re != b.rows[k].re or a.rows[k].im != b.rows[k].im) r.fail = 1;
        for (auto k = static_cast<std::size_t>(0); k < a.T.size(); ++k)
            if (a.T[k].re != b.T[k].re or a.T[k].im != b.T[k].im) r.fail = 1;
        results.push_back(r);
    }

    {
        SetResult r{};
        r.name = "Batch";
        const DevOut a{dev_run(d, {}, {}, true, meo, true, false, true, 0.0)};
        for (auto lane = 1; lane < n and not r.fail; ++lane)
        {
            for (auto k = static_cast<std::int64_t>(0); k < d.compact; ++k)
            {
                const cf32 v0{a.rows[static_cast<std::size_t>(k)]};
                const cf32 vl{a.rows[static_cast<std::size_t>(lane) * d.compact + k]};
                if (v0.re != vl.re or v0.im != vl.im)
                {
                    r.fail = 1;
                    break;
                }
            }
        }
        results.push_back(r);
    }

    {
        SetResult r{};
        r.name = "Wave";
        const DevOut a{dev_run(d, {}, {}, false, 4, true, false, true, 0.0)};
        const DevOut b{dev_run(d, {}, {}, false, 8, true, false, true, 0.0)};
        bool biteq{true};
        for (auto k = static_cast<std::size_t>(0); k < a.rows.size(); ++k)
            if (a.rows[k].re != b.rows[k].re or a.rows[k].im != b.rows[k].im) biteq = false;
        for (auto k = static_cast<std::size_t>(0); k < a.T.size(); ++k)
            if (a.T[k].re != b.T[k].re or a.T[k].im != b.T[k].im) biteq = false;
        r.fail = biteq ? 0 : 1;
        r.worst = biteq ? "BITEQ" : "DIVERGED";
        results.push_back(r);
    }

    {
        SetResult r{};
        r.name = "F_sector";
        const std::vector<std::complex<double>> masked{
            host_gram(base.rows, d.compact, n, layout, spins, 0.0, true)
        };
        const std::vector<std::complex<double>> full{
            host_gram(base.rows, d.compact, n, layout, spins, 0.0, false)
        };
        int n_diff_pairs{0};
        double masking_gap{0.0};
        for (auto s = 0; s < n; ++s)
        {
            for (auto u = 0; u < n; ++u)
            {
                const cf32 dv{base.T[static_cast<std::size_t>(s) * n + u]};
                const double rr{rel_of(dv.re, masked[static_cast<std::size_t>(s) * n + u].real())};
                r.max_rel = std::max(r.max_rel, rr);
                if (rr > tol_gram) r.fail = 1;
                const double gap{std::abs(
                    masked[static_cast<std::size_t>(s) * n + u]
                    - full[static_cast<std::size_t>(s) * n + u]
                )};
                if (gap > 1.0e-6)
                {
                    ++n_diff_pairs;
                    masking_gap = std::max(masking_gap, gap);
                }
            }
        }
        if (n_diff_pairs == 0)
        {
            r.fail = 1;
            r.worst = "NO-MASKING-EXERCISED";
        }
        std::printf(
            "[gate3]   F_sector masked_pairs=%d masking_gap=%.3e\n", n_diff_pairs, masking_gap
        );
        results.push_back(r);
    }

    dev_teardown(d);

    int fails{0};
    double max_row{0.0};
    double max_gram{0.0};
    for (const SetResult& r : results)
    {
        set_line(r);
        if (r.fail) ++fails;
        if (r.name.rfind("C_rows", 0) == 0) max_row = std::max(max_row, r.max_rel);
        if (r.name.rfind("D_gram", 0) == 0) max_gram = std::max(max_gram, r.max_rel);
    }
    std::printf(
        "[gate3] CASE lx=%d ly=%d D=%d chi=%d n=%d meo=%d tolRow=%.1e tolGram=%.1e "
        "sets_failed=%d maxRow=%.3e maxGram=%.3e => %s\n",
        lx,
        ly,
        dim_bond,
        chi,
        n,
        meo,
        tol_row,
        tol_gram,
        fails,
        max_row,
        max_gram,
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
    const double noise{argc > 7 ? std::atof(argv[7]) : 0.002};
    const double tol_row{argc > 8 ? std::atof(argv[8]) : 1.0e-4};
    const double tol_gram{argc > 9 ? std::atof(argv[9]) : 5.0e-4};
    const double lambda_pos{argc > 10 ? std::atof(argv[10]) : 0.1};
    return run_case(lx, ly, dim_bond, chi, n, meo, noise, tol_row, tol_gram, lambda_pos);
}
