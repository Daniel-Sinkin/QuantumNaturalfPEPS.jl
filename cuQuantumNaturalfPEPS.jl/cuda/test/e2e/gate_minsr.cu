#include "../../core/types.cuh"
#include "capi/qnpeps.h"
#include "dans_qnpeps_e2e.h"
#include "dans_qnpeps_eloc.h"
#include "gate_fixture.cuh"
#include "peps_minsr_host.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <random>
#include <string>
#include <vector>

namespace
{

using gate::bad_num;
using gate::bd;
using gate::gen_samples;
using gate::GenPeps;
using gate::make_peps;
using gate::to_ints;

using cdb = std::complex<double>;

struct cf32
{
    float re{};
    float im{};
};

struct Bond
{
    int ai{};
    int aj{};
    int bi{};
    int bj{};
    double j{};
};

auto site0(int i, int j, int ly) -> int
{
    return (i - 1) * ly + (j - 1);
}

auto bonds_nn(int lx, int ly, double j1) -> std::vector<Bond>
{
    std::vector<Bond> out{};
    for (auto i = 1; i <= lx; ++i)
    {
        for (auto j = 1; j < ly; ++j)
            out.push_back({i, j, i, j + 1, j1});
    }
    for (auto i = 1; i < lx; ++i)
    {
        for (auto j = 1; j <= ly; ++j)
            out.push_back({i, j, i + 1, j, j1});
    }
    return out;
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
    {
        QnpepsElocFlipTerm t{};
        t.n_flips = 2;
        t.flip_site[0] = site0(b.ai, b.aj, ly);
        t.flip_site[1] = site0(b.bi, b.bj, ly);
        t.flip_value[0] = -1;
        t.flip_value[1] = -1;
        t.mask_a = t.flip_site[0];
        t.mask_b = t.flip_site[1];
        t.coeff_re = 2.0 * b.j;
        t.coeff_im = 0.0;
        out.push_back(t);
    }
    return out;
}

struct Blk
{
    int i{};
    int c{};
    long slice{};
    long compact_off{};
    long dense_off{};
};

auto build_layout(int lx, int ly, int dim_bond, int dim_phys, long& compact, long& dense)
    -> std::vector<Blk>
{
    std::vector<Blk> blocks{};
    long cpos{0};
    long dpos{0};
    for (auto i = 0; i < lx; ++i)
    {
        for (auto c = 0; c < ly; ++c)
        {
            const long slice{
                static_cast<long>(bd(ly, c, dim_bond)) * bd(lx, i + 1, dim_bond)
                * bd(ly, c + 1, dim_bond) * bd(lx, i, dim_bond)
            };
            blocks.push_back(Blk{i, c, slice, cpos, dpos});
            cpos += slice;
            dpos += static_cast<long>(dim_phys) * slice;
        }
    }
    compact = cpos;
    dense = dpos;
    return blocks;
}

auto ck(cudaError_t e, const char* what) -> void
{
    if (e != cudaSuccess) std::printf("[gate_e2e] CUDA %s: %s\n", what, cudaGetErrorString(e));
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
        "[gate_e2e] SET %-14s maxRel=%.3e worst=%s => %s\n",
        r.name.c_str(),
        r.max_rel,
        r.worst.empty() ? "-" : r.worst.c_str(),
        r.fail ? "FAIL" : "PASS"
    );
}

auto hash_bytes(const void* data, std::size_t bytes) -> std::uint64_t
{
    const auto* p{static_cast<const std::uint8_t*>(data)};
    std::uint64_t h{1469598103934665603ull};
    for (std::size_t k{0}; k < bytes; ++k)
    {
        h ^= p[k];
        h *= 1099511628211ull;
    }
    return h;
}

auto different_words(const void* a, const void* b, std::size_t bytes, std::size_t word_bytes)
    -> std::size_t
{
    const auto* pa{static_cast<const std::uint8_t*>(a)};
    const auto* pb{static_cast<const std::uint8_t*>(b)};
    std::size_t different{};
    for (std::size_t offset{}; offset < bytes; offset += word_bytes)
        if (std::memcmp(pa + offset, pb + offset, word_bytes) != 0) ++different;
    return different;
}

auto run_case(
    int lx,
    int ly,
    int dim_bond,
    int chi,
    int n,
    int meo,
    double noise,
    double tol_td,
    double rel_cut,
    double abs_cut
) -> int
{
    const int dim_phys{2};
    const std::uint64_t peps_seed{
        0xE2Eull ^ (static_cast<std::uint64_t>(lx) * 131 + ly * 17 + dim_bond * 7 + n * 3)
    };
    GenPeps g{make_peps(lx, ly, dim_bond, dim_phys, chi, 2, noise, peps_seed)};

    const std::vector<std::uint8_t> samp{
        gen_samples(lx, ly, dim_phys, n, 0xA11CEull ^ peps_seed, false)
    };

    long compact{0};
    long dense{0};
    const std::vector<Blk> blocks{build_layout(lx, ly, dim_bond, dim_phys, compact, dense)};

    QnpepsE2eConfig ecfg{};
    ecfg.struct_size = sizeof(QnpepsE2eConfig);
    ecfg.lx = lx;
    ecfg.ly = ly;
    ecfg.dim_phys = dim_phys;
    ecfg.dim_bond = dim_bond;
    ecfg.chi_s = chi;
    ecfg.chi_dl = chi;
    ecfg.chi_eo = chi;
    ecfg.meo = meo;
    ecfg.seed = peps_seed;

    std::int64_t e2e_compact{0};
    std::int64_t e2e_dense{0};
    qnpeps_e2e_compact_count(&ecfg, &e2e_compact);
    qnpeps_e2e_dense_count(&ecfg, &e2e_dense);

    QnpepsElocConfig lcfg{};
    lcfg.struct_size = sizeof(QnpepsElocConfig);
    lcfg.lx = lx;
    lcfg.ly = ly;
    lcfg.dim_phys = dim_phys;
    lcfg.dim_bond = dim_bond;
    lcfg.chi_eo = chi;
    lcfg.meo = meo;

    std::int64_t eloc_compact{0};
    qnpeps_eloc_compact_count(&lcfg, &eloc_compact);

    std::printf(
        "[gate_e2e] fixture lx=%d ly=%d D=%d chi=%d n=%d meo=%d compact(e2e=%lld eloc=%lld "
        "layout=%ld) dense(e2e=%lld layout=%ld)\n",
        lx,
        ly,
        dim_bond,
        chi,
        n,
        meo,
        static_cast<long long>(e2e_compact),
        static_cast<long long>(eloc_compact),
        compact,
        static_cast<long long>(e2e_dense),
        dense
    );
    if (e2e_compact != eloc_compact or e2e_compact != compact or e2e_dense != dense)
    {
        std::printf("[gate_e2e] CASE => FAIL (layout mismatch)\n");
        return 1;
    }

    const std::size_t total{g.flat.size()};
    auto hpeps = std::vector<float>(2 * total);
    auto irng = std::mt19937_64(peps_seed ^ 0x5D5D5D5Dull);
    auto igauss = std::normal_distribution<double>(0.0, 1.0);
    for (std::size_t k{0}; k < total; ++k)
    {
        hpeps[2 * k] = static_cast<float>(g.flat[k]);
        hpeps[2 * k + 1] = static_cast<float>(0.35 * igauss(irng));
    }

    void* d_peps{};
    std::uint8_t* d_samp{};
    double* d_lp{};
    double* d_el{};
    double* d_logq{};
    cf32* d_rows{};
    cf32* d_gram{};
    cf32* d_theta{};
    ck(cudaMalloc(&d_peps, hpeps.size() * sizeof(float)), "malloc peps");
    ck(cudaMemcpy(d_peps, hpeps.data(), hpeps.size() * sizeof(float), cudaMemcpyHostToDevice),
       "cpy peps");
    ck(cudaMalloc(reinterpret_cast<void**>(&d_samp), samp.size()), "malloc samp");
    ck(cudaMemcpy(d_samp, samp.data(), samp.size(), cudaMemcpyHostToDevice), "cpy samp");
    ck(cudaMalloc(
           reinterpret_cast<void**>(&d_lp), static_cast<std::size_t>(2 * n) * sizeof(double)
       ),
       "malloc lp");
    ck(cudaMalloc(
           reinterpret_cast<void**>(&d_el), static_cast<std::size_t>(2 * n) * sizeof(double)
       ),
       "malloc el");
    ck(cudaMalloc(reinterpret_cast<void**>(&d_logq), static_cast<std::size_t>(n) * sizeof(double)),
       "malloc logq");
    ck(cudaMalloc(
           reinterpret_cast<void**>(&d_rows), static_cast<std::size_t>(n) * compact * sizeof(cf32)
       ),
       "malloc rows");
    ck(cudaMalloc(
           reinterpret_cast<void**>(&d_gram), static_cast<std::size_t>(n) * n * sizeof(cf32)
       ),
       "malloc gram");
    ck(cudaMalloc(
           reinterpret_cast<void**>(&d_theta), static_cast<std::size_t>(dense) * sizeof(cf32)
       ),
       "malloc theta");
    auto h_rows{
        static_cast<cf32*>(std::malloc(static_cast<std::size_t>(n) * compact * sizeof(cf32)))
    };

    const std::vector<Bond> bonds{bonds_nn(lx, ly, 1.0)};
    const std::vector<QnpepsElocDiagBond> diag{diag_bonds_of(bonds, ly)};
    const std::vector<QnpepsElocFlipTerm> flips{masked_flips_of(bonds, ly)};
    QnpepsElocTermTable tt{};
    tt.n_diag = static_cast<int32_t>(diag.size());
    tt.diag = diag.data();
    tt.n_flip = static_cast<int32_t>(flips.size());
    tt.flip = flips.data();

    const qnpeps_eloc_status est{qnpeps_eloc_run(
        &lcfg,
        static_cast<const qnpeps_eloc_cbuf*>(d_peps),
        d_samp,
        n,
        &tt,
        d_lp,
        d_el,
        reinterpret_cast<qnpeps_eloc_cbuf*>(d_rows),
        reinterpret_cast<qnpeps_eloc_cbuf*>(h_rows),
        reinterpret_cast<qnpeps_eloc_cbuf*>(d_gram),
        0.0,
        nullptr
    )};
    if (est != QNPEPS_ELOC_OK)
    {
        std::printf("[gate_e2e] eloc error: %s\n", qnpeps_eloc_strerror(est));
        return 1;
    }

    auto lp = std::vector<double>(static_cast<std::size_t>(2 * n));
    auto el = std::vector<double>(static_cast<std::size_t>(2 * n));
    auto rows = std::vector<cf32>(static_cast<std::size_t>(n) * compact);
    auto gram = std::vector<cf32>(static_cast<std::size_t>(n) * n);
    ck(cudaMemcpy(lp.data(), d_lp, lp.size() * sizeof(double), cudaMemcpyDeviceToHost), "d2h lp");
    ck(cudaMemcpy(el.data(), d_el, el.size() * sizeof(double), cudaMemcpyDeviceToHost), "d2h el");
    ck(cudaMemcpy(rows.data(), d_rows, rows.size() * sizeof(cf32), cudaMemcpyDeviceToHost),
       "d2h rows");
    ck(cudaMemcpy(gram.data(), d_gram, gram.size() * sizeof(cf32), cudaMemcpyDeviceToHost),
       "d2h gram");

    auto logq = std::vector<double>(static_cast<std::size_t>(n));
    auto orng = std::mt19937_64(peps_seed ^ 0x0FF5E7ull);
    auto ogauss = std::normal_distribution<double>(0.0, 0.75);
    for (auto j = 0; j < n; ++j)
    {
        logq[static_cast<std::size_t>(j)] =
            2.0 * lp[static_cast<std::size_t>(2 * j)] + ogauss(orng);
    }
    ck(cudaMemcpy(d_logq, logq.data(), logq.size() * sizeof(double), cudaMemcpyHostToDevice),
       "cpy logq");

    const long long tile_tiny{4ll * compact * static_cast<long long>(sizeof(cf32))};
    qnpeps::CuArray<double, 2> e_mean_d{};
    double e_var_d{};
    double ess_d{};
    const qnpeps_e2e_status s1{qnpeps_e2e_minsr(
        &ecfg,
        n,
        d_samp,
        d_lp,
        d_el,
        d_logq,
        reinterpret_cast<const qnpeps_e2e_cbuf*>(d_gram),
        reinterpret_cast<const qnpeps_e2e_cbuf*>(d_rows),
        nullptr,
        0,
        rel_cut,
        abs_cut,
        reinterpret_cast<qnpeps_e2e_cbuf*>(d_theta),
        e_mean_d.data(),
        &e_var_d,
        &ess_d,
        nullptr
    )};
    if (s1 != QNPEPS_E2E_OK)
    {
        std::printf("[gate_e2e] e2e device-rows error: %s\n", qnpeps_e2e_strerror(s1));
        return 1;
    }
    auto theta_dev = std::vector<cf32>(static_cast<std::size_t>(dense));
    ck(cudaMemcpy(
           theta_dev.data(), d_theta, theta_dev.size() * sizeof(cf32), cudaMemcpyDeviceToHost
       ),
       "d2h theta dev");
    const qnpeps::CuArray<double, 4> stats_dev{e_mean_d[0], e_mean_d[1], e_var_d, ess_d};
    std::printf(
        "[gate_e2e] HASH one_shot theta=%016llx stats=%016llx gram=%016llx rows=%016llx\n",
        static_cast<unsigned long long>(
            hash_bytes(theta_dev.data(), theta_dev.size() * sizeof(cf32))
        ),
        static_cast<unsigned long long>(hash_bytes(stats_dev.data(), sizeof(stats_dev))),
        static_cast<unsigned long long>(hash_bytes(gram.data(), gram.size() * sizeof(cf32))),
        static_cast<unsigned long long>(hash_bytes(rows.data(), rows.size() * sizeof(cf32)))
    );
    if (hash_bytes(theta_dev.data(), theta_dev.size() * sizeof(cf32)) != 0xd78cf952e625ac39ull
        or hash_bytes(stats_dev.data(), sizeof(stats_dev)) != 0x90142b51ef2df83cull)
        return 1;

    QnpepsMinsrDesc root_desc{};
    root_desc.struct_size = sizeof(QnpepsMinsrDesc);
    root_desc.lx = lx;
    root_desc.ly = ly;
    root_desc.dim_phys = dim_phys;
    root_desc.dim_bond = dim_bond;
    root_desc.n_samples = n;
    QnpepsMinsrArgs root_args{};
    root_args.struct_size = sizeof(QnpepsMinsrArgs);
    root_args.samples = d_samp;
    root_args.samples_bytes = samp.size();
    root_args.logpsi = d_lp;
    root_args.logpsi_bytes = static_cast<std::uint64_t>(2 * n) * sizeof(double);
    root_args.e_loc = d_el;
    root_args.e_loc_bytes = static_cast<std::uint64_t>(2 * n) * sizeof(double);
    root_args.logq = d_logq;
    root_args.logq_bytes = static_cast<std::uint64_t>(n) * sizeof(double);
    root_args.gram = d_gram;
    root_args.gram_bytes = static_cast<std::uint64_t>(n) * n * sizeof(cf32);
    root_args.o_rows_device = d_rows;
    root_args.o_rows_bytes = static_cast<std::uint64_t>(n) * compact * sizeof(cf32);
    root_args.theta_dot_out = d_theta;
    root_args.theta_dot_out_bytes = static_cast<std::uint64_t>(dense) * sizeof(cf32);
    root_args.relative_cut = rel_cut;
    root_args.absolute_cut = abs_cut;
    qnpeps::CuArray<double, 2> root_e_mean{};
    double root_e_var{};
    double root_ess{};
    root_args.e_mean_out = root_e_mean.data();
    root_args.e_var_out = &root_e_var;
    root_args.ess_out = &root_ess;
    ck(cudaMemset(d_theta, 0, static_cast<std::size_t>(dense) * sizeof(cf32)), "memset theta");
    const qnpeps_status root_status{qnpeps_minsr(&root_desc, &root_args)};
    if (root_status != QNPEPS_OK)
    {
        std::printf("[gate_e2e] root device-rows error: %s\n", qnpeps_strerror(root_status));
        return 1;
    }
    auto theta_root = std::vector<cf32>(static_cast<std::size_t>(dense));
    ck(cudaMemcpy(
           theta_root.data(), d_theta, theta_root.size() * sizeof(cf32), cudaMemcpyDeviceToHost
       ),
       "d2h theta root");
    const qnpeps::CuArray<double, 4> root_stats{
        root_e_mean[0], root_e_mean[1], root_e_var, root_ess
    };
    std::printf(
        "[gate_e2e] HASH root_same_input theta=%016llx stats=%016llx\n",
        static_cast<unsigned long long>(
            hash_bytes(theta_root.data(), theta_root.size() * sizeof(cf32))
        ),
        static_cast<unsigned long long>(hash_bytes(root_stats.data(), sizeof(root_stats)))
    );
    const std::size_t root_theta_words{different_words(
        theta_root.data(), theta_dev.data(), theta_root.size() * sizeof(cf32), sizeof(float)
    )};
    const std::size_t root_stats_words{
        different_words(root_stats.data(), stats_dev.data(), sizeof(root_stats), sizeof(double))
    };
    std::printf(
        "[gate_e2e] root_vs_e2e theta_words=%zu stats_words=%zu\n",
        root_theta_words,
        root_stats_words
    );
    if (root_theta_words != 0 or root_stats_words != 0) return 1;

    qnpeps::CuArray<double, 2> e_mean_h{};
    double e_var_h{};
    double ess_h{};
    ck(cudaMemset(d_theta, 0, static_cast<std::size_t>(dense) * sizeof(cf32)), "memset theta");
    const qnpeps_e2e_status s2{qnpeps_e2e_minsr(
        &ecfg,
        n,
        d_samp,
        d_lp,
        d_el,
        d_logq,
        reinterpret_cast<const qnpeps_e2e_cbuf*>(d_gram),
        nullptr,
        reinterpret_cast<const qnpeps_e2e_cbuf*>(h_rows),
        tile_tiny,
        rel_cut,
        abs_cut,
        reinterpret_cast<qnpeps_e2e_cbuf*>(d_theta),
        e_mean_h.data(),
        &e_var_h,
        &ess_h,
        nullptr
    )};
    if (s2 != QNPEPS_E2E_OK)
    {
        std::printf("[gate_e2e] e2e host-rows error: %s\n", qnpeps_e2e_strerror(s2));
        return 1;
    }
    auto theta_host = std::vector<cf32>(static_cast<std::size_t>(dense));
    ck(cudaMemcpy(
           theta_host.data(), d_theta, theta_host.size() * sizeof(cf32), cudaMemcpyDeviceToHost
       ),
       "d2h theta host");

    qnpeps::CuArray<double, 2> e_mean_h0{};
    double e_var_h0{};
    double ess_h0{};
    ck(cudaMemset(d_theta, 0, static_cast<std::size_t>(dense) * sizeof(cf32)), "memset theta");
    const qnpeps_e2e_status s2b{qnpeps_e2e_minsr(
        &ecfg,
        n,
        d_samp,
        d_lp,
        d_el,
        d_logq,
        reinterpret_cast<const qnpeps_e2e_cbuf*>(d_gram),
        nullptr,
        reinterpret_cast<const qnpeps_e2e_cbuf*>(h_rows),
        0,
        rel_cut,
        abs_cut,
        reinterpret_cast<qnpeps_e2e_cbuf*>(d_theta),
        e_mean_h0.data(),
        &e_var_h0,
        &ess_h0,
        nullptr
    )};
    if (s2b != QNPEPS_E2E_OK)
    {
        std::printf("[gate_e2e] e2e host-rows(default) error: %s\n", qnpeps_e2e_strerror(s2b));
        return 1;
    }
    auto theta_host0 = std::vector<cf32>(static_cast<std::size_t>(dense));
    ck(cudaMemcpy(
           theta_host0.data(), d_theta, theta_host0.size() * sizeof(cf32), cudaMemcpyDeviceToHost
       ),
       "d2h theta host0");

    qnpeps::CuArray<double, 2> e_mean_r{};
    double e_var_r{};
    double ess_r{};
    ck(cudaMemset(d_theta, 0, static_cast<std::size_t>(dense) * sizeof(cf32)), "memset theta");
    qnpeps_e2e_minsr(
        &ecfg,
        n,
        d_samp,
        d_lp,
        d_el,
        d_logq,
        reinterpret_cast<const qnpeps_e2e_cbuf*>(d_gram),
        reinterpret_cast<const qnpeps_e2e_cbuf*>(d_rows),
        nullptr,
        0,
        rel_cut,
        abs_cut,
        reinterpret_cast<qnpeps_e2e_cbuf*>(d_theta),
        e_mean_r.data(),
        &e_var_r,
        &ess_r,
        nullptr
    );
    auto theta_rep = std::vector<cf32>(static_cast<std::size_t>(dense));
    ck(cudaMemcpy(
           theta_rep.data(), d_theta, theta_rep.size() * sizeof(cf32), cudaMemcpyDeviceToHost
       ),
       "d2h theta rep");

    auto theta_ctx_first = std::vector<cf32>(static_cast<std::size_t>(dense));
    auto theta_ctx_changed = std::vector<cf32>(static_cast<std::size_t>(dense));
    auto theta_one_changed = std::vector<cf32>(static_cast<std::size_t>(dense));
    auto theta_ctx_rebuilt = std::vector<cf32>(static_cast<std::size_t>(dense));
    qnpeps::CuArray<double, 4> stats_ctx_first{};
    qnpeps::CuArray<double, 4> stats_ctx_changed{};
    qnpeps::CuArray<double, 4> stats_one_changed{};
    qnpeps::CuArray<double, 4> stats_ctx_rebuilt{};
    qnpeps_e2e_status ctx_create_status{QNPEPS_E2E_ERR_INTERNAL};
    qnpeps_e2e_status ctx_first_status{QNPEPS_E2E_ERR_INTERNAL};
    qnpeps_e2e_status ctx_changed_status{QNPEPS_E2E_ERR_INTERNAL};
    qnpeps_e2e_status one_changed_status{QNPEPS_E2E_ERR_INTERNAL};
    qnpeps_e2e_status ctx_recreate_status{QNPEPS_E2E_ERR_INTERNAL};
    qnpeps_e2e_status ctx_rebuilt_status{QNPEPS_E2E_ERR_INTERNAL};
    qnpeps_e2e_status ctx_wrong_device_status{QNPEPS_E2E_OK};
    bool ctx_wrong_device_checked{false};

    qnpeps_e2e_minsr_ctx* minsr_ctx{};
    ctx_create_status = qnpeps_e2e_minsr_ctx_create(&ecfg, n, 0, nullptr, &minsr_ctx);
    if (ctx_create_status == QNPEPS_E2E_OK)
    {
        ctx_first_status = qnpeps_e2e_minsr_ctx_run(
            minsr_ctx,
            d_samp,
            d_lp,
            d_el,
            d_logq,
            reinterpret_cast<const qnpeps_e2e_cbuf*>(d_gram),
            reinterpret_cast<const qnpeps_e2e_cbuf*>(d_rows),
            nullptr,
            rel_cut,
            abs_cut,
            reinterpret_cast<qnpeps_e2e_cbuf*>(d_theta),
            stats_ctx_first.data(),
            &stats_ctx_first[2],
            &stats_ctx_first[3]
        );
        if (ctx_first_status == QNPEPS_E2E_OK)
        {
            ck(cudaMemcpy(
                   theta_ctx_first.data(),
                   d_theta,
                   theta_ctx_first.size() * sizeof(cf32),
                   cudaMemcpyDeviceToHost
               ),
               "d2h theta ctx first");
        }

        int original_device{};
        int device_count{};
        ck(cudaGetDevice(&original_device), "get original device");
        ck(cudaGetDeviceCount(&device_count), "get device count");
        if (device_count > 1)
        {
            const int other_device{original_device == 0 ? 1 : 0};
            ck(cudaSetDevice(other_device), "set other device");
            ctx_wrong_device_status = qnpeps_e2e_minsr_ctx_run(
                minsr_ctx,
                d_samp,
                d_lp,
                d_el,
                d_logq,
                reinterpret_cast<const qnpeps_e2e_cbuf*>(d_gram),
                reinterpret_cast<const qnpeps_e2e_cbuf*>(d_rows),
                nullptr,
                rel_cut,
                abs_cut,
                reinterpret_cast<qnpeps_e2e_cbuf*>(d_theta),
                stats_ctx_first.data(),
                &stats_ctx_first[2],
                &stats_ctx_first[3]
            );
            ctx_wrong_device_checked = true;
            ck(cudaSetDevice(original_device), "restore original device");
        }
    }

    const std::vector<std::uint8_t> samp_changed{
        gen_samples(lx, ly, dim_phys, n, 0xC0176ull ^ peps_seed, false)
    };
    ck(cudaMemcpy(d_samp, samp_changed.data(), samp_changed.size(), cudaMemcpyHostToDevice),
       "cpy changed samp");
    const qnpeps_eloc_status est_changed{qnpeps_eloc_run(
        &lcfg,
        static_cast<const qnpeps_eloc_cbuf*>(d_peps),
        d_samp,
        n,
        &tt,
        d_lp,
        d_el,
        reinterpret_cast<qnpeps_eloc_cbuf*>(d_rows),
        reinterpret_cast<qnpeps_eloc_cbuf*>(h_rows),
        reinterpret_cast<qnpeps_eloc_cbuf*>(d_gram),
        0.0,
        nullptr
    )};
    auto lp_changed = std::vector<double>(static_cast<std::size_t>(2 * n));
    auto rows_changed = std::vector<cf32>(static_cast<std::size_t>(n) * compact);
    auto gram_changed = std::vector<cf32>(static_cast<std::size_t>(n) * n);
    auto logq_changed = std::vector<double>(static_cast<std::size_t>(n));
    if (est_changed == QNPEPS_ELOC_OK)
    {
        ck(cudaMemcpy(
               lp_changed.data(), d_lp, lp_changed.size() * sizeof(double), cudaMemcpyDeviceToHost
           ),
           "d2h changed lp");
        ck(cudaMemcpy(
               rows_changed.data(),
               d_rows,
               rows_changed.size() * sizeof(cf32),
               cudaMemcpyDeviceToHost
           ),
           "d2h changed rows");
        ck(cudaMemcpy(
               gram_changed.data(),
               d_gram,
               gram_changed.size() * sizeof(cf32),
               cudaMemcpyDeviceToHost
           ),
           "d2h changed gram");
        for (int j{0}; j < n; ++j)
        {
            logq_changed[static_cast<std::size_t>(j)] =
                2.0 * lp_changed[static_cast<std::size_t>(2 * j)]
                + static_cast<double>((j * 11 + 7) % 13 - 6) * 0.125;
        }
        ck(cudaMemcpy(
               d_logq,
               logq_changed.data(),
               logq_changed.size() * sizeof(double),
               cudaMemcpyHostToDevice
           ),
           "cpy changed logq");

        if (minsr_ctx)
        {
            ctx_changed_status = qnpeps_e2e_minsr_ctx_run(
                minsr_ctx,
                d_samp,
                d_lp,
                d_el,
                d_logq,
                reinterpret_cast<const qnpeps_e2e_cbuf*>(d_gram),
                reinterpret_cast<const qnpeps_e2e_cbuf*>(d_rows),
                nullptr,
                rel_cut,
                abs_cut,
                reinterpret_cast<qnpeps_e2e_cbuf*>(d_theta),
                stats_ctx_changed.data(),
                &stats_ctx_changed[2],
                &stats_ctx_changed[3]
            );
            if (ctx_changed_status == QNPEPS_E2E_OK)
            {
                ck(cudaMemcpy(
                       theta_ctx_changed.data(),
                       d_theta,
                       theta_ctx_changed.size() * sizeof(cf32),
                       cudaMemcpyDeviceToHost
                   ),
                   "d2h theta ctx changed");
            }
        }

        one_changed_status = qnpeps_e2e_minsr(
            &ecfg,
            n,
            d_samp,
            d_lp,
            d_el,
            d_logq,
            reinterpret_cast<const qnpeps_e2e_cbuf*>(d_gram),
            reinterpret_cast<const qnpeps_e2e_cbuf*>(d_rows),
            nullptr,
            0,
            rel_cut,
            abs_cut,
            reinterpret_cast<qnpeps_e2e_cbuf*>(d_theta),
            stats_one_changed.data(),
            &stats_one_changed[2],
            &stats_one_changed[3],
            nullptr
        );
        if (one_changed_status == QNPEPS_E2E_OK)
        {
            ck(cudaMemcpy(
                   theta_one_changed.data(),
                   d_theta,
                   theta_one_changed.size() * sizeof(cf32),
                   cudaMemcpyDeviceToHost
               ),
               "d2h theta one changed");
        }
    }

    qnpeps_e2e_minsr_ctx_destroy(minsr_ctx);
    minsr_ctx = nullptr;
    cudaStream_t rebuilt_stream{};
    ck(cudaStreamCreateWithFlags(&rebuilt_stream, cudaStreamNonBlocking), "create rebuilt stream");
    if (est_changed == QNPEPS_ELOC_OK)
    {
        ctx_recreate_status =
            qnpeps_e2e_minsr_ctx_create(&ecfg, n, tile_tiny, rebuilt_stream, &minsr_ctx);
        if (ctx_recreate_status == QNPEPS_E2E_OK)
        {
            ctx_rebuilt_status = qnpeps_e2e_minsr_ctx_run(
                minsr_ctx,
                d_samp,
                d_lp,
                d_el,
                d_logq,
                reinterpret_cast<const qnpeps_e2e_cbuf*>(d_gram),
                nullptr,
                reinterpret_cast<const qnpeps_e2e_cbuf*>(h_rows),
                rel_cut,
                abs_cut,
                reinterpret_cast<qnpeps_e2e_cbuf*>(d_theta),
                stats_ctx_rebuilt.data(),
                &stats_ctx_rebuilt[2],
                &stats_ctx_rebuilt[3]
            );
            if (ctx_rebuilt_status == QNPEPS_E2E_OK)
            {
                ck(cudaMemcpy(
                       theta_ctx_rebuilt.data(),
                       d_theta,
                       theta_ctx_rebuilt.size() * sizeof(cf32),
                       cudaMemcpyDeviceToHost
                   ),
                   "d2h theta ctx rebuilt");
            }
        }
    }
    qnpeps_e2e_minsr_ctx_destroy(minsr_ctx);
    qnpeps_e2e_minsr_ctx_destroy(nullptr);
    ck(cudaStreamDestroy(rebuilt_stream), "destroy rebuilt stream");

    const std::size_t ctx_first_theta_diff{different_words(
        theta_ctx_first.data(), theta_dev.data(), theta_dev.size() * sizeof(cf32), sizeof(float)
    )};
    const std::size_t ctx_first_stats_diff{
        different_words(stats_ctx_first.data(), stats_dev.data(), sizeof(stats_dev), sizeof(double))
    };
    const std::size_t ctx_changed_theta_diff{different_words(
        theta_ctx_changed.data(),
        theta_one_changed.data(),
        theta_one_changed.size() * sizeof(cf32),
        sizeof(float)
    )};
    const std::size_t ctx_changed_stats_diff{different_words(
        stats_ctx_changed.data(),
        stats_one_changed.data(),
        sizeof(stats_one_changed),
        sizeof(double)
    )};
    const std::size_t ctx_rebuilt_theta_diff{different_words(
        theta_ctx_rebuilt.data(),
        theta_one_changed.data(),
        theta_one_changed.size() * sizeof(cf32),
        sizeof(float)
    )};
    const std::size_t ctx_rebuilt_stats_diff{different_words(
        stats_ctx_rebuilt.data(),
        stats_one_changed.data(),
        sizeof(stats_one_changed),
        sizeof(double)
    )};
    const std::size_t freshness_theta_words{different_words(
        theta_ctx_first.data(),
        theta_ctx_changed.data(),
        theta_ctx_changed.size() * sizeof(cf32),
        sizeof(float)
    )};
    const std::size_t freshness_stats_words{different_words(
        stats_ctx_first.data(), stats_ctx_changed.data(), sizeof(stats_ctx_changed), sizeof(double)
    )};
    const std::size_t freshness_input_words{
        different_words(samp.data(), samp_changed.data(), samp.size(), sizeof(std::uint8_t))
        + different_words(
            rows.data(), rows_changed.data(), rows.size() * sizeof(cf32), sizeof(float)
        )
        + different_words(
            gram.data(), gram_changed.data(), gram.size() * sizeof(cf32), sizeof(float)
        )
    };
    std::printf(
        "[gate_e2e] EXACT ctx_first theta_words=%zu stats_words=%zu status=%d\n",
        ctx_first_theta_diff,
        ctx_first_stats_diff,
        static_cast<int>(ctx_first_status)
    );
    std::printf(
        "[gate_e2e] EXACT ctx_changed theta_words=%zu stats_words=%zu status=%d "
        "fresh_input_words=%zu fresh_output_words=%zu\n",
        ctx_changed_theta_diff,
        ctx_changed_stats_diff,
        static_cast<int>(ctx_changed_status),
        freshness_input_words,
        freshness_theta_words + freshness_stats_words
    );
    std::printf(
        "[gate_e2e] EXACT ctx_rebuilt theta_words=%zu stats_words=%zu status=%d\n",
        ctx_rebuilt_theta_diff,
        ctx_rebuilt_stats_diff,
        static_cast<int>(ctx_rebuilt_status)
    );

    std::vector<std::vector<eoh::T>> pv(
        static_cast<std::size_t>(lx), std::vector<eoh::T>(static_cast<std::size_t>(ly))
    );
    for (const Blk& blk : blocks)
    {
        eoh::T& site = pv[static_cast<std::size_t>(blk.i)][static_cast<std::size_t>(blk.c)];
        site.dim = {dim_phys, static_cast<int>(blk.slice)};
    }

    minsr::UpdateInput in{};
    in.per_sample.resize(static_cast<std::size_t>(n));
    in.spins.resize(static_cast<std::size_t>(n));
    in.e_loc.resize(static_cast<std::size_t>(n));
    in.logpsi_re.resize(static_cast<std::size_t>(n));
    in.logpc.resize(static_cast<std::size_t>(n));
    for (auto j = 0; j < n; ++j)
    {
        const std::vector<int> sp{to_ints(samp, j, lx, ly)};
        in.spins[static_cast<std::size_t>(j)].assign(
            static_cast<std::size_t>(lx), std::vector<int>(static_cast<std::size_t>(ly), 0)
        );
        for (auto r = 0; r < lx; ++r)
        {
            for (auto c = 0; c < ly; ++c)
            {
                in.spins[static_cast<std::size_t>(j)][static_cast<std::size_t>(r)]
                        [static_cast<std::size_t>(c)] = sp[static_cast<std::size_t>(r * ly + c)];
            }
        }

        auto& eo{in.per_sample[static_cast<std::size_t>(j)]};
        eo.ok.assign(
            static_cast<std::size_t>(lx), std::vector<eoh::T>(static_cast<std::size_t>(ly))
        );
        for (const Blk& blk : blocks)
        {
            eoh::T& ok{eo.ok[static_cast<std::size_t>(blk.i)][static_cast<std::size_t>(blk.c)]};
            ok.dim = {static_cast<int>(blk.slice)};
            ok.v.assign(static_cast<std::size_t>(blk.slice), eoh::cfl{0.0f, 0.0f});
            for (long k = 0; k < blk.slice; ++k)
            {
                const cf32 rv{rows[static_cast<std::size_t>(j) * compact + blk.compact_off + k]};
                ok.v[static_cast<std::size_t>(k)] = eoh::cfl{rv.re, rv.im};
            }
        }
        in.e_loc[static_cast<std::size_t>(j)] =
            cdb{el[static_cast<std::size_t>(2 * j)], el[static_cast<std::size_t>(2 * j + 1)]};
        in.logpsi_re[static_cast<std::size_t>(j)] = lp[static_cast<std::size_t>(2 * j)];
        in.logpc[static_cast<std::size_t>(j)] = logq[static_cast<std::size_t>(j)];
    }
    const minsr::UpdateOutput ref{minsr::natural_gradient_direction(pv, in, rel_cut, abs_cut)};

    double ref_sumw{0.0};
    double ref_sumw2{0.0};
    for (double wv : ref.w)
    {
        ref_sumw += wv;
        ref_sumw2 += wv * wv;
    }
    const double ref_ess{ref_sumw * ref_sumw / ref_sumw2};

    std::vector<SetResult> results{};

    {
        SetResult r{};
        r.name = "ctx_first";
        if (ctx_create_status != QNPEPS_E2E_OK or ctx_first_status != QNPEPS_E2E_OK
            or ctx_first_theta_diff != 0 or ctx_first_stats_diff != 0)
        {
            r.fail = 1;
            r.max_rel = 1.0;
        }
        r.worst = "theta_words=" + std::to_string(ctx_first_theta_diff)
                  + ",stats_words=" + std::to_string(ctx_first_stats_diff);
        results.push_back(r);
    }
    {
        SetResult r{};
        r.name = "ctx_changed";
        if (est_changed != QNPEPS_ELOC_OK or ctx_changed_status != QNPEPS_E2E_OK
            or one_changed_status != QNPEPS_E2E_OK or ctx_changed_theta_diff != 0
            or ctx_changed_stats_diff != 0 or freshness_input_words == 0
            or freshness_theta_words + freshness_stats_words == 0)
        {
            r.fail = 1;
            r.max_rel = 1.0;
        }
        r.worst = "theta_words=" + std::to_string(ctx_changed_theta_diff)
                  + ",stats_words=" + std::to_string(ctx_changed_stats_diff)
                  + ",fresh=" + std::to_string(freshness_theta_words + freshness_stats_words);
        results.push_back(r);
    }
    {
        SetResult r{};
        r.name = "ctx_rebuilt";
        if (ctx_recreate_status != QNPEPS_E2E_OK or ctx_rebuilt_status != QNPEPS_E2E_OK
            or ctx_rebuilt_theta_diff != 0 or ctx_rebuilt_stats_diff != 0)
        {
            r.fail = 1;
            r.max_rel = 1.0;
        }
        r.worst = "theta_words=" + std::to_string(ctx_rebuilt_theta_diff)
                  + ",stats_words=" + std::to_string(ctx_rebuilt_stats_diff);
        results.push_back(r);
    }
    {
        SetResult r{};
        r.name = "ctx_device";
        if (ctx_wrong_device_checked)
        {
            if (ctx_wrong_device_status != QNPEPS_E2E_ERR_BAD_CONFIG)
            {
                r.fail = 1;
                r.max_rel = 1.0;
            }
            r.worst = "status=" + std::to_string(static_cast<int>(ctx_wrong_device_status));
        }
        else
        {
            r.worst = "one_visible_device";
        }
        results.push_back(r);
    }

    {
        SetResult r{};
        r.name = "theta_dot";
        double scale{1e-30};
        for (auto p = static_cast<long>(0); p < dense; ++p)
            scale = std::max(scale, std::abs(ref.theta_dot[static_cast<std::size_t>(p)]));
        for (auto p = static_cast<long>(0); p < dense; ++p)
        {
            const cf32 dv{theta_dev[static_cast<std::size_t>(p)]};
            const cdb rf{ref.theta_dot[static_cast<std::size_t>(p)]};
            const double err{
                std::sqrt(
                    (dv.re - rf.real()) * (dv.re - rf.real())
                    + (dv.im - rf.imag()) * (dv.im - rf.imag())
                )
                / scale
            };
            if (err > r.max_rel)
            {
                r.max_rel = err;
                char w[24]{};
                std::snprintf(w, sizeof(w), "p%ld", p);
                r.worst = w;
            }
            if (bad_num(dv.re) or bad_num(dv.im) or err > tol_td) r.fail = 1;
        }
        results.push_back(r);
    }

    {
        SetResult r{};
        r.name = "e_mean";
        r.max_rel = std::abs(e_mean_d[0] - ref.e_mean_re) / std::max(1.0, std::abs(ref.e_mean_re));
        if (r.max_rel > 1e-10 or bad_num(e_mean_d[0])) r.fail = 1;
        results.push_back(r);
    }
    {
        SetResult r{};
        r.name = "e_var";
        r.max_rel = std::abs(e_var_d - ref.e_var) / std::max(1.0, std::abs(ref.e_var));
        if (r.max_rel > 1e-10 or bad_num(e_var_d)) r.fail = 1;
        results.push_back(r);
    }
    {
        SetResult r{};
        r.name = "ess";
        r.max_rel = std::abs(ess_d - ref_ess) / std::max(1.0, std::abs(ref_ess));
        if (r.max_rel > 1e-10 or bad_num(ess_d)) r.fail = 1;
        r.worst = std::to_string(ess_d) + "vs" + std::to_string(ref_ess);
        results.push_back(r);
    }

    {
        SetResult r{};
        r.name = "dev_vs_host";
        for (auto p = static_cast<long>(0); p < dense; ++p)
        {
            if (theta_dev[static_cast<std::size_t>(p)].re
                    != theta_host[static_cast<std::size_t>(p)].re
                or theta_dev[static_cast<std::size_t>(p)].im
                       != theta_host[static_cast<std::size_t>(p)].im)
            {
                r.fail = 1;
                r.max_rel = 1.0;
                char w[24]{};
                std::snprintf(w, sizeof(w), "p%ld", p);
                r.worst = w;
                break;
            }
        }
        if (e_mean_d[0] != e_mean_h[0] or e_var_d != e_var_h or ess_d != ess_h) r.fail = 1;
        results.push_back(r);
    }

    {
        SetResult r{};
        r.name = "host_default";
        for (auto p = static_cast<long>(0); p < dense; ++p)
        {
            if (theta_dev[static_cast<std::size_t>(p)].re
                    != theta_host0[static_cast<std::size_t>(p)].re
                or theta_dev[static_cast<std::size_t>(p)].im
                       != theta_host0[static_cast<std::size_t>(p)].im)
            {
                r.fail = 1;
                r.max_rel = 1.0;
                char w[24]{};
                std::snprintf(w, sizeof(w), "p%ld", p);
                r.worst = w;
                break;
            }
        }
        if (e_mean_d[0] != e_mean_h0[0] or e_var_d != e_var_h0 or ess_d != ess_h0) r.fail = 1;
        results.push_back(r);
    }

    {
        SetResult r{};
        r.name = "repeat";
        for (auto p = static_cast<long>(0); p < dense; ++p)
        {
            if (theta_dev[static_cast<std::size_t>(p)].re
                    != theta_rep[static_cast<std::size_t>(p)].re
                or theta_dev[static_cast<std::size_t>(p)].im
                       != theta_rep[static_cast<std::size_t>(p)].im)
            {
                r.fail = 1;
                r.max_rel = 1.0;
                break;
            }
        }
        results.push_back(r);
    }

    {
        SetResult r{};
        r.name = "gram_finish";
        auto lr = std::vector<double>(static_cast<std::size_t>(n));
        for (auto j = 0; j < n; ++j)
        {
            lr[static_cast<std::size_t>(j)] =
                2.0 * lp[static_cast<std::size_t>(2 * j)] - logq[static_cast<std::size_t>(j)];
        }
        double mmax{lr[0]};
        for (double v : lr)
            mmax = std::max(mmax, v);
        double se{0.0};
        for (double v : lr)
            se += std::exp(v - mmax);
        const double logz{mmax + std::log(se) - std::log(static_cast<double>(n))};
        auto w = std::vector<double>(static_cast<std::size_t>(n));
        double wsum{0.0};
        for (auto j = 0; j < n; ++j)
        {
            w[static_cast<std::size_t>(j)] = std::exp(lr[static_cast<std::size_t>(j)] - logz);
            wsum += w[static_cast<std::size_t>(j)];
        }
        const double wmean{wsum / n};
        for (double& v : w)
            v /= wmean;

        auto G = std::vector<cdb>(static_cast<std::size_t>(n) * n, cdb{0.0, 0.0});
        auto spflat = std::vector<std::vector<int>>(static_cast<std::size_t>(n));
        for (auto j = 0; j < n; ++j)
            spflat[static_cast<std::size_t>(j)] = to_ints(samp, j, lx, ly);
        for (auto s = 0; s < n; ++s)
        {
            for (auto u = 0; u < n; ++u)
            {
                cdb acc{0.0, 0.0};
                for (const Blk& blk : blocks)
                {
                    if (spflat[static_cast<std::size_t>(s)]
                              [static_cast<std::size_t>(blk.i * ly + blk.c)]
                        != spflat[static_cast<std::size_t>(u)]
                                 [static_cast<std::size_t>(blk.i * ly + blk.c)])
                        continue;
                    for (long k = 0; k < blk.slice; ++k)
                    {
                        const cf32 rs{
                            rows[static_cast<std::size_t>(s) * compact + blk.compact_off + k]
                        };
                        const cf32 ru{
                            rows[static_cast<std::size_t>(u) * compact + blk.compact_off + k]
                        };
                        acc += std::conj(cdb{rs.re, rs.im}) * cdb{ru.re, ru.im};
                    }
                }
                G[static_cast<std::size_t>(s) * n + u] = acc;
            }
        }

        auto beta = std::vector<cdb>(static_cast<std::size_t>(n), cdb{0.0, 0.0});
        for (auto j = 0; j < n; ++j)
        {
            cdb b{0.0, 0.0};
            for (auto l = 0; l < n; ++l)
                b += w[static_cast<std::size_t>(l)] * G[static_cast<std::size_t>(j) * n + l];
            beta[static_cast<std::size_t>(j)] = b / static_cast<double>(n);
        }
        cdb mu{0.0, 0.0};
        for (auto j = 0; j < n; ++j)
            mu += w[static_cast<std::size_t>(j)] * beta[static_cast<std::size_t>(j)];
        mu /= static_cast<double>(n);
        auto T_id = std::vector<cdb>(static_cast<std::size_t>(n) * n, cdb{0.0, 0.0});
        for (auto j = 0; j < n; ++j)
        {
            for (auto k = 0; k < n; ++k)
            {
                const cdb s_jk{
                    std::sqrt(w[static_cast<std::size_t>(j)] * w[static_cast<std::size_t>(k)])
                    * (G[static_cast<std::size_t>(j) * n + k] - beta[static_cast<std::size_t>(j)]
                       - std::conj(beta[static_cast<std::size_t>(k)]) + mu)
                };
                T_id[static_cast<std::size_t>(j) * n + k] = std::conj(s_jk);
            }
        }

        const long np{dense};
        auto O = std::vector<cdb>(static_cast<std::size_t>(np) * n, cdb{0.0, 0.0});
        for (int j = 0; j < n; ++j)
            for (const Blk& blk : blocks)
            {
                const int spin{spflat[static_cast<std::size_t>(j)]
                                     [static_cast<std::size_t>(blk.i * ly + blk.c)]};
                for (long k = 0; k < blk.slice; ++k)
                {
                    const cf32 rv{
                        rows[static_cast<std::size_t>(j) * compact + blk.compact_off + k]
                    };
                    const long pidx{blk.dense_off + spin + dim_phys * k};
                    O[static_cast<std::size_t>(pidx + np * j)] = cdb{rv.re, rv.im};
                }
            }
        for (auto p = static_cast<long>(0); p < np; ++p)
        {
            cdb mp{0.0, 0.0};
            for (auto j = 0; j < n; ++j)
                mp += w[static_cast<std::size_t>(j)] * O[static_cast<std::size_t>(p + np * j)];
            mp /= static_cast<double>(n);
            for (auto j = 0; j < n; ++j)
            {
                O[static_cast<std::size_t>(p + np * j)] =
                    (O[static_cast<std::size_t>(p + np * j)] - mp)
                    * std::sqrt(w[static_cast<std::size_t>(j)]);
            }
        }
        auto T_dense = std::vector<cdb>(static_cast<std::size_t>(n) * n, cdb{0.0, 0.0});
        for (auto j = 0; j < n; ++j)
        {
            for (auto k = 0; k < n; ++k)
            {
                cdb acc{0.0, 0.0};
                for (auto p = static_cast<long>(0); p < np; ++p)
                {
                    acc += std::conj(O[static_cast<std::size_t>(p + np * j)])
                           * O[static_cast<std::size_t>(p + np * k)];
                }
                T_dense[static_cast<std::size_t>(j) * n + k] = std::conj(acc);
            }
        }

        double scale{1e-30};
        for (const cdb& v : T_dense)
            scale = std::max(scale, std::abs(v));
        for (auto idx = static_cast<std::size_t>(0); idx < T_id.size(); ++idx)
        {
            const double err{std::abs(T_id[idx] - T_dense[idx]) / scale};
            if (err > r.max_rel) r.max_rel = err;
        }
        if (r.max_rel > 1e-10) r.fail = 1;
        results.push_back(r);
    }

    cudaFree(d_peps);
    cudaFree(d_samp);
    cudaFree(d_lp);
    cudaFree(d_el);
    cudaFree(d_logq);
    cudaFree(d_rows);
    cudaFree(d_gram);
    cudaFree(d_theta);
    std::free(h_rows);

    int fails{0};
    double max_td{0.0};
    for (const SetResult& r : results)
    {
        set_line(r);
        if (r.fail) ++fails;
        if (r.name == "theta_dot") max_td = r.max_rel;
    }
    std::printf(
        "[gate_e2e] CASE lx=%d ly=%d D=%d chi=%d n=%d meo=%d tolTd=%.1e ess=%.4f sets_failed=%d "
        "maxTd=%.3e => %s\n",
        lx,
        ly,
        dim_bond,
        chi,
        n,
        meo,
        tol_td,
        ess_d,
        fails,
        max_td,
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
    const double noise{argc > 7 ? std::atof(argv[7]) : 0.02};
    const double tol_td{argc > 8 ? std::atof(argv[8]) : 1.0e-4};
    const double rel_cut{argc > 9 ? std::atof(argv[9]) : 1.0e-1};
    const double abs_cut{argc > 10 ? std::atof(argv[10]) : 1.0e-8};
    return run_case(lx, ly, dim_bond, chi, n, meo, noise, tol_td, rel_cut, abs_cut);
}
