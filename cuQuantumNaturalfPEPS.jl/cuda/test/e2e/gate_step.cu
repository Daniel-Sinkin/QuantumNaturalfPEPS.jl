#include "../../core/types.cuh"
#include "capi/qnpeps.h"
#include "dans_qnpeps_e2e.h"
#include "dans_qnpeps_eloc.h"
#include "gate_fixture.cuh"

#include <algorithm>
#include <cmath>
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

using gate::bd;
using gate::gen_samples;
using gate::GenPeps;
using gate::make_peps;

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

auto ck(cudaError_t e, const char* what) -> void
{
    if (e != cudaSuccess) std::printf("[gate_step] CUDA %s: %s\n", what, cudaGetErrorString(e));
}

template <class T>
auto dmalloc(std::size_t n) -> T*
{
    void* p{};
    ck(cudaMalloc(&p, n * sizeof(T)), "malloc");
    return static_cast<T*>(p);
}

struct StepOut
{
    std::vector<cf32> theta{};
    qnpeps::CuArray<double, 2> e_mean{};
    double e_var{};
    double ess{};
    std::vector<std::uint8_t> samples{};
    std::vector<double> logq{};
    std::vector<double> loggauge{};
    std::vector<double> logpsi{};
    std::vector<double> eloc{};
};

auto run_step(
    const QnpepsE2eConfig& cfg,
    const void* d_peps,
    int ns,
    const QnpepsElocTermTable& tt,
    double rel,
    double abs_cut,
    std::int64_t dense,
    int sites,
    StepOut& out
) -> qnpeps_e2e_status
{
    auto d_theta{dmalloc<cf32>(static_cast<std::size_t>(dense))};
    auto d_samp{dmalloc<std::uint8_t>(static_cast<std::size_t>(ns) * sites)};
    auto d_logq{dmalloc<double>(static_cast<std::size_t>(ns))};
    auto d_logg{dmalloc<double>(static_cast<std::size_t>(ns))};
    auto d_lp{dmalloc<double>(static_cast<std::size_t>(2 * ns))};
    auto d_el{dmalloc<double>(static_cast<std::size_t>(2 * ns))};

    const qnpeps_e2e_status st{qnpeps_e2e_step(
        &cfg,
        d_peps,
        ns,
        &tt,
        0,
        rel,
        abs_cut,
        reinterpret_cast<qnpeps_e2e_cbuf*>(d_theta),
        out.e_mean.data(),
        &out.e_var,
        &out.ess,
        d_samp,
        d_logq,
        d_logg,
        d_lp,
        d_el,
        nullptr,
        nullptr
    )};
    if (st == QNPEPS_E2E_OK)
    {
        out.theta.assign(static_cast<std::size_t>(dense), cf32{});
        out.samples.assign(static_cast<std::size_t>(ns) * sites, 0);
        out.logq.assign(static_cast<std::size_t>(ns), 0.0);
        out.loggauge.assign(static_cast<std::size_t>(ns), 0.0);
        out.logpsi.assign(static_cast<std::size_t>(2 * ns), 0.0);
        out.eloc.assign(static_cast<std::size_t>(2 * ns), 0.0);
        ck(cudaMemcpy(
               out.theta.data(), d_theta, out.theta.size() * sizeof(cf32), cudaMemcpyDeviceToHost
           ),
           "d2h theta");
        ck(cudaMemcpy(out.samples.data(), d_samp, out.samples.size(), cudaMemcpyDeviceToHost),
           "d2h samp");
        ck(cudaMemcpy(
               out.logq.data(), d_logq, out.logq.size() * sizeof(double), cudaMemcpyDeviceToHost
           ),
           "d2h logq");
        ck(cudaMemcpy(
               out.loggauge.data(),
               d_logg,
               out.loggauge.size() * sizeof(double),
               cudaMemcpyDeviceToHost
           ),
           "d2h logg");
        ck(cudaMemcpy(
               out.logpsi.data(), d_lp, out.logpsi.size() * sizeof(double), cudaMemcpyDeviceToHost
           ),
           "d2h lp");
        ck(cudaMemcpy(
               out.eloc.data(), d_el, out.eloc.size() * sizeof(double), cudaMemcpyDeviceToHost
           ),
           "d2h el");
    }
    cudaFree(d_theta);
    cudaFree(d_samp);
    cudaFree(d_logq);
    cudaFree(d_logg);
    cudaFree(d_lp);
    cudaFree(d_el);
    return st;
}

auto run_manual(
    const QnpepsE2eConfig& cfg,
    const void* d_peps,
    int ns,
    const QnpepsElocTermTable& tt,
    double rel,
    double abs_cut,
    std::int64_t dense,
    std::int64_t compact,
    int sites,
    StepOut& out
) -> qnpeps_e2e_status
{
    QnpepsConfig scfg{};
    scfg.struct_size = sizeof(QnpepsConfig);
    scfg.lx = cfg.lx;
    scfg.ly = cfg.ly;
    scfg.dim_phys = cfg.dim_phys;
    scfg.dim_bond = cfg.dim_bond;
    scfg.chi_s = cfg.chi_s;
    scfg.chi_dl = cfg.chi_dl;
    scfg.seed = cfg.seed;
    scfg.sampling_mode = cfg.sampling_mode;
    scfg.chi_c = cfg.contract_dim;

    QnpepsElocConfig lcfg{};
    lcfg.struct_size = sizeof(QnpepsElocConfig);
    lcfg.lx = cfg.lx;
    lcfg.ly = cfg.ly;
    lcfg.dim_phys = cfg.dim_phys;
    lcfg.dim_bond = cfg.dim_bond;
    lcfg.chi_eo = cfg.chi_eo;
    lcfg.meo = cfg.meo;

    const std::int64_t dlenv_bytes{qnpeps_dlenv_bytes(&scfg)};
    const auto dim_batch =
        static_cast<std::uint64_t>(cfg.sample_batch > 0 ? cfg.sample_batch : std::min(ns, 2048));
    const std::int64_t scratch_bytes{qnpeps_sample_scratch_bytes(&scfg, dim_batch)};
    const std::int64_t samples_bytes{qnpeps_sample_bytes(&scfg, static_cast<std::uint64_t>(ns))};

    auto dlenv{dmalloc<std::uint8_t>(static_cast<std::size_t>(dlenv_bytes))};
    auto rowlogs{dmalloc<double>(static_cast<std::size_t>(cfg.lx - 1))};
    auto scratch{dmalloc<std::uint8_t>(static_cast<std::size_t>(scratch_bytes))};
    auto samples{dmalloc<std::uint8_t>(static_cast<std::size_t>(samples_bytes))};
    auto logq{dmalloc<double>(static_cast<std::size_t>(ns))};
    auto logg{dmalloc<double>(static_cast<std::size_t>(ns))};
    auto lp{dmalloc<double>(static_cast<std::size_t>(2 * ns))};
    auto el{dmalloc<double>(static_cast<std::size_t>(2 * ns))};
    auto rows{dmalloc<cf32>(static_cast<std::size_t>(ns) * compact)};
    auto gram{dmalloc<cf32>(static_cast<std::size_t>(ns) * ns)};
    auto theta{dmalloc<cf32>(static_cast<std::size_t>(dense))};

    qnpeps_e2e_status st{QNPEPS_E2E_OK};
    const qnpeps_status s1{qnpeps_build_dlenv(
        &scfg,
        static_cast<const qnpeps_device_peps*>(d_peps),
        reinterpret_cast<qnpeps_device_dlenv*>(dlenv),
        rowlogs,
        nullptr
    )};
    const QnpepsSampleArgs sample_args{
        .struct_size = sizeof(QnpepsSampleArgs),
        .peps = static_cast<const qnpeps_device_peps*>(d_peps),
        .dlenv = reinterpret_cast<const qnpeps_device_dlenv*>(dlenv),
        .gpus = 1,
        .scratch = scratch,
        .scratch_bytes = static_cast<std::uint64_t>(scratch_bytes),
        .samples_out = samples,
        .log_prob_config = logq,
        .log_gauge = logg,
        .n_samples = static_cast<std::uint64_t>(ns),
        .batch_base = 0,
        .dim_batch = dim_batch,
        .stream = nullptr
    };
    const qnpeps_status s2{qnpeps_sample(&scfg, &sample_args)};
    const qnpeps_eloc_status s3{qnpeps_eloc_run(
        &lcfg,
        static_cast<const qnpeps_eloc_cbuf*>(d_peps),
        samples,
        ns,
        &tt,
        lp,
        el,
        reinterpret_cast<qnpeps_eloc_cbuf*>(rows),
        nullptr,
        reinterpret_cast<qnpeps_eloc_cbuf*>(gram),
        0.0,
        nullptr
    )};
    const qnpeps_e2e_status s4{qnpeps_e2e_minsr(
        &cfg,
        ns,
        samples,
        lp,
        el,
        logq,
        reinterpret_cast<const qnpeps_e2e_cbuf*>(gram),
        reinterpret_cast<const qnpeps_e2e_cbuf*>(rows),
        nullptr,
        0,
        rel,
        abs_cut,
        reinterpret_cast<qnpeps_e2e_cbuf*>(theta),
        out.e_mean.data(),
        &out.e_var,
        &out.ess,
        nullptr
    )};
    if (s1 or s2 or s3 or s4) st = QNPEPS_E2E_ERR_INTERNAL;

    if (st == QNPEPS_E2E_OK)
    {
        out.theta.assign(static_cast<std::size_t>(dense), cf32{});
        out.samples.assign(static_cast<std::size_t>(ns) * sites, 0);
        out.logq.assign(static_cast<std::size_t>(ns), 0.0);
        out.loggauge.assign(static_cast<std::size_t>(ns), 0.0);
        out.logpsi.assign(static_cast<std::size_t>(2 * ns), 0.0);
        out.eloc.assign(static_cast<std::size_t>(2 * ns), 0.0);
        ck(cudaMemcpy(
               out.theta.data(), theta, out.theta.size() * sizeof(cf32), cudaMemcpyDeviceToHost
           ),
           "d2h theta m");
        ck(cudaMemcpy(out.samples.data(), samples, out.samples.size(), cudaMemcpyDeviceToHost),
           "d2h samp m");
        ck(cudaMemcpy(
               out.logq.data(), logq, out.logq.size() * sizeof(double), cudaMemcpyDeviceToHost
           ),
           "d2h logq m");
        ck(cudaMemcpy(
               out.loggauge.data(),
               logg,
               out.loggauge.size() * sizeof(double),
               cudaMemcpyDeviceToHost
           ),
           "d2h logg m");
        ck(cudaMemcpy(
               out.logpsi.data(), lp, out.logpsi.size() * sizeof(double), cudaMemcpyDeviceToHost
           ),
           "d2h lp m");
        ck(cudaMemcpy(
               out.eloc.data(), el, out.eloc.size() * sizeof(double), cudaMemcpyDeviceToHost
           ),
           "d2h el m");
    }
    cudaFree(dlenv);
    cudaFree(rowlogs);
    cudaFree(scratch);
    cudaFree(samples);
    cudaFree(logq);
    cudaFree(logg);
    cudaFree(lp);
    cudaFree(el);
    cudaFree(rows);
    cudaFree(gram);
    cudaFree(theta);
    return st;
}

auto theta_bitwise_eq(const std::vector<cf32>& a, const std::vector<cf32>& b) -> bool
{
    if (a.size() != b.size()) return false;
    for (std::size_t k{0}; k < a.size(); ++k)
        if (a[k].re != b[k].re or a[k].im != b[k].im) return false;
    return true;
}

auto stats_bitwise_eq(const StepOut& a, const StepOut& b) -> bool
{
    return a.e_mean[0] == b.e_mean[0] and a.e_mean[1] == b.e_mean[1] and a.e_var == b.e_var
           and a.ess == b.ess;
}

auto set_line(const char* name, bool pass, const char* note) -> void
{
    std::printf(
        "[gate_step] SET %-16s => %s%s%s\n",
        name,
        pass ? "PASS" : "FAIL",
        note and note[0] ? "  " : "",
        note ? note : ""
    );
}

auto set_report_line(const char* name, const char* note) -> void
{
    std::printf(
        "[gate_step] SET %-16s => REPORT%s%s\n",
        name,
        note and note[0] ? "  " : "",
        note ? note : ""
    );
}

auto dump_julia_fixture(
    const char* dir,
    const QnpepsE2eConfig& cfg,
    int ns,
    double rel,
    double abs_cut,
    double j1,
    double j2,
    const std::vector<float>& hpeps,
    const StepOut& s
) -> void
{
    const std::string d{dir};
    auto pf{std::fopen((d + "/params.txt").c_str(), "w")};
    if (pf)
    {
        std::fprintf(
            pf,
            "%d %d %d %d %d %d %d %d %llu %.17g %.17g %.17g %.17g %d\n",
            cfg.lx,
            cfg.ly,
            cfg.dim_bond,
            cfg.chi_s,
            cfg.chi_dl,
            cfg.chi_eo,
            cfg.meo,
            ns,
            static_cast<unsigned long long>(cfg.seed),
            rel,
            abs_cut,
            j1,
            j2,
            0
        );
        std::fclose(pf);
    }
    auto bf{std::fopen((d + "/peps.bin").c_str(), "wb")};
    if (bf)
    {
        std::fwrite(hpeps.data(), sizeof(float), hpeps.size(), bf);
        std::fclose(bf);
    }
    auto tf{std::fopen((d + "/theta.bin").c_str(), "wb")};
    if (tf)
    {
        std::fwrite(s.theta.data(), sizeof(cf32), s.theta.size(), tf);
        std::fclose(tf);
    }
    auto sf{std::fopen((d + "/stats.txt").c_str(), "w")};
    if (sf)
    {
        std::fprintf(sf, "%.17g %.17g %.17g %.17g\n", s.e_mean[0], s.e_mean[1], s.e_var, s.ess);
        std::fclose(sf);
    }
    std::printf(
        "[gate_step] dumped Julia fixture to %s (peps %zu float, theta %zu cf32)\n",
        dir,
        hpeps.size(),
        s.theta.size()
    );
}

}

#include "ms3_compare.cuh"

namespace
{

auto run_case(int lx, int ly, int D, int chi, int ns, int meo, double rel, double abs_cut) -> int
{
    const int dim_phys{2};
    const int sites{lx * ly};
    const std::uint64_t seed{
        0xE2E57Eull ^ (static_cast<std::uint64_t>(lx) * 131 + ly * 17 + D * 7 + ns * 3)
    };

    GenPeps g{make_peps(lx, ly, D, dim_phys, chi, 2, 0.02, seed)};
    const std::size_t total{g.flat.size()};
    auto hpeps = std::vector<float>(2 * total);
    auto irng = std::mt19937_64(seed ^ 0x5D5D5D5Dull);
    auto igauss = std::normal_distribution<double>(0.0, 1.0);
    for (std::size_t k{0}; k < total; ++k)
    {
        hpeps[2 * k] = static_cast<float>(g.flat[k]);
        hpeps[2 * k + 1] = static_cast<float>(0.35 * igauss(irng));
    }

    QnpepsE2eConfig cfg{};
    cfg.struct_size = sizeof(QnpepsE2eConfig);
    cfg.lx = lx;
    cfg.ly = ly;
    cfg.dim_phys = dim_phys;
    cfg.dim_bond = D;
    cfg.chi_s = chi;
    cfg.chi_dl = chi;
    cfg.chi_eo = chi;
    cfg.meo = meo;
    cfg.seed = seed;
    if (std::getenv("QNPEPS_GATE_FULL"))
    {
        cfg.sampling_mode = QNPEPS_SAMPLING_FULL;
        cfg.contract_dim = 3 * D;
        cfg.sample_batch = std::min(ns, 8);
    }

    std::int64_t dense{0};
    std::int64_t compact{0};
    qnpeps_e2e_dense_count(&cfg, &dense);
    qnpeps_e2e_compact_count(&cfg, &compact);

    void* d_peps{};
    ck(cudaMalloc(&d_peps, hpeps.size() * sizeof(float)), "malloc peps");
    ck(cudaMemcpy(d_peps, hpeps.data(), hpeps.size() * sizeof(float), cudaMemcpyHostToDevice),
       "cpy peps");

    const double j1{1.0};
    const std::vector<Bond> bonds{bonds_nn(lx, ly, j1)};
    const std::vector<QnpepsElocDiagBond> diag{diag_bonds_of(bonds, ly)};
    const std::vector<QnpepsElocFlipTerm> flips{masked_flips_of(bonds, ly)};
    QnpepsElocTermTable tt{};
    tt.n_diag = static_cast<int32_t>(diag.size());
    tt.diag = diag.data();
    tt.n_flip = static_cast<int32_t>(flips.size());
    tt.flip = flips.data();

    std::printf(
        "[gate_step] fixture lx=%d ly=%d D=%d chi=%d ns=%d meo=%d mode=%s "
        "contract=%d sample_batch=%d dense=%lld compact=%lld\n",
        lx,
        ly,
        D,
        chi,
        ns,
        meo,
        cfg.sampling_mode == QNPEPS_SAMPLING_FULL ? "full" : "fast",
        cfg.contract_dim,
        cfg.sample_batch,
        static_cast<long long>(dense),
        static_cast<long long>(compact)
    );

    StepOut step{};
    StepOut manual{};
    const qnpeps_e2e_status ss{run_step(cfg, d_peps, ns, tt, rel, abs_cut, dense, sites, step)};
    if (ss != QNPEPS_E2E_OK)
    {
        std::printf("[gate_step] step error: %s\n", qnpeps_e2e_strerror(ss));
        cudaFree(d_peps);
        return 1;
    }
    const qnpeps_e2e_status sm{
        run_manual(cfg, d_peps, ns, tt, rel, abs_cut, dense, compact, sites, manual)
    };
    if (sm != QNPEPS_E2E_OK)
    {
        std::printf("[gate_step] manual error\n");
        cudaFree(d_peps);
        return 1;
    }

    int fails{0};

    {
        const bool th{theta_bitwise_eq(step.theta, manual.theta)};
        const bool sa{step.samples == manual.samples};
        const bool lq{step.logq == manual.logq};
        const bool lg{step.loggauge == manual.loggauge};
        const bool lp{step.logpsi == manual.logpsi};
        const bool el{step.eloc == manual.eloc};
        const bool stt{stats_bitwise_eq(step, manual)};
        const bool pass{th and sa and lq and lg and lp and el and stt};
        char note[128]{};
        std::snprintf(
            note,
            sizeof(note),
            "theta=%d samp=%d logq=%d logg=%d logpsi=%d eloc=%d stats=%d",
            th,
            sa,
            lq,
            lg,
            lp,
            el,
            stt
        );
        set_line("d_composition", pass, note);
        if (not pass) ++fails;
    }

    {
        StepOut rep{};
        run_step(cfg, d_peps, ns, tt, rel, abs_cut, dense, sites, rep);
        const bool pass{
            theta_bitwise_eq(step.theta, rep.theta) and stats_bitwise_eq(step, rep)
            and step.samples == rep.samples and step.logq == rep.logq
        };
        set_line("determinism", pass, nullptr);
        if (not pass) ++fails;
    }

    {
        std::vector<QnpepsElocDiagBond> diag2{diag};
        std::vector<QnpepsElocFlipTerm> flips2{flips};
        if (not diag2.empty()) diag2[0].coeff *= 1.5;
        if (not flips2.empty()) flips2[0].coeff_re *= 1.5;
        QnpepsElocTermTable tt2{};
        tt2.n_diag = static_cast<int32_t>(diag2.size());
        tt2.diag = diag2.data();
        tt2.n_flip = static_cast<int32_t>(flips2.size());
        tt2.flip = flips2.data();
        StepOut c{};
        run_step(cfg, d_peps, ns, tt2, rel, abs_cut, dense, sites, c);
        const bool samp_same{c.samples == step.samples and c.logq == step.logq};
        const bool eloc_moved{c.eloc != step.eloc};
        const bool theta_moved{not theta_bitwise_eq(c.theta, step.theta)};
        const bool pass{samp_same and eloc_moved and theta_moved};
        char note[96]{};
        std::snprintf(
            note,
            sizeof(note),
            "sampPristine=%d elocMoved=%d thetaMoved=%d",
            samp_same,
            eloc_moved,
            theta_moved
        );
        set_line("probe_termcoeff", pass, note);
        if (not pass) ++fails;
    }

    {
        StepOut c{};
        run_step(cfg, d_peps, ns, tt, rel * 20.0 + 0.05, abs_cut, dense, sites, c);
        const bool stats_same{
            stats_bitwise_eq(step, c) and c.samples == step.samples and c.logq == step.logq
            and c.eloc == step.eloc
        };
        const bool theta_moved{not theta_bitwise_eq(c.theta, step.theta)};
        const bool pass{stats_same and theta_moved};
        char note[96]{};
        std::snprintf(
            note, sizeof(note), "otherPristine=%d thetaMoved=%d", stats_same, theta_moved
        );
        set_line("probe_relcut", pass, note);
        if (not pass) ++fails;
    }

    {
        QnpepsE2eConfig cfg2{cfg};
        cfg2.seed = cfg.seed ^ 0x9E3779B97F4A7C15ull;
        StepOut c{};
        run_step(cfg2, d_peps, ns, tt, rel, abs_cut, dense, sites, c);
        const bool samp_moved{c.samples != step.samples};
        set_line("probe_seed", samp_moved, samp_moved ? "samplesDiverged" : "samplesSAME(!)");
        if (not samp_moved) ++fails;
    }

    if (auto dd{std::getenv("E2E_DUMP_DIR")}; dd and dd[0])
        dump_julia_fixture(dd, cfg, ns, rel, abs_cut, j1, 0.0, hpeps, step);

    if (auto mb{std::getenv("MS3_BIN")}; mb and mb[0])
        fails += ms3::compare_case(cfg, ns, hpeps, g, mb);

    std::printf(
        "[gate_step] CASE lx=%d ly=%d D=%d chi=%d ns=%d meo=%d rel=%.3g => %s (fails=%d)\n",
        lx,
        ly,
        D,
        chi,
        ns,
        meo,
        rel,
        fails == 0 ? "PASS" : "FAIL",
        fails
    );
    cudaFree(d_peps);
    return fails == 0 ? 0 : 1;
}

}

auto main(int argc, char** argv) -> int
{
    const int lx{argc > 1 ? std::atoi(argv[1]) : 4};
    const int ly{argc > 2 ? std::atoi(argv[2]) : 4};
    const int D{argc > 3 ? std::atoi(argv[3]) : 2};
    const int chi{argc > 4 ? std::atoi(argv[4]) : 8};
    const int ns{argc > 5 ? std::atoi(argv[5]) : 16};
    const int meo{argc > 6 ? std::atoi(argv[6]) : 4};
    const double rel{argc > 7 ? std::atof(argv[7]) : 0.1};
    const double abs_cut{argc > 8 ? std::atof(argv[8]) : 1.0e-8};
    return run_case(lx, ly, D, chi, ns, meo, rel, abs_cut);
}
