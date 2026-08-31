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
#include <vector>

namespace
{

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
    if (e != cudaSuccess) std::printf("[gate_mg] CUDA %s: %s\n", what, cudaGetErrorString(e));
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

auto capture(
    StepOut& out,
    cf32* d_theta,
    std::uint8_t* d_samp,
    double* d_logq,
    double* d_logg,
    double* d_lp,
    double* d_el,
    std::int64_t dense,
    int ns,
    int sites
) -> void
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
           out.loggauge.data(), d_logg, out.loggauge.size() * sizeof(double), cudaMemcpyDeviceToHost
       ),
       "d2h logg");
    ck(cudaMemcpy(
           out.logpsi.data(), d_lp, out.logpsi.size() * sizeof(double), cudaMemcpyDeviceToHost
       ),
       "d2h lp");
    ck(cudaMemcpy(out.eloc.data(), d_el, out.eloc.size() * sizeof(double), cudaMemcpyDeviceToHost),
       "d2h el");
}

auto run_single(
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
        capture(out, d_theta, d_samp, d_logq, d_logg, d_lp, d_el, dense, ns, sites);
    cudaFree(d_theta);
    cudaFree(d_samp);
    cudaFree(d_logq);
    cudaFree(d_logg);
    cudaFree(d_lp);
    cudaFree(d_el);
    return st;
}

auto run_mg(
    const QnpepsE2eConfig& cfg,
    const void* d_peps,
    int ns,
    const QnpepsElocTermTable& tt,
    int gpus,
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
    const qnpeps_e2e_status st{qnpeps_e2e_step_multigpu(
        &cfg,
        d_peps,
        ns,
        &tt,
        gpus,
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
        nullptr
    )};
    if (st == QNPEPS_E2E_OK)
        capture(out, d_theta, d_samp, d_logq, d_logg, d_lp, d_el, dense, ns, sites);
    cudaFree(d_theta);
    cudaFree(d_samp);
    cudaFree(d_logq);
    cudaFree(d_logg);
    cudaFree(d_lp);
    cudaFree(d_el);
    return st;
}

auto run_dist(
    const QnpepsE2eConfig& cfg,
    const void* d_peps,
    int ns,
    const QnpepsElocTermTable& tt,
    int gpus,
    std::int64_t peer_tile_bytes,
    double rel,
    double abs_cut,
    std::int64_t dense,
    int sites,
    StepOut& out,
    QnpepsE2eDistTimings* tm
) -> qnpeps_e2e_status
{
    auto d_theta{dmalloc<cf32>(static_cast<std::size_t>(dense))};
    auto d_samp{dmalloc<std::uint8_t>(static_cast<std::size_t>(ns) * sites)};
    auto d_logq{dmalloc<double>(static_cast<std::size_t>(ns))};
    auto d_logg{dmalloc<double>(static_cast<std::size_t>(ns))};
    auto d_lp{dmalloc<double>(static_cast<std::size_t>(2 * ns))};
    auto d_el{dmalloc<double>(static_cast<std::size_t>(2 * ns))};
    const qnpeps_e2e_status st{qnpeps_e2e_step_multigpu_dist(
        &cfg,
        d_peps,
        ns,
        &tt,
        gpus,
        peer_tile_bytes,
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
        tm
    )};
    if (st == QNPEPS_E2E_OK)
        capture(out, d_theta, d_samp, d_logq, d_logg, d_lp, d_el, dense, ns, sites);
    cudaFree(d_theta);
    cudaFree(d_samp);
    cudaFree(d_logq);
    cudaFree(d_logg);
    cudaFree(d_lp);
    cudaFree(d_el);
    return st;
}

auto print_dist_timings(int gpus, const QnpepsE2eDistTimings& tm) -> void
{
    std::printf(
        "[gate_mg] DIST gpus=%d sample=%.4f eo=%.4f gram=%.4f pull_sum=%.4f pull_max=%.4f "
        "gather=%.4f solve=%.4f scatter=%.4f acc=%.4f pull_bytes=%llu acc_bytes=%llu\n",
        gpus,
        tm.sample_s,
        tm.eo_s,
        tm.gram_s,
        tm.gram_rows_peer_sum_s,
        tm.gram_rows_peer_max_s,
        tm.gram_gather_s,
        tm.solve_s,
        tm.scatter_s,
        tm.scatter_acc_peer_s,
        static_cast<unsigned long long>(tm.gram_rows_peer_bytes),
        static_cast<unsigned long long>(tm.scatter_acc_peer_bytes)
    );
}

auto theta_eq(const std::vector<cf32>& a, const std::vector<cf32>& b) -> bool
{
    if (a.size() != b.size()) return false;
    for (std::size_t k{0}; k < a.size(); ++k)
        if (a[k].re != b[k].re or a[k].im != b[k].im) return false;
    return true;
}

auto stats_eq(const StepOut& a, const StepOut& b) -> bool
{
    return a.e_mean[0] == b.e_mean[0] and a.e_mean[1] == b.e_mean[1] and a.e_var == b.e_var
           and a.ess == b.ess;
}

auto all_eq(const StepOut& a, const StepOut& b, char* note, std::size_t n) -> bool
{
    const bool th{theta_eq(a.theta, b.theta)};
    const bool st{stats_eq(a, b)};
    const bool sa{a.samples == b.samples};
    const bool lq{a.logq == b.logq};
    const bool lg{a.loggauge == b.loggauge};
    const bool lp{a.logpsi == b.logpsi};
    const bool el{a.eloc == b.eloc};
    std::snprintf(
        note,
        n,
        "theta=%d stats=%d samp=%d logq=%d logg=%d logpsi=%d eloc=%d",
        th,
        st,
        sa,
        lq,
        lg,
        lp,
        el
    );
    return th and st and sa and lq and lg and lp and el;
}

auto set_line(const char* name, bool pass, const char* note) -> void
{
    std::printf(
        "[gate_mg] SET %-22s => %s%s%s\n",
        name,
        pass ? "PASS" : "FAIL",
        note and note[0] ? "  " : "",
        note ? note : ""
    );
}

auto probe_p2p(int a, int b) -> int
{
    const std::size_t bytes{1u << 20};
    void* da{};
    void* db{};
    if (cudaSetDevice(a) != cudaSuccess) return 1;
    if (cudaMalloc(&da, bytes) != cudaSuccess) return 1;
    if (cudaSetDevice(b) != cudaSuccess or cudaMalloc(&db, bytes) != cudaSuccess)
    {
        cudaSetDevice(a);
        cudaFree(da);
        return 1;
    }
    auto* host = static_cast<unsigned char*>(std::malloc(bytes));
    cudaSetDevice(a);
    cudaMemset(da, 0xAB, bytes);
    cudaDeviceSynchronize();
    cudaError_t err{cudaMemcpyPeer(db, b, da, a, bytes)};
    cudaSetDevice(b);
    const cudaError_t se{cudaDeviceSynchronize()};
    if (err == cudaSuccess) err = se;
    const cudaError_t ce{
        host ? cudaMemcpy(host, db, bytes, cudaMemcpyDeviceToHost) : cudaErrorInvalidValue
    };
    if (err == cudaSuccess) err = ce;
    int ok{err == cudaSuccess ? 1 : 0};
    if (ok and host)
    {
        for (std::size_t k{0}; k < bytes; ++k)
        {
            if (host[k] != 0xAB)
            {
                ok = 0;
                break;
            }
        }
    }
    std::free(host);
    cudaSetDevice(a);
    cudaFree(da);
    cudaSetDevice(b);
    cudaFree(db);
    if (ok) return 0;
    return err != cudaSuccess ? 1 : 2;
}

auto print_shards(int ns, int gpus, int meo) -> void
{
    const int total_waves{(ns + meo - 1) / meo};
    int wbase{0};
    int covered{0};
    std::printf("[gate_mg] shards ns=%d gpus=%d meo=%d:", ns, gpus, meo);
    for (int g{0}; g < gpus; ++g)
    {
        const int remaining{gpus - g};
        const int waves_here{(total_waves - wbase + remaining - 1) / remaining};
        const int wend{wbase + waves_here};
        int base{wbase * meo};
        int end{wend * meo};
        if (base > ns) base = ns;
        if (end > ns) end = ns;
        std::printf(" g%d=[%d,%d)", g, base, end);
        covered += end - base;
        wbase = wend;
    }
    std::printf(
        "  covered=%d%s\n", covered, covered == ns ? " (partition OK)" : " (PARTITION BUG)"
    );
}

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

    int ndev{0};
    cudaGetDeviceCount(&ndev);
    std::printf(
        "[gate_mg] fixture lx=%d ly=%d D=%d chi=%d ns=%d meo=%d mode=%s "
        "contract=%d sample_batch=%d dense=%lld compact=%lld ndev=%d\n",
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
        static_cast<long long>(compact),
        ndev
    );
    print_shards(ns, 2, meo);
    if (ndev >= 4) print_shards(ns, 4, meo);

    StepOut single{};
    const qnpeps_e2e_status ss{run_single(cfg, d_peps, ns, tt, rel, abs_cut, dense, sites, single)};
    if (ss != QNPEPS_E2E_OK)
    {
        std::printf("[gate_mg] single error: %s\n", qnpeps_e2e_strerror(ss));
        cudaFree(d_peps);
        return 1;
    }

    int fails{0};

    bool dist_only{false};
    if (auto dl{std::getenv("QNPEPS_GATE_DIST_ONLY")}; dl and dl[0] == '1') dist_only = true;

    qnpeps::CuArray<int, 3> gpu_list{1, 2, 4};
    for (int gi{0}; gi < 3 and not dist_only; ++gi)
    {
        const int gpus{gpu_list[gi]};
        if (gpus > ndev) continue;
        StepOut mg{};
        const qnpeps_e2e_status sm{
            run_mg(cfg, d_peps, ns, tt, gpus, rel, abs_cut, dense, sites, mg)
        };
        char note[160]{};
        char label[32]{};
        std::snprintf(label, sizeof(label), "mg_gpus%d_vs_single", gpus);
        if (sm != QNPEPS_E2E_OK)
        {
            std::snprintf(note, sizeof(note), "status=%s", qnpeps_e2e_strerror(sm));
            set_line(label, false, note);
            ++fails;
            continue;
        }
        const bool pass{all_eq(single, mg, note, sizeof(note))};
        set_line(label, pass, note);
        if (not pass) ++fails;
    }

    if (ndev >= 2 and not dist_only)
    {
        setenv("QNPEPS_E2E_FORCE_STAGED", "1", 1);
        StepOut mgs{};
        const qnpeps_e2e_status sm{run_mg(cfg, d_peps, ns, tt, 2, rel, abs_cut, dense, sites, mgs)};
        unsetenv("QNPEPS_E2E_FORCE_STAGED");
        char note[160]{};
        if (sm != QNPEPS_E2E_OK)
        {
            std::snprintf(note, sizeof(note), "status=%s", qnpeps_e2e_strerror(sm));
            set_line("mg_gpus2_staged", false, note);
            ++fails;
        }
        else
        {
            const bool pass{all_eq(single, mgs, note, sizeof(note))};
            set_line("mg_gpus2_staged", pass, note);
            if (not pass) ++fails;
        }
    }

    if (ndev >= 2 and not dist_only)
    {
        StepOut a{};
        StepOut b{};
        const qnpeps_e2e_status sa{run_mg(cfg, d_peps, ns, tt, 2, rel, abs_cut, dense, sites, a)};
        const qnpeps_e2e_status sb{run_mg(cfg, d_peps, ns, tt, 2, rel, abs_cut, dense, sites, b)};
        char note[160]{};
        const bool ok{sa == QNPEPS_E2E_OK and sb == QNPEPS_E2E_OK};
        const bool pass{ok and all_eq(a, b, note, sizeof(note))};
        if (not ok) std::snprintf(note, sizeof(note), "status a=%d b=%d", sa, sb);
        set_line("mg_repeat", pass, note);
        if (not pass) ++fails;
    }

    if (ndev >= 2 and not dist_only)
    {
        setenv("QNPEPS_E2E_INJECT_OOM_GPU", "1", 1);
        StepOut junk{};
        const qnpeps_e2e_status sm{
            run_mg(cfg, d_peps, ns, tt, 2, rel, abs_cut, dense, sites, junk)
        };
        unsetenv("QNPEPS_E2E_INJECT_OOM_GPU");
        const bool pass{sm == QNPEPS_E2E_ERR_OOM};
        set_line("mg_oom_injection", pass, qnpeps_e2e_strerror(sm));
        if (not pass) ++fails;
    }

    for (int gi{0}; gi < 3; ++gi)
    {
        const int gpus{gpu_list[gi]};
        if (gpus > ndev) continue;
        if (dist_only and gpus == 1) continue;
        StepOut ds{};
        QnpepsE2eDistTimings tm{};
        tm.struct_size = sizeof(QnpepsE2eDistTimings);
        const qnpeps_e2e_status sd{
            run_dist(cfg, d_peps, ns, tt, gpus, 0, rel, abs_cut, dense, sites, ds, &tm)
        };
        char note[160]{};
        char label[32]{};
        std::snprintf(label, sizeof(label), "dist_gpus%d_vs_single", gpus);
        if (sd != QNPEPS_E2E_OK)
        {
            std::snprintf(note, sizeof(note), "status=%s", qnpeps_e2e_strerror(sd));
            set_line(label, false, note);
            ++fails;
            continue;
        }
        const bool pass{all_eq(single, ds, note, sizeof(note))};
        set_line(label, pass, note);
        if (gpus > 1) print_dist_timings(gpus, tm);
        if (not pass) ++fails;
    }

    if (ndev >= 2 and not dist_only)
    {
        StepOut ds{};
        const qnpeps_e2e_status sd{
            run_dist(cfg, d_peps, ns, tt, 2, 1, rel, abs_cut, dense, sites, ds, nullptr)
        };
        char note[160]{};
        if (sd != QNPEPS_E2E_OK)
        {
            std::snprintf(note, sizeof(note), "status=%s", qnpeps_e2e_strerror(sd));
            set_line("dist_gpus2_tiled", false, note);
            ++fails;
        }
        else
        {
            const bool pass{all_eq(single, ds, note, sizeof(note))};
            set_line("dist_gpus2_tiled", pass, note);
            if (not pass) ++fails;
        }
    }

    if (ndev >= 2 and not dist_only)
    {
        setenv("QNPEPS_E2E_FORCE_DIST", "1", 1);
        StepOut ds{};
        QnpepsE2eDistTimings tm{};
        tm.struct_size = sizeof(QnpepsE2eDistTimings);
        const qnpeps_e2e_status sd{
            run_dist(cfg, d_peps, ns, tt, 2, 0, rel, abs_cut, dense, sites, ds, &tm)
        };
        unsetenv("QNPEPS_E2E_FORCE_DIST");
        char note[160]{};
        if (sd != QNPEPS_E2E_OK)
        {
            std::snprintf(note, sizeof(note), "status=%s", qnpeps_e2e_strerror(sd));
            set_line("dist_gpus2_forced", false, note);
            ++fails;
        }
        else
        {
            const bool pass{all_eq(single, ds, note, sizeof(note))};
            set_line("dist_gpus2_forced", pass, note);
            print_dist_timings(2, tm);
            if (not pass) ++fails;
        }
    }

    if (ndev >= 2 and not dist_only)
    {
        setenv("QNPEPS_E2E_FORCE_DIST", "1", 1);
        StepOut ds{};
        const qnpeps_e2e_status sd{
            run_dist(cfg, d_peps, ns, tt, 2, 1, rel, abs_cut, dense, sites, ds, nullptr)
        };
        unsetenv("QNPEPS_E2E_FORCE_DIST");
        char note[160]{};
        if (sd != QNPEPS_E2E_OK)
        {
            std::snprintf(note, sizeof(note), "status=%s", qnpeps_e2e_strerror(sd));
            set_line("dist_gpus2_forced_tiled", false, note);
            ++fails;
        }
        else
        {
            const bool pass{all_eq(single, ds, note, sizeof(note))};
            set_line("dist_gpus2_forced_tiled", pass, note);
            if (not pass) ++fails;
        }
    }

    if (ndev >= 2 and not dist_only)
    {
        setenv("QNPEPS_E2E_FORCE_STAGED", "1", 1);
        StepOut ds{};
        const qnpeps_e2e_status sd{
            run_dist(cfg, d_peps, ns, tt, 2, 0, rel, abs_cut, dense, sites, ds, nullptr)
        };
        unsetenv("QNPEPS_E2E_FORCE_STAGED");
        char note[160]{};
        if (sd != QNPEPS_E2E_OK)
        {
            std::snprintf(note, sizeof(note), "status=%s", qnpeps_e2e_strerror(sd));
            set_line("dist_staged_delegate", false, note);
            ++fails;
        }
        else
        {
            const bool pass{all_eq(single, ds, note, sizeof(note))};
            set_line("dist_staged_delegate", pass, note);
            if (not pass) ++fails;
        }
    }

    if (ndev >= 2 and not dist_only)
    {
        StepOut a{};
        StepOut b{};
        const qnpeps_e2e_status sa{
            run_dist(cfg, d_peps, ns, tt, 2, 0, rel, abs_cut, dense, sites, a, nullptr)
        };
        const qnpeps_e2e_status sb{
            run_dist(cfg, d_peps, ns, tt, 2, 0, rel, abs_cut, dense, sites, b, nullptr)
        };
        char note[160]{};
        const bool ok{sa == QNPEPS_E2E_OK and sb == QNPEPS_E2E_OK};
        const bool pass{ok and all_eq(a, b, note, sizeof(note))};
        if (not ok) std::snprintf(note, sizeof(note), "status a=%d b=%d", sa, sb);
        set_line("dist_repeat", pass, note);
        if (not pass) ++fails;
    }

    if (ndev >= 2 and not dist_only)
    {
        setenv("QNPEPS_E2E_INJECT_OOM_GPU", "1", 1);
        setenv("QNPEPS_E2E_FORCE_DIST", "1", 1);
        StepOut junk{};
        const qnpeps_e2e_status sd{
            run_dist(cfg, d_peps, ns, tt, 2, 0, rel, abs_cut, dense, sites, junk, nullptr)
        };
        unsetenv("QNPEPS_E2E_INJECT_OOM_GPU");
        unsetenv("QNPEPS_E2E_FORCE_DIST");
        const bool pass{sd == QNPEPS_E2E_ERR_OOM};
        set_line("dist_oom_injection", pass, qnpeps_e2e_strerror(sd));
        if (not pass) ++fails;
    }

    if (ndev >= 2)
    {
        const int fwd{probe_p2p(0, 1)};
        const int bwd{probe_p2p(1, 0)};
        cudaSetDevice(0);
        auto verdict{
            (fwd == 0 and bwd == 0)  ? "P2P VERIFIED CLEAN (library will trust cudaMemcpyPeer)"
            : (fwd == 2 or bwd == 2) ? "P2P DATA MISMATCH (library falls back to host-staged)"
                                     : "P2P CUDA FAILURE (library falls back to host-staged)"
        };
        std::printf("[gate_mg] PROBE 0<->1 rc_fwd=%d rc_bwd=%d :: %s\n", fwd, bwd, verdict);
    }

    std::printf(
        "[gate_mg] CASE lx=%d ly=%d D=%d chi=%d ns=%d meo=%d rel=%.3g => %s (fails=%d)\n",
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
