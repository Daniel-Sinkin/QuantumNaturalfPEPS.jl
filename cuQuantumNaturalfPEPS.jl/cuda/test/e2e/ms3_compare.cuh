#ifndef QNPEPS_E2E_MS3_COMPARE_CUH
#define QNPEPS_E2E_MS3_COMPARE_CUH

#include "../../core/types.cuh"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace ms3
{

using ::cf32;

struct SiteDims
{
    int dw{};
    int ds{};
    int de{};
    int dn{};
};

inline auto site_dims(int lx, int ly, int D, int r, int c) -> SiteDims
{
    SiteDims s{};
    s.dw = gate::bd(ly, c, D);
    s.ds = gate::bd(lx, r + 1, D);
    s.de = gate::bd(ly, c + 1, D);
    s.dn = gate::bd(lx, r, D);
    return s;
}

inline auto write_fixture(
    const std::string& path,
    int lx,
    int ly,
    int D,
    int seed,
    int dim_phys,
    const std::vector<float>& hpeps
) -> bool
{
    auto f{std::fopen(path.c_str(), "w")};
    if (not f) return false;
    std::fprintf(f, "FIXTURE %d %d %d %d\n", lx, ly, D, seed);
    std::size_t off{0};
    for (auto r = 0; r < lx; ++r)
    {
        for (auto c = 0; c < ly; ++c)
        {
            const SiteDims sd{site_dims(lx, ly, D, r, c)};
            const std::size_t n{static_cast<std::size_t>(sd.dw) * sd.ds * sd.de * sd.dn * dim_phys};
            std::fprintf(
                f, "SITE %d %d %d %d %d %d %d", r, c, sd.dw, sd.ds, sd.de, sd.dn, dim_phys
            );
            for (auto k = static_cast<std::size_t>(0); k < n; ++k)
            {
                std::fprintf(
                    f,
                    " %.10g %.10g",
                    static_cast<double>(hpeps[2 * (off + k)]),
                    static_cast<double>(hpeps[2 * (off + k) + 1])
                );
            }
            std::fprintf(f, "\n");
            off += n;
        }
    }
    std::fprintf(f, "END\n");
    std::fclose(f);
    return true;
}

struct M3Samples
{
    std::vector<std::uint8_t> spins{};
    std::vector<double> logq{};
    int ok{};
};

inline auto parse_samples(const std::string& path, int lx, int ly, int ns) -> M3Samples
{
    M3Samples out{};
    auto f{std::fopen(path.c_str(), "r")};
    if (not f) return out;
    char tag[32]{};
    int flx{};
    int fly{};
    int m{};
    int batches{};
    if (std::fscanf(f, "%31s %d %d %d %d", tag, &flx, &fly, &m, &batches) != 5)
    {
        std::fclose(f);
        return out;
    }
    const int sites{lx * ly};
    out.spins.assign(static_cast<std::size_t>(ns) * sites, 0);
    out.logq.assign(static_cast<std::size_t>(ns), 0.0);
    for (auto bb = 0; bb < batches; ++bb)
    {
        for (auto b = 0; b < m; ++b)
        {
            const int idx{bb * m + b};
            double lpc{};
            if (std::fscanf(f, "%lf", &lpc) != 1)
            {
                std::fclose(f);
                return out;
            }
            if (idx < ns) out.logq[static_cast<std::size_t>(idx)] = lpc;
            for (auto r = 0; r < flx; ++r)
            {
                unsigned long long word{};
                if (std::fscanf(f, "%llu", &word) != 1)
                {
                    std::fclose(f);
                    return out;
                }
                if (idx < ns)
                {
                    for (auto c = 0; c < fly; ++c)
                    {
                        out.spins[static_cast<std::size_t>(idx) * sites + r * fly + c] =
                            static_cast<std::uint8_t>((word >> c) & 1ull);
                    }
                }
            }
        }
    }
    out.ok = 1;
    std::fclose(f);
    return out;
}

struct M3Eloc
{
    std::vector<double> e_re{};
    std::vector<double> e_im{};
    std::vector<double> logpsi{};
    int ok{};
};

inline auto parse_eloc(const std::string& path, int lx, int ns) -> M3Eloc
{
    M3Eloc out{};
    auto f{std::fopen(path.c_str(), "r")};
    if (not f) return out;
    char tag[32]{};
    int a{};
    int b2{};
    int c2{};
    int d2{};
    unsigned long long sd{};
    long n{};
    if (std::fscanf(f, "%31s %d %d %d %d %llu %ld", tag, &a, &b2, &c2, &d2, &sd, &n) != 7)
    {
        std::fclose(f);
        return out;
    }
    out.e_re.assign(static_cast<std::size_t>(ns), 0.0);
    out.e_im.assign(static_cast<std::size_t>(ns), 0.0);
    out.logpsi.assign(static_cast<std::size_t>(ns), 0.0);
    for (auto i = static_cast<long>(0); i < n; ++i)
    {
        double er{};
        double ei{};
        double lp{};
        if (std::fscanf(f, "%lf %lf %lf", &er, &ei, &lp) != 3)
        {
            std::fclose(f);
            return out;
        }
        if (i < ns)
        {
            out.e_re[static_cast<std::size_t>(i)] = er;
            out.e_im[static_cast<std::size_t>(i)] = ei;
            out.logpsi[static_cast<std::size_t>(i)] = lp;
        }
        for (auto r = 0; r < lx; ++r)
        {
            long pos{std::ftell(f)};
            unsigned long long w{};
            int c{std::getc(f)};
            while (c == ' ' or c == '\t')
                c = std::getc(f);
            if (c == '\n' or c == EOF)
            {
                std::fseek(f, pos, SEEK_SET);
                break;
            }
            std::ungetc(c, f);
            if (std::fscanf(f, "%llu", &w) != 1)
            {
                std::fseek(f, pos, SEEK_SET);
                break;
            }
        }
    }
    out.ok = 1;
    std::fclose(f);
    return out;
}

inline auto read_thetadot(const std::string& path, std::vector<float>& buf) -> bool
{
    auto f{std::fopen(path.c_str(), "rb")};
    if (not f) return false;
    std::fseek(f, 0, SEEK_END);
    const long bytes{std::ftell(f)};
    std::fseek(f, 0, SEEK_SET);
    buf.assign(static_cast<std::size_t>(bytes) / sizeof(float), 0.0f);
    const std::size_t got{std::fread(buf.data(), sizeof(float), buf.size(), f)};
    std::fclose(f);
    return got == buf.size();
}

inline auto theta_vs_m3(
    const std::vector<cf32>& ours_theta,
    const std::vector<float>& td,
    int lx,
    int ly,
    int D,
    int dim_phys,
    int& ok_site_out
) -> double
{
    int ok_site{0};
    for (auto r = 0; r < lx; ++r)
    {
        for (auto c = 0; c < ly; ++c)
        {
            const SiteDims sd{site_dims(lx, ly, D, r, c)};
            ok_site = std::max(ok_site, sd.dw * sd.ds * sd.de * sd.dn);
        }
    }
    ok_site_out = ok_site;
    double scale{1e-30};
    for (const cf32& v : ours_theta)
    {
        scale = std::max(
            scale, std::sqrt(static_cast<double>(v.re) * v.re + static_cast<double>(v.im) * v.im)
        );
    }
    double td_max{0.0};
    std::int64_t dense_off{0};
    for (auto r = 0; r < lx; ++r)
    {
        for (auto c = 0; c < ly; ++c)
        {
            const int q{r * ly + c};
            const SiteDims sd{site_dims(lx, ly, D, r, c)};
            const int X{sd.dw * sd.ds * sd.de * sd.dn};
            for (auto gidx = 0; gidx < X; ++gidx)
            {
                for (auto p = 0; p < dim_phys; ++p)
                {
                    const std::size_t ours_i{
                        static_cast<std::size_t>(dense_off) + dim_phys * gidx + p
                    };
                    const std::int64_t m3c{
                        static_cast<std::int64_t>(2) * q * ok_site + p * X + gidx
                    };
                    const std::size_t fr{static_cast<std::size_t>(2 * m3c)};
                    if (fr + 1 >= td.size()) continue;
                    const double dr{static_cast<double>(ours_theta[ours_i].re) - td[fr]};
                    const double di{static_cast<double>(ours_theta[ours_i].im) - td[fr + 1]};
                    td_max = std::max(td_max, std::sqrt(dr * dr + di * di) / scale);
                }
            }
            dense_off += static_cast<std::int64_t>(dim_phys) * X;
        }
    }
    return td_max;
}

inline auto logq_feed_probe(
    const QnpepsE2eConfig& cfg,
    const void* d_peps,
    int ns,
    const QnpepsElocTermTable& tt,
    const std::vector<double>& m3_logq,
    const std::vector<double>& m3_eloc,
    const std::vector<double>& m3_logpsi,
    std::int64_t dense,
    std::int64_t compact,
    double rel,
    double& feed_max_out,
    std::vector<cf32>& theta_m3q_out,
    std::vector<cf32>& theta_m3qe_out,
    std::vector<cf32>& theta_m3full_out
) -> bool
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
    const std::uint64_t dim_batch{static_cast<std::uint64_t>(
        cfg.sample_batch > 0 ? cfg.sample_batch : (ns > 2048 ? 2048 : ns)
    )};
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

    bool ok{true};
    ok = ok
         and qnpeps_build_dlenv(
                 &scfg,
                 static_cast<const qnpeps_device_peps*>(d_peps),
                 reinterpret_cast<qnpeps_device_dlenv*>(dlenv),
                 rowlogs,
                 nullptr
             ) == QNPEPS_OK;
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
        .dim_batch = 0,
        .stream = nullptr
    };
    ok = ok and qnpeps_sample(&scfg, &sample_args) == QNPEPS_OK;
    ok = ok
         and qnpeps_eloc_run(
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
             ) == QNPEPS_ELOC_OK;

    auto theta_own = std::vector<cf32>(static_cast<std::size_t>(dense));
    auto theta_m3q = std::vector<cf32>(static_cast<std::size_t>(dense));
    qnpeps::CuArray<double, 2> em{};
    double ev{};
    double es{};
    if (ok)
    {
        ok = qnpeps_e2e_minsr(
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
                 0.0,
                 reinterpret_cast<qnpeps_e2e_cbuf*>(theta),
                 em.data(),
                 &ev,
                 &es,
                 nullptr
             )
             == QNPEPS_E2E_OK;
        ck(cudaMemcpy(
               theta_own.data(), theta, theta_own.size() * sizeof(cf32), cudaMemcpyDeviceToHost
           ),
           "probe d2h own");
    }
    if (ok)
    {
        ck(cudaMemcpy(
               logq,
               m3_logq.data(),
               sizeof(double) * static_cast<std::size_t>(ns),
               cudaMemcpyHostToDevice
           ),
           "probe h2d m3 logq");
        ok = qnpeps_e2e_minsr(
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
                 0.0,
                 reinterpret_cast<qnpeps_e2e_cbuf*>(theta),
                 em.data(),
                 &ev,
                 &es,
                 nullptr
             )
             == QNPEPS_E2E_OK;
        ck(cudaMemcpy(
               theta_m3q.data(), theta, theta_m3q.size() * sizeof(cf32), cudaMemcpyDeviceToHost
           ),
           "probe d2h m3q");
    }
    if (ok)
    {
        double scale{1e-30};
        for (const cf32& v : theta_own)
        {
            scale = std::max(
                scale,
                std::sqrt(static_cast<double>(v.re) * v.re + static_cast<double>(v.im) * v.im)
            );
        }
        double m{0.0};
        for (auto k = static_cast<std::size_t>(0); k < theta_own.size(); ++k)
        {
            const double dr{static_cast<double>(theta_own[k].re) - theta_m3q[k].re};
            const double di{static_cast<double>(theta_own[k].im) - theta_m3q[k].im};
            m = std::max(m, std::sqrt(dr * dr + di * di) / scale);
        }
        feed_max_out = m;
        theta_m3q_out = theta_m3q;
    }
    if (ok and not m3_eloc.empty())
    {
        ck(cudaMemcpy(
               el,
               m3_eloc.data(),
               sizeof(double) * static_cast<std::size_t>(2 * ns),
               cudaMemcpyHostToDevice
           ),
           "probe h2d m3 eloc");
        auto theta_m3qe = std::vector<cf32>(static_cast<std::size_t>(dense));
        ok = qnpeps_e2e_minsr(
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
                 0.0,
                 reinterpret_cast<qnpeps_e2e_cbuf*>(theta),
                 em.data(),
                 &ev,
                 &es,
                 nullptr
             )
             == QNPEPS_E2E_OK;
        ck(cudaMemcpy(
               theta_m3qe.data(), theta, theta_m3qe.size() * sizeof(cf32), cudaMemcpyDeviceToHost
           ),
           "probe d2h m3qe");
        if (ok) theta_m3qe_out = theta_m3qe;
    }
    if (ok and not m3_eloc.empty() and not m3_logpsi.empty())
    {
        auto h_lp = std::vector<double>(static_cast<std::size_t>(2 * ns));
        auto h_rows = std::vector<cf32>(static_cast<std::size_t>(ns) * compact);
        auto h_gram = std::vector<cf32>(static_cast<std::size_t>(ns) * ns);
        ck(cudaMemcpy(h_lp.data(), lp, h_lp.size() * sizeof(double), cudaMemcpyDeviceToHost),
           "probe d2h lp");
        ck(cudaMemcpy(h_rows.data(), rows, h_rows.size() * sizeof(cf32), cudaMemcpyDeviceToHost),
           "probe d2h rows");
        ck(cudaMemcpy(h_gram.data(), gram, h_gram.size() * sizeof(cf32), cudaMemcpyDeviceToHost),
           "probe d2h gram");
        auto s = std::vector<double>(static_cast<std::size_t>(ns));
        for (auto j = 0; j < ns; ++j)
        {
            s[static_cast<std::size_t>(j)] = std::exp(
                m3_logpsi[static_cast<std::size_t>(j)] - h_lp[static_cast<std::size_t>(2 * j)]
            );
            h_lp[static_cast<std::size_t>(2 * j)] = m3_logpsi[static_cast<std::size_t>(j)];
        }
        for (auto j = 0; j < ns; ++j)
        {
            const float sj{static_cast<float>(s[static_cast<std::size_t>(j)])};
            for (auto k = static_cast<std::int64_t>(0); k < compact; ++k)
            {
                auto& v{h_rows[static_cast<std::size_t>(j) * compact + k]};
                v.re *= sj;
                v.im *= sj;
            }
        }
        for (auto a = 0; a < ns; ++a)
        {
            for (auto b = 0; b < ns; ++b)
            {
                const float sab{static_cast<float>(
                    s[static_cast<std::size_t>(a)] * s[static_cast<std::size_t>(b)]
                )};
                auto& v{h_gram[static_cast<std::size_t>(a) * ns + b]};
                v.re *= sab;
                v.im *= sab;
            }
        }
        ck(cudaMemcpy(lp, h_lp.data(), h_lp.size() * sizeof(double), cudaMemcpyHostToDevice),
           "probe h2d lp full");
        ck(cudaMemcpy(rows, h_rows.data(), h_rows.size() * sizeof(cf32), cudaMemcpyHostToDevice),
           "probe h2d rows full");
        ck(cudaMemcpy(gram, h_gram.data(), h_gram.size() * sizeof(cf32), cudaMemcpyHostToDevice),
           "probe h2d gram full");
        auto theta_m3full = std::vector<cf32>(static_cast<std::size_t>(dense));
        ok = qnpeps_e2e_minsr(
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
                 0.0,
                 reinterpret_cast<qnpeps_e2e_cbuf*>(theta),
                 em.data(),
                 &ev,
                 &es,
                 nullptr
             )
             == QNPEPS_E2E_OK;
        ck(cudaMemcpy(
               theta_m3full.data(),
               theta,
               theta_m3full.size() * sizeof(cf32),
               cudaMemcpyDeviceToHost
           ),
           "probe d2h m3full");
        if (ok) theta_m3full_out = theta_m3full;
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
    return ok;
}

inline auto compare_case(
    const QnpepsE2eConfig& cfg_in,
    int ns,
    const std::vector<float>& hpeps,
    const GenPeps&,
    const char* ms3_bin
) -> int
{
    const int lx{cfg_in.lx};
    const int ly{cfg_in.ly};
    const int D{cfg_in.dim_bond};
    const int dim_phys{cfg_in.dim_phys};
    const int sites{lx * ly};
    const int m3_seed{1};
    const double rel{1.0e-6};
    auto rr_env{std::getenv("E2E_MS3_REL_REG")};
    const double rel_reg{rr_env and rr_env[0] ? std::atof(rr_env) : 0.1};
    const double abs_cut{0.0};

    QnpepsE2eConfig cfg{cfg_in};
    cfg.seed = static_cast<std::uint64_t>(m3_seed);

    std::int64_t dense{0};
    std::int64_t compact{0};
    qnpeps_e2e_dense_count(&cfg, &dense);
    qnpeps_e2e_compact_count(&cfg, &compact);

    void* d_peps{};
    ck(cudaMalloc(&d_peps, hpeps.size() * sizeof(float)), "ms3 malloc peps");
    ck(cudaMemcpy(d_peps, hpeps.data(), hpeps.size() * sizeof(float), cudaMemcpyHostToDevice),
       "ms3 cpy peps");

    const std::vector<Bond> bonds{bonds_nn(lx, ly, 1.0)};
    const std::vector<QnpepsElocDiagBond> diag{diag_bonds_of(bonds, ly)};
    const std::vector<QnpepsElocFlipTerm> flips{masked_flips_of(bonds, ly)};
    QnpepsElocTermTable tt{};
    tt.n_diag = static_cast<int32_t>(diag.size());
    tt.diag = diag.data();
    tt.n_flip = static_cast<int32_t>(flips.size());
    tt.flip = flips.data();

    StepOut ours{};
    const qnpeps_e2e_status ss{run_step(cfg, d_peps, ns, tt, rel, abs_cut, dense, sites, ours)};
    StepOut ours_reg{};
    const qnpeps_e2e_status sr{
        run_step(cfg, d_peps, ns, tt, rel_reg, abs_cut, dense, sites, ours_reg)
    };
    if (ss != QNPEPS_E2E_OK or sr != QNPEPS_E2E_OK)
    {
        std::printf("[gate_step] MS3 our-step error: %s\n", qnpeps_e2e_strerror(ss ? ss : sr));
        cudaFree(d_peps);
        return 1;
    }

    auto work{std::getenv("MS3_WORK")};
    const std::string wd{work and work[0] ? work : "/tmp"};
    const std::string tag{
        std::to_string(lx) + "_" + std::to_string(ly) + "_" + std::to_string(D) + "_ns"
        + std::to_string(ns)
    };
    const std::string fix{wd + "/fixture_" + tag + ".txt"};
    const std::string ds{wd + "/samples_" + tag + ".txt"};
    const std::string de{wd + "/eloc_" + tag + ".txt"};
    const std::string dt{wd + "/thetadot_" + tag + ".bin"};
    const std::string dt_reg{wd + "/thetadot_" + tag + "_reg.bin"};
    const std::string lg{wd + "/ms3run_" + tag + ".log"};
    const std::string lg_reg{wd + "/ms3run_" + tag + "_reg.log"};

    if (not write_fixture(fix, lx, ly, D, m3_seed, dim_phys, hpeps))
    {
        std::printf("[gate_step] MS3 cannot write fixture %s\n", fix.c_str());
        cudaFree(d_peps);
        return 1;
    }

    const auto run_ms3{
        [&](double cut, const std::string& dtp, const std::string& lgp, bool with_stage_dumps)
            -> void
        {
            char cmd[2048]{};
            std::snprintf(
                cmd,
                sizeof(cmd),
                "%s --lx=%d --ly=%d --d=%d --D=%d --dcd=%d --sd=%d --chieo=%d --chitop=%d "
                "--m=%d --batches=1 --meo=%d --seed=%d --eo=1 --okout=1 --update=1 --dl_rf=1 "
                "--solvercut=%.17g --validate=0 --mode=full --graph=0 --loadpeps=%s "
                "%s%s %s%s --dump_thetadot=%s > %s 2>&1",
                ms3_bin,
                lx,
                ly,
                dim_phys,
                D,
                cfg.chi_dl,
                cfg.chi_s,
                cfg.chi_eo,
                cfg.chi_s,
                ns,
                cfg.meo,
                m3_seed,
                cut,
                fix.c_str(),
                with_stage_dumps ? "--dump_samples=" : "",
                with_stage_dumps ? ds.c_str() : "",
                with_stage_dumps ? "--dump_eloc=" : "",
                with_stage_dumps ? de.c_str() : "",
                dtp.c_str(),
                lgp.c_str()
            );
            std::printf("[gate_step] MS3 run: %s\n", cmd);
            const int rc{std::system(cmd)};
            if (rc != 0) std::printf("[gate_step] MS3 binary exit=%d (see %s)\n", rc, lgp.c_str());
        }
    };
    run_ms3(rel, dt, lg, true);
    run_ms3(rel_reg, dt_reg, lg_reg, false);

    const M3Samples ms{parse_samples(ds, lx, ly, ns)};
    const M3Eloc me{parse_eloc(de, lx, ns)};
    std::vector<double> m3_eloc{};
    std::vector<double> m3_logpsi{};
    if (me.ok)
    {
        m3_eloc.assign(static_cast<std::size_t>(2 * ns), 0.0);
        for (auto i = 0; i < ns; ++i)
        {
            m3_eloc[static_cast<std::size_t>(2 * i)] = me.e_re[static_cast<std::size_t>(i)];
            m3_eloc[static_cast<std::size_t>(2 * i + 1)] = me.e_im[static_cast<std::size_t>(i)];
        }
        m3_logpsi = me.logpsi;
    }

    int fails{0};

    {
        bool pass{ms.ok == 1};
        int samp_mism{0};
        double logq_max{0.0};
        if (ms.ok)
        {
            for (auto k = static_cast<std::size_t>(0);
                 k < ms.spins.size() and k < ours.samples.size();
                 ++k)
                if (ms.spins[k] != ours.samples[k]) ++samp_mism;
            for (auto i = 0; i < ns; ++i)
            {
                logq_max = std::max(
                    logq_max,
                    std::abs(
                        ms.logq[static_cast<std::size_t>(i)]
                        - ours.logq[static_cast<std::size_t>(i)]
                    )
                );
            }
            pass = (samp_mism == 0) and (logq_max <= 5e-2);
        }
        char note[128]{};
        std::snprintf(note, sizeof(note), "sampMism=%d logqMax=%.3e", samp_mism, logq_max);
        set_line("a_samples", pass, note);
        if (not pass) ++fails;
    }

    {
        bool pass{me.ok == 1};
        double e_max{0.0};
        double lp_max{0.0};
        if (me.ok)
        {
            for (auto i = 0; i < ns; ++i)
            {
                const double er{ours.eloc[static_cast<std::size_t>(2 * i)]};
                const double ei{ours.eloc[static_cast<std::size_t>(2 * i + 1)]};
                const double mr{me.e_re[static_cast<std::size_t>(i)]};
                const double mi{me.e_im[static_cast<std::size_t>(i)]};
                const double sc{std::max(1.0, std::abs(mr) + std::abs(mi))};
                e_max = std::max(e_max, (std::abs(er - mr) + std::abs(ei - mi)) / sc);
                const double olp{ours.logpsi[static_cast<std::size_t>(2 * i)]};
                const double mlp{me.logpsi[static_cast<std::size_t>(i)]};
                lp_max = std::max(lp_max, std::abs(olp - mlp) / std::max(1.0, std::abs(mlp)));
            }
            pass = (e_max <= 1e-3) and (lp_max <= 1e-3);
        }
        char note[128]{};
        std::snprintf(note, sizeof(note), "elocMax=%.3e logpsiMax=%.3e", e_max, lp_max);
        set_line("b_eloc", pass, note);
        if (not pass) ++fails;
    }

    {
        std::vector<float> td{};
        const bool have{read_thetadot(dt, td)};
        int ok_site{0};
        bool pass{have};
        double td_max{0.0};
        if (have)
        {
            td_max = theta_vs_m3(ours.theta, td, lx, ly, D, dim_phys, ok_site);
            pass = td_max <= 5e-3;
        }
        char note[96]{};
        std::snprintf(note, sizeof(note), "thetaMax=%.3e okSite=%d", td_max, ok_site);
        set_line("c_thetadot", pass, note);
        if (not pass) ++fails;
    }

    {
        std::vector<float> td{};
        const bool have{read_thetadot(dt_reg, td)};
        int ok_site{0};
        bool pass{have};
        double td_max{0.0};
        if (have)
        {
            td_max = theta_vs_m3(ours_reg.theta, td, lx, ly, D, dim_phys, ok_site);
            pass = td_max <= 1e-2;
        }
        char note[96]{};
        std::snprintf(note, sizeof(note), "thetaMax=%.3e okSite=%d", td_max, ok_site);
        set_line("c_thetadot_reg", pass, note);
        if (not pass) ++fails;
    }

    {
        double feed_reg{-1.0};
        double feed_unreg{-1.0};
        std::vector<cf32> theta_m3q_reg{};
        std::vector<cf32> theta_m3qe_reg{};
        std::vector<cf32> theta_m3full_reg{};
        std::vector<cf32> theta_m3q_unreg{};
        std::vector<cf32> theta_m3qe_unreg{};
        std::vector<cf32> theta_m3full_unreg{};
        bool ok{ms.ok == 1 and me.ok == 1};
        if (ok)
        {
            ok = logq_feed_probe(
                cfg,
                d_peps,
                ns,
                tt,
                ms.logq,
                m3_eloc,
                m3_logpsi,
                dense,
                compact,
                rel_reg,
                feed_reg,
                theta_m3q_reg,
                theta_m3qe_reg,
                theta_m3full_reg
            );
        }
        if (ok)
        {
            ok = logq_feed_probe(
                cfg,
                d_peps,
                ns,
                tt,
                ms.logq,
                m3_eloc,
                m3_logpsi,
                dense,
                compact,
                rel,
                feed_unreg,
                theta_m3q_unreg,
                theta_m3qe_unreg,
                theta_m3full_unreg
            );
        }
        {
            char note[128]{};
            std::snprintf(
                note, sizeof(note), "reg=%.3e unreg=%.3e (report-only)", feed_reg, feed_unreg
            );
            set_report_line("logq_feed", note);
        }
        if (ok)
        {
            std::vector<float> td{};
            int ok_site{0};
            const bool have_r{read_thetadot(dt_reg, td)};
            double m3q_max{-1.0};
            double m3qe_max{-1.0};
            double m3full_max{-1.0};
            if (have_r)
            {
                m3q_max = theta_vs_m3(theta_m3q_reg, td, lx, ly, D, dim_phys, ok_site);
                m3qe_max = theta_vs_m3(theta_m3qe_reg, td, lx, ly, D, dim_phys, ok_site);
                m3full_max = theta_vs_m3(theta_m3full_reg, td, lx, ly, D, dim_phys, ok_site);
            }
            {
                char note[96]{};
                std::snprintf(note, sizeof(note), "thetaMax=%.3e (logq-matched, report)", m3q_max);
                set_report_line("c_reg_matched", note);
            }
            {
                char note[96]{};
                std::snprintf(
                    note,
                    sizeof(note),
                    "thetaMax=%.3e (logq+eloc-matched, report, rel_reg=%g)",
                    m3qe_max,
                    rel_reg
                );
                set_report_line("c_reg_matched2", note);
            }
            {
                char note[96]{};
                std::snprintf(
                    note,
                    sizeof(note),
                    "thetaMax=%.3e (logq+eloc+logpsi/rows-matched, report, rel_reg=%g)",
                    m3full_max,
                    rel_reg
                );
                set_report_line("c_reg_matched3", note);
            }

            std::vector<float> tdu{};
            int ok_site_u{0};
            const bool have_u{read_thetadot(dt, tdu)};
            double td_max_u{-1.0};
            if (have_u)
                td_max_u = theta_vs_m3(theta_m3q_unreg, tdu, lx, ly, D, dim_phys, ok_site_u);
            char note_u[96]{};
            std::snprintf(
                note_u, sizeof(note_u), "thetaMax=%.3e (logq-matched, report-only)", td_max_u
            );
            set_report_line("c_unreg_matched", note_u);
        }
    }

    cudaFree(d_peps);
    std::printf("[gate_step] MS3 stages done (fails=%d) tag=%s\n", fails, tag.c_str());
    return fails;
}

}

#endif
