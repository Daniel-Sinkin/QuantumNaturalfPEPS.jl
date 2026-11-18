#include "../../core/types.cuh"
#include "capi/qnpeps.h"
#include "dans_qnpeps_e2e.h"
#include "dans_qnpeps_eloc.h"
#include "e2e/common.cuh"
#include "gate_fixture.cuh"
#include "term_builders.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <numbers>
#include <random>
#include <string>
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
    int ai{}, aj{}, bi{}, bj{};
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

auto heisenberg_of(const std::vector<Bond>& bonds, int ly) -> eo_test::TermTableBuilder
{
    eo_test::TermTableBuilder terms{};
    for (const Bond& b : bonds)
        terms.add_heisenberg_bond(site0(b.ai, b.aj, ly), site0(b.bi, b.bj, ly), b.j);
    return terms;
}

auto ck(cudaError_t e, const char* what) -> void
{
    if (e != cudaSuccess) std::printf("[gate_node] CUDA %s: %s\n", what, cudaGetErrorString(e));
}

template <class T>
auto dmalloc(std::size_t n) -> T*
{
    void* p{};
    ck(cudaMalloc(&p, (n < 1 ? 1 : n) * sizeof(T)), "malloc");
    return static_cast<T*>(p);
}

auto scfg_of(const QnpepsE2eConfig& c) -> QnpepsConfig
{
    QnpepsConfig s{};
    s.struct_size = sizeof(QnpepsConfig);
    s.lx = c.lx;
    s.ly = c.ly;
    s.dim_phys = c.dim_phys;
    s.dim_bond = c.dim_bond;
    s.chi_s = c.chi_s;
    s.chi_dl = c.chi_dl;
    s.seed = c.seed;
    s.sampling_mode = c.sampling_mode;
    s.chi_c = c.contract_dim;
    return s;
}

auto lcfg_of(const QnpepsE2eConfig& c) -> QnpepsElocConfig
{
    QnpepsElocConfig e{};
    e.struct_size = sizeof(QnpepsElocConfig);
    e.lx = c.lx;
    e.ly = c.ly;
    e.dim_phys = c.dim_phys;
    e.dim_bond = c.dim_bond;
    e.chi_eo = c.chi_eo;
    e.meo = qn_e2e::local_energy_wave_lanes(c);
    e.truncation_route = c.truncation_route;
    return e;
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
    std::vector<std::int64_t> epoch{};
    qnpeps_status status{QNPEPS_OK};
};

auto theta_eq(const std::vector<cf32>& a, const std::vector<cf32>& b) -> bool
{
    if (a.size() != b.size()) return false;
    for (std::size_t k{0}; k < a.size(); ++k)
        if (a[k].re != b[k].re or a[k].im != b[k].im) return false;
    return true;
}

auto all_eq(const StepOut& a, const StepOut& b, char* note, std::size_t n) -> bool
{
    const bool th{theta_eq(a.theta, b.theta)};
    const bool st{
        a.e_mean[0] == b.e_mean[0] and a.e_mean[1] == b.e_mean[1] and a.e_var == b.e_var
        and a.ess == b.ess
    };
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

constexpr double k_wave_energy_tolerance{1.0e-5};
constexpr double k_wave_logpsi_tolerance{1.0e-5};
constexpr double k_wave_statistics_tolerance{1.0e-4};
constexpr double k_wave_theta_tolerance{1.0e-3};

auto relative_gap(double observed, double expected) -> double
{
    return std::abs(observed - expected) / std::max(1.0, std::abs(expected));
}

auto logpsi_gap(const std::vector<double>& a, const std::vector<double>& b) -> double
{
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    double gap{};
    for (std::size_t k{0}; k + 1 < a.size(); k += 2)
    {
        const double phase{std::remainder(a[k + 1] - b[k + 1], 2.0 * std::numbers::pi)};
        gap = std::max({gap, std::abs(a[k] - b[k]), std::abs(phase)});
    }
    return gap;
}

auto energy_gap(const std::vector<double>& a, const std::vector<double>& b) -> double
{
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    double gap{};
    for (std::size_t k{0}; k + 1 < a.size(); k += 2)
    {
        const double scale{std::max(1.0, std::hypot(b[k], b[k + 1]))};
        gap = std::max(gap, std::hypot(a[k] - b[k], a[k + 1] - b[k + 1]) / scale);
    }
    return gap;
}

auto theta_gap(const std::vector<cf32>& a, const std::vector<cf32>& b) -> double
{
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    double difference{};
    double norm{};
    for (std::size_t k{0}; k < a.size(); ++k)
    {
        const double re{static_cast<double>(a[k].re) - static_cast<double>(b[k].re)};
        const double im{static_cast<double>(a[k].im) - static_cast<double>(b[k].im)};
        difference += re * re + im * im;
        norm += static_cast<double>(b[k].re) * b[k].re + static_cast<double>(b[k].im) * b[k].im;
    }
    return std::sqrt(difference) / std::max(std::sqrt(norm), std::numeric_limits<double>::min());
}

auto close_eq(const StepOut& a, const StepOut& b, char* note, std::size_t n) -> bool
{
    const double th{theta_gap(a.theta, b.theta)};
    const double st{std::max(
        {relative_gap(a.e_mean[0], b.e_mean[0]),
         relative_gap(a.e_mean[1], b.e_mean[1]),
         relative_gap(a.e_var, b.e_var),
         relative_gap(a.ess, b.ess)}
    )};
    const double lp{logpsi_gap(a.logpsi, b.logpsi)};
    const double el{energy_gap(a.eloc, b.eloc)};
    const bool inputs{a.samples == b.samples and a.logq == b.logq and a.loggauge == b.loggauge};
    std::snprintf(
        note, n, "inputs=%d theta=%.1e stats=%.1e logpsi=%.1e eloc=%.1e", inputs, th, st, lp, el
    );
    return inputs and th <= k_wave_theta_tolerance and st <= k_wave_statistics_tolerance
           and lp <= k_wave_logpsi_tolerance and el <= k_wave_energy_tolerance;
}

auto set_line(const char* name, bool pass, const char* note) -> void
{
    std::printf(
        "[gate_node] SET %-26s => %s%s%s\n",
        name,
        pass ? "PASS" : "FAIL",
        note and note[0] ? "  " : "",
        note ? note : ""
    );
}

auto ref_step(
    const QnpepsE2eConfig& cfg,
    const void* d_peps,
    int ns,
    const QnpepsElocTermTable& tt,
    double rel,
    double abs_cut,
    std::int64_t base,
    std::int64_t dim_batch,
    std::int64_t dense,
    std::int64_t compact,
    int sites,
    StepOut& out
) -> qnpeps_status
{
    const QnpepsConfig scfg{scfg_of(cfg)};
    const QnpepsElocConfig lcfg{lcfg_of(cfg)};
    const std::int64_t dlenv_bytes{qnpeps_dlenv_bytes(&scfg)};
    const std::int64_t scratch_bytes{
        qnpeps_sample_scratch_bytes(&scfg, static_cast<std::uint64_t>(dim_batch))
    };

    auto dlenv{dmalloc<std::uint8_t>(static_cast<std::size_t>(dlenv_bytes))};
    auto rowlogs{dmalloc<double>(static_cast<std::size_t>(cfg.lx - 1))};
    auto scratch{dmalloc<std::uint8_t>(static_cast<std::size_t>(scratch_bytes))};
    auto samples{dmalloc<std::uint8_t>(static_cast<std::size_t>(ns) * sites)};
    auto logq{dmalloc<double>(static_cast<std::size_t>(ns))};
    auto logg{dmalloc<double>(static_cast<std::size_t>(ns))};
    auto logpsi{dmalloc<double>(static_cast<std::size_t>(2 * ns))};
    auto eloc{dmalloc<double>(static_cast<std::size_t>(2 * ns))};
    auto rows{dmalloc<cf32>(static_cast<std::size_t>(ns) * compact)};
    auto gram{dmalloc<cf32>(static_cast<std::size_t>(ns) * ns)};
    auto theta{dmalloc<cf32>(static_cast<std::size_t>(dense))};

    qnpeps_status rc{QNPEPS_OK};
    if (qnpeps_build_dlenv(
            &scfg,
            static_cast<const qnpeps_device_peps*>(d_peps),
            reinterpret_cast<qnpeps_device_dlenv*>(dlenv),
            rowlogs,
            nullptr
        )
        != QNPEPS_OK)
        rc = QNPEPS_ERR_INTERNAL;
    cudaDeviceSynchronize();
    const QnpepsSampleArgs sample_args_a{
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
        .batch_base = static_cast<std::uint64_t>(base),
        .dim_batch = static_cast<std::uint64_t>(dim_batch),
        .stream = nullptr
    };
    if (rc == QNPEPS_OK and qnpeps_sample(&scfg, &sample_args_a) != QNPEPS_OK)
        rc = QNPEPS_ERR_INTERNAL;
    if (rc == QNPEPS_OK
        and qnpeps_eloc_run(
                &lcfg,
                static_cast<const qnpeps_eloc_cbuf*>(d_peps),
                samples,
                ns,
                &tt,
                logpsi,
                eloc,
                reinterpret_cast<qnpeps_eloc_cbuf*>(rows),
                nullptr,
                reinterpret_cast<qnpeps_eloc_cbuf*>(gram),
                0.0,
                nullptr
            ) != QNPEPS_OK)
        rc = QNPEPS_ERR_INTERNAL;
    if (rc == QNPEPS_OK)
    {
        rc = qnpeps_e2e_minsr(
            &cfg,
            ns,
            samples,
            logpsi,
            eloc,
            logq,
            reinterpret_cast<qnpeps_e2e_cbuf*>(gram),
            reinterpret_cast<qnpeps_e2e_cbuf*>(rows),
            nullptr,
            0,
            rel,
            abs_cut,
            reinterpret_cast<qnpeps_e2e_cbuf*>(theta),
            out.e_mean.data(),
            &out.e_var,
            &out.ess,
            nullptr
        );
    }

    if (rc == QNPEPS_OK)
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
           "ref theta");
        ck(cudaMemcpy(out.samples.data(), samples, out.samples.size(), cudaMemcpyDeviceToHost),
           "ref samp");
        ck(cudaMemcpy(
               out.logq.data(), logq, out.logq.size() * sizeof(double), cudaMemcpyDeviceToHost
           ),
           "ref logq");
        ck(cudaMemcpy(
               out.loggauge.data(),
               logg,
               out.loggauge.size() * sizeof(double),
               cudaMemcpyDeviceToHost
           ),
           "ref logg");
        ck(cudaMemcpy(
               out.logpsi.data(), logpsi, out.logpsi.size() * sizeof(double), cudaMemcpyDeviceToHost
           ),
           "ref lp");
        ck(cudaMemcpy(
               out.eloc.data(), eloc, out.eloc.size() * sizeof(double), cudaMemcpyDeviceToHost
           ),
           "ref el");
    }
    out.status = rc;
    cudaFree(dlenv);
    cudaFree(rowlogs);
    cudaFree(scratch);
    cudaFree(samples);
    cudaFree(logq);
    cudaFree(logg);
    cudaFree(logpsi);
    cudaFree(eloc);
    cudaFree(rows);
    cudaFree(gram);
    cudaFree(theta);
    return rc;
}

auto node_step_capture(
    qnpeps_e2e_node* node,
    int ns,
    double rel,
    double abs_cut,
    std::int64_t dense,
    int sites,
    StepOut& out
) -> qnpeps_status
{
    auto theta{dmalloc<cf32>(static_cast<std::size_t>(dense))};
    auto samp{dmalloc<std::uint8_t>(static_cast<std::size_t>(ns) * sites)};
    auto logq{dmalloc<double>(static_cast<std::size_t>(ns))};
    auto logg{dmalloc<double>(static_cast<std::size_t>(ns))};
    auto lp{dmalloc<double>(static_cast<std::size_t>(2 * ns))};
    auto el{dmalloc<double>(static_cast<std::size_t>(2 * ns))};
    out.epoch.assign(static_cast<std::size_t>(ns), 0);
    const qnpeps_status st{qnpeps_e2e_node_step(
        node,
        ns,
        rel,
        abs_cut,
        reinterpret_cast<qnpeps_e2e_cbuf*>(theta),
        out.e_mean.data(),
        &out.e_var,
        &out.ess,
        samp,
        logq,
        logg,
        lp,
        el,
        nullptr,
        out.epoch.data()
    )};
    out.status = st;
    if (st == QNPEPS_OK)
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
           "nd theta");
        ck(cudaMemcpy(out.samples.data(), samp, out.samples.size(), cudaMemcpyDeviceToHost),
           "nd samp");
        ck(cudaMemcpy(
               out.logq.data(), logq, out.logq.size() * sizeof(double), cudaMemcpyDeviceToHost
           ),
           "nd logq");
        ck(cudaMemcpy(
               out.loggauge.data(),
               logg,
               out.loggauge.size() * sizeof(double),
               cudaMemcpyDeviceToHost
           ),
           "nd logg");
        ck(cudaMemcpy(
               out.logpsi.data(), lp, out.logpsi.size() * sizeof(double), cudaMemcpyDeviceToHost
           ),
           "nd lp");
        ck(cudaMemcpy(
               out.eloc.data(), el, out.eloc.size() * sizeof(double), cudaMemcpyDeviceToHost
           ),
           "nd el");
    }
    cudaFree(theta);
    cudaFree(samp);
    cudaFree(logq);
    cudaFree(logg);
    cudaFree(lp);
    cudaFree(el);
    return st;
}

auto euler_update(std::vector<float>& hpeps, const std::vector<cf32>& theta_dot, double lr) -> void
{
    const std::size_t n{theta_dot.size()};
    for (std::size_t i{0}; i < n; ++i)
    {
        hpeps[2 * i] += static_cast<float>(lr * theta_dot[i].re);
        hpeps[2 * i + 1] += static_cast<float>(lr * theta_dot[i].im);
    }
}

auto boundary_dim(int length, int position, int bond) -> int
{
    return position <= 0 or position >= length ? 1 : bond;
}

auto separately_rounded_f32(float value, float rate, float direction) -> float
{
    volatile float product{rate * direction};
    volatile float result{value + product};
    return result;
}

auto separately_rounded_f64(double value, double rate, float direction) -> double
{
    volatile double product{rate * static_cast<double>(direction)};
    volatile double result{value + product};
    return result;
}

template <class Scalar, class Update>
auto layout_update(
    const QnpepsE2eConfig& cfg,
    std::vector<Scalar> state,
    const std::vector<cf32>& theta_dot,
    Update update
) -> std::vector<Scalar>
{
    std::int64_t site_offset{};
    for (int row{}; row < cfg.lx; ++row)
    {
        for (int column{}; column < cfg.ly; ++column)
        {
            const int dw{boundary_dim(cfg.ly, column, cfg.dim_bond)};
            const int ds{boundary_dim(cfg.lx, row + 1, cfg.dim_bond)};
            const int de{boundary_dim(cfg.ly, column + 1, cfg.dim_bond)};
            const int dn{boundary_dim(cfg.lx, row, cfg.dim_bond)};
            const int dp{cfg.dim_phys};
            const std::int64_t site_count{static_cast<std::int64_t>(dw) * ds * de * dn * dp};
            for (int east{}; east < de; ++east)
            {
                for (int south{}; south < ds; ++south)
                {
                    for (int north{}; north < dn; ++north)
                    {
                        for (int west{}; west < dw; ++west)
                        {
                            for (int physical{}; physical < dp; ++physical)
                            {
                                const std::int64_t theta_local{
                                    physical
                                    + static_cast<std::int64_t>(dp)
                                          * (west
                                             + static_cast<std::int64_t>(dw)
                                                   * (north
                                                      + static_cast<std::int64_t>(dn)
                                                            * (south
                                                               + static_cast<std::int64_t>(ds)
                                                                     * east)))
                                };
                                const std::int64_t fixture_local{
                                    west
                                    + static_cast<std::int64_t>(dw)
                                          * (south
                                             + static_cast<std::int64_t>(ds)
                                                   * (east
                                                      + static_cast<std::int64_t>(de)
                                                            * (north
                                                               + static_cast<std::int64_t>(dn)
                                                                     * physical)))
                                };
                                const std::size_t source{
                                    static_cast<std::size_t>(site_offset + theta_local)
                                };
                                const std::size_t destination{
                                    static_cast<std::size_t>(site_offset + fixture_local)
                                };
                                update(state, destination, theta_dot[source]);
                            }
                        }
                    }
                }
            }
            site_offset += site_count;
        }
    }
    return state;
}

auto layout_update_f32(
    const QnpepsE2eConfig& cfg,
    const std::vector<float>& initial,
    const std::vector<cf32>& theta_dot,
    double learning_rate
) -> std::vector<float>
{
    const float rate{static_cast<float>(learning_rate)};
    return layout_update(
        cfg,
        initial,
        theta_dot,
        [rate](std::vector<float>& state, std::size_t index, const cf32& direction)
        {
            state[2 * index] = separately_rounded_f32(state[2 * index], rate, direction.re);
            state[2 * index + 1] = separately_rounded_f32(state[2 * index + 1], rate, direction.im);
        }
    );
}

auto layout_update_f64(
    const QnpepsE2eConfig& cfg,
    const std::vector<double>& initial,
    const std::vector<cf32>& theta_dot,
    double learning_rate
) -> std::vector<double>
{
    return layout_update(
        cfg,
        initial,
        theta_dot,
        [learning_rate](std::vector<double>& state, std::size_t index, const cf32& direction)
        {
            state[2 * index] =
                separately_rounded_f64(state[2 * index], learning_rate, direction.re);
            state[2 * index + 1] =
                separately_rounded_f64(state[2 * index + 1], learning_rate, direction.im);
        }
    );
}

auto quantize_f64(const std::vector<double>& state) -> std::vector<float>
{
    auto result = std::vector<float>(state.size());
    for (std::size_t index{}; index < state.size(); ++index)
        result[index] = static_cast<float>(state[index]);
    return result;
}

template <class T>
auto bytes_equal(const std::vector<T>& left, const std::vector<T>& right) -> bool
{
    return left.size() == right.size()
           and std::memcmp(left.data(), right.data(), left.size() * sizeof(T)) == 0;
}

auto ess_from(const std::vector<double>& logpsi, const std::vector<double>& logq, int ns) -> double
{
    auto lr = std::vector<double>(static_cast<std::size_t>(ns));
    for (int j{0}; j < ns; ++j)
    {
        lr[static_cast<std::size_t>(j)] =
            2.0 * logpsi[static_cast<std::size_t>(2 * j)] - logq[static_cast<std::size_t>(j)];
    }
    double m{lr[0]};
    for (double v : lr)
        m = v > m ? v : m;
    double se{0.0};
    for (double v : lr)
        se += std::exp(v - m);
    const double logz{m + std::log(se) - std::log(static_cast<double>(ns))};
    double sw{0.0};
    double sw2{0.0};
    double sum{0.0};
    auto w = std::vector<double>(static_cast<std::size_t>(ns));
    for (int j{0}; j < ns; ++j)
    {
        w[static_cast<std::size_t>(j)] = std::exp(lr[static_cast<std::size_t>(j)] - logz);
        sum += w[static_cast<std::size_t>(j)];
    }
    const double wmean{sum / static_cast<double>(ns)};
    for (int j{0}; j < ns; ++j)
    {
        const double wv{w[static_cast<std::size_t>(j)] / wmean};
        sw += wv;
        sw2 += wv * wv;
    }
    return sw * sw / sw2;
}

auto make_hpeps(int lx, int ly, int D, int chi, std::uint64_t seed) -> std::vector<float>
{
    GenPeps g{make_peps(lx, ly, D, 2, chi, 2, 0.02, seed)};
    const std::size_t total{g.flat.size()};
    auto hpeps = std::vector<float>(2 * total);
    auto irng = std::mt19937_64(seed ^ 0x5D5D5D5Dull);
    auto igauss = std::normal_distribution<double>(0.0, 1.0);
    for (std::size_t k{0}; k < total; ++k)
    {
        hpeps[2 * k] = static_cast<float>(g.flat[k]);
        hpeps[2 * k + 1] = static_cast<float>(0.35 * igauss(irng));
    }
    return hpeps;
}

auto make_cfg(int lx, int ly, int D, int chi, int meo, std::uint64_t seed) -> QnpepsE2eConfig
{
    QnpepsE2eConfig cfg{};
    cfg.struct_size = sizeof(QnpepsE2eConfig);
    cfg.lx = lx;
    cfg.ly = ly;
    cfg.dim_phys = 2;
    cfg.dim_bond = D;
    cfg.chi_s = chi;
    cfg.chi_dl = chi;
    cfg.chi_eo = chi;
    cfg.meo = meo;
    cfg.seed = seed;
    return cfg;
}

struct Fix
{
    QnpepsE2eConfig cfg{};
    std::vector<float> hpeps0{};
    eo_test::TermTableBuilder terms{};
    QnpepsElocTermTable tt{};
    std::int64_t dense{};
    std::int64_t compact{};
    int sites{};
};

auto build_fixture(int lx, int ly, int D, int chi, int meo) -> Fix
{
    Fix f{};
    const std::uint64_t seed{
        0xE2E4A5Eull ^ (static_cast<std::uint64_t>(lx) * 131 + ly * 17 + D * 7 + chi * 3)
    };
    f.cfg = make_cfg(lx, ly, D, chi, meo, seed);
    f.hpeps0 = make_hpeps(lx, ly, D, chi, seed);
    f.sites = lx * ly;
    qnpeps_e2e_dense_count(&f.cfg, &f.dense);
    qnpeps_e2e_compact_count(&f.cfg, &f.compact);
    const std::vector<Bond> bonds{bonds_nn(lx, ly, 1.0)};
    f.terms = heisenberg_of(bonds, ly);
    f.tt = f.terms.table();
    return f;
}

auto run_schedule(
    const Fix& f,
    int gpus,
    int ns,
    std::int64_t ns_cap,
    std::int64_t ns_ahead,
    std::int64_t dim_batch,
    double rel,
    double abs_cut,
    double lr,
    int iters,
    std::vector<StepOut>& outs
) -> qnpeps_status
{
    auto d_theta{dmalloc<cf32>(static_cast<std::size_t>(f.dense))};
    std::vector<float> hpeps{f.hpeps0};

    qnpeps_e2e_node* node{};
    const qnpeps_status cs{
        qnpeps_e2e_node_create(&f.cfg, gpus, ns_cap, ns_ahead, dim_batch, 0, &f.tt, &node)
    };
    if (cs != QNPEPS_OK)
    {
        cudaFree(d_theta);
        return cs;
    }
    qnpeps_status rc{QNPEPS_OK};
    for (int k{0}; k < iters and rc == QNPEPS_OK; ++k)
    {
        ck(cudaMemcpy(d_theta, hpeps.data(), hpeps.size() * sizeof(float), cudaMemcpyHostToDevice),
           "sched upload");
        rc = qnpeps_e2e_node_submit_theta(node, d_theta);
        if (rc != QNPEPS_OK) break;
        StepOut o{};
        rc = node_step_capture(node, ns, rel, abs_cut, f.dense, f.sites, o);
        outs.push_back(o);
        if (rc != QNPEPS_OK) break;
        euler_update(hpeps, o.theta, lr);
    }
    qnpeps_e2e_node_destroy(node);
    cudaFree(d_theta);
    return rc;
}

auto case_a(
    const Fix& f,
    int gpus,
    int ns,
    double rel,
    double abs_cut,
    double lr,
    const char* name,
    bool is_bitwise
) -> int
{
    int fails{0};
    const std::int64_t ns_cap{static_cast<std::int64_t>(ns) + 8};
    auto d_theta{dmalloc<cf32>(static_cast<std::size_t>(f.dense))};
    std::vector<float> hpeps{f.hpeps0};
    qnpeps_e2e_node* node{};
    const qnpeps_status cs{qnpeps_e2e_node_create(&f.cfg, gpus, ns_cap, 0, ns, 0, &f.tt, &node)};
    if (cs != QNPEPS_OK)
    {
        std::printf("[gate_node] case_a create err %s\n", qnpeps_last_error());
        cudaFree(d_theta);
        return 1;
    }
    for (int k{0}; k < 3; ++k)
    {
        ck(cudaMemcpy(d_theta, hpeps.data(), hpeps.size() * sizeof(float), cudaMemcpyHostToDevice),
           "a upload");
        qnpeps_e2e_node_submit_theta(node, d_theta);
        StepOut nd{};
        const qnpeps_status ns_st{node_step_capture(node, ns, rel, abs_cut, f.dense, f.sites, nd)};
        StepOut rf{};
        const qnpeps_status rf_st{
            ref_step(f.cfg, d_theta, ns, f.tt, rel, abs_cut, k, ns, f.dense, f.compact, f.sites, rf)
        };
        char note[160]{};
        char label[48]{};
        std::snprintf(label, sizeof(label), "%s_iter%d_base%d_vs_ref", name, k + 1, k);
        if (ns_st != QNPEPS_OK or rf_st != QNPEPS_OK)
        {
            std::snprintf(
                note, sizeof(note), "node=%s ref=%s", qnpeps_last_error(), qnpeps_last_error()
            );
            set_line(label, false, note);
            ++fails;
        }
        else
        {
            const bool pass{
                is_bitwise ? all_eq(nd, rf, note, sizeof(note))
                           : close_eq(nd, rf, note, sizeof(note))
            };
            set_line(label, pass, note);
            if (not pass) ++fails;
            euler_update(hpeps, nd.theta, lr);
        }
        if (ns_st != QNPEPS_OK) break;
    }
    qnpeps_e2e_node_destroy(node);
    cudaFree(d_theta);
    return fails;
}

auto determinism_case(
    const Fix& f,
    int gpus,
    int ns,
    std::int64_t ns_cap,
    std::int64_t ns_ahead,
    std::int64_t dim_batch,
    double rel,
    double abs_cut,
    double lr,
    const char* label
) -> int
{
    std::vector<StepOut> a{};
    std::vector<StepOut> b{};
    const qnpeps_status ra{
        run_schedule(f, gpus, ns, ns_cap, ns_ahead, dim_batch, rel, abs_cut, lr, 3, a)
    };
    const qnpeps_status rb{
        run_schedule(f, gpus, ns, ns_cap, ns_ahead, dim_batch, rel, abs_cut, lr, 3, b)
    };
    char note[160]{};
    if (ra != QNPEPS_OK or rb != QNPEPS_OK or a.size() != b.size() or a.empty())
    {
        std::snprintf(
            note,
            sizeof(note),
            "run_status a=%s b=%s n=%zu",
            qnpeps_last_error(),
            qnpeps_last_error(),
            a.size()
        );
        set_line(label, false, note);
        return 1;
    }
    bool pass{true};
    for (std::size_t k{0}; k < a.size(); ++k)
    {
        char n2[160]{};
        if (not all_eq(a[k], b[k], n2, sizeof(n2)) or a[k].epoch != b[k].epoch) pass = false;
    }
    std::snprintf(note, sizeof(note), "iters=%zu", a.size());
    set_line(label, pass, note);
    return pass ? 0 : 1;
}

auto case_c(
    const Fix& f, int gpus, int ns, std::int64_t dim_batch, double rel, double abs_cut, double lr
) -> int
{
    int fails{0};
    const std::int64_t ns_cap{2 * static_cast<std::int64_t>(ns) + 8};
    const std::int64_t nb{(static_cast<std::int64_t>(ns) + dim_batch - 1) / dim_batch};

    auto d_theta{dmalloc<cf32>(static_cast<std::size_t>(f.dense))};
    std::vector<float> hpeps{f.hpeps0};
    std::vector<std::vector<float>> theta_hist{};
    qnpeps_e2e_node* node{};
    const qnpeps_status cs{
        qnpeps_e2e_node_create(&f.cfg, gpus, ns_cap, ns, dim_batch, 0, &f.tt, &node)
    };
    if (cs != QNPEPS_OK)
    {
        std::printf("[gate_node] case_c create err %s\n", qnpeps_last_error());
        cudaFree(d_theta);
        return 1;
    }
    std::vector<StepOut> lag1{};
    for (int k{0}; k < 3; ++k)
    {
        theta_hist.push_back(hpeps);
        ck(cudaMemcpy(d_theta, hpeps.data(), hpeps.size() * sizeof(float), cudaMemcpyHostToDevice),
           "c upload");
        qnpeps_e2e_node_submit_theta(node, d_theta);
        StepOut o{};
        node_step_capture(node, ns, rel, abs_cut, f.dense, f.sites, o);
        lag1.push_back(o);
        if (o.status != QNPEPS_OK) break;
        euler_update(hpeps, o.theta, lr);
    }
    qnpeps_e2e_node_destroy(node);

    bool ess_ok{true};
    for (const StepOut& o : lag1)
    {
        if (o.status == QNPEPS_OK)
        {
            const double e{ess_from(o.logpsi, o.logq, ns)};
            if (e != o.ess) ess_ok = false;
        }
    }
    set_line("c_ess_recompute", ess_ok, ess_ok ? "" : "referee ess != node ess");
    if (not ess_ok) ++fails;

    if (lag1.size() >= 2 and lag1[1].status == QNPEPS_OK)
    {
        auto d0{dmalloc<cf32>(static_cast<std::size_t>(f.dense))};
        ck(cudaMemcpy(
               d0,
               theta_hist[0].data(),
               theta_hist[0].size() * sizeof(float),
               cudaMemcpyHostToDevice
           ),
           "c theta0");
        const QnpepsConfig scfg{scfg_of(f.cfg)};
        const std::int64_t dlb{qnpeps_dlenv_bytes(&scfg)};
        const std::int64_t scb{
            qnpeps_sample_scratch_bytes(&scfg, static_cast<std::uint64_t>(dim_batch))
        };
        auto dlenv{dmalloc<std::uint8_t>(static_cast<std::size_t>(dlb))};
        auto rl{dmalloc<double>(static_cast<std::size_t>(f.cfg.lx - 1))};
        auto sc{dmalloc<std::uint8_t>(static_cast<std::size_t>(scb))};
        auto samp{dmalloc<std::uint8_t>(static_cast<std::size_t>(ns) * f.sites)};
        auto rq{dmalloc<double>(static_cast<std::size_t>(ns))};
        auto rg{dmalloc<double>(static_cast<std::size_t>(ns))};
        qnpeps_build_dlenv(
            &scfg,
            reinterpret_cast<qnpeps_device_peps*>(d0),
            reinterpret_cast<qnpeps_device_dlenv*>(dlenv),
            rl,
            nullptr
        );
        cudaDeviceSynchronize();
        const QnpepsSampleArgs sample_args_b{
            .struct_size = sizeof(QnpepsSampleArgs),
            .peps = reinterpret_cast<qnpeps_device_peps*>(d0),
            .dlenv = reinterpret_cast<qnpeps_device_dlenv*>(dlenv),
            .gpus = 1,
            .scratch = sc,
            .scratch_bytes = static_cast<std::uint64_t>(scb),
            .samples_out = samp,
            .log_prob_config = rq,
            .log_gauge = rg,
            .n_samples = static_cast<std::uint64_t>(ns),
            .batch_base = static_cast<std::uint64_t>(nb),
            .dim_batch = static_cast<std::uint64_t>(dim_batch),
            .stream = nullptr
        };
        qnpeps_sample(&scfg, &sample_args_b);
        auto hsamp = std::vector<std::uint8_t>(static_cast<std::size_t>(ns) * f.sites);
        auto hq = std::vector<double>(static_cast<std::size_t>(ns));
        cudaMemcpy(hsamp.data(), samp, hsamp.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(hq.data(), rq, hq.size() * sizeof(double), cudaMemcpyDeviceToHost);
        bool epoch_ok{true};
        for (int j{0}; j < ns; ++j)
            if (lag1[1].epoch[static_cast<std::size_t>(j)] != 0) epoch_ok = false;
        const bool samp_ok{hsamp == lag1[1].samples};
        const bool logq_ok{hq == lag1[1].logq};
        char note[160]{};
        std::snprintf(
            note, sizeof(note), "stale_samp=%d stale_logq=%d epoch0=%d", samp_ok, logq_ok, epoch_ok
        );
        const bool pass{samp_ok and logq_ok and epoch_ok};
        set_line("c_stale_replay_old_env", pass, note);
        if (not pass) ++fails;
        cudaFree(d0);
        cudaFree(dlenv);
        cudaFree(rl);
        cudaFree(sc);
        cudaFree(samp);
        cudaFree(rq);
        cudaFree(rg);
    }

    std::vector<StepOut> lag0{};
    {
        std::vector<float> hp{f.hpeps0};
        auto dt{dmalloc<cf32>(static_cast<std::size_t>(f.dense))};
        qnpeps_e2e_node* n0{};
        qnpeps_e2e_node_create(&f.cfg, gpus, ns_cap, 0, dim_batch, 0, &f.tt, &n0);
        for (int k{0}; k < 3; ++k)
        {
            cudaMemcpy(dt, hp.data(), hp.size() * sizeof(float), cudaMemcpyHostToDevice);
            qnpeps_e2e_node_submit_theta(n0, dt);
            StepOut o{};
            node_step_capture(n0, ns, rel, abs_cut, f.dense, f.sites, o);
            lag0.push_back(o);
            if (o.status != QNPEPS_OK) break;
            euler_update(hp, o.theta, lr);
        }
        qnpeps_e2e_node_destroy(n0);
        cudaFree(dt);
    }
    bool ratio_ok{true};
    for (std::size_t k{1}; k < lag1.size() and k < lag0.size(); ++k)
    {
        const double r{lag1[k].ess / lag0[k].ess};
        std::printf(
            "[gate_node] RATIO iter%zu ess_lag1=%.6g ess_lag0=%.6g ratio=%.4f\n",
            k + 1,
            lag1[k].ess,
            lag0[k].ess,
            r
        );
        if (lag1[k].ess > lag0[k].ess * (1.0 + 1e-9)) ratio_ok = false;
    }
    set_line("c_ess_lag1_le_lag0", ratio_ok, "");
    if (not ratio_ok) ++fails;

    cudaFree(d_theta);
    return fails;
}

auto case_d(
    const Fix& f, int gpus, int ns, std::int64_t dim_batch, double rel, double abs_cut, double lr
) -> int
{
    const std::int64_t ns_cap{2 * static_cast<std::int64_t>(ns) + 8};
    std::vector<StepOut> base{};
    std::vector<StepOut> delayed{};
    const qnpeps_status rb{
        run_schedule(f, gpus, ns, ns_cap, ns, dim_batch, rel, abs_cut, lr, 3, base)
    };
    setenv("QNPEPS_E2E_INJECT_DELAY_GPU", "1:40", 1);
    const qnpeps_status rd{
        run_schedule(f, gpus, ns, ns_cap, ns, dim_batch, rel, abs_cut, lr, 3, delayed)
    };
    unsetenv("QNPEPS_E2E_INJECT_DELAY_GPU");
    char note[160]{};
    if (rb != QNPEPS_OK or rd != QNPEPS_OK or base.size() != delayed.size() or base.empty())
    {
        std::snprintf(
            note,
            sizeof(note),
            "run_status base=%s delay=%s",
            qnpeps_last_error(),
            qnpeps_last_error()
        );
        set_line("d_delay_content_invariant", false, note);
        return 1;
    }
    bool pass{true};
    for (std::size_t k{0}; k < base.size(); ++k)
    {
        char n2[160]{};
        if (not all_eq(base[k], delayed[k], n2, sizeof(n2)) or base[k].epoch != delayed[k].epoch)
            pass = false;
    }
    std::snprintf(note, sizeof(note), "iters=%zu (worker1 +40ms)", base.size());
    set_line("d_delay_content_invariant", pass, note);
    return pass ? 0 : 1;
}

auto case_f(const Fix& f, int ns, double rel, double abs_cut, double learning_rate) -> int
{
    int fails{};
    const std::int64_t ns_cap{static_cast<std::int64_t>(ns) + 8};
    const std::uint64_t peps_bytes{static_cast<std::uint64_t>(f.dense) * sizeof(cf32)};
    const std::uint64_t state_bytes{static_cast<std::uint64_t>(2 * f.dense) * sizeof(double)};

    qnpeps_e2e_node* granular{};
    qnpeps_e2e_node* fused_f64{};
    qnpeps_e2e_node* fused_f32{};
    const qnpeps_status create_granular{
        qnpeps_e2e_node_create(&f.cfg, 1, ns_cap, 0, ns, 0, &f.tt, &granular)
    };
    const qnpeps_status create_f64{
        qnpeps_e2e_node_create(&f.cfg, 1, ns_cap, 0, ns, 0, &f.tt, &fused_f64)
    };
    const qnpeps_status create_f32{
        qnpeps_e2e_node_create(&f.cfg, 1, ns_cap, 0, ns, 0, &f.tt, &fused_f32)
    };
    const bool create_ok{
        create_granular == QNPEPS_OK and create_f64 == QNPEPS_OK and create_f32 == QNPEPS_OK
    };
    set_line("f_one_step_create", create_ok, create_ok ? "" : "node create failed");
    if (not create_ok)
    {
        qnpeps_e2e_node_destroy(granular);
        qnpeps_e2e_node_destroy(fused_f64);
        qnpeps_e2e_node_destroy(fused_f32);
        return 1;
    }

    auto d_initial{dmalloc<cf32>(static_cast<std::size_t>(f.dense))};
    auto d_peps_f64{dmalloc<cf32>(static_cast<std::size_t>(f.dense))};
    auto d_peps_f32{dmalloc<cf32>(static_cast<std::size_t>(f.dense))};
    auto d_theta_f64{dmalloc<cf32>(static_cast<std::size_t>(f.dense))};
    auto d_theta_f32{dmalloc<cf32>(static_cast<std::size_t>(f.dense))};
    auto d_master{dmalloc<double>(static_cast<std::size_t>(2 * f.dense))};

    auto initial_f64 = std::vector<double>(f.hpeps0.size());
    for (std::size_t index{}; index < f.hpeps0.size(); ++index)
        initial_f64[index] = static_cast<double>(f.hpeps0[index]);

    cudaMemcpy(d_initial, f.hpeps0.data(), peps_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_master, initial_f64.data(), state_bytes, cudaMemcpyHostToDevice);
    qnpeps_e2e_node_submit_theta(granular, d_initial);
    qnpeps_e2e_node_submit_theta(fused_f64, d_initial);
    qnpeps_e2e_node_submit_theta(fused_f32, d_initial);

    StepOut reference{};
    const qnpeps_status reference_status{
        node_step_capture(granular, ns, rel, abs_cut, f.dense, f.sites, reference)
    };
    set_line("f_granular_producer", reference_status == QNPEPS_OK, qnpeps_last_error());
    if (reference_status != QNPEPS_OK) ++fails;

    qnpeps::CuArray<double, 2> e_mean_f64{};
    double e_var_f64{};
    double ess_f64{};
    QnpepsE2eEulerStepArgs args_f64{};
    args_f64.struct_size = sizeof(args_f64);
    args_f64.precision = QNPEPS_E2E_UPDATE_F64_MASTER;
    args_f64.n_samples = ns;
    args_f64.relative_cut = rel;
    args_f64.absolute_cut = abs_cut;
    args_f64.learning_rate = learning_rate;
    args_f64.state_f64_io = d_master;
    args_f64.state_f64_bytes = state_bytes - sizeof(double);
    args_f64.peps_f32_io = reinterpret_cast<qnpeps_e2e_cbuf*>(d_peps_f64);
    args_f64.peps_f32_bytes = peps_bytes;
    args_f64.theta_output = reinterpret_cast<qnpeps_e2e_cbuf*>(d_theta_f64);
    args_f64.theta_dot_bytes = peps_bytes;
    args_f64.energy_mean_output = e_mean_f64.data();
    args_f64.energy_variance_output = &e_var_f64;
    args_f64.ess_output = &ess_f64;

    const qnpeps_status bad_f64{qnpeps_e2e_node_step_euler(fused_f64, &args_f64)};
    set_line("f_wrong_f64_width", bad_f64 == QNPEPS_ERR_BAD_CONFIG, qnpeps_last_error());
    if (bad_f64 != QNPEPS_ERR_BAD_CONFIG) ++fails;

    args_f64.state_f64_bytes = state_bytes;
    const qnpeps_status f64_status{qnpeps_e2e_node_step_euler(fused_f64, &args_f64)};
    auto theta_f64 = std::vector<cf32>(static_cast<std::size_t>(f.dense));
    auto state_f64 = std::vector<double>(initial_f64.size());
    auto peps_f64 = std::vector<float>(f.hpeps0.size());
    cudaMemcpy(theta_f64.data(), d_theta_f64, peps_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(state_f64.data(), d_master, state_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(peps_f64.data(), d_peps_f64, peps_bytes, cudaMemcpyDeviceToHost);

    const std::vector<double> expected_f64{
        layout_update_f64(f.cfg, initial_f64, reference.theta, learning_rate)
    };
    const std::vector<float> expected_f64_emit{quantize_f64(expected_f64)};
    const bool f64_direction{f64_status == QNPEPS_OK and theta_eq(theta_f64, reference.theta)};
    const bool f64_stats{
        f64_status == QNPEPS_OK and e_mean_f64[0] == reference.e_mean[0]
        and e_mean_f64[1] == reference.e_mean[1] and e_var_f64 == reference.e_var
        and ess_f64 == reference.ess
    };
    const bool f64_state{f64_status == QNPEPS_OK and bytes_equal(state_f64, expected_f64)};
    const bool f64_emit{f64_status == QNPEPS_OK and bytes_equal(peps_f64, expected_f64_emit)};
    set_line("f_f64_direction", f64_direction, "");
    set_line("f_f64_stats", f64_stats, "");
    set_line("f_f64_layout_state", f64_state, "");
    set_line("f_f64_emit", f64_emit, "");
    fails += not f64_direction;
    fails += not f64_stats;
    fails += not f64_state;
    fails += not f64_emit;

    if (f64_status == QNPEPS_OK)
    {
        qnpeps_e2e_node_submit_theta(granular, d_peps_f64);
        StepOut granular_next{};
        StepOut fused_next{};
        const qnpeps_status granular_next_status{
            node_step_capture(granular, ns, rel, abs_cut, f.dense, f.sites, granular_next)
        };
        const qnpeps_status fused_next_status{
            node_step_capture(fused_f64, ns, rel, abs_cut, f.dense, f.sites, fused_next)
        };
        char note[160]{};
        const bool submitted{
            granular_next_status == QNPEPS_OK and fused_next_status == QNPEPS_OK
            and all_eq(granular_next, fused_next, note, sizeof(note))
        };
        set_line("f_theta_next_submitted", submitted, note);
        if (not submitted) ++fails;
    }

    qnpeps::CuArray<double, 2> e_mean_f32{};
    double e_var_f32{};
    double ess_f32{};
    QnpepsE2eEulerStepArgs args_f32{};
    args_f32.struct_size = sizeof(args_f32);
    args_f32.precision = QNPEPS_E2E_UPDATE_F32;
    args_f32.n_samples = ns;
    args_f32.relative_cut = rel;
    args_f32.absolute_cut = abs_cut;
    args_f32.learning_rate = learning_rate;
    args_f32.peps_f32_io = reinterpret_cast<qnpeps_e2e_cbuf*>(d_peps_f32);
    args_f32.peps_f32_bytes = peps_bytes;
    args_f32.theta_output = reinterpret_cast<qnpeps_e2e_cbuf*>(d_theta_f32);
    args_f32.theta_dot_bytes = state_bytes;
    args_f32.energy_mean_output = e_mean_f32.data();
    args_f32.energy_variance_output = &e_var_f32;
    args_f32.ess_output = &ess_f32;
    const qnpeps_status bad_f32{qnpeps_e2e_node_step_euler(fused_f32, &args_f32)};
    set_line("f_wrong_f32_width", bad_f32 == QNPEPS_ERR_BAD_CONFIG, qnpeps_last_error());
    if (bad_f32 != QNPEPS_ERR_BAD_CONFIG) ++fails;

    args_f32.theta_dot_bytes = peps_bytes;
    const qnpeps_status f32_status{qnpeps_e2e_node_step_euler(fused_f32, &args_f32)};
    auto theta_f32 = std::vector<cf32>(static_cast<std::size_t>(f.dense));
    auto peps_f32 = std::vector<float>(f.hpeps0.size());
    cudaMemcpy(theta_f32.data(), d_theta_f32, peps_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(peps_f32.data(), d_peps_f32, peps_bytes, cudaMemcpyDeviceToHost);
    const std::vector<float> expected_f32{
        layout_update_f32(f.cfg, f.hpeps0, reference.theta, learning_rate)
    };
    const bool f32_direction{f32_status == QNPEPS_OK and theta_eq(theta_f32, reference.theta)};
    const bool f32_stats{
        f32_status == QNPEPS_OK and e_mean_f32[0] == reference.e_mean[0]
        and e_mean_f32[1] == reference.e_mean[1] and e_var_f32 == reference.e_var
        and ess_f32 == reference.ess
    };
    const bool f32_state{f32_status == QNPEPS_OK and bytes_equal(peps_f32, expected_f32)};
    set_line("f_f32_direction", f32_direction, "");
    set_line("f_f32_stats", f32_stats, "");
    set_line("f_f32_layout_state", f32_state, "");
    fails += not f32_direction;
    fails += not f32_stats;
    fails += not f32_state;

    cudaFree(d_initial);
    cudaFree(d_peps_f64);
    cudaFree(d_peps_f32);
    cudaFree(d_theta_f64);
    cudaFree(d_theta_f32);
    cudaFree(d_master);
    qnpeps_e2e_node_destroy(granular);
    qnpeps_e2e_node_destroy(fused_f64);
    qnpeps_e2e_node_destroy(fused_f32);
    return fails;
}

auto case_e(const Fix& f, int gpus, int ns, std::int64_t dim_batch, double rel, double abs_cut)
    -> int
{
    int fails{0};
    const std::int64_t ns_cap{2 * static_cast<std::int64_t>(ns) + 8};

    std::size_t free0{0};
    std::size_t total0{0};
    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaMemGetInfo(&free0, &total0);
    for (int rep{0}; rep < 5; ++rep)
    {
        qnpeps_e2e_node* n{};
        const qnpeps_status cs{
            qnpeps_e2e_node_create(&f.cfg, gpus, ns_cap, ns, dim_batch, 0, &f.tt, &n)
        };
        if (cs != QNPEPS_OK)
        {
            set_line("e_create_destroy_cycle", false, qnpeps_last_error());
            return 1;
        }
        qnpeps_e2e_node_destroy(n);
    }
    std::size_t free1{0};
    std::size_t total1{0};
    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaMemGetInfo(&free1, &total1);
    const long long drift{static_cast<long long>(free0) - static_cast<long long>(free1)};
    const bool leak_ok{std::llabs(drift) < (16ll << 20)};
    char note[160]{};
    std::snprintf(note, sizeof(note), "memGetInfo drift=%lld bytes over 5 cycles", drift);
    set_line("e_create_destroy_cycle", leak_ok, note);
    if (not leak_ok) ++fails;

    if (gpus >= 2)
    {
        setenv("QNPEPS_E2E_INJECT_OOM_GPU", "1", 1);
        qnpeps_e2e_node* n{};
        qnpeps_e2e_node_create(&f.cfg, gpus, ns_cap, 0, dim_batch, 0, &f.tt, &n);
        unsetenv("QNPEPS_E2E_INJECT_OOM_GPU");
        auto d_theta{dmalloc<cf32>(static_cast<std::size_t>(f.dense))};
        cudaMemcpy(
            d_theta, f.hpeps0.data(), f.hpeps0.size() * sizeof(float), cudaMemcpyHostToDevice
        );
        qnpeps_e2e_node_submit_theta(n, d_theta);
        StepOut o1{};
        const qnpeps_status s1{node_step_capture(n, ns, rel, abs_cut, f.dense, f.sites, o1)};
        StepOut o2{};
        const qnpeps_status s2{node_step_capture(n, ns, rel, abs_cut, f.dense, f.sites, o2)};
        const bool dead_ok{s1 == QNPEPS_ERR_OOM and s2 == QNPEPS_ERR_OOM};
        std::snprintf(
            note,
            sizeof(note),
            "step1=%d step2=%d error=%s",
            static_cast<int>(s1),
            static_cast<int>(s2),
            qnpeps_last_error()
        );
        set_line("e_oom_step_cleanly_dead", dead_ok, note);
        if (not dead_ok) ++fails;
        const qnpeps_status ds{qnpeps_e2e_node_destroy(n)};
        set_line("e_destroy_after_fail", ds == QNPEPS_OK, qnpeps_last_error());
        if (ds != QNPEPS_OK) ++fails;
        cudaFree(d_theta);
    }
    return fails;
}

struct DeviceMemory
{
    std::size_t outside_free{};
    QnpepsArenaReading arena{};
};

auto device_memory(int gpus) -> std::vector<DeviceMemory>
{
    std::vector<DeviceMemory> memory(static_cast<std::size_t>(gpus));
    for (int g{0}; g < gpus; ++g)
    {
        auto& entry{memory[static_cast<std::size_t>(g)]};
        std::size_t total{0};
        cudaSetDevice(g);
        cudaDeviceSynchronize();
        cudaMemGetInfo(&entry.outside_free, &total);
        entry.arena.struct_size = sizeof(QnpepsArenaReading);
        qnpeps_device_arena_reading(g, &entry.arena);
    }
    cudaSetDevice(0);
    return memory;
}

auto arena_lease_case(const Fix& f, int gpus, int ns) -> int
{
    constexpr std::int64_t k_lease_capacity{8192};
    constexpr long long k_outside_growth_bound{512ll << 20};
    const std::int64_t dim_batch{ns / 2};
    const auto gram_bytes{
        static_cast<unsigned long long>(k_lease_capacity * k_lease_capacity) * sizeof(cf32)
    };
    int fails{0};
    char note[256]{};

    const auto create = [&](std::int64_t capacity, qnpeps_e2e_node** node)
    { return qnpeps_e2e_node_create(&f.cfg, gpus, capacity, 0, dim_batch, 0, &f.tt, node); };

    {
        qnpeps_e2e_node* warm{};
        if (create(k_lease_capacity, &warm) != QNPEPS_OK)
        {
            set_line("g_lease_create", false, qnpeps_last_error());
            return 1;
        }
        qnpeps_e2e_node_destroy(warm);
    }

    const auto before{device_memory(gpus)};
    qnpeps_e2e_node* first{};
    if (create(k_lease_capacity, &first) != QNPEPS_OK)
    {
        set_line("g_lease_create", false, qnpeps_last_error());
        return 1;
    }
    const auto held{device_memory(gpus)};
    qnpeps_e2e_node_destroy(first);
    const auto released{device_memory(gpus)};
    qnpeps_e2e_node* second{};
    const qnpeps_status second_status{create(k_lease_capacity, &second)};
    const auto held_again{device_memory(gpus)};
    qnpeps_e2e_node_destroy(second);
    const auto released_again{device_memory(gpus)};

    long long worst_outside_growth{0};
    bool leases_grow{true};
    bool leases_return{true};
    bool leases_repeat{second_status == QNPEPS_OK};
    for (int g{0}; g < gpus; ++g)
    {
        const auto index{static_cast<std::size_t>(g)};
        const auto& start{before[index]};
        const long long outside_growth{
            static_cast<long long>(start.outside_free)
            - static_cast<long long>(held[index].outside_free)
        };
        worst_outside_growth = std::max(worst_outside_growth, outside_growth);
        const auto leased_bytes{held[index].arena.in_use_bytes - start.arena.in_use_bytes};
        const auto minimum_leased{g == 0 ? gram_bytes : 1ull};
        if (held[index].arena.in_use_bytes < start.arena.in_use_bytes) leases_grow = false;
        if (leased_bytes < minimum_leased) leases_grow = false;
        for (const auto* after : {&released[index], &released_again[index]})
        {
            if (after->arena.in_use_bytes != start.arena.in_use_bytes) leases_return = false;
            if (after->arena.quarantined_count != start.arena.quarantined_count)
                leases_return = false;
        }
        if (held_again[index].arena.in_use_bytes != held[index].arena.in_use_bytes)
            leases_repeat = false;
        if (held_again[index].arena.top_bytes != held[index].arena.top_bytes) leases_repeat = false;
    }
    const bool outside_ok{worst_outside_growth <= k_outside_growth_bound};
    std::snprintf(
        note,
        sizeof(note),
        "worst outside-arena growth=%lld bytes, bound=%lld",
        worst_outside_growth,
        k_outside_growth_bound
    );
    set_line("g_node_outside_arena", outside_ok, note);
    if (not outside_ok) ++fails;
    std::snprintf(
        note,
        sizeof(note),
        "gpu0 leased=%llu bytes, gram=%llu",
        static_cast<unsigned long long>(held[0].arena.in_use_bytes - before[0].arena.in_use_bytes),
        gram_bytes
    );
    set_line("g_node_leases_in_arena", leases_grow, note);
    if (not leases_grow) ++fails;
    set_line("g_destroy_returns_leases", leases_return, "");
    if (not leases_return) ++fails;
    set_line("g_create_again_same_leases", leases_repeat, qnpeps_last_error());
    if (not leases_repeat) ++fails;

    const auto oversize_capacity{static_cast<std::int64_t>(
        std::sqrt(static_cast<double>(before[0].arena.capacity_bytes) / sizeof(cf32))
        + k_lease_capacity
    )};
    qnpeps_e2e_node* oversize{};
    const qnpeps_status oversize_status{create(oversize_capacity, &oversize)};
    const std::string oversize_message{qnpeps_last_error()};
    const auto after_oversize{device_memory(gpus)};
    bool oversize_ok{oversize_status == QNPEPS_ERR_OOM and oversize == nullptr};
    oversize_ok = oversize_ok and oversize_message.find("node buffer gram") != std::string::npos;
    oversize_ok = oversize_ok and oversize_message.find("bytes free") != std::string::npos;
    for (int g{0}; g < gpus; ++g)
    {
        const auto index{static_cast<std::size_t>(g)};
        const auto& start{before[index].arena};
        if (after_oversize[index].arena.in_use_bytes != start.in_use_bytes) oversize_ok = false;
        if (after_oversize[index].arena.quarantined_count != start.quarantined_count)
            oversize_ok = false;
    }
    std::snprintf(
        note,
        sizeof(note),
        "capacity=%lld status=%d %s",
        static_cast<long long>(oversize_capacity),
        static_cast<int>(oversize_status),
        oversize_message.c_str()
    );
    set_line("g_oversize_names_buffer", oversize_ok, note);
    if (not oversize_ok) ++fails;
    return fails;
}

auto run_case(int lx, int ly, int D, int chi, int ns, int meo, double rel, double abs_cut) -> int
{
    int ndev{0};
    cudaGetDeviceCount(&ndev);
    const int gpus{ndev >= 2 ? 2 : 1};
    const Fix f{build_fixture(lx, ly, D, chi, meo)};
    std::printf(
        "[gate_node] fixture lx=%d ly=%d D=%d chi=%d ns=%d meo=%d dense=%lld compact=%lld "
        "gpus=%d\n",
        lx,
        ly,
        D,
        chi,
        ns,
        meo,
        static_cast<long long>(f.dense),
        static_cast<long long>(f.compact),
        gpus
    );
    double lr{0.2};
    if (auto e{std::getenv("QNPEPS_E2E_GATE_LR")}; e and e[0]) lr = std::atof(e);
    int fails{0};

    fails += case_a(f, gpus, ns, rel, abs_cut, lr, "a", gpus == 1);
    {
        Fix rangefinder{f};
        rangefinder.cfg.truncation_route = QNPEPS_TRUNCATION_RANGEFINDER;
        rangefinder.tt = rangefinder.terms.table();
        fails += case_a(rangefinder, gpus, ns, rel, abs_cut, lr, "a_rangefinder", true);
    }
    fails +=
        determinism_case(f, gpus, ns, 2 * ns + 8, ns, ns / 2, rel, abs_cut, lr, "b_lag1_twice");
    fails += determinism_case(
        f, gpus, ns, 2 * ns + 8, ns / 2, ns / 4, rel, abs_cut, lr, "b_mixed_twice"
    );
    {
        const std::int64_t dcarry{ns % 6 == 0 ? 5 : 6};
        fails += determinism_case(
            f, gpus, ns, 2 * ns + 16, ns / 2, dcarry, rel, abs_cut, lr, "b_carryover_twice"
        );
    }
    fails += case_c(f, gpus, ns, ns / 2, rel, abs_cut, lr);
    if (gpus >= 2) fails += case_d(f, gpus, ns, ns / 2, rel, abs_cut, lr);
    fails += case_e(f, gpus, ns, ns / 2, rel, abs_cut);
    fails += arena_lease_case(f, gpus, ns);
    const Fix layout_fixture{build_fixture(2, 3, 2, 8, 4)};
    fails += case_f(layout_fixture, 16, rel, abs_cut, 0.0312500074505806);

    std::printf(
        "[gate_node] CASE lx=%d ly=%d D=%d chi=%d ns=%d meo=%d rel=%.3g => %s (fails=%d)\n",
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
