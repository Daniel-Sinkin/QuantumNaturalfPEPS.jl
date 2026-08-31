#ifndef QNPEPS_ELOC_GATE_FIXTURE_CUH
#define QNPEPS_ELOC_GATE_FIXTURE_CUH

#include "../../core/types.cuh"
#include "boundary.hpp"
#include "tensor.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <random>
#include <vector>

namespace gate
{

using peps::EnvMPS;
using peps::Fixture;
using peps::Index;
using peps::NamedTensor;
using peps::PepsSite;

constexpr long long k_h_base{10000000};
constexpr long long k_v_base{20000000};
constexpr long long k_p_base{30000000};
constexpr long long k_b_base{40000000};

inline auto bd(int axis_len, int pos, int dim_bond) -> int
{
    if (pos <= 0 or pos >= axis_len) return 1;
    return dim_bond;
}

inline auto mk_index(long long id, int dim) -> Index
{
    Index ix{};
    ix.id = id;
    ix.prime = 0;
    ix.dim = dim;
    ix.tag = "";
    return ix;
}

struct GenPeps
{
    Fixture fx{};
    std::vector<double> flat{};
};

inline auto make_peps(
    int lx, int ly, int dim_bond, int dim_phys, int chi, int mode, double noise, std::uint64_t seed
) -> GenPeps
{
    GenPeps g{};
    g.fx.lx = lx;
    g.fx.ly = ly;
    g.fx.bond_dim = dim_bond;
    g.fx.contract_dim = chi;
    g.fx.contract_cutoff = 0.0;
    g.fx.peps.assign(
        static_cast<std::size_t>(lx), std::vector<PepsSite>(static_cast<std::size_t>(ly))
    );

    auto rng = std::mt19937_64(seed);
    auto gauss = std::normal_distribution<double>(0.0, 1.0);
    long long b_counter{0};

    for (auto i = 1; i <= lx; ++i)
    {
        for (auto j = 1; j <= ly; ++j)
        {
            const int row0{i - 1};
            const int col0{j - 1};
            const int wdim{bd(ly, col0, dim_bond)};
            const int sdim{bd(lx, row0 + 1, dim_bond)};
            const int edim{bd(ly, col0 + 1, dim_bond)};
            const int ndim{bd(lx, row0, dim_bond)};

            const long long w_id{
                (j > 1) ? (k_h_base + (i - 1) * ly + (j - 2)) : (k_b_base + b_counter++)
            };
            const long long e_id{
                (j < ly) ? (k_h_base + (i - 1) * ly + (j - 1)) : (k_b_base + b_counter++)
            };
            const long long n_id{
                (i > 1) ? (k_v_base + (i - 2) * ly + (j - 1)) : (k_b_base + b_counter++)
            };
            const long long s_id{
                (i < lx) ? (k_v_base + (i - 1) * ly + (j - 1)) : (k_b_base + b_counter++)
            };
            const long long p_id{k_p_base + (i - 1) * ly + (j - 1)};

            PepsSite site{};
            site.i = i;
            site.j = j;
            site.phys = mk_index(p_id, dim_phys);
            site.tensor.inds = {
                mk_index(w_id, wdim),
                mk_index(s_id, sdim),
                mk_index(e_id, edim),
                mk_index(n_id, ndim),
                site.phys
            };
            const std::size_t count{static_cast<std::size_t>(wdim) * sdim * edim * ndim * dim_phys};
            site.tensor.data.resize(count);
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
                if (mode == 1 and wi == ei and si == ni) v += 1.0;
                if (mode == 2)
                {
                    const qnpeps::CuArray<double, 4> a_arr{1.0, 0.7, 1.3, 0.85};
                    const double uw{1.0 + 0.35 * wi / std::max(1, wdim - 1)};
                    const double vs{1.0 - 0.25 * si / std::max(1, sdim - 1)};
                    const double xe{1.0 + 0.15 * ei / std::max(1, edim - 1)};
                    const double yn{1.0 - 0.45 * ni / std::max(1, ndim - 1)};
                    v += a_arr[pi % 4] * uw * vs * xe * yn;
                }
                site.tensor.data[k] = v;
                g.flat.push_back(v);
            }
            g.fx.peps[static_cast<std::size_t>(i - 1)][static_cast<std::size_t>(j - 1)] =
                std::move(site);
        }
    }
    return g;
}

inline auto frob2(const NamedTensor& t) -> double
{
    double s{0.0};
    for (const double v : t.data)
        s += v * v;
    return s;
}

inline auto host_logpsi(const Fixture& fx, const std::vector<int>& sample) -> std::complex<double>
{
    const std::vector<EnvMPS> et{peps::build_env_top(fx, sample)};
    const std::vector<EnvMPS> ed{peps::build_env_down(fx, sample)};
    int pos{(fx.lx - 1) / 2};
    if (pos < 1) pos = 1;
    const std::size_t ti{static_cast<std::size_t>(pos - 1)};
    const std::size_t di{static_cast<std::size_t>(fx.lx - pos - 1)};
    const auto& a{et[ti].tensors};
    const auto& b{ed[di].tensors};

    NamedTensor acc{peps::contract(a[0], b[0])};
    double log_tot{0.0};
    for (std::size_t j{1}; j < a.size(); ++j)
    {
        const double nrm{std::sqrt(frob2(acc))};
        if (nrm > 0.0)
        {
            for (double& v : acc.data)
                v /= nrm;
            log_tot += std::log(nrm);
        }
        acc = peps::contract(acc, a[j]);
        acc = peps::contract(acc, b[j]);
    }
    const double nrm{std::sqrt(frob2(acc))};
    if (nrm > 0.0)
    {
        for (double& v : acc.data)
            v /= nrm;
        log_tot += std::log(nrm);
    }
    const double s{peps::scalar_value(acc)};
    std::complex<double> out{log_tot + et[ti].f + ed[di].f, 0.0};
    out += std::log(std::complex<double>(s, 0.0));
    return out;
}

inline auto gen_samples(int lx, int ly, int dim_phys, int n, std::uint64_t seed, bool identical)
    -> std::vector<std::uint8_t>
{
    const std::size_t sites{static_cast<std::size_t>(lx) * ly};
    auto s = std::vector<std::uint8_t>(static_cast<std::size_t>(n) * sites);
    auto rng = std::mt19937_64(seed);
    auto spin = std::uniform_int_distribution<int>(0, dim_phys - 1);
    if (identical)
    {
        auto one = std::vector<std::uint8_t>(sites);
        for (auto& v : one)
            v = static_cast<std::uint8_t>(spin(rng));
        for (int lane{0}; lane < n; ++lane)
        {
            for (std::size_t k{0}; k < sites; ++k)
                s[static_cast<std::size_t>(lane) * sites + k] = one[k];
        }
    }
    else
    {
        for (auto& v : s)
            v = static_cast<std::uint8_t>(spin(rng));
    }
    return s;
}

inline auto to_ints(const std::vector<std::uint8_t>& s, int lane, int lx, int ly)
    -> std::vector<int>
{
    const std::size_t sites{static_cast<std::size_t>(lx) * ly};
    auto out = std::vector<int>(sites);
    for (std::size_t k{0}; k < sites; ++k)
        out[k] = static_cast<int>(s[static_cast<std::size_t>(lane) * sites + k]);
    return out;
}

inline auto ang_dist(double a, double b) -> double
{
    double d{a - b};
    while (d > M_PI)
        d -= 2.0 * M_PI;
    while (d <= -M_PI)
        d += 2.0 * M_PI;
    return std::abs(d);
}

inline auto bad_num(double x) -> bool
{
    return std::isnan(x) or std::isinf(x);
}

}

#endif
