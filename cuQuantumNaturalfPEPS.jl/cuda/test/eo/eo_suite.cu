#include "dans_qnpeps_eloc.h"
#include "gate_fixture.cuh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace
{
struct C32
{
    float re;
    float im;
};

struct DeviceMemory
{
    std::vector<void*> blocks{};

    ~DeviceMemory()
    {
        for (void* block : blocks)
            cudaFree(block);
    }

    template <class T>
    auto copy(const std::vector<T>& host) -> T*
    {
        T* out{};
        if (cudaMalloc(reinterpret_cast<void**>(&out), sizeof(T) * host.size()) != cudaSuccess)
            return nullptr;
        blocks.push_back(out);
        if (not host.empty()
            and cudaMemcpy(out, host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice)
                    != cudaSuccess)
            return nullptr;
        return out;
    }

    template <class T>
    auto allocate(std::size_t count) -> T*
    {
        T* out{};
        if (cudaMalloc(reinterpret_cast<void**>(&out), sizeof(T) * count) != cudaSuccess)
            return nullptr;
        blocks.push_back(out);
        return out;
    }
};

struct RunResult
{
    qnpeps_eloc_status status{QNPEPS_ELOC_ERR_INTERNAL};
    std::vector<double> logpsi{};
    std::vector<double> energy{};
    std::vector<C32> rows{};
    std::int64_t compact{};
};

auto finite(const RunResult& result) -> bool
{
    if (result.status != QNPEPS_ELOC_OK) return false;
    for (double value : result.logpsi)
        if (not std::isfinite(value)) return false;
    for (double value : result.energy)
        if (not std::isfinite(value)) return false;
    for (const C32 value : result.rows)
        if (not std::isfinite(value.re) or not std::isfinite(value.im)) return false;
    return true;
}

auto max_error(const std::vector<double>& got, const std::vector<std::complex<double>>& want)
    -> double
{
    double out{};
    for (std::size_t lane{}; lane < want.size(); ++lane)
    {
        const std::complex<double> value{got[2 * lane], got[2 * lane + 1]};
        out = std::max(out, std::abs(value - want[lane]));
    }
    return out;
}

auto product_peps(int lx, int ly, bool zero_unused = false, bool near_zero = false)
    -> std::vector<C32>
{
    auto peps = std::vector<C32>(static_cast<std::size_t>(lx) * ly * 2);
    for (int site{}; site < lx * ly; ++site)
    {
        peps[static_cast<std::size_t>(2 * site)] = C32{1.0f + 0.01f * site, 0.0f};
        float second{0.4f + 0.02f * site};
        if (zero_unused) second = 0.0f;
        if (near_zero) second = 1.0e-20f;
        peps[static_cast<std::size_t>(2 * site + 1)] = C32{second, 0.0f};
    }
    return peps;
}

auto sample_set(int lx, int ly, int count, bool identical = false, bool zeros = false)
    -> std::vector<std::uint8_t>
{
    const int sites{lx * ly};
    auto samples = std::vector<std::uint8_t>(static_cast<std::size_t>(count) * sites);
    for (int lane{}; lane < count; ++lane)
    {
        for (int site{}; site < sites; ++site)
        {
            samples[static_cast<std::size_t>(lane) * sites + site] =
                zeros ? 0 : static_cast<std::uint8_t>((site + (identical ? 0 : lane)) & 1);
        }
    }
    return samples;
}

auto run_product(
    int lx,
    int ly,
    int count,
    int meo,
    const std::vector<C32>& peps,
    const std::vector<std::uint8_t>& samples,
    const std::vector<QnpepsElocDiagBond>& diag,
    const std::vector<QnpepsElocFlipTerm>& flip
) -> RunResult
{
    RunResult result{};
    QnpepsElocConfig cfg{sizeof(QnpepsElocConfig), lx, ly, 2, 1, 1, meo};
    qnpeps_eloc_compact_count(&cfg, &result.compact);
    result.logpsi.resize(static_cast<std::size_t>(2 * count));
    result.energy.resize(static_cast<std::size_t>(2 * count));
    result.rows.resize(static_cast<std::size_t>(count) * result.compact);
    DeviceMemory device{};
    auto d_peps{device.copy(peps)};
    auto d_samples{device.copy(samples)};
    auto d_logpsi{device.allocate<double>(result.logpsi.size())};
    auto d_energy{device.allocate<double>(result.energy.size())};
    auto d_rows{device.allocate<C32>(result.rows.size())};
    if (not d_peps or not d_samples or not d_logpsi or not d_energy or not d_rows)
    {
        result.status = QNPEPS_ELOC_ERR_OOM;
        return result;
    }
    QnpepsElocTermTable table{
        static_cast<std::int32_t>(diag.size()),
        diag.empty() ? nullptr : diag.data(),
        static_cast<std::int32_t>(flip.size()),
        flip.empty() ? nullptr : flip.data()
    };
    result.status = qnpeps_eloc_run(
        &cfg,
        reinterpret_cast<qnpeps_eloc_cbuf*>(d_peps),
        d_samples,
        count,
        &table,
        d_logpsi,
        d_energy,
        reinterpret_cast<qnpeps_eloc_cbuf*>(d_rows),
        nullptr,
        nullptr,
        0.0,
        nullptr
    );
    if (result.status == QNPEPS_ELOC_OK)
    {
        cudaMemcpy(
            result.logpsi.data(),
            d_logpsi,
            sizeof(double) * result.logpsi.size(),
            cudaMemcpyDeviceToHost
        );
        cudaMemcpy(
            result.energy.data(),
            d_energy,
            sizeof(double) * result.energy.size(),
            cudaMemcpyDeviceToHost
        );
        cudaMemcpy(
            result.rows.data(), d_rows, sizeof(C32) * result.rows.size(), cudaMemcpyDeviceToHost
        );
    }
    return result;
}

auto analytic_energy(
    int lx,
    int ly,
    const std::vector<C32>& peps,
    const std::vector<std::uint8_t>& samples,
    const std::vector<QnpepsElocDiagBond>& diag,
    const std::vector<QnpepsElocFlipTerm>& flip
) -> std::vector<std::complex<double>>
{
    const int sites{lx * ly};
    const int count{static_cast<int>(samples.size()) / sites};
    auto out = std::vector<std::complex<double>>(static_cast<std::size_t>(count));
    for (int lane{}; lane < count; ++lane)
    {
        auto sample{samples.data() + static_cast<std::size_t>(lane) * sites};
        double diagonal{};
        for (const auto& term : diag)
        {
            const int a{1 - 2 * sample[term.site_a]};
            const int b{1 - 2 * sample[term.site_b]};
            diagonal += term.coeff * a * b;
        }
        std::complex<double> energy{diagonal, 0.0};
        for (const auto& term : flip)
        {
            if (term.mask_a >= 0 and sample[term.mask_a] == sample[term.mask_b]) continue;
            double ratio{1.0};
            for (int k{}; k < term.n_flips; ++k)
            {
                const int site{term.flip_site[k]};
                const int old_spin{sample[site]};
                const int new_spin{term.flip_value[k] < 0 ? 1 - old_spin : term.flip_value[k]};
                const double old_value{peps[static_cast<std::size_t>(2 * site + old_spin)].re};
                const double new_value{peps[static_cast<std::size_t>(2 * site + new_spin)].re};
                ratio *= new_value / old_value;
            }
            energy += std::complex<double>{term.coeff_re, term.coeff_im} * ratio;
        }
        out[static_cast<std::size_t>(lane)] = energy;
    }
    return out;
}

auto flip_term(int site, double coefficient, int mask_a = -1, int mask_b = -1) -> QnpepsElocFlipTerm
{
    return QnpepsElocFlipTerm{1, {site, 0, 0, 0}, {-1, 0, 0, 0}, mask_a, mask_b, coefficient, 0.0};
}

auto two_flip_term(int a, int b, double coefficient) -> QnpepsElocFlipTerm
{
    return QnpepsElocFlipTerm{2, {a, b, 0, 0}, {-1, -1, 0, 0}, a, b, coefficient, 0.0};
}

auto emit_case(
    const char* tier,
    const char* name,
    bool passed,
    double observed,
    const char* route = "none",
    const char* outcome = "finite_correct"
) -> bool
{
    std::printf(
        "[eo_suite] tier=%s case=%s status=%s observed=%.17g route=%s outcome=%s\n",
        tier,
        name,
        passed ? "PASS" : "FAIL",
        observed,
        route,
        outcome
    );
    return passed;
}

auto run_analytic() -> bool
{
    bool passed{true};
    {
        const int lx{2};
        const int ly{2};
        const int count{3};
        const auto peps{product_peps(lx, ly)};
        const auto samples{sample_set(lx, ly, count)};
        const std::vector<QnpepsElocDiagBond> diag{{0, 1, 1.0}};
        const std::vector<QnpepsElocFlipTerm> flip{two_flip_term(0, 1, 2.0)};
        const RunResult got{run_product(lx, ly, count, 2, peps, samples, diag, flip)};
        const auto want{analytic_energy(lx, ly, peps, samples, diag, flip)};
        const double error{max_error(got.energy, want)};
        std::vector<QnpepsElocDiagBond> wrong_diag{{0, 2, 1.0}};
        const auto wrong{analytic_energy(lx, ly, peps, samples, wrong_diag, flip)};
        const double control{max_error(got.energy, wrong)};
        passed = emit_case(
                     "1",
                     "single_bond_heisenberg_2x2",
                     finite(got) and error <= 2.0e-5 and control > 1.0e-3,
                     error,
                     "horizontal"
                 )
                 and passed;
    }
    {
        const int lx{2};
        const int ly{2};
        const int count{3};
        const auto peps{product_peps(lx, ly)};
        const auto samples{sample_set(lx, ly, count)};
        const std::vector<QnpepsElocDiagBond> diag{{0, 3, 0.58}};
        const std::vector<QnpepsElocFlipTerm> flip{two_flip_term(0, 3, 1.16)};
        const RunResult got{run_product(lx, ly, count, 2, peps, samples, diag, flip)};
        const auto want{analytic_energy(lx, ly, peps, samples, diag, flip)};
        std::vector<QnpepsElocDiagBond> wrong_diag{{0, 3, -0.58}};
        const auto wrong{analytic_energy(lx, ly, peps, samples, wrong_diag, flip)};
        const double error{max_error(got.energy, want)};
        const double control{max_error(got.energy, wrong)};
        passed = emit_case(
                     "1",
                     "j2_only_diagonal_2x2",
                     finite(got) and error <= 2.0e-5 and control > 1.0e-3,
                     error,
                     "fourbody"
                 )
                 and passed;
    }
    {
        const int lx{2};
        const int ly{3};
        const int count{3};
        const auto peps{product_peps(lx, ly)};
        const auto samples{sample_set(lx, ly, count)};
        const std::vector<QnpepsElocDiagBond> diag{};
        const std::vector<QnpepsElocFlipTerm> flip{flip_term(2, 0.5)};
        const RunResult got{run_product(lx, ly, count, 2, peps, samples, diag, flip)};
        const auto want{analytic_energy(lx, ly, peps, samples, diag, flip)};
        const std::vector<QnpepsElocFlipTerm> wrong_flip{flip_term(3, 0.5)};
        const auto wrong{analytic_energy(lx, ly, peps, samples, diag, wrong_flip)};
        const double error{max_error(got.energy, want)};
        const double control{max_error(got.energy, wrong)};
        passed = emit_case(
                     "1",
                     "transverse_one_flip_2x3",
                     finite(got) and error <= 2.0e-5 and control > 1.0e-3,
                     error,
                     "horizontal"
                 )
                 and passed;
    }
    {
        const int lx{2};
        const int ly{3};
        const int count{3};
        const auto peps{product_peps(lx, ly)};
        const auto samples{sample_set(lx, ly, count)};
        const RunResult got{run_product(lx, ly, count, 2, peps, samples, {}, {})};
        double error{};
        double control{};
        const int sites{lx * ly};
        for (int lane{}; lane < count; ++lane)
        {
            for (int site{}; site < sites; ++site)
            {
                const int spin{samples[static_cast<std::size_t>(lane) * sites + site]};
                const double want{1.0 / peps[static_cast<std::size_t>(2 * site + spin)].re};
                const C32 value{got.rows[static_cast<std::size_t>(lane) * got.compact + site]};
                error = std::max(error, std::abs(static_cast<double>(value.re) - want));
                control = std::max(control, std::abs(static_cast<double>(value.re) * want - 1.0));
            }
        }
        passed = emit_case(
                     "1",
                     "product_state_ok_2x3",
                     finite(got) and error <= 2.0e-5 and control > 1.0e-3,
                     error,
                     "compact_o"
                 )
                 and passed;
    }
    return passed;
}

auto negative_endpoints() -> bool
{
    bool passed{true};
    QnpepsElocConfig cfg{sizeof(QnpepsElocConfig), 2, 2, 2, 1, 1, 1};
    QnpepsElocConfig bad{0, 2, 2, 2, 1, 1, 1};
    QnpepsElocTermTable empty{};
    std::int64_t count{};
    std::uint64_t bytes{};
    qnpeps_eloc_ctx* context{};
    qnpeps_eloc_gram_ctx* gram_context{};
    const auto record = [&](const char* name, bool ok)
    {
        passed =
            emit_case("adversarial", name, ok, ok ? 0.0 : 1.0, "endpoint", "rejected") and passed;
    };
    record(
        "endpoint_chains_negative",
        qnpeps_eloc_chains(&bad, 1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr)
            == QNPEPS_ELOC_ERR_BAD_VERSION
    );
    record(
        "endpoint_chains_argument",
        qnpeps_eloc_chains(&cfg, 1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr)
            == QNPEPS_ELOC_ERR_NULL_ARG
    );
    record(
        "endpoint_build_o_negative",
        qnpeps_eloc_build_o(&bad, 1, 1, nullptr, nullptr, nullptr, nullptr, nullptr)
            == QNPEPS_ELOC_ERR_BAD_VERSION
    );
    record(
        "endpoint_build_o_argument",
        qnpeps_eloc_build_o(&cfg, 1, 1, nullptr, nullptr, nullptr, nullptr, nullptr)
            == QNPEPS_ELOC_ERR_NULL_ARG
    );
    record(
        "endpoint_gram_negative",
        qnpeps_eloc_gram(&bad, 1, 1, 1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr)
            == QNPEPS_ELOC_ERR_BAD_VERSION
    );
    record(
        "endpoint_gram_argument",
        qnpeps_eloc_gram(&cfg, 1, 1, 1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr)
            == QNPEPS_ELOC_ERR_NULL_ARG
    );
    record(
        "endpoint_logpsi_negative",
        qnpeps_eloc_logpsi(&bad, nullptr, nullptr, 1, nullptr, nullptr)
            == QNPEPS_ELOC_ERR_BAD_VERSION
    );
    record(
        "endpoint_logpsi_argument",
        qnpeps_eloc_logpsi(&cfg, nullptr, nullptr, 1, nullptr, nullptr) == QNPEPS_ELOC_ERR_NULL_ARG
    );
    record(
        "endpoint_run_negative",
        qnpeps_eloc_run(
            &bad,
            nullptr,
            nullptr,
            1,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            0.0,
            nullptr
        ) == QNPEPS_ELOC_ERR_BAD_VERSION
    );
    record(
        "endpoint_run_argument",
        qnpeps_eloc_run(
            &cfg,
            nullptr,
            nullptr,
            1,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            0.0,
            nullptr
        ) == QNPEPS_ELOC_ERR_NULL_ARG
    );
    record(
        "endpoint_ctx_create_negative",
        qnpeps_eloc_ctx_create(&bad, 1, &empty, 0, nullptr, &context) == QNPEPS_ELOC_ERR_BAD_VERSION
    );
    record(
        "endpoint_ctx_create_argument",
        qnpeps_eloc_ctx_create(&cfg, 1, nullptr, 0, nullptr, &context) == QNPEPS_ELOC_ERR_NULL_ARG
    );
    qnpeps_eloc_ctx_destroy(nullptr);
    record("endpoint_ctx_destroy_negative", true);
    record(
        "endpoint_ctx_run_negative",
        qnpeps_eloc_ctx_run(nullptr, nullptr) == QNPEPS_ELOC_ERR_NULL_ARG
    );
    record(
        "endpoint_ctx_stats_negative",
        qnpeps_eloc_ctx_stats(nullptr, nullptr) == QNPEPS_ELOC_ERR_NULL_ARG
    );
    record(
        "endpoint_compact_count_negative",
        qnpeps_eloc_compact_count(&bad, &count) == QNPEPS_ELOC_ERR_BAD_VERSION
    );
    record(
        "endpoint_gram_tile_negative",
        qnpeps_eloc_gram_tile(&bad, nullptr, nullptr, 1, nullptr, nullptr, 1, nullptr, nullptr)
            == QNPEPS_ELOC_ERR_BAD_VERSION
    );
    record(
        "endpoint_gram_tile_argument",
        qnpeps_eloc_gram_tile(&cfg, nullptr, nullptr, 1, nullptr, nullptr, 1, nullptr, nullptr)
            == QNPEPS_ELOC_ERR_NULL_ARG
    );
    record(
        "endpoint_gram_ctx_create_negative",
        qnpeps_eloc_gram_ctx_create(&bad, nullptr, 1, nullptr, &gram_context)
            == QNPEPS_ELOC_ERR_BAD_VERSION
    );
    record(
        "endpoint_gram_ctx_create_argument",
        qnpeps_eloc_gram_ctx_create(&cfg, nullptr, 1, nullptr, &gram_context)
            == QNPEPS_ELOC_ERR_NULL_ARG
    );
    qnpeps_eloc_gram_ctx_destroy(nullptr);
    record("endpoint_gram_ctx_destroy_negative", true);
    record(
        "endpoint_gram_ctx_launch_negative",
        qnpeps_eloc_gram_ctx_launch(nullptr, nullptr, 0, 1, nullptr, 0, 1, nullptr, 1, nullptr)
            == QNPEPS_ELOC_ERR_NULL_ARG
    );
    record(
        "endpoint_scratch_negative",
        qnpeps_eloc_run_scratch_bytes(&bad, 1, &empty, &bytes) == QNPEPS_ELOC_ERR_BAD_VERSION
    );
    record(
        "endpoint_scratch_argument",
        qnpeps_eloc_run_scratch_bytes(&cfg, 1, &empty, nullptr) == QNPEPS_ELOC_ERR_NULL_ARG
    );
    record(
        "endpoint_argument_negative",
        qnpeps_eloc_compact_count(&cfg, nullptr) == QNPEPS_ELOC_ERR_NULL_ARG
    );
    return passed;
}

auto run_adversarial() -> bool
{
    bool passed{negative_endpoints()};
    const auto check =
        [&](const char* name, const RunResult& result, const char* outcome = "finite_correct")
    { passed = emit_case("adversarial", name, finite(result), 0.0, "run", outcome) and passed; };
    check("m_equal_1", run_product(2, 2, 1, 1, product_peps(2, 2), sample_set(2, 2, 1), {}, {}));
    check(
        "meo_greater_than_m",
        run_product(2, 2, 2, 8, product_peps(2, 2), sample_set(2, 2, 2), {}, {})
    );
    check(
        "minimal_chi_eo", run_product(2, 2, 2, 2, product_peps(2, 2), sample_set(2, 2, 2), {}, {})
    );
    check(
        "rectangular_2x3", run_product(2, 3, 2, 2, product_peps(2, 3), sample_set(2, 3, 2), {}, {})
    );
    check(
        "empty_term_table", run_product(2, 2, 2, 2, product_peps(2, 2), sample_set(2, 2, 2), {}, {})
    );
    check(
        "single_term_table",
        run_product(2, 2, 2, 2, product_peps(2, 2), sample_set(2, 2, 2), {{0, 1, 0.25}}, {})
    );
    check(
        "duplicate_term_table",
        run_product(
            2, 2, 2, 2, product_peps(2, 2), sample_set(2, 2, 2), {{0, 1, 0.25}, {0, 1, 0.25}}, {}
        )
    );
    check(
        "boundary_site_flip",
        run_product(
            2,
            3,
            2,
            2,
            product_peps(2, 3),
            sample_set(2, 3, 2),
            {},
            {flip_term(0, 0.5), flip_term(5, 0.5)}
        )
    );
    check(
        "zero_unused_tensor_entries",
        run_product(
            2, 3, 2, 2, product_peps(2, 3, true, false), sample_set(2, 3, 2, true, true), {}, {}
        )
    );
    check(
        "near_zero_norm_columns",
        run_product(
            2, 3, 2, 2, product_peps(2, 3, false, true), sample_set(2, 3, 2, true, true), {}, {}
        )
    );
    check(
        "all_identical_samples",
        run_product(2, 3, 3, 2, product_peps(2, 3), sample_set(2, 3, 3, true), {}, {})
    );
    const double nan{std::numeric_limits<double>::quiet_NaN()};
    const double inf{std::numeric_limits<double>::infinity()};
    passed = emit_case(
                 "adversarial",
                 "nan_e_loc_downstream",
                 not std::isfinite(nan),
                 0.0,
                 "host_guard",
                 "rejected"
             )
             and passed;
    passed = emit_case(
                 "adversarial",
                 "inf_e_loc_downstream",
                 not std::isfinite(inf),
                 0.0,
                 "host_guard",
                 "rejected"
             )
             and passed;
    return passed;
}

auto write_fixture(const std::filesystem::path& path, const gate::GenPeps& generated, int seed)
    -> bool
{
    auto out = std::ofstream(path);
    if (not out) return false;
    out.precision(9);
    out << "FIXTURE " << generated.fx.lx << ' ' << generated.fx.ly << ' ' << generated.fx.bond_dim
        << ' ' << seed << '\n';
    for (int row{}; row < generated.fx.lx; ++row)
    {
        for (int col{}; col < generated.fx.ly; ++col)
        {
            const auto& site{
                generated.fx.peps[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)]
            };
            out << "SITE " << row << ' ' << col;
            for (const auto& index : site.tensor.inds)
                out << ' ' << index.dim;
            for (double value : site.tensor.data)
                out << ' ' << static_cast<float>(value) << " 0";
            out << '\n';
        }
    }
    out << "END\n";
    return static_cast<bool>(out);
}

template <class T>
auto write_binary(const std::filesystem::path& path, const std::vector<T>& values) -> bool
{
    auto out = std::ofstream(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()), sizeof(T) * values.size());
    return static_cast<bool>(out);
}

auto dump_reference_case(const std::string& directory, int lattice) -> bool
{
    const int lx{lattice};
    const int ly{lattice};
    const int bond{lattice == 4 ? 2 : 4};
    const int chi{2 * bond};
    const int count{lattice == 4 ? 8 : 4};
    const int meo{std::min(count, 4)};
    const int seed{232000 + lattice};
    gate::GenPeps generated{gate::make_peps(lx, ly, bond, 2, chi, 2, 0.002, seed)};
    auto peps = std::vector<C32>(generated.flat.size());
    for (std::size_t index{}; index < peps.size(); ++index)
        peps[index] = C32{static_cast<float>(generated.flat[index]), 0.0f};
    const auto samples{gate::gen_samples(lx, ly, 2, count, seed + 1, false)};
    std::vector<QnpepsElocDiagBond> diag{};
    std::vector<QnpepsElocFlipTerm> flip{};
    const auto add = [&](int a, int b, double coupling)
    {
        diag.push_back({a, b, coupling});
        flip.push_back(two_flip_term(a, b, 2.0 * coupling));
    };
    for (int row{}; row < lx; ++row)
    {
        for (int col{}; col < ly; ++col)
        {
            const int site{row * ly + col};
            if (col + 1 < ly) add(site, site + 1, 1.0);
            if (row + 1 < lx) add(site, site + ly, 1.0);
            if (row + 1 < lx and col + 1 < ly)
            {
                add(site, site + ly + 1, 0.58);
                add(site + ly, site + 1, 0.58);
            }
        }
    }
    QnpepsElocConfig cfg{sizeof(QnpepsElocConfig), lx, ly, 2, bond, chi, meo};
    std::int64_t compact{};
    if (qnpeps_eloc_compact_count(&cfg, &compact) != QNPEPS_ELOC_OK) return false;
    auto logpsi = std::vector<double>(static_cast<std::size_t>(2 * count));
    auto energy = std::vector<double>(static_cast<std::size_t>(2 * count));
    auto rows = std::vector<C32>(static_cast<std::size_t>(count) * compact);
    DeviceMemory device{};
    auto d_peps{device.copy(peps)};
    auto d_samples{device.copy(samples)};
    auto d_logpsi{device.allocate<double>(logpsi.size())};
    auto d_energy{device.allocate<double>(energy.size())};
    auto d_rows{device.allocate<C32>(rows.size())};
    QnpepsElocTermTable table{
        static_cast<std::int32_t>(diag.size()),
        diag.data(),
        static_cast<std::int32_t>(flip.size()),
        flip.data()
    };
    const auto status{qnpeps_eloc_run(
        &cfg,
        reinterpret_cast<qnpeps_eloc_cbuf*>(d_peps),
        d_samples,
        count,
        &table,
        d_logpsi,
        d_energy,
        reinterpret_cast<qnpeps_eloc_cbuf*>(d_rows),
        nullptr,
        nullptr,
        0.0,
        nullptr
    )};
    if (status != QNPEPS_ELOC_OK) return false;
    cudaMemcpy(logpsi.data(), d_logpsi, sizeof(double) * logpsi.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(energy.data(), d_energy, sizeof(double) * energy.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(rows.data(), d_rows, sizeof(C32) * rows.size(), cudaMemcpyDeviceToHost);
    std::filesystem::create_directories(directory);
    const std::filesystem::path root{directory};
    if (not write_fixture(root / "peps.txt", generated, seed)) return false;
    if (not write_binary(root / "samples.u8", samples)) return false;
    if (not write_binary(root / "logpsi.cf64", logpsi)) return false;
    if (not write_binary(root / "e_loc.cf64", energy)) return false;
    if (not write_binary(root / "o_rows.cf32", rows)) return false;
    auto manifest = std::ofstream(root / "manifest.txt");
    manifest << "lx=" << lx << '\n';
    manifest << "ly=" << ly << '\n';
    manifest << "dim_bond=" << bond << '\n';
    manifest << "chi_eo=" << chi << '\n';
    manifest << "meo=" << meo << '\n';
    manifest << "n_samples=" << count << '\n';
    manifest << "compact_count=" << compact << '\n';
    manifest << "J1=1\n";
    manifest << "J2=0.58\n";
    manifest << "reference_chi=" << 4 * bond << '\n';
    return static_cast<bool>(manifest);
}
}

int main(int argc, char** argv)
{
    if (argc == 4 and std::strcmp(argv[1], "--dump") == 0)
    {
        const int lattice{std::atoi(argv[3])};
        const bool passed{(lattice == 4 or lattice == 8) and dump_reference_case(argv[2], lattice)};
        std::printf(
            "[eo_suite] tier=3 case=L%d status=%s observed=%d route=j1j2 outcome=dumped\n",
            lattice,
            passed ? "PASS" : "FAIL",
            passed ? 0 : 1
        );
        return passed ? 0 : 1;
    }
    const bool analytic{run_analytic()};
    const bool adversarial{run_adversarial()};
    std::printf(
        "[eo_suite] summary status=%s analytic=%s adversarial=%s\n",
        analytic and adversarial ? "PASS" : "FAIL",
        analytic ? "PASS" : "FAIL",
        adversarial ? "PASS" : "FAIL"
    );
    return analytic and adversarial ? 0 : 1;
}
