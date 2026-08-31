#include "../core/types.cuh"
#include "capi/qnpeps.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace
{
using cfl = std::complex<float>;
using cdb = std::complex<double>;

auto bond(int length, int position, int dimension) -> int
{
    return position <= 0 or position >= length ? 1 : dimension;
}

auto cuda_ok(cudaError_t status, const char* operation) -> bool
{
    if (status == cudaSuccess) return true;
    std::fprintf(
        stderr, "[minsr_gate] CUDA failure in %s with %s\n", operation, cudaGetErrorString(status)
    );
    return false;
}

auto hash_bytes(const void* data, std::size_t bytes) -> std::uint64_t
{
    const auto* pointer{static_cast<const std::uint8_t*>(data)};
    std::uint64_t hash{1469598103934665603ull};
    for (std::size_t index{}; index < bytes; ++index)
    {
        hash ^= pointer[index];
        hash *= 1099511628211ull;
    }
    return hash;
}

template <class T>
auto device_allocate(std::size_t count) -> T*
{
    T* pointer{};
    if (not cuda_ok(
            cudaMalloc(&pointer, std::max<std::size_t>(count, 1) * sizeof(T)), "allocation"
        ))
        return nullptr;
    return pointer;
}

struct Fixture
{
    QnpepsMinsrDesc descriptor{};
    int samples{};
    int sites{};
    std::int64_t compact{};
    std::int64_t dense{};
    std::vector<int> slices{};
    std::vector<int> slot_site{};
    std::vector<std::uint8_t> spins{};
    std::vector<cfl> rows{};
    std::vector<double> logpsi{};
    std::vector<double> e_loc{};
    std::vector<double> logq{};
    std::vector<cfl> gram{};
};

auto make_fixture(int lx, int ly, int dim_bond, int samples, double spread, std::uint64_t seed)
    -> Fixture
{
    Fixture fixture{};
    fixture.descriptor.struct_size = sizeof(QnpepsMinsrDesc);
    fixture.descriptor.lx = lx;
    fixture.descriptor.ly = ly;
    fixture.descriptor.dim_phys = 2;
    fixture.descriptor.dim_bond = dim_bond;
    fixture.descriptor.diagnostics = 0;
    fixture.descriptor.reserved = 0;
    fixture.descriptor.n_samples = samples;
    fixture.descriptor.host_tile_bytes = 0;
    fixture.samples = samples;
    fixture.sites = lx * ly;

    for (int row{}; row < lx; ++row)
    {
        for (int column{}; column < ly; ++column)
        {
            fixture.slices.push_back(
                bond(ly, column, dim_bond) * bond(lx, row + 1, dim_bond)
                * bond(ly, column + 1, dim_bond) * bond(lx, row, dim_bond)
            );
        }
    }
    fixture.compact = 0;
    for (int site{}; site < fixture.sites; ++site)
    {
        for (int local{}; local < fixture.slices[static_cast<std::size_t>(site)]; ++local)
            fixture.slot_site.push_back(site);
        fixture.compact += fixture.slices[static_cast<std::size_t>(site)];
    }
    fixture.dense = 2 * fixture.compact;

    std::mt19937_64 engine{seed};
    std::normal_distribution<double> gauss{0.0, 1.0};
    std::uniform_int_distribution<int> spin{0, 1};

    fixture.spins.resize(static_cast<std::size_t>(fixture.samples) * fixture.sites);
    for (std::size_t index{}; index < fixture.spins.size(); ++index)
        fixture.spins[index] = static_cast<std::uint8_t>(spin(engine));

    fixture.rows.resize(static_cast<std::size_t>(fixture.samples) * fixture.compact);
    for (int sample{}; sample < fixture.samples; ++sample)
    {
        for (std::int64_t local{}; local < fixture.compact; ++local)
        {
            const double scale{
                spread > 0.0 ? std::pow(10.0, spread * (static_cast<double>(local % 7) / 6.0 - 0.5))
                             : 1.0
            };
            fixture.rows[static_cast<std::size_t>(sample * fixture.compact + local)] =
                cfl{static_cast<float>(scale * (0.31 * gauss(engine) + 0.07)),
                    static_cast<float>(scale * (0.23 * gauss(engine) - 0.11))};
        }
    }

    fixture.logpsi.resize(static_cast<std::size_t>(2 * fixture.samples));
    fixture.e_loc.resize(static_cast<std::size_t>(2 * fixture.samples));
    fixture.logq.resize(static_cast<std::size_t>(fixture.samples));
    for (int sample{}; sample < fixture.samples; ++sample)
    {
        const double real{-0.21 * sample + 0.013 * sample * sample};
        fixture.logpsi[static_cast<std::size_t>(2 * sample)] = real;
        fixture.logpsi[static_cast<std::size_t>(2 * sample + 1)] = 0.37 * gauss(engine);
        fixture.e_loc[static_cast<std::size_t>(2 * sample)] = -1.7 + 0.19 * gauss(engine);
        fixture.e_loc[static_cast<std::size_t>(2 * sample + 1)] = 0.41 * gauss(engine);
        fixture.logq[static_cast<std::size_t>(sample)] = 2.0 * real + 0.75 * gauss(engine);
    }

    fixture.gram.resize(static_cast<std::size_t>(fixture.samples) * fixture.samples);
    for (int left{}; left < fixture.samples; ++left)
    {
        for (int right{}; right < fixture.samples; ++right)
        {
            cdb accumulator{0.0, 0.0};
            std::int64_t offset{0};
            for (int site{}; site < fixture.sites; ++site)
            {
                if (fixture.spins[static_cast<std::size_t>(left * fixture.sites + site)]
                    == fixture.spins[static_cast<std::size_t>(right * fixture.sites + site)])
                {
                    for (int local{}; local < fixture.slices[static_cast<std::size_t>(site)];
                         ++local)
                    {
                        const cfl a{fixture.rows[static_cast<std::size_t>(
                            left * fixture.compact + offset + local
                        )]};
                        const cfl b{fixture.rows[static_cast<std::size_t>(
                            right * fixture.compact + offset + local
                        )]};
                        accumulator += std::conj(cdb{a.real(), a.imag()}) * cdb{b.real(), b.imag()};
                    }
                }
                offset += fixture.slices[static_cast<std::size_t>(site)];
            }
            fixture.gram[static_cast<std::size_t>(left * fixture.samples + right)] =
                cfl{static_cast<float>(accumulator.real()), static_cast<float>(accumulator.imag())};
        }
    }
    return fixture;
}

auto jacobi_hermitian(
    std::vector<cdb>& matrix, int order, std::vector<double>& eigenvalues, std::vector<cdb>& vectors
) -> void
{
    const auto n = static_cast<std::size_t>(order);
    vectors.assign(n * n, cdb{0.0, 0.0});
    for (std::size_t index{}; index < n; ++index)
        vectors[index + index * n] = cdb{1.0, 0.0};

    for (int sweep{}; sweep < 100; ++sweep)
    {
        double off{0.0};
        double diagonal{0.0};
        for (std::size_t column{}; column < n; ++column)
        {
            for (std::size_t row{}; row < n; ++row)
            {
                if (row != column)
                    off += std::norm(matrix[row + column * n]);
                else
                    diagonal += std::norm(matrix[row + column * n]);
            }
        }
        if (off <= 1.0e-30 * std::max(diagonal, 1.0e-300)) break;

        for (std::size_t p{}; p + 1 < n; ++p)
        {
            for (std::size_t q{p + 1}; q < n; ++q)
            {
                const cdb apq{matrix[p + q * n]};
                const double magnitude{std::abs(apq)};
                if (magnitude <= 1.0e-300) continue;
                const cdb phase{apq / magnitude};
                const double app{matrix[p + p * n].real()};
                const double aqq{matrix[q + q * n].real()};
                const double tau{(aqq - app) / (2.0 * magnitude)};
                const double t{
                    tau == 0.0
                        ? 1.0
                        : (tau > 0.0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau * tau))
                };
                const double c{1.0 / std::sqrt(1.0 + t * t)};
                const double s{t * c};
                const cdb conjugate_phase{std::conj(phase)};
                const cdb u11{c, 0.0};
                const cdb u12{s, 0.0};
                const cdb u21{-s * conjugate_phase};
                const cdb u22{c * conjugate_phase};

                for (std::size_t row{}; row < n; ++row)
                {
                    const cdb left{matrix[row + p * n]};
                    const cdb right{matrix[row + q * n]};
                    matrix[row + p * n] = left * u11 + right * u21;
                    matrix[row + q * n] = left * u12 + right * u22;
                }
                for (std::size_t column{}; column < n; ++column)
                {
                    const cdb top{matrix[p + column * n]};
                    const cdb bottom{matrix[q + column * n]};
                    matrix[p + column * n] = std::conj(u11) * top + std::conj(u21) * bottom;
                    matrix[q + column * n] = std::conj(u12) * top + std::conj(u22) * bottom;
                }
                for (std::size_t row{}; row < n; ++row)
                {
                    const cdb left{vectors[row + p * n]};
                    const cdb right{vectors[row + q * n]};
                    vectors[row + p * n] = left * u11 + right * u21;
                    vectors[row + q * n] = left * u12 + right * u22;
                }
            }
        }
    }

    auto order_index = std::vector<std::size_t>(n);
    for (std::size_t index{}; index < n; ++index)
        order_index[index] = index;
    auto raw = std::vector<double>(n);
    for (std::size_t index{}; index < n; ++index)
        raw[index] = matrix[index + index * n].real();
    std::sort(
        order_index.begin(),
        order_index.end(),
        [&raw](std::size_t left, std::size_t right) { return raw[left] < raw[right]; }
    );

    eigenvalues.assign(n, 0.0);
    auto sorted = std::vector<cdb>(n * n, cdb{0.0, 0.0});
    for (std::size_t column{}; column < n; ++column)
    {
        eigenvalues[column] = raw[order_index[column]];
        for (std::size_t row{}; row < n; ++row)
            sorted[row + column * n] = vectors[row + order_index[column] * n];
    }
    vectors = sorted;
}

struct Reference
{
    std::vector<cdb> theta_dot{};
    double e_mean_real{};
    double e_mean_imag{};
    double e_var{};
    double ess{};
    bool solved{};
};

auto host_reference(
    const Fixture& fixture, double relative_cut, double absolute_cut, bool swapped_scatter
) -> Reference
{
    Reference reference{};
    const int ns{fixture.samples};
    const auto count = static_cast<std::size_t>(ns);
    reference.theta_dot.assign(static_cast<std::size_t>(fixture.dense), cdb{0.0, 0.0});

    auto ratio = std::vector<double>(count);
    for (int j{}; j < ns; ++j)
    {
        ratio[static_cast<std::size_t>(j)] = 2.0 * fixture.logpsi[static_cast<std::size_t>(2 * j)]
                                             - fixture.logq[static_cast<std::size_t>(j)];
    }
    double maximum{ratio[0]};
    for (int j{}; j < ns; ++j)
        maximum = std::max(maximum, ratio[static_cast<std::size_t>(j)]);
    double sum_exponential{0.0};
    for (int j{}; j < ns; ++j)
        sum_exponential += std::exp(ratio[static_cast<std::size_t>(j)] - maximum);
    const double logz{maximum + std::log(sum_exponential) - std::log(static_cast<double>(ns))};
    auto weights = std::vector<double>(count);
    double weight_sum{0.0};
    for (int j{}; j < ns; ++j)
    {
        weights[static_cast<std::size_t>(j)] = std::exp(ratio[static_cast<std::size_t>(j)] - logz);
        weight_sum += weights[static_cast<std::size_t>(j)];
    }
    const double weight_mean{weight_sum / static_cast<double>(ns)};
    double sum_w{0.0};
    double sum_w2{0.0};
    for (int j{}; j < ns; ++j)
    {
        weights[static_cast<std::size_t>(j)] /= weight_mean;
        sum_w += weights[static_cast<std::size_t>(j)];
        sum_w2 += weights[static_cast<std::size_t>(j)] * weights[static_cast<std::size_t>(j)];
    }
    reference.ess = sum_w * sum_w / sum_w2;

    cdb e_mean{0.0, 0.0};
    for (int j{}; j < ns; ++j)
    {
        e_mean += weights[static_cast<std::size_t>(j)]
                  * cdb{
                      fixture.e_loc[static_cast<std::size_t>(2 * j)],
                      fixture.e_loc[static_cast<std::size_t>(2 * j + 1)]
                  };
    }
    e_mean /= static_cast<double>(ns);
    double e_var{0.0};
    for (int j{}; j < ns; ++j)
    {
        const cdb delta{
            cdb{fixture.e_loc[static_cast<std::size_t>(2 * j)],
                fixture.e_loc[static_cast<std::size_t>(2 * j + 1)]}
            - e_mean
        };
        e_var += weights[static_cast<std::size_t>(j)] * std::norm(delta);
    }
    e_var = e_var / static_cast<double>(ns)
            * (static_cast<double>(ns) / (static_cast<double>(ns) - 1.0));
    reference.e_mean_real = e_mean.real();
    reference.e_mean_imag = e_mean.imag();
    reference.e_var = e_var;

    auto centered = std::vector<cdb>(count);
    for (int j{}; j < ns; ++j)
    {
        centered[static_cast<std::size_t>(j)] =
            (cdb{fixture.e_loc[static_cast<std::size_t>(2 * j)],
                 fixture.e_loc[static_cast<std::size_t>(2 * j + 1)]}
             - e_mean)
            * std::sqrt(weights[static_cast<std::size_t>(j)]);
    }

    const auto gram_at = [&fixture, ns](int j, int k) -> cdb
    {
        const cfl value{fixture.gram[static_cast<std::size_t>(j * ns + k)]};
        return cdb{value.real(), value.imag()};
    };

    auto beta = std::vector<cdb>(count, cdb{0.0, 0.0});
    for (int j{}; j < ns; ++j)
    {
        double real{0.0};
        double imag{0.0};
        for (int l{}; l < ns; ++l)
        {
            const cdb value{gram_at(j, l)};
            real += value.real() * weights[static_cast<std::size_t>(l)];
            imag += value.imag() * weights[static_cast<std::size_t>(l)];
        }
        beta[static_cast<std::size_t>(j)] =
            cdb{real / static_cast<double>(ns), imag / static_cast<double>(ns)};
    }
    cdb mu{0.0, 0.0};
    for (int j{}; j < ns; ++j)
        mu += weights[static_cast<std::size_t>(j)] * beta[static_cast<std::size_t>(j)];
    mu /= static_cast<double>(ns);

    auto matrix = std::vector<cdb>(count * count, cdb{0.0, 0.0});
    for (int j{}; j < ns; ++j)
    {
        for (int k{}; k < ns; ++k)
        {
            const cdb value{gram_at(j, k)};
            const cdb bj{beta[static_cast<std::size_t>(j)]};
            const cdb bk{beta[static_cast<std::size_t>(k)]};
            const double sw{std::sqrt(
                weights[static_cast<std::size_t>(j)] * weights[static_cast<std::size_t>(k)]
            )};
            const double real{(value.real() - bj.real() - bk.real() + mu.real()) * sw};
            const double imag{(value.imag() - bj.imag() + bk.imag() + mu.imag()) * sw};
            matrix[static_cast<std::size_t>(j) + static_cast<std::size_t>(k) * count] =
                cdb{real, -imag};
        }
    }
    for (std::size_t column{}; column < count; ++column)
    {
        for (std::size_t row{column + 1}; row < count; ++row)
            matrix[column + row * count] = std::conj(matrix[row + column * count]);
    }
    for (std::size_t index{}; index < count; ++index)
        matrix[index + index * count] = cdb{matrix[index + index * count].real(), 0.0};

    std::vector<double> eigenvalues{};
    std::vector<cdb> vectors{};
    jacobi_hermitian(matrix, ns, eigenvalues, vectors);
    const double lambda_max{eigenvalues[count - 1]};
    if (not(lambda_max > 0.0)) return reference;
    reference.solved = true;

    auto projected = std::vector<cdb>(count, cdb{0.0, 0.0});
    for (std::size_t column{}; column < count; ++column)
    {
        cdb accumulator{0.0, 0.0};
        for (std::size_t row{}; row < count; ++row)
            accumulator += std::conj(vectors[row + column * count]) * centered[row];
        projected[column] = accumulator;
    }
    for (std::size_t index{}; index < count; ++index)
    {
        const double lambda{eigenvalues[index]};
        double inverse{0.0};
        if (lambda / lambda_max >= 1.0e-13)
        {
            const double soft{
                std::pow((lambda_max * relative_cut + absolute_cut) / std::abs(lambda), 6.0)
            };
            inverse = 1.0 / (lambda * (1.0 + soft));
        }
        projected[index] *= inverse;
    }
    auto raw = std::vector<cdb>(count, cdb{0.0, 0.0});
    for (std::size_t row{}; row < count; ++row)
    {
        cdb accumulator{0.0, 0.0};
        for (std::size_t column{}; column < count; ++column)
            accumulator += vectors[row + column * count] * projected[column];
        raw[row] = -accumulator;
    }

    auto scaled = std::vector<cdb>(count, cdb{0.0, 0.0});
    cdb scaled_sum{0.0, 0.0};
    for (std::size_t index{}; index < count; ++index)
    {
        scaled[index] = std::sqrt(weights[index]) * raw[index];
        scaled_sum += scaled[index];
    }
    auto coefficients = std::vector<cdb>(count, cdb{0.0, 0.0});
    for (std::size_t index{}; index < count; ++index)
        coefficients[index] =
            scaled[index] - (scaled_sum / static_cast<double>(ns)) * weights[index];

    for (std::int64_t slot{}; slot < fixture.compact; ++slot)
    {
        const int site{fixture.slot_site[static_cast<std::size_t>(slot)]};
        for (int j{}; j < ns; ++j)
        {
            const int value{
                static_cast<int>(fixture.spins[static_cast<std::size_t>(j * fixture.sites + site)])
            };
            const cfl row{fixture.rows[static_cast<std::size_t>(j * fixture.compact + slot)]};
            const std::int64_t index{
                swapped_scatter ? value * fixture.compact + slot : 2 * slot + value
            };
            reference.theta_dot[static_cast<std::size_t>(index)] +=
                coefficients[static_cast<std::size_t>(j)] * std::conj(cdb{row.real(), row.imag()});
        }
    }
    return reference;
}

struct Recorder
{
    bool pass{true};

    auto add(const char* check, const char* expected, const std::string& observed, bool ok) -> void
    {
        std::printf(
            "[minsr_gate] check=%s expected=%s observed=%s status=%s\n",
            check,
            expected,
            observed.c_str(),
            ok ? "PASS" : "FAIL"
        );
        pass = pass and ok;
    }
};

auto format(double value) -> std::string
{
    char text[64]{};
    std::snprintf(text, sizeof(text), "%.9g", value);
    return text;
}

auto normalized_error(const std::vector<cfl>& observed, const std::vector<cdb>& reference) -> double
{
    double scale{1.0e-30};
    for (const cdb& value : reference)
        scale = std::max(scale, std::abs(value));
    double worst{0.0};
    for (std::size_t index{}; index < observed.size(); ++index)
    {
        const cdb delta{cdb{observed[index].real(), observed[index].imag()} - reference[index]};
        worst = std::max(worst, std::abs(delta) / scale);
    }
    return worst;
}

struct Device
{
    std::uint8_t* spins{};
    double* logpsi{};
    double* e_loc{};
    double* logq{};
    cuFloatComplex* gram{};
    cuFloatComplex* rows{};
    cuFloatComplex* theta{};
};

auto upload(const Fixture& fixture, Device& device) -> bool
{
    device.spins = device_allocate<std::uint8_t>(fixture.spins.size());
    device.logpsi = device_allocate<double>(fixture.logpsi.size());
    device.e_loc = device_allocate<double>(fixture.e_loc.size());
    device.logq = device_allocate<double>(fixture.logq.size());
    device.gram = device_allocate<cuFloatComplex>(fixture.gram.size());
    device.rows = device_allocate<cuFloatComplex>(fixture.rows.size());
    device.theta = device_allocate<cuFloatComplex>(static_cast<std::size_t>(fixture.dense));
    if (not device.spins or not device.logpsi or not device.e_loc or not device.logq
        or not device.gram or not device.rows or not device.theta)
        return false;
    cudaMemcpy(device.spins, fixture.spins.data(), fixture.spins.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(
        device.logpsi,
        fixture.logpsi.data(),
        fixture.logpsi.size() * sizeof(double),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        device.e_loc,
        fixture.e_loc.data(),
        fixture.e_loc.size() * sizeof(double),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        device.logq,
        fixture.logq.data(),
        fixture.logq.size() * sizeof(double),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        device.gram,
        fixture.gram.data(),
        fixture.gram.size() * sizeof(cuFloatComplex),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        device.rows,
        fixture.rows.data(),
        fixture.rows.size() * sizeof(cuFloatComplex),
        cudaMemcpyHostToDevice
    );
    return true;
}

auto release(Device& device) -> void
{
    cudaFree(device.theta);
    cudaFree(device.rows);
    cudaFree(device.gram);
    cudaFree(device.logq);
    cudaFree(device.e_loc);
    cudaFree(device.logpsi);
    cudaFree(device.spins);
}

auto make_args(
    const Fixture& fixture,
    const Device& device,
    double relative_cut,
    double absolute_cut,
    double* e_mean,
    double* e_var,
    double* ess
) -> QnpepsMinsrArgs
{
    QnpepsMinsrArgs args{};
    args.struct_size = sizeof(QnpepsMinsrArgs);
    args.reserved = 0;
    args.samples = device.spins;
    args.samples_bytes = fixture.spins.size();
    args.logpsi = device.logpsi;
    args.logpsi_bytes = fixture.logpsi.size() * sizeof(double);
    args.e_loc = device.e_loc;
    args.e_loc_bytes = fixture.e_loc.size() * sizeof(double);
    args.logq = device.logq;
    args.logq_bytes = fixture.logq.size() * sizeof(double);
    args.gram = device.gram;
    args.gram_bytes = fixture.gram.size() * sizeof(cuFloatComplex);
    args.o_rows_device = device.rows;
    args.o_rows_host = nullptr;
    args.o_rows_bytes = fixture.rows.size() * sizeof(cuFloatComplex);
    args.theta_dot_out = device.theta;
    args.theta_dot_out_bytes = static_cast<std::uint64_t>(fixture.dense) * sizeof(cuFloatComplex);
    args.relative_cut = relative_cut;
    args.absolute_cut = absolute_cut;
    args.e_mean_out = e_mean;
    args.e_var_out = e_var;
    args.ess_out = ess;
    args.stream = nullptr;
    return args;
}

auto download(const Fixture& fixture, const Device& device) -> std::vector<cfl>
{
    auto theta = std::vector<cfl>(static_cast<std::size_t>(fixture.dense));
    cudaMemcpy(
        theta.data(), device.theta, theta.size() * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost
    );
    return theta;
}

auto relative_l2(const std::vector<cfl>& observed, const std::vector<cfl>& reference) -> double
{
    double numerator{};
    double denominator{};
    for (std::size_t index{}; index < observed.size(); ++index)
    {
        numerator += std::norm(
            cdb{observed[index].real(), observed[index].imag()}
            - cdb{reference[index].real(), reference[index].imag()}
        );
        denominator += std::norm(cdb{reference[index].real(), reference[index].imag()});
    }
    return std::sqrt(numerator / std::max(denominator, 1.0e-300));
}

auto vector_norm(const std::vector<cfl>& values) -> double
{
    double total{};
    for (const cfl value : values)
        total += std::norm(cdb{value.real(), value.imag()});
    return std::sqrt(total);
}

auto soft_inverse(double lambda, double lambda_max, double relative_cut, double absolute_cut)
    -> double
{
    if (not(lambda_max > 0.0) or lambda / lambda_max < 1.0e-13) return 0.0;
    const double scale{lambda_max * relative_cut + absolute_cut};
    return 1.0 / (lambda * (1.0 + std::pow(scale / std::abs(lambda), 6.0)));
}

auto analytic_fixture(int samples) -> Fixture
{
    Fixture fixture{make_fixture(2, 2, 1, samples, 0.0, 0xA11CEull)};
    std::fill(fixture.spins.begin(), fixture.spins.end(), std::uint8_t{0});
    std::fill(fixture.rows.begin(), fixture.rows.end(), cfl{0.0f, 0.0f});
    std::fill(fixture.logpsi.begin(), fixture.logpsi.end(), 0.0);
    std::fill(fixture.e_loc.begin(), fixture.e_loc.end(), 0.0);
    std::fill(fixture.logq.begin(), fixture.logq.end(), 0.0);
    std::fill(fixture.gram.begin(), fixture.gram.end(), cfl{0.0f, 0.0f});
    return fixture;
}

auto gram_from_rows(Fixture& fixture) -> void
{
    for (int left{}; left < fixture.samples; ++left)
    {
        for (int right{}; right < fixture.samples; ++right)
        {
            cdb total{0.0, 0.0};
            for (std::int64_t slot{}; slot < fixture.compact; ++slot)
            {
                const cfl a{fixture.rows[static_cast<std::size_t>(left * fixture.compact + slot)]};
                const cfl b{fixture.rows[static_cast<std::size_t>(right * fixture.compact + slot)]};
                total += std::conj(cdb{a.real(), a.imag()}) * cdb{b.real(), b.imag()};
            }
            fixture.gram[static_cast<std::size_t>(left * fixture.samples + right)] =
                cfl{static_cast<float>(total.real()), static_cast<float>(total.imag())};
        }
    }
}

struct AnalyticRun
{
    qnpeps_status status{QNPEPS_ERR_INTERNAL};
    std::vector<cfl> theta{};
    qnpeps::CuArray<double, 2> e_mean{};
    double e_var{};
    double ess{};
};

auto run_analytic(const Fixture& fixture, double relative_cut, double absolute_cut) -> AnalyticRun
{
    AnalyticRun result{};
    Device device{};
    if (not upload(fixture, device)) return result;
    QnpepsMinsrArgs args{make_args(
        fixture,
        device,
        relative_cut,
        absolute_cut,
        result.e_mean.data(),
        &result.e_var,
        &result.ess
    )};
    result.status = qnpeps_minsr(&fixture.descriptor, &args);
    if (result.status == QNPEPS_OK) result.theta = download(fixture, device);
    release(device);
    return result;
}

auto analytic_case_identity_diagonal(Recorder& recorder) -> bool
{
    const double relative_cut{0.2};
    const double absolute_cut{0.05};
    Fixture identity{analytic_fixture(4)};
    const qnpeps::CuArray<double, 4> energies{1.0, -1.0, 0.5, -0.5};
    for (int sample{}; sample < 4; ++sample)
    {
        identity.rows[static_cast<std::size_t>(sample * identity.compact + sample)] =
            cfl{1.0f, 0.0f};
        identity.e_loc[static_cast<std::size_t>(2 * sample)] = energies[sample];
    }
    gram_from_rows(identity);
    const AnalyticRun identity_run{run_analytic(identity, relative_cut, absolute_cut)};
    auto identity_expected = std::vector<cfl>(static_cast<std::size_t>(identity.dense));
    const double identity_inverse{soft_inverse(1.0, 1.0, relative_cut, absolute_cut)};
    for (int sample{}; sample < 4; ++sample)
    {
        identity_expected[static_cast<std::size_t>(2 * sample)] =
            cfl{static_cast<float>(-identity_inverse * energies[sample]), 0.0f};
    }
    const double identity_error{
        identity_run.status == QNPEPS_OK ? relative_l2(identity_run.theta, identity_expected)
                                         : std::numeric_limits<double>::infinity()
    };
    const bool identity_ok{identity_run.status == QNPEPS_OK and identity_error <= 2.0e-6};
    recorder.add(
        "analytic_case1_identity", "relative_l2_le_2e-6", format(identity_error), identity_ok
    );

    Fixture diagonal{analytic_fixture(2)};
    diagonal.rows[0] = cfl{2.0f, 0.0f};
    diagonal.rows[static_cast<std::size_t>(diagonal.compact + 1)] = cfl{1.0f, 0.0f};
    diagonal.e_loc[0] = 1.0;
    diagonal.e_loc[2] = -1.0;
    gram_from_rows(diagonal);
    const AnalyticRun diagonal_run{run_analytic(diagonal, relative_cut, absolute_cut)};
    const double lambda{2.5};
    const double inverse{soft_inverse(lambda, lambda, relative_cut, absolute_cut)};
    auto diagonal_expected = std::vector<cfl>(static_cast<std::size_t>(diagonal.dense));
    diagonal_expected[0] = cfl{static_cast<float>(-2.0 * inverse), 0.0f};
    diagonal_expected[2] = cfl{static_cast<float>(inverse), 0.0f};
    const double diagonal_error{
        diagonal_run.status == QNPEPS_OK ? relative_l2(diagonal_run.theta, diagonal_expected)
                                         : std::numeric_limits<double>::infinity()
    };
    const bool diagonal_ok{diagonal_run.status == QNPEPS_OK and diagonal_error <= 2.0e-6};
    recorder.add(
        "analytic_case1_diagonal", "relative_l2_le_2e-6", format(diagonal_error), diagonal_ok
    );
    for (cfl& value : diagonal_expected)
        value = -value;
    const double wrong_error{
        diagonal_run.status == QNPEPS_OK ? relative_l2(diagonal_run.theta, diagonal_expected) : 0.0
    };
    const bool control_ok{std::isfinite(wrong_error) and wrong_error > 1.0};
    recorder.add(
        "analytic_case1_wrong_sign_control", "reference_mismatch", format(wrong_error), control_ok
    );
    return identity_ok and diagonal_ok and control_ok;
}

auto analytic_case_filter_curve(Recorder& recorder) -> bool
{
    const double relative_cut{0.1};
    const double absolute_cut{0.05};
    const qnpeps::CuArray<double, 3> eigenvalues{4.0, 1.0, 0.25};
    const qnpeps::CuArray<double, 3> roots{2.0, 1.0, 0.5};
    const qnpeps::CuArray<double, 3> coefficients{1.0, -0.5, 0.25};
    const qnpeps::CuArray<qnpeps::CuArray<double, 4>, 3> modes{
        {{0.5, 0.5, -0.5, -0.5}, {0.5, -0.5, 0.5, -0.5}, {0.5, -0.5, -0.5, 0.5}}
    };
    Fixture fixture{analytic_fixture(4)};
    for (int sample{}; sample < 4; ++sample)
    {
        double energy{};
        for (int mode{}; mode < 3; ++mode)
        {
            fixture.rows[static_cast<std::size_t>(sample * fixture.compact + mode)] =
                cfl{static_cast<float>(modes[mode][sample] * roots[mode]), 0.0f};
            energy += coefficients[mode] * modes[mode][sample];
        }
        fixture.e_loc[static_cast<std::size_t>(2 * sample)] = energy;
    }
    gram_from_rows(fixture);
    const AnalyticRun run{run_analytic(fixture, relative_cut, absolute_cut)};
    auto expected = std::vector<cfl>(static_cast<std::size_t>(fixture.dense));
    auto hard_expected = std::vector<cfl>(static_cast<std::size_t>(fixture.dense));
    const double configured{eigenvalues[0] * relative_cut + absolute_cut};
    for (int mode{}; mode < 3; ++mode)
    {
        expected[static_cast<std::size_t>(2 * mode)] =
            cfl{static_cast<float>(
                    -coefficients[mode] * roots[mode]
                    * soft_inverse(eigenvalues[mode], eigenvalues[0], relative_cut, absolute_cut)
                ),
                0.0f};
        const double hard_inverse{eigenvalues[mode] >= configured ? 1.0 / eigenvalues[mode] : 0.0};
        hard_expected[static_cast<std::size_t>(2 * mode)] =
            cfl{static_cast<float>(-coefficients[mode] * roots[mode] * hard_inverse), 0.0f};
    }
    const double error{
        run.status == QNPEPS_OK ? relative_l2(run.theta, expected)
                                : std::numeric_limits<double>::infinity()
    };
    const double hard_error{run.status == QNPEPS_OK ? relative_l2(run.theta, hard_expected) : 0.0};
    const bool curve_ok{run.status == QNPEPS_OK and error <= 2.0e-6};
    const bool control_ok{std::isfinite(hard_error) and hard_error > 1.0e-3};
    recorder.add("analytic_case2_filter_curve", "relative_l2_le_2e-6", format(error), curve_ok);
    recorder.add(
        "analytic_case2_hard_cut_control", "reference_mismatch", format(hard_error), control_ok
    );
    return curve_ok and control_ok;
}

auto analytic_case_rank1(Recorder& recorder) -> bool
{
    const qnpeps::CuArray<double, 4> mode{0.5, 0.5, -0.5, -0.5};
    const double lambda{4.0};
    const double coefficient{1.25};
    const double relative_cut{0.01};
    const double absolute_cut{1.0e-8};
    Fixture fixture{analytic_fixture(4)};
    for (int sample{}; sample < 4; ++sample)
    {
        fixture.rows[static_cast<std::size_t>(sample * fixture.compact)] =
            cfl{static_cast<float>(2.0 * mode[sample]), 0.0f};
        fixture.e_loc[static_cast<std::size_t>(2 * sample)] = coefficient * mode[sample];
    }
    gram_from_rows(fixture);
    const AnalyticRun run{run_analytic(fixture, relative_cut, absolute_cut)};
    const double expected_value{
        -coefficient * 2.0 * soft_inverse(lambda, lambda, relative_cut, absolute_cut)
    };
    auto expected = std::vector<cfl>(static_cast<std::size_t>(fixture.dense));
    expected[0] = cfl{static_cast<float>(expected_value), 0.0f};
    const double error{
        run.status == QNPEPS_OK ? relative_l2(run.theta, expected)
                                : std::numeric_limits<double>::infinity()
    };
    double off_axis{};
    if (run.status == QNPEPS_OK)
    {
        for (std::size_t index{1}; index < run.theta.size(); ++index)
            off_axis = std::max(off_axis, static_cast<double>(std::abs(run.theta[index])));
    }
    const bool rank_ok{run.status == QNPEPS_OK and error <= 2.0e-6 and off_axis == 0.0};
    recorder.add("analytic_case3_rank1", "collinear_relative_l2_le_2e-6", format(error), rank_ok);
    expected[0] = -expected[0];
    const double wrong_error{run.status == QNPEPS_OK ? relative_l2(run.theta, expected) : 0.0};
    const bool control_ok{std::isfinite(wrong_error) and wrong_error > 1.0};
    recorder.add(
        "analytic_case3_wrong_direction_control",
        "reference_mismatch",
        format(wrong_error),
        control_ok
    );
    return rank_ok and control_ok;
}

auto analytic_case_zero_energy(Recorder& recorder) -> bool
{
    Fixture fixture{analytic_fixture(4)};
    for (int sample{}; sample < 4; ++sample)
    {
        fixture.rows[static_cast<std::size_t>(sample * fixture.compact + sample)] =
            cfl{1.0f, 0.25f};
        fixture.e_loc[static_cast<std::size_t>(2 * sample)] = 3.25;
        fixture.e_loc[static_cast<std::size_t>(2 * sample + 1)] = -0.75;
    }
    gram_from_rows(fixture);
    const AnalyticRun run{run_analytic(fixture, 0.01, 1.0e-8)};
    const double norm{
        run.status == QNPEPS_OK ? vector_norm(run.theta) : std::numeric_limits<double>::infinity()
    };
    const bool zero_ok{run.status == QNPEPS_OK and norm == 0.0};
    recorder.add("analytic_case4_zero_centered_energy", "bit_zero", format(norm), zero_ok);
    const double wrong_uncentered_norm{std::abs(cdb{3.25, -0.75})};
    const bool control_ok{wrong_uncentered_norm > 0.0};
    recorder.add(
        "analytic_case4_uncentered_control", "nonzero", format(wrong_uncentered_norm), control_ok
    );
    return zero_ok and control_ok;
}

auto analytic_case_large_shift(Recorder& recorder) -> bool
{
    const qnpeps::CuArray<double, 4> mode{0.5, 0.5, -0.5, -0.5};
    Fixture fixture{analytic_fixture(4)};
    for (int sample{}; sample < 4; ++sample)
    {
        fixture.rows[static_cast<std::size_t>(sample * fixture.compact)] =
            cfl{static_cast<float>(2.0 * mode[sample]), 0.0f};
        fixture.e_loc[static_cast<std::size_t>(2 * sample)] = mode[sample];
    }
    gram_from_rows(fixture);
    const AnalyticRun first{run_analytic(fixture, 0.0, 100.0)};
    const AnalyticRun second{run_analytic(fixture, 0.0, 1000.0)};
    const AnalyticRun third{run_analytic(fixture, 0.0, 10000.0)};
    const double first_norm{first.status == QNPEPS_OK ? vector_norm(first.theta) : 0.0};
    const double second_norm{second.status == QNPEPS_OK ? vector_norm(second.theta) : 0.0};
    const double third_norm{third.status == QNPEPS_OK ? vector_norm(third.theta) : 0.0};
    const double first_slope{
        first_norm > 0.0 and second_norm > 0.0 ? std::log(second_norm / first_norm) / std::log(10.0)
                                               : std::numeric_limits<double>::infinity()
    };
    const double second_slope{
        second_norm > 0.0 and third_norm > 0.0 ? std::log(third_norm / second_norm) / std::log(10.0)
                                               : std::numeric_limits<double>::infinity()
    };
    const bool sixth_power{
        first.status == QNPEPS_OK and second.status == QNPEPS_OK and third.status == QNPEPS_OK
        and std::abs(first_slope + 6.0) <= 1.0e-4 and std::abs(second_slope + 6.0) <= 1.0e-4
    };
    const bool rejects_linear{
        std::abs(first_slope + 1.0) > 1.0 and std::abs(second_slope + 1.0) > 1.0
    };
    const std::string observed{format(first_slope) + "," + format(second_slope)};
    recorder.add("analytic_case5_large_shift_slope", "two_decade_slopes_-6", observed, sixth_power);
    recorder.add(
        "analytic_case5_linear_slope_control", "reference_mismatch", observed, rejects_linear
    );
    return sixth_power and rejects_linear;
}

auto permuted_fixture(const Fixture& source, const std::vector<int>& permutation) -> Fixture
{
    Fixture target{source};
    for (int sample{}; sample < source.samples; ++sample)
    {
        const int original{permutation[static_cast<std::size_t>(sample)]};
        std::copy_n(
            source.spins.begin() + static_cast<std::ptrdiff_t>(original * source.sites),
            source.sites,
            target.spins.begin() + static_cast<std::ptrdiff_t>(sample * source.sites)
        );
        std::copy_n(
            source.rows.begin() + static_cast<std::ptrdiff_t>(original * source.compact),
            source.compact,
            target.rows.begin() + static_cast<std::ptrdiff_t>(sample * source.compact)
        );
        target.logpsi[static_cast<std::size_t>(2 * sample)] =
            source.logpsi[static_cast<std::size_t>(2 * original)];
        target.logpsi[static_cast<std::size_t>(2 * sample + 1)] =
            source.logpsi[static_cast<std::size_t>(2 * original + 1)];
        target.e_loc[static_cast<std::size_t>(2 * sample)] =
            source.e_loc[static_cast<std::size_t>(2 * original)];
        target.e_loc[static_cast<std::size_t>(2 * sample + 1)] =
            source.e_loc[static_cast<std::size_t>(2 * original + 1)];
        target.logq[static_cast<std::size_t>(sample)] =
            source.logq[static_cast<std::size_t>(original)];
    }
    for (int left{}; left < source.samples; ++left)
    {
        for (int right{}; right < source.samples; ++right)
        {
            target.gram[static_cast<std::size_t>(left * source.samples + right)] =
                source.gram[static_cast<std::size_t>(
                    permutation[static_cast<std::size_t>(left)] * source.samples
                    + permutation[static_cast<std::size_t>(right)]
                )];
        }
    }
    return target;
}

auto analytic_case_permutation(Recorder& recorder) -> bool
{
    const Fixture fixture{make_fixture(2, 2, 1, 8, 0.0, 0x6E7Aull)};
    const std::vector<int> permutation{5, 0, 7, 2, 4, 1, 6, 3};
    const Fixture permuted{permuted_fixture(fixture, permutation)};
    const AnalyticRun baseline{run_analytic(fixture, 0.01, 1.0e-8)};
    const AnalyticRun reordered{run_analytic(permuted, 0.01, 1.0e-8)};
    const double error{
        baseline.status == QNPEPS_OK and reordered.status == QNPEPS_OK
            ? relative_l2(reordered.theta, baseline.theta)
            : std::numeric_limits<double>::infinity()
    };
    const bool invariant{
        baseline.status == QNPEPS_OK and reordered.status == QNPEPS_OK and error <= 1.0e-5
    };
    recorder.add(
        "analytic_case6_sample_permutation", "relative_l2_le_1e-5", format(error), invariant
    );
    Fixture wrong{fixture};
    for (int sample{}; sample < fixture.samples; ++sample)
    {
        const int original{permutation[static_cast<std::size_t>(sample)]};
        wrong.e_loc[static_cast<std::size_t>(2 * sample)] =
            fixture.e_loc[static_cast<std::size_t>(2 * original)];
        wrong.e_loc[static_cast<std::size_t>(2 * sample + 1)] =
            fixture.e_loc[static_cast<std::size_t>(2 * original + 1)];
    }
    const AnalyticRun wrong_run{run_analytic(wrong, 0.01, 1.0e-8)};
    const double wrong_error{
        baseline.status == QNPEPS_OK and wrong_run.status == QNPEPS_OK
            ? relative_l2(wrong_run.theta, baseline.theta)
            : 0.0
    };
    const bool control_ok{std::isfinite(wrong_error) and wrong_error > 1.0e-5};
    recorder.add(
        "analytic_case6_energy_only_permutation_control",
        "reference_mismatch",
        format(wrong_error),
        control_ok
    );
    return invariant and control_ok;
}

auto analytic_case_weights(Recorder& recorder) -> bool
{
    const qnpeps::CuArray<double, 4> weights{0.25, 0.75, 1.25, 1.75};
    const qnpeps::CuArray<cdb, 4> energies{
        cdb{-2.0, 0.5}, cdb{-0.25, -1.0}, cdb{1.5, 0.75}, cdb{3.0, -0.25}
    };
    Fixture fixture{analytic_fixture(4)};
    for (int sample{}; sample < 4; ++sample)
    {
        fixture.rows[static_cast<std::size_t>(sample * fixture.compact + sample)] = cfl{1.0f, 0.0f};
        fixture.logpsi[static_cast<std::size_t>(2 * sample)] = 0.5 * std::log(weights[sample]);
        fixture.e_loc[static_cast<std::size_t>(2 * sample)] = energies[sample].real();
        fixture.e_loc[static_cast<std::size_t>(2 * sample + 1)] = energies[sample].imag();
    }
    gram_from_rows(fixture);
    const AnalyticRun run{run_analytic(fixture, 0.01, 1.0e-8)};
    cdb mean{};
    double sumw{};
    double sumw2{};
    for (int sample{}; sample < 4; ++sample)
    {
        mean += weights[sample] * energies[sample];
        sumw += weights[sample];
        sumw2 += weights[sample] * weights[sample];
    }
    mean /= 4.0;
    double variance{};
    for (int sample{}; sample < 4; ++sample)
        variance += weights[sample] * std::norm(energies[sample] - mean);
    variance = variance / 4.0 * (4.0 / 3.0);
    const double expected_ess{sumw * sumw / sumw2};
    const double stats_error{
        run.status == QNPEPS_OK
            ? std::max(
                  std::max(
                      std::abs(run.e_mean[0] - mean.real()), std::abs(run.e_mean[1] - mean.imag())
                  ),
                  std::max(std::abs(run.e_var - variance), std::abs(run.ess - expected_ess))
              )
            : std::numeric_limits<double>::infinity()
    };
    const bool stats_ok{run.status == QNPEPS_OK and stats_error <= 1.0e-12};
    recorder.add(
        "analytic_case7_weight_statistics", "absolute_le_1e-12", format(stats_error), stats_ok
    );
    cdb uniform_mean{};
    for (const cdb energy : energies)
        uniform_mean += energy;
    uniform_mean /= 4.0;
    const double wrong_error{std::abs(uniform_mean - mean)};
    const bool control_ok{wrong_error > 1.0e-2};
    recorder.add(
        "analytic_case7_uniform_weight_control",
        "reference_mismatch",
        format(wrong_error),
        control_ok
    );
    return stats_ok and control_ok;
}

auto run_analytic_cases(Recorder& recorder) -> bool
{
    const bool case1{analytic_case_identity_diagonal(recorder)};
    const bool case2{analytic_case_filter_curve(recorder)};
    const bool case3{analytic_case_rank1(recorder)};
    const bool case4{analytic_case_zero_energy(recorder)};
    const bool case5{analytic_case_large_shift(recorder)};
    const bool case6{analytic_case_permutation(recorder)};
    const bool case7{analytic_case_weights(recorder)};
    return case1 and case2 and case3 and case4 and case5 and case6 and case7;
}

auto run_precision(Recorder& recorder) -> bool
{
    const double relative_cut{1.0e-2};
    const double absolute_cut{1.0e-8};
    const Fixture fixture{make_fixture(3, 3, 2, 12, 0.0, 0x51D5E0ull)};
    Device device{};
    if (not upload(fixture, device)) return false;

    qnpeps::CuArray<double, 2> e_mean{};
    double e_var{};
    double ess{};
    QnpepsMinsrArgs args{
        make_args(fixture, device, relative_cut, absolute_cut, e_mean.data(), &e_var, &ess)
    };

    const std::int64_t dense_count{qnpeps_minsr_dense_count(&fixture.descriptor)};
    const std::int64_t compact_count{qnpeps_minsr_compact_count(&fixture.descriptor)};
    const std::int64_t scratch_bytes{qnpeps_minsr_scratch_bytes(&fixture.descriptor)};
    const bool sizers_ok{
        dense_count == fixture.dense and compact_count == fixture.compact and scratch_bytes > 0
    };
    recorder.add(
        "sizers",
        "match_layout",
        std::to_string(dense_count) + "," + std::to_string(compact_count) + ","
            + std::to_string(scratch_bytes),
        sizers_ok
    );

    const qnpeps_status device_status{qnpeps_minsr(&fixture.descriptor, &args)};
    const std::vector<cfl> theta_device{download(fixture, device)};
    const qnpeps::CuArray<double, 4> stats_device{e_mean[0], e_mean[1], e_var, ess};
    std::printf(
        "[minsr_gate] HASH theta=%016llx stats=%016llx\n",
        static_cast<unsigned long long>(
            hash_bytes(theta_device.data(), theta_device.size() * sizeof(cfl))
        ),
        static_cast<unsigned long long>(hash_bytes(stats_device.data(), sizeof(stats_device)))
    );
    if (hash_bytes(theta_device.data(), theta_device.size() * sizeof(cfl)) != 0x74b245373ae56963ull
        or hash_bytes(stats_device.data(), sizeof(stats_device)) != 0x1b74cfe68bdc80bcull)
        return false;
    recorder.add(
        "device_rows_run",
        "status_0",
        std::to_string(static_cast<int>(device_status)),
        device_status == QNPEPS_OK
    );
    if (device_status != QNPEPS_OK)
    {
        release(device);
        return false;
    }

    const Reference reference{host_reference(fixture, relative_cut, absolute_cut, false)};
    const double theta_error{normalized_error(theta_device, reference.theta_dot)};
    const bool theta_ok{reference.solved and std::isfinite(theta_error) and theta_error <= 1.0e-4};
    recorder.add("theta_dot", "normalized_le_1e-4", format(theta_error), theta_ok);

    const double mean_error{
        std::abs(e_mean[0] - reference.e_mean_real) / std::max(1.0, std::abs(reference.e_mean_real))
    };
    const double mean_imag_error{
        std::abs(e_mean[1] - reference.e_mean_imag) / std::max(1.0, std::abs(reference.e_mean_imag))
    };
    const double var_error{
        std::abs(e_var - reference.e_var) / std::max(1.0, std::abs(reference.e_var))
    };
    const double ess_error{std::abs(ess - reference.ess) / std::max(1.0, std::abs(reference.ess))};
    const double stats_error{
        std::max(std::max(mean_error, mean_imag_error), std::max(var_error, ess_error))
    };
    recorder.add("stats", "relative_le_1e-10", format(stats_error), stats_error <= 1.0e-10);

    const Reference swapped{host_reference(fixture, relative_cut, absolute_cut, true)};
    const double swapped_error{normalized_error(theta_device, swapped.theta_dot)};
    const bool swapped_rejected{not std::isfinite(swapped_error) or swapped_error > 1.0e-4};
    recorder.add(
        "wrong_path_scatter_order", "reference_mismatch", format(swapped_error), swapped_rejected
    );

    cudaMemset(device.theta, 0, static_cast<std::size_t>(fixture.dense) * sizeof(cuFloatComplex));
    qnpeps::CuArray<double, 2> repeat_mean{};
    double repeat_var{};
    double repeat_ess{};
    QnpepsMinsrArgs repeat_args{args};
    repeat_args.e_mean_out = repeat_mean.data();
    repeat_args.e_var_out = &repeat_var;
    repeat_args.ess_out = &repeat_ess;
    const qnpeps_status repeat_status{qnpeps_minsr(&fixture.descriptor, &repeat_args)};
    const std::vector<cfl> theta_repeat{download(fixture, device)};
    const qnpeps::CuArray<double, 4> stats_repeat{
        repeat_mean[0], repeat_mean[1], repeat_var, repeat_ess
    };
    const bool repeat_exact{
        repeat_status == QNPEPS_OK
        and std::memcmp(theta_device.data(), theta_repeat.data(), theta_repeat.size() * sizeof(cfl))
                == 0
        and std::memcmp(stats_device.data(), stats_repeat.data(), sizeof(stats_device)) == 0
    };
    recorder.add(
        "repeat_bitwise", "identical", repeat_exact ? "identical" : "different", repeat_exact
    );

    cudaMemset(device.theta, 0, static_cast<std::size_t>(fixture.dense) * sizeof(cuFloatComplex));
    qnpeps::CuArray<double, 2> host_mean{};
    double host_var{};
    double host_ess{};
    QnpepsMinsrArgs host_args{args};
    host_args.o_rows_device = nullptr;
    host_args.o_rows_host = fixture.rows.data();
    host_args.e_mean_out = host_mean.data();
    host_args.e_var_out = &host_var;
    host_args.ess_out = &host_ess;
    QnpepsMinsrDesc tiled{fixture.descriptor};
    tiled.host_tile_bytes = 4 * fixture.compact * static_cast<std::int64_t>(sizeof(cuFloatComplex));
    const qnpeps_status host_status{qnpeps_minsr(&tiled, &host_args)};
    const std::vector<cfl> theta_host{download(fixture, device)};
    const qnpeps::CuArray<double, 4> stats_host{host_mean[0], host_mean[1], host_var, host_ess};
    const bool host_exact{
        host_status == QNPEPS_OK
        and std::memcmp(theta_device.data(), theta_host.data(), theta_host.size() * sizeof(cfl))
                == 0
        and std::memcmp(stats_device.data(), stats_host.data(), sizeof(stats_device)) == 0
    };
    recorder.add(
        "host_rows_bitwise", "identical", host_exact ? "identical" : "different", host_exact
    );

    cudaMemset(device.theta, 0, static_cast<std::size_t>(fixture.dense) * sizeof(cuFloatComplex));
    qnpeps::CuArray<double, 2> default_mean{};
    double default_var{};
    double default_ess{};
    QnpepsMinsrArgs default_args{host_args};
    default_args.e_mean_out = default_mean.data();
    default_args.e_var_out = &default_var;
    default_args.ess_out = &default_ess;
    const qnpeps_status default_status{qnpeps_minsr(&fixture.descriptor, &default_args)};
    const std::vector<cfl> theta_default{download(fixture, device)};
    const qnpeps::CuArray<double, 4> stats_default{
        default_mean[0], default_mean[1], default_var, default_ess
    };
    const bool default_exact{
        default_status == QNPEPS_OK
        and std::memcmp(
                theta_device.data(), theta_default.data(), theta_default.size() * sizeof(cfl)
            ) == 0
        and std::memcmp(stats_device.data(), stats_default.data(), sizeof(stats_device)) == 0
    };
    recorder.add(
        "host_tile_default_bitwise",
        "identical",
        default_exact ? "identical" : "different",
        default_exact
    );

    cudaMemset(device.theta, 0, static_cast<std::size_t>(fixture.dense) * sizeof(cuFloatComplex));
    qnpeps::CuArray<double, 2> diagnostic_mean{};
    double diagnostic_var{};
    double diagnostic_ess{};
    QnpepsMinsrArgs diagnostic_args{args};
    diagnostic_args.e_mean_out = diagnostic_mean.data();
    diagnostic_args.e_var_out = &diagnostic_var;
    diagnostic_args.ess_out = &diagnostic_ess;
    QnpepsMinsrDesc diagnostic{fixture.descriptor};
    diagnostic.diagnostics = 1;
    const qnpeps_status diagnostic_status{qnpeps_minsr(&diagnostic, &diagnostic_args)};
    const std::vector<cfl> theta_diagnostic{download(fixture, device)};
    const bool diagnostic_exact{
        diagnostic_status == QNPEPS_OK
        and std::memcmp(
                theta_device.data(), theta_diagnostic.data(), theta_diagnostic.size() * sizeof(cfl)
            ) == 0
    };
    recorder.add(
        "diagnostics_field_bitwise",
        "identical",
        diagnostic_exact ? "identical" : "different",
        diagnostic_exact
    );

    release(device);
    return sizers_ok and theta_ok and stats_error <= 1.0e-10 and swapped_rejected and repeat_exact
           and host_exact and default_exact and diagnostic_exact;
}

auto run_negatives(Recorder& recorder) -> bool
{
    const Fixture fixture{make_fixture(2, 3, 2, 8, 0.0, 0xBADC0DEull)};
    Device device{};
    if (not upload(fixture, device)) return false;

    qnpeps::CuArray<double, 2> e_mean{};
    double e_var{};
    double ess{};
    const QnpepsMinsrArgs args{
        make_args(fixture, device, 1.0e-2, 1.0e-8, e_mean.data(), &e_var, &ess)
    };

    QnpepsMinsrDesc version{fixture.descriptor};
    version.struct_size = 0;
    const bool descriptor_version{
        qnpeps_minsr(&version, &args) == QNPEPS_ERR_BAD_VERSION
        and qnpeps_minsr_dense_count(&version) == -1 and qnpeps_minsr_compact_count(&version) == -1
        and qnpeps_minsr_scratch_bytes(&version) == -1
    };
    recorder.add(
        "negative_descriptor_version",
        "rejected",
        descriptor_version ? "rejected" : "accepted",
        descriptor_version
    );

    QnpepsMinsrDesc reserved{fixture.descriptor};
    reserved.reserved = 1;
    QnpepsMinsrDesc diagnostics{fixture.descriptor};
    diagnostics.diagnostics = 7;
    QnpepsMinsrDesc lattice{fixture.descriptor};
    lattice.lx = 1;
    QnpepsMinsrDesc tile{fixture.descriptor};
    tile.host_tile_bytes = -1;
    const bool descriptor_config{
        qnpeps_minsr(&reserved, &args) == QNPEPS_ERR_BAD_CONFIG
        and qnpeps_minsr(&diagnostics, &args) == QNPEPS_ERR_BAD_CONFIG
        and qnpeps_minsr(&lattice, &args) == QNPEPS_ERR_BAD_CONFIG
        and qnpeps_minsr(&tile, &args) == QNPEPS_ERR_BAD_CONFIG
    };
    recorder.add(
        "negative_descriptor_config",
        "rejected",
        descriptor_config ? "rejected" : "accepted",
        descriptor_config
    );

    QnpepsMinsrArgs args_version{args};
    args_version.struct_size = 0;
    QnpepsMinsrArgs args_reserved{args};
    args_reserved.reserved = 1;
    const bool args_rejected{
        qnpeps_minsr(&fixture.descriptor, &args_version) == QNPEPS_ERR_BAD_VERSION
        and qnpeps_minsr(&fixture.descriptor, &args_reserved) == QNPEPS_ERR_BAD_CONFIG
    };
    recorder.add(
        "negative_args_version", "rejected", args_rejected ? "rejected" : "accepted", args_rejected
    );

    QnpepsMinsrArgs both_null{args};
    both_null.o_rows_device = nullptr;
    both_null.o_rows_host = nullptr;
    QnpepsMinsrArgs both_set{args};
    both_set.o_rows_host = fixture.rows.data();
    const bool rows_rejected{
        qnpeps_minsr(&fixture.descriptor, &both_null) == QNPEPS_ERR_NULL_ARG
        and qnpeps_minsr(&fixture.descriptor, &both_set) == QNPEPS_ERR_NULL_ARG
    };
    recorder.add(
        "negative_row_source", "rejected", rows_rejected ? "rejected" : "accepted", rows_rejected
    );

    QnpepsMinsrArgs short_theta{args};
    short_theta.theta_dot_out_bytes -= 1;
    QnpepsMinsrArgs short_gram{args};
    short_gram.gram_bytes -= 1;
    QnpepsMinsrArgs short_rows{args};
    short_rows.o_rows_bytes -= 1;
    QnpepsMinsrArgs short_samples{args};
    short_samples.samples_bytes -= 1;
    QnpepsMinsrArgs short_logq{args};
    short_logq.logq_bytes -= 1;
    const bool undersized_rejected{
        qnpeps_minsr(&fixture.descriptor, &short_theta) == QNPEPS_ERR_BAD_CONFIG
        and qnpeps_minsr(&fixture.descriptor, &short_gram) == QNPEPS_ERR_BAD_CONFIG
        and qnpeps_minsr(&fixture.descriptor, &short_rows) == QNPEPS_ERR_BAD_CONFIG
        and qnpeps_minsr(&fixture.descriptor, &short_samples) == QNPEPS_ERR_BAD_CONFIG
        and qnpeps_minsr(&fixture.descriptor, &short_logq) == QNPEPS_ERR_BAD_CONFIG
    };
    recorder.add(
        "negative_undersized",
        "rejected",
        undersized_rejected ? "rejected" : "accepted",
        undersized_rejected
    );

    qnpeps_minsr_ctx* context{};
    const qnpeps_status create_status{
        qnpeps_minsr_ctx_create(&fixture.descriptor, nullptr, &context)
    };
    bool context_ok{create_status == QNPEPS_OK};
    if (context_ok) context_ok = qnpeps_minsr_ctx_run(context, &args) == QNPEPS_OK;
    if (context_ok)
        context_ok = qnpeps_minsr_ctx_run(context, &short_theta) == QNPEPS_ERR_BAD_CONFIG;
    recorder.add("context_run", "status_0", context_ok ? "status_0" : "error", context_ok);

    int original{};
    int devices{};
    cudaGetDevice(&original);
    cudaGetDeviceCount(&devices);
    bool wrong_device_ok{true};
    std::string wrong_device_observed{"one_visible_device"};
    if (context_ok and devices > 1)
    {
        cudaSetDevice(original == 0 ? 1 : 0);
        const qnpeps_status status{qnpeps_minsr_ctx_run(context, &args)};
        wrong_device_ok = status == QNPEPS_ERR_BAD_CONFIG;
        wrong_device_observed = std::to_string(static_cast<int>(status));
        cudaSetDevice(original);
    }
    recorder.add("negative_wrong_device", "rejected", wrong_device_observed, wrong_device_ok);

    qnpeps_minsr_ctx_destroy(context);
    qnpeps_minsr_ctx_destroy(nullptr);
    release(device);
    return descriptor_version and descriptor_config and args_rejected and rows_rejected
           and undersized_rejected and context_ok and wrong_device_ok;
}

auto run_magnitude_spread(Recorder& recorder) -> bool
{
    const double relative_cut{1.0e-2};
    const double absolute_cut{1.0e-8};
    const Fixture fixture{make_fixture(3, 3, 2, 12, 4.0, 0x5F8EADull)};
    Device device{};
    if (not upload(fixture, device)) return false;

    qnpeps::CuArray<double, 2> e_mean{};
    double e_var{};
    double ess{};
    const QnpepsMinsrArgs args{
        make_args(fixture, device, relative_cut, absolute_cut, e_mean.data(), &e_var, &ess)
    };
    const qnpeps_status status{qnpeps_minsr(&fixture.descriptor, &args)};
    const std::vector<cfl> theta{download(fixture, device)};
    const Reference reference{host_reference(fixture, relative_cut, absolute_cut, false)};
    const double error{normalized_error(theta, reference.theta_dot)};
    const bool ok{
        status == QNPEPS_OK and reference.solved and std::isfinite(error) and error <= 1.0e-2
    };
    recorder.add("magnitude_spread", "normalized_le_1e-2", format(error), ok);
    release(device);
    return ok;
}
}

auto main() -> int
{
    Recorder recorder{};
    const bool analytic{run_analytic_cases(recorder)};
    const bool precision{run_precision(recorder)};
    const bool negatives{run_negatives(recorder)};
    const bool spread{run_magnitude_spread(recorder)};
    const bool pass{recorder.pass and analytic and precision and negatives and spread};
    std::printf("[minsr_gate] RESULT=%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
