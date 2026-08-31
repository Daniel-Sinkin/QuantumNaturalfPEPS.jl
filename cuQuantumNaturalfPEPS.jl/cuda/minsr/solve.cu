#include "core/defer.cuh"
#include "core/predicates.cuh"
#include "linalg/transfer.cuh"
#include "minsr/kernels.cuh"
#include "minsr/solve.cuh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <new>
#include <vector>

namespace qnpeps::minsr
{
using cdh = std::complex<f64>;

namespace
{
constexpr usize k_complex_components{2};

struct SolveStatistics
{
    cdh energy_mean;
    f64 energy_variance;
    f64 ess;
};
}

class MinsrSolveContext
{
  public:
    MinsrSolveContext(int samples, Linalg& linalg) : sample_count_{samples}, linalg_{linalg}
    {
        const auto sample_count = static_cast<usize>(sample_count_);
        const auto paired_sample_count = k_complex_components * sample_count;
        host_log_amplitudes_.resize(paired_sample_count);
        host_local_energies_.resize(paired_sample_count);
        host_log_proposals_.resize(sample_count);
        log_importance_ratios_.resize(sample_count);
        importance_weights_.resize(sample_count);
        host_centered_energies_.resize(sample_count);
        host_gram_means_.resize(sample_count);
        host_eigenvalues_.resize(sample_count);
        host_raw_coefficients_.resize(sample_count);
        weighted_coefficients_.resize(sample_count);
        sample_coefficients_.resize(sample_count);

        auto& persistent_arena = linalg_.persistent_arena();
        device_importance_weights_ = persistent_arena.take<f64>(sample_count);
        device_gram_means_ = persistent_arena.take<cd>(sample_count);
        device_solve_matrix_ = persistent_arena.take<cd>(sample_count * sample_count);
        device_eigenvalues_ = persistent_arena.take<f64>(sample_count);
        device_centered_energies_ = persistent_arena.take<cd>(sample_count);
        device_eigen_coefficients_ = persistent_arena.take<cd>(sample_count);
        device_raw_coefficients_ = persistent_arena.take<cd>(sample_count);
        device_solver_info_ = persistent_arena.take<int>(1);
        if (err_state() != QNPEPS_E2E_OK) return;

        workspace_count_ = linalg_.diagonalize_workspace_count(
            CuMatrixCF64{device_solve_matrix_, sample_count_, sample_count_}, device_eigenvalues_
        );
        if (err_state() != QNPEPS_E2E_OK) return;
        const auto workspace_count = static_cast<usize>(std::max(workspace_count_, 1));
        device_workspace_ = persistent_arena.take<cd>(workspace_count);
    }

    MinsrSolveContext(const MinsrSolveContext&) = delete;
    auto operator=(const MinsrSolveContext&) -> MinsrSolveContext& = delete;

    [[nodiscard]] auto samples() const noexcept -> int { return sample_count_; }

    [[nodiscard]] auto coefficients() const noexcept -> const std::vector<cd>&
    {
        return sample_coefficients_;
    }

    [[nodiscard]] auto transient_arena() const -> TransientArenaCursor
    {
        return linalg_.transient_arena();
    }

    [[nodiscard]] auto linalg() const noexcept -> Linalg& { return linalg_; }

    auto solve(
        const f64* device_log_amplitudes,
        const f64* device_local_energies,
        const f64* device_log_proposals,
        const cf* device_gram,
        f64 relative_cut,
        f64 absolute_cut,
        f64* energy_mean_output,
        f64* energy_variance_output,
        f64* ess_output,
        bool& did_solve_output
    ) -> void
    {
        did_solve_output = false;
        std::fill(
            sample_coefficients_.begin(), sample_coefficients_.end(), make_cuDoubleComplex(0.0, 0.0)
        );
        const auto valid = _solve_check(
            device_log_amplitudes,
            device_local_energies,
            device_log_proposals,
            device_gram,
            energy_mean_output,
            energy_variance_output,
            ess_output
        );
        if (not valid)
        {
            if (err_state() == QNPEPS_E2E_OK) set_err(QNPEPS_E2E_ERR_NULL_ARG);
            return;
        }

        LinalgState linalg_state{};
        if (not linalg_.enter_default_state(linalg_state)) return;
        DEFER([&] { linalg_.restore_state(linalg_state); });

        download_inputs(device_log_amplitudes, device_local_energies, device_log_proposals);
        if (err_state() != QNPEPS_E2E_OK) return;

        const auto statistics = compute_statistics();
        energy_mean_output[0] = statistics.energy_mean.real();
        energy_mean_output[1] = statistics.energy_mean.imag();
        energy_variance_output[0] = statistics.energy_variance;
        ess_output[0] = statistics.ess;

        upload_solve_inputs();
        build_solve_matrix(device_gram);
        if (not diagonalize()) return;
        const auto solved = solve_coefficients(relative_cut, absolute_cut);
        did_solve_output = solved and err_state() == QNPEPS_E2E_OK;
    }

  private:
    [[nodiscard]] auto _solve_check(
        const f64* device_log_amplitudes,
        const f64* device_local_energies,
        const f64* device_log_proposals,
        const cf* device_gram,
        const f64* energy_mean_output,
        const f64* energy_variance_output,
        const f64* ess_output
    ) const noexcept -> bool
    {
        const auto valid_inputs = device_log_amplitudes and device_local_energies
                                  and device_log_proposals and device_gram;
        const auto valid_outputs = energy_mean_output and energy_variance_output and ess_output;
        return err_state() == QNPEPS_E2E_OK and valid_inputs and valid_outputs;
    }

    auto download_inputs(
        const f64* device_log_amplitudes,
        const f64* device_local_energies,
        const f64* device_log_proposals
    ) -> void
    {
        const auto sample_count = static_cast<usize>(sample_count_);
        const auto paired_sample_count = k_complex_components * sample_count;
        download_async(
            linalg_, host_log_amplitudes_.data(), device_log_amplitudes, paired_sample_count
        );
        download_async(
            linalg_, host_local_energies_.data(), device_local_energies, paired_sample_count
        );
        download_async(linalg_, host_log_proposals_.data(), device_log_proposals, sample_count);
        CUDA_CHECK(cudaStreamSynchronize(linalg_.stream()));
    }

    [[nodiscard]] auto compute_statistics() noexcept -> SolveStatistics
    {
        const auto sample_count = static_cast<usize>(sample_count_);
        const auto sample_count_real = static_cast<f64>(sample_count_);
        for (auto sample = 0_uz; sample < sample_count; ++sample)
        {
            log_importance_ratios_[sample] =
                static_cast<f64>(k_complex_components)
                    * host_log_amplitudes_[k_complex_components * sample]
                - host_log_proposals_[sample];
        }
        const auto maximum_log_importance_ratio = [&]
        {
            auto maximum = log_importance_ratios_[0];
            for (auto sample = 0_uz; sample < sample_count; ++sample)
            {
                maximum = maximum > log_importance_ratios_[sample] ? maximum
                                                                   : log_importance_ratios_[sample];
            }
            return maximum;
        }();
        const auto shifted_exponential_sum = [&]
        {
            f64 sum{0.0};
            for (auto sample = 0_uz; sample < sample_count; ++sample)
            {
                sum += std::exp(log_importance_ratios_[sample] - maximum_log_importance_ratio);
            }
            return sum;
        }();
        const auto log_mean_importance_weight = maximum_log_importance_ratio
                                                + std::log(shifted_exponential_sum)
                                                - std::log(sample_count_real);
        const auto importance_weight_sum = [&]
        {
            f64 sum{0.0};
            for (auto sample = 0_uz; sample < sample_count; ++sample)
            {
                importance_weights_[sample] =
                    std::exp(log_importance_ratios_[sample] - log_mean_importance_weight);
                sum += importance_weights_[sample];
            }
            return sum;
        }();
        const auto mean_importance_weight = importance_weight_sum / sample_count_real;
        f64 normalized_weight_sum{0.0};
        f64 squared_weight_sum{0.0};
        for (auto sample = 0_uz; sample < sample_count; ++sample)
        {
            importance_weights_[sample] /= mean_importance_weight;
            normalized_weight_sum += importance_weights_[sample];
            squared_weight_sum += importance_weights_[sample] * importance_weights_[sample];
        }
        const auto ess = normalized_weight_sum * normalized_weight_sum / squared_weight_sum;

        const auto energy_mean = [&]
        {
            cdh mean{0.0, 0.0};
            for (auto sample = 0_uz; sample < sample_count; ++sample)
            {
                mean += importance_weights_[sample]
                        * cdh{
                            host_local_energies_[k_complex_components * sample],
                            host_local_energies_[k_complex_components * sample + 1]
                        };
            }
            return mean / sample_count_real;
        }();
        const auto energy_variance = [&]
        {
            f64 variance{0.0};
            for (auto sample = 0_uz; sample < sample_count; ++sample)
            {
                const cdh deviation{
                    cdh{host_local_energies_[k_complex_components * sample],
                        host_local_energies_[k_complex_components * sample + 1]}
                    - energy_mean
                };
                variance += importance_weights_[sample] * std::norm(deviation);
            }
            const auto bessel_factor = sample_count_real / (sample_count_real - 1.0);
            return variance / sample_count_real * bessel_factor;
        }();

        for (auto sample = 0_uz; sample < sample_count; ++sample)
        {
            const cdh centered_energy{
                (cdh{host_local_energies_[k_complex_components * sample],
                     host_local_energies_[k_complex_components * sample + 1]}
                 - energy_mean)
                * std::sqrt(importance_weights_[sample])
            };
            host_centered_energies_[sample] =
                make_cuDoubleComplex(centered_energy.real(), centered_energy.imag());
        }
        return SolveStatistics{energy_mean, energy_variance, ess};
    }

    auto upload_solve_inputs() -> void
    {
        const auto sample_count = static_cast<usize>(sample_count_);
        upload_async(linalg_, device_importance_weights_, importance_weights_.data(), sample_count);
        upload_async(
            linalg_, device_centered_energies_, host_centered_energies_.data(), sample_count
        );
    }

    auto build_solve_matrix(const cf* device_gram) -> void
    {
        launch_beta(
            linalg_,
            {
                .output = device_gram_means_,
                .gram = device_gram,
                .weights = device_importance_weights_,
                .sample_count = sample_count_,
            }
        );
        const auto sample_count = static_cast<usize>(sample_count_);
        download_async(linalg_, host_gram_means_.data(), device_gram_means_, sample_count);
        CUDA_CHECK(cudaStreamSynchronize(linalg_.stream()));
        if (err_state() != QNPEPS_E2E_OK) return;
        const auto total_mean = [&]
        {
            cdh mean{0.0, 0.0};
            for (auto sample = 0_uz; sample < sample_count; ++sample)
            {
                const auto gram_mean = host_gram_means_[sample];
                mean += importance_weights_[sample] * cdh{cuCreal(gram_mean), cuCimag(gram_mean)};
            }
            return mean / static_cast<f64>(sample_count_);
        }();
        launch_build_matrix(
            linalg_,
            {
                .output = device_solve_matrix_,
                .gram = device_gram,
                .gram_means = device_gram_means_,
                .weights = device_importance_weights_,
                .total_mean = make_cuDoubleComplex(total_mean.real(), total_mean.imag()),
                .sample_count = sample_count_,
            }
        );
    }

    [[nodiscard]] auto diagonalize() -> bool
    {
        linalg_.diagonalize(
            CuMatrixCF64{device_solve_matrix_, sample_count_, sample_count_},
            {
                .eigenvalues = device_eigenvalues_,
                .workspace = device_workspace_,
                .workspace_count = workspace_count_,
                .info = device_solver_info_,
            }
        );
        if (err_state() == QNPEPS_E2E_OK)
        {
            const auto status =
                launch_canonicalize_eigenvectors(linalg_, device_solve_matrix_, sample_count_);
            if (status != QNPEPS_OK) set_err(static_cast<qnpeps_e2e_status>(status));
        }

        const auto sample_count = static_cast<usize>(sample_count_);
        download_async(linalg_, host_eigenvalues_.data(), device_eigenvalues_, sample_count);
        int solver_info{};
        download_async(linalg_, &solver_info, device_solver_info_, usize{1});
        CUDA_CHECK(cudaStreamSynchronize(linalg_.stream()));
        if (err_state() == QNPEPS_E2E_OK and solver_info != 0)
        {
            set_err(QNPEPS_E2E_ERR_INTERNAL);
        }
        return err_state() == QNPEPS_E2E_OK;
    }

    [[nodiscard]] auto solve_coefficients(f64 relative_cut, f64 absolute_cut) -> bool
    {
        const auto sample_count = static_cast<usize>(sample_count_);
        const auto largest_eigenvalue = host_eigenvalues_[sample_count - 1];
        if (not all_positive(largest_eigenvalue)) return false;

        linalg_.gemv(
            CuMatrixCF64Const{device_solve_matrix_, sample_count_, sample_count_},
            device_centered_energies_,
            device_eigen_coefficients_,
            {.op = BlasOp::conj_trans}
        );
        launch_apply_inverse(
            linalg_,
            {
                .values = device_eigen_coefficients_,
                .eigenvalues = device_eigenvalues_,
                .largest_eigenvalue = largest_eigenvalue,
                .relative_cut = relative_cut,
                .absolute_cut = absolute_cut,
                .sample_count = sample_count_,
            }
        );
        linalg_.gemv(
            CuMatrixCF64Const{device_solve_matrix_, sample_count_, sample_count_},
            device_eigen_coefficients_,
            device_raw_coefficients_,
            {.alpha_real = -1.0}
        );
        download_async(
            linalg_, host_raw_coefficients_.data(), device_raw_coefficients_, sample_count
        );
        CUDA_CHECK(cudaStreamSynchronize(linalg_.stream()));

        const auto weighted_coefficient_sum = [&]
        {
            cdh sum{0.0, 0.0};
            for (auto sample = 0_uz; sample < sample_count; ++sample)
            {
                const auto raw_coefficient = host_raw_coefficients_[sample];
                weighted_coefficients_[sample] =
                    std::sqrt(importance_weights_[sample])
                    * cdh{cuCreal(raw_coefficient), cuCimag(raw_coefficient)};
                sum += weighted_coefficients_[sample];
            }
            return sum;
        }();
        const auto sample_count_real = static_cast<f64>(sample_count_);
        for (auto sample = 0_uz; sample < sample_count; ++sample)
        {
            const auto coefficient =
                weighted_coefficients_[sample]
                - (weighted_coefficient_sum / sample_count_real) * importance_weights_[sample];
            sample_coefficients_[sample] =
                make_cuDoubleComplex(coefficient.real(), coefficient.imag());
        }
        return true;
    }

    int sample_count_{};
    Linalg& linalg_;
    int workspace_count_{};
    f64* device_importance_weights_{};
    cd* device_gram_means_{};
    cd* device_solve_matrix_{};
    f64* device_eigenvalues_{};
    cd* device_centered_energies_{};
    cd* device_eigen_coefficients_{};
    cd* device_raw_coefficients_{};
    int* device_solver_info_{};
    cd* device_workspace_{};
    std::vector<f64> host_log_amplitudes_{};
    std::vector<f64> host_local_energies_{};
    std::vector<f64> host_log_proposals_{};
    std::vector<f64> log_importance_ratios_{};
    std::vector<f64> importance_weights_{};
    std::vector<cd> host_centered_energies_{};
    std::vector<cd> host_gram_means_{};
    std::vector<f64> host_eigenvalues_{};
    std::vector<cd> host_raw_coefficients_{};
    std::vector<cdh> weighted_coefficients_{};
    std::vector<cd> sample_coefficients_{};
};

[[nodiscard]] auto make_solve_context(int samples, Linalg& linalg)
    -> std::unique_ptr<MinsrSolveContext>
{
    try
    {
        std::unique_ptr<MinsrSolveContext> solve{new (std::nothrow)
                                                     MinsrSolveContext{samples, linalg}};
        if (not solve) set_err(QNPEPS_E2E_ERR_OOM);
        return solve;
    }
    catch (const std::bad_alloc&)
    {
        set_err(QNPEPS_E2E_ERR_OOM);
        return {};
    }
}

[[nodiscard]] auto prepared_solve_create(i64 samples, Linalg& linalg) -> MinsrSolveContext*
{
    auto solve = make_solve_context(static_cast<int>(samples), linalg);
    return solve.release();
}

auto prepared_solve_run(
    MinsrSolveContext& solve,
    const f64* device_log_amplitudes,
    const f64* device_local_energies,
    const f64* device_log_proposals,
    const cf* device_gram,
    f64 relative_cut,
    f64 absolute_cut,
    f64* energy_mean_output,
    f64* energy_variance_output,
    f64* ess_output,
    std::vector<cd>& coefficient_output,
    bool& did_solve_output
) -> void
{
    solve.solve(
        device_log_amplitudes,
        device_local_energies,
        device_log_proposals,
        device_gram,
        relative_cut,
        absolute_cut,
        energy_mean_output,
        energy_variance_output,
        ess_output,
        did_solve_output
    );
    try
    {
        coefficient_output = solve.coefficients();
    }
    catch (const std::bad_alloc&)
    {
        set_err(QNPEPS_E2E_ERR_OOM);
        did_solve_output = false;
    }
}

[[nodiscard]] auto prepared_solve_transient_arena(const MinsrSolveContext& solve)
    -> TransientArenaCursor
{
    return solve.transient_arena();
}

[[nodiscard]] auto prepared_solve_samples(const MinsrSolveContext& solve) noexcept -> i64
{
    return solve.samples();
}

[[nodiscard]] auto prepared_solve_linalg(const MinsrSolveContext& solve) noexcept -> Linalg&
{
    return solve.linalg();
}

auto prepared_solve_destroy(MinsrSolveContext* solve) noexcept -> void
{
    delete solve;
}
}
