#ifndef QNPEPS_SAMPLER_STATE_CUH
#define QNPEPS_SAMPLER_STATE_CUH

#include "core/arena_cursor.cuh"
#include "linalg/linalg.cuh"
#include "tensor/permutation.cuh"
#include "linalg/rangefinder_rng.cuh"
#include "sampler/kernels.cuh"
#include "tensor/tensor.cuh"
#include "core/types.cuh"

#include <algorithm>
#include <cassert>
#include <map>
#include <utility>
#include <vector>

namespace qnpeps
{
struct HostEnvRow
{
    std::vector<Shape> site_shapes{};
    f64 lognorm{};
};

struct SamplerConfig
{
    int lx{};
    int ly{};
    int dim_phys{};
    int dim_bond{};
    int chi_dl{};
    int chi_s{};
    int dim_batch{};
    int row_spin_stride{};
    int lane_base{};
    int batches{};
    bool fast_mode{};
    int chi_c{};
    u64 seed{};
    u64 batch_base{};
    bool density_route{};
    f64 state_density_cutoff{};
    f64 projected_density_cutoff{};

    [[nodiscard]] constexpr auto num_sites() const noexcept -> int { return lx * ly; }
};

struct DlEnvView
{
    const int* dims{};
    const cuFloatComplex* values{};
};

enum class PepsLayout
{
    canonical,
    reverse_packed
};

[[nodiscard]] auto upload_to_device(ArenaCursor& arena, const HostTensor& host_tensor)
    -> cuFloatComplex*;

class Sampler
{
  public:
    [[nodiscard]] auto cfg() noexcept -> SamplerConfig& { return cfg_; }
    [[nodiscard]] auto linalg() noexcept -> Linalg& { return *linalg_; }
    [[nodiscard]] auto permutation_cache() noexcept -> PermutationCache&
    {
        return permutation_cache_;
    }
    auto bind_linalg(Linalg& linalg) noexcept -> void { linalg_ = &linalg; }
    auto bind_arena(ArenaCursor& arena) noexcept -> void { arena_ = &arena; }

    [[nodiscard]] auto mpo() noexcept -> std::vector<std::vector<cuFloatComplex*>>& { return mpo_; }
    [[nodiscard]] auto peps_shapes() noexcept -> std::vector<std::vector<Shape>>&
    {
        return peps_shapes_;
    }
    [[nodiscard]] auto dlenv_host() noexcept -> std::vector<HostEnvRow>& { return dlenv_host_; }
    [[nodiscard]] auto ket_row0() noexcept -> std::vector<cuFloatComplex*>& { return ket_row0_; }

    [[nodiscard]] auto env_above() noexcept -> CuArray<CuSpanCF32, 2>& { return env_above_; }

    [[nodiscard]] auto ket() noexcept -> CuSpanCF32& { return ket_; }
    [[nodiscard]] auto env_unsampled() noexcept -> CuSpanCF32& { return env_unsampled_; }
    [[nodiscard]] auto sigma() noexcept -> CuSpanCF32& { return sigma_; }
    [[nodiscard]] auto sigma_full() noexcept -> CuSpanCF32& { return sigma_full_; }
    [[nodiscard]] auto sigma_full_scratch() noexcept -> CuSpanCF32& { return sigma_full_scratch_; }
    [[nodiscard]] auto rho() noexcept -> CuSpanCF32& { return rho_; }
    [[nodiscard]] auto rfactor() noexcept -> CuSpanCF32& { return rfactor_; }
    [[nodiscard]] auto tmp_a() noexcept -> CuSpanCF32& { return tmp_a_; }
    [[nodiscard]] auto tmp_b() noexcept -> CuSpanCF32& { return tmp_b_; }
    [[nodiscard]] auto reduce_input() noexcept -> CuSpanCF32& { return reduce_input_; }
    [[nodiscard]] auto sketch() noexcept -> CuSpanCF32& { return sketch_; }
    [[nodiscard]] auto projection() noexcept -> CuSpanCF32& { return projection_; }
    [[nodiscard]] auto rfactor_next() noexcept -> CuSpanCF32& { return rfactor_next_; }
    [[nodiscard]] auto gram() noexcept -> CuSpanCF32& { return gram_; }

    [[nodiscard]] auto gram_ptrs() noexcept -> cuFloatComplex**& { return gram_ptrs_; }
    [[nodiscard]] auto sketch_ptrs() noexcept -> cuFloatComplex**& { return sketch_ptrs_; }
    [[nodiscard]] auto tmp_a_ptrs() noexcept -> cuFloatComplex**& { return tmp_a_ptrs_; }
    [[nodiscard]] auto tmp_b_ptrs() noexcept -> cuFloatComplex**& { return tmp_b_ptrs_; }
    [[nodiscard]] auto dl_unit_ptrs() noexcept -> cuFloatComplex**& { return dl_unit_ptrs_; }

    [[nodiscard]] auto envu_ptrs() noexcept -> std::vector<cuFloatComplex**>& { return envu_ptrs_; }
    [[nodiscard]] auto ket_row0_ptrs() noexcept -> std::vector<cuFloatComplex**>&
    {
        return ket_row0_ptrs_;
    }
    [[nodiscard]] auto mpo_ptrs() noexcept -> std::vector<std::vector<cuFloatComplex**>>&
    {
        return mpo_ptrs_;
    }
    [[nodiscard]] auto dlenv_env_ptrs() noexcept -> std::vector<std::vector<cuFloatComplex**>>&
    {
        return dlenv_env_ptrs_;
    }
    [[nodiscard]] auto dlenv_sigma_ptrs() noexcept -> std::vector<std::vector<cuFloatComplex**>>&
    {
        return dlenv_sigma_ptrs_;
    }

    [[nodiscard]] auto info() noexcept -> int*& { return info_; }
    [[nodiscard]] auto fail() noexcept -> int*& { return fail_; }
    [[nodiscard]] auto drawn_spin() noexcept -> int*& { return drawn_spin_; }
    [[nodiscard]] auto row_spins() noexcept -> int*& { return row_spins_; }
    [[nodiscard]] auto logpc() noexcept -> f64*& { return logpc_; }
    [[nodiscard]] auto lognorm() noexcept -> f64*& { return lognorm_; }
    [[nodiscard]] auto samples() noexcept -> u8*& { return samples_; }

    [[nodiscard]] auto max_env_above_site() noexcept -> i64& { return max_env_above_site_; }
    [[nodiscard]] auto max_ket_site() noexcept -> i64& { return max_ket_site_; }
    [[nodiscard]] auto max_env_unsampled() noexcept -> i64& { return max_env_unsampled_; }
    [[nodiscard]] auto max_reduce_input() noexcept -> i64& { return max_reduce_input_; }
    [[nodiscard]] auto max_rfactor() noexcept -> i64& { return max_rfactor_; }
    [[nodiscard]] auto max_sketch() noexcept -> i64& { return max_sketch_; }
    [[nodiscard]] auto max_rho() noexcept -> i64& { return max_rho_; }
    [[nodiscard]] auto max_sigma() noexcept -> i64& { return max_sigma_; }
    [[nodiscard]] auto max_sigma_full() noexcept -> i64& { return max_sigma_full_; }
    [[nodiscard]] auto max_tmp() noexcept -> i64& { return max_tmp_; }

    [[nodiscard]] auto omega(int cols, int rank) -> cuFloatComplex*
    {
        const auto valid = linalg_ and arena_ and cols > 0 and rank > 0 and rank <= cols;
        if (not valid)
        {
            set_err(QNPEPS_ERR_INTERNAL);
            return nullptr;
        }
        const std::pair key{cols, rank};
        if (const auto it = omegas_.find(key); it != omegas_.end())
        {
            return it->second;
        }

        auto rng = RangefinderRng::from_seed_and_width(cfg_.seed, cols);
        HostTensor host_omega{Shape{cols, rank}};
        rng.fill_complex_normal(host_omega.values());
        auto* device_omega = upload_to_device(*arena_, host_omega);
        omegas_.emplace(key, device_omega);
        return device_omega;
    }

    auto reduce(
        CuMatrixBatchedCF32Const input,
        int rank,
        CuMatrixBatchedCF32 q_out,
        CuMatrixBatchedCF32 r_out,
        int dim_batch
    ) -> void
    {
        const auto valid = linalg_ and input.data() and input.rows() > 0 and input.cols() > 0
                           and rank > 0 and rank <= std::min(input.rows(), input.cols())
                           and q_out.data() and r_out.data() and dim_batch > 0;
        if (not valid)
        {
            set_err(QNPEPS_ERR_INTERNAL);
            return;
        }
        batched_rangefinder(
            *linalg_,
            {
                .input = input,
                .rank = rank,
                .omega = omega(input.cols(), rank),
                .q_out = q_out,
                .r_out = r_out,
                .dim_batch = dim_batch,
                .sketch = sketch_,
                .projection = projection_,
                .gram = gram_,
                .gram_ptrs = gram_ptrs_,
                .sketch_ptrs = sketch_ptrs_,
                .info = info_,
                .fail_flag = fail_,
            }
        );
    }

  private:
    SamplerConfig cfg_{};
    Linalg* linalg_{};
    PermutationCache permutation_cache_{};
    ArenaCursor* arena_{};

    std::vector<std::vector<cuFloatComplex*>> mpo_{};
    std::vector<std::vector<Shape>> peps_shapes_{};
    std::vector<HostEnvRow> dlenv_host_{};
    std::vector<cuFloatComplex*> ket_row0_{};
    std::map<std::pair<int, int>, cuFloatComplex*> omegas_{};

    CuArray<CuSpanCF32, 2> env_above_{};
    CuSpanCF32 ket_{};
    CuSpanCF32 env_unsampled_{};
    CuSpanCF32 sigma_{};
    CuSpanCF32 sigma_full_{};
    CuSpanCF32 sigma_full_scratch_{};
    CuSpanCF32 rho_{};
    CuSpanCF32 rfactor_{};
    CuSpanCF32 tmp_a_{};
    CuSpanCF32 tmp_b_{};
    CuSpanCF32 reduce_input_{};
    CuSpanCF32 sketch_{};
    CuSpanCF32 projection_{};
    CuSpanCF32 rfactor_next_{};
    CuSpanCF32 gram_{};

    cuFloatComplex** gram_ptrs_{};
    cuFloatComplex** sketch_ptrs_{};
    cuFloatComplex** tmp_a_ptrs_{};
    cuFloatComplex** tmp_b_ptrs_{};
    cuFloatComplex** dl_unit_ptrs_{};

    std::vector<cuFloatComplex**> envu_ptrs_{};
    std::vector<cuFloatComplex**> ket_row0_ptrs_{};
    std::vector<std::vector<cuFloatComplex**>> mpo_ptrs_{};
    std::vector<std::vector<cuFloatComplex**>> dlenv_env_ptrs_{};
    std::vector<std::vector<cuFloatComplex**>> dlenv_sigma_ptrs_{};

    int* info_{};
    int* fail_{};
    int* drawn_spin_{};
    int* row_spins_{};
    f64* logpc_{};
    f64* lognorm_{};
    u8* samples_{};

    i64 max_env_above_site_{};
    i64 max_ket_site_{};
    i64 max_env_unsampled_{};
    i64 max_reduce_input_{};
    i64 max_rfactor_{};
    i64 max_sketch_{};
    i64 max_rho_{};
    i64 max_sigma_{};
    i64 max_sigma_full_{};
    i64 max_tmp_{};
};
}

#endif
