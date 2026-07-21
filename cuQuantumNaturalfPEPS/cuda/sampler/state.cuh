#ifndef QNPEPS_SAMPLER_STATE_CUH
#define QNPEPS_SAMPLER_STATE_CUH

#include "arena_cursor.cuh"
#include "linalg.cuh"
#include "permutation.cuh"
#include "rangefinder_rng.cuh"
#include "sampler/kernels.cuh"
#include "tensor.cuh"
#include "types.cuh"

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
    int batches{};
    bool fast_mode{};
    int chi_c{};
    u64 seed{};
    u64 batch_base{};

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

auto upload_to_device(ArenaCursor& arena, const HostTensor& host_tensor) -> cuFloatComplex*;

class Sampler
{
  public:
    auto cfg() -> SamplerConfig& { return cfg_; }
    auto linalg() -> Linalg& { return *linalg_; }
    auto permutation_cache() -> PermutationCache& { return permutation_cache_; }
    auto bind_linalg(Linalg& linalg) -> void { linalg_ = &linalg; }
    auto bind_arena(ArenaCursor& arena) -> void { arena_ = &arena; }

    // clang-format off
    auto mpo()         -> std::vector<std::vector<cuFloatComplex*>>& { return mpo_; }
    auto peps_shapes() -> std::vector<std::vector<Shape>>& { return peps_shapes_; }
    auto dlenv_host()  -> std::vector<HostEnvRow>& { return dlenv_host_; }
    auto ket_row0()    -> std::vector<cuFloatComplex*>& { return ket_row0_; }

    auto env_above() -> CuArray (&)[2] { return env_above_; }

    auto ket()                -> CuArray& { return ket_; }
    auto env_unsampled()      -> CuArray& { return env_unsampled_; }
    auto sigma()              -> CuArray& { return sigma_; }
    auto sigma_full()         -> CuArray& { return sigma_full_; }
    auto sigma_full_scratch() -> CuArray& { return sigma_full_scratch_; }
    auto rho()                -> CuArray& { return rho_; }
    auto rfactor()            -> CuArray& { return rfactor_; }
    auto tmp_a()              -> CuArray& { return tmp_a_; }
    auto tmp_b()              -> CuArray& { return tmp_b_; }
    auto reduce_input()       -> CuArray& { return reduce_input_; }
    auto sketch()             -> CuArray& { return sketch_; }
    auto projection()         -> CuArray& { return projection_; }
    auto rfactor_next()       -> CuArray& { return rfactor_next_; }
    auto gram()               -> CuArray& { return gram_; }

    auto gram_ptrs()    -> cuFloatComplex**& { return gram_ptrs_; }
    auto sketch_ptrs()  -> cuFloatComplex**& { return sketch_ptrs_; }
    auto tmp_a_ptrs()   -> cuFloatComplex**& { return tmp_a_ptrs_; }
    auto tmp_b_ptrs()   -> cuFloatComplex**& { return tmp_b_ptrs_; }
    auto dl_unit_ptrs() -> cuFloatComplex**& { return dl_unit_ptrs_; }

    auto envu_ptrs()        -> std::vector<cuFloatComplex**>& { return envu_ptrs_; }
    auto ket_row0_ptrs()    -> std::vector<cuFloatComplex**>& { return ket_row0_ptrs_; }
    auto mpo_ptrs()         -> std::vector<std::vector<cuFloatComplex**>>& { return mpo_ptrs_; }
    auto dlenv_env_ptrs()   -> std::vector<std::vector<cuFloatComplex**>>& { return dlenv_env_ptrs_; }
    auto dlenv_sigma_ptrs() -> std::vector<std::vector<cuFloatComplex**>>& { return dlenv_sigma_ptrs_; }

    auto info()       -> int*& { return info_; }
    auto fail()       -> int*& { return fail_; }
    auto drawn_spin() -> int*& { return drawn_spin_; }
    auto row_spins()  -> int*& { return row_spins_; }
    auto logpc()      -> f64*& { return logpc_; }
    auto lognorm()    -> f64*& { return lognorm_; }
    auto samples()    ->  u8*&{ return samples_; }

    auto max_env_above_site() -> i64& { return max_env_above_site_; }
    auto max_ket_site()       -> i64& { return max_ket_site_; }
    auto max_env_unsampled()  -> i64& { return max_env_unsampled_; }
    auto max_reduce_input()   -> i64& { return max_reduce_input_; }
    auto max_rfactor()        -> i64& { return max_rfactor_; }
    auto max_sketch()         -> i64& { return max_sketch_; }
    auto max_rho()            -> i64& { return max_rho_; }
    auto max_sigma()          -> i64& { return max_sigma_; }
    auto max_sigma_full()     -> i64& { return max_sigma_full_; }
    auto max_tmp()            -> i64& { return max_tmp_; }
    // clang-format on

    auto omega(int cols, int rank) -> cuFloatComplex*
    {
        const auto valid = linalg_ and arena_ and cols > 0 and rank > 0 and rank <= cols;
        if (not valid)
        {
            set_err(QNPEPS_ERR_INTERNAL);
            return nullptr;
        }
        const auto key = std::pair{cols, rank};
        if (const auto it = omegas_.find(key); it != omegas_.end())
        {
            return it->second;
        }

        auto rng = RangefinderRng::from_seed_and_width(cfg_.seed, cols);
        auto host_omega = HostTensor{Shape{cols, rank}};
        rng.fill_complex_normal(host_omega.values());
        auto* device_omega = upload_to_device(*arena_, host_omega);
        omegas_.emplace(key, device_omega);
        return device_omega;
    }

    auto reduce(
        CuMatrixConstBatched input,
        int rank,
        CuMatrixBatched q_out,
        CuMatrixBatched r_out,
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

    CuArray env_above_[2]{};
    CuArray ket_{};
    CuArray env_unsampled_{};
    CuArray sigma_{};
    CuArray sigma_full_{};
    CuArray sigma_full_scratch_{};
    CuArray rho_{};
    CuArray rfactor_{};
    CuArray tmp_a_{};
    CuArray tmp_b_{};
    CuArray reduce_input_{};
    CuArray sketch_{};
    CuArray projection_{};
    CuArray rfactor_next_{};
    CuArray gram_{};

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
