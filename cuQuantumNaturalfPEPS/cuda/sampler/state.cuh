#ifndef QNPEPS_SAMPLER_STATE_CUH
#define QNPEPS_SAMPLER_STATE_CUH

#include "dtensor.cuh"
#include "linalg.cuh"
#include "sampler/kernels.cuh"
#include "types.cuh"

#include <map>
#include <random>
#include <utility>
#include <vector>

namespace qnpeps
{
struct HostEnvRow
{
    std::vector<HostTensor> site{};
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
    u64 seed{};
    u64 batch_base{};

    [[nodiscard]] constexpr auto num_sites() const noexcept -> int { return lx * ly; }
};

struct DlEnvView
{
    const int* dims{};
    const cuFloatComplex* values{};
};

auto arena_upload(BumpArena& arena, const HostTensor& host_tensor) -> cf*;

class Sampler
{
  public:
    auto cfg() -> SamplerConfig& { return cfg_; }
    auto linalg() -> Linalg& { return *linalg_; }
    auto permutation_cache() -> PermutationCache& { return permutation_cache_; }
    auto bind(Linalg& linalg, BumpArena& arena) -> void
    {
        linalg_ = &linalg;
        arena_ = &arena;
    }

    auto mpo() -> std::vector<std::vector<cf*>>& { return mpo_; }
    auto mpo_host() -> std::vector<std::vector<HostTensor>>& { return mpo_host_; }
    auto dlenv_host() -> std::vector<HostEnvRow>& { return dlenv_host_; }
    auto ket_row0() -> std::vector<cf*>& { return ket_row0_; }
    auto ket_row0_host() -> std::vector<HostTensor>& { return ket_row0_host_; }

    auto env_above() -> CuArray (&)[2] { return env_above_; }
    auto ket() -> CuArray& { return ket_; }
    auto env_unsampled() -> CuArray& { return env_unsampled_; }
    auto sigma() -> CuArray& { return sigma_; }
    auto sigma_full() -> CuArray& { return sigma_full_; }
    auto sigma_full_scratch() -> CuArray& { return sigma_full_scratch_; }
    auto rho() -> CuArray& { return rho_; }
    auto rfactor() -> CuArray& { return rfactor_; }
    auto tmp_a() -> CuArray& { return tmp_a_; }
    auto tmp_b() -> CuArray& { return tmp_b_; }
    auto reduce_input() -> CuArray& { return reduce_input_; }
    auto sketch() -> CuArray& { return sketch_; }
    auto proj() -> CuArray& { return proj_; }
    auto rfactor_next() -> CuArray& { return rfactor_next_; }
    auto gram() -> CuArray& { return gram_; }
    auto gram_ptrs() -> cf**& { return gram_ptrs_; }
    auto sketch_ptrs() -> cf**& { return sketch_ptrs_; }
    auto tmp_a_ptrs() -> cf**& { return tmp_a_ptrs_; }
    auto tmp_b_ptrs() -> cf**& { return tmp_b_ptrs_; }
    auto dl_unit_ptrs() -> cf**& { return dl_unit_ptrs_; }
    auto envu_ptrs() -> std::vector<cf**>& { return envu_ptrs_; }
    auto ket_row0_ptrs() -> std::vector<cf**>& { return ket_row0_ptrs_; }
    auto mpo_ptrs() -> std::vector<std::vector<cf**>>& { return mpo_ptrs_; }
    auto dlenv_env_ptrs() -> std::vector<std::vector<cf**>>& { return dlenv_env_ptrs_; }
    auto dlenv_sigma_ptrs() -> std::vector<std::vector<cf**>>& { return dlenv_sigma_ptrs_; }
    auto info() -> int*& { return info_; }
    auto fail() -> int*& { return fail_; }
    auto drawn_spin() -> int*& { return drawn_spin_; }
    auto row_spins() -> int*& { return row_spins_; }
    auto logpc() -> f64*& { return logpc_; }
    auto lognorm() -> f64*& { return lognorm_; }
    auto samples() -> u8*& { return samples_; }

    auto max_env_above_site() -> i64& { return max_env_above_site_; }
    auto max_ket_site() -> i64& { return max_ket_site_; }
    auto max_env_unsampled() -> i64& { return max_env_unsampled_; }
    auto max_reduce_input() -> i64& { return max_reduce_input_; }
    auto max_rfactor() -> i64& { return max_rfactor_; }
    auto max_sketch() -> i64& { return max_sketch_; }
    auto max_rho() -> i64& { return max_rho_; }
    auto max_sigma() -> i64& { return max_sigma_; }
    auto max_sigma_full() -> i64& { return max_sigma_full_; }
    auto max_tmp() -> i64& { return max_tmp_; }

    auto omega(int N, int k) -> cf*
    {
        const std::pair<int, int> key{N, k};
        auto it = omegas_.find(key);
        if (it != omegas_.end()) return it->second;

        std::mt19937_64 rng(cfg_.seed ^ (static_cast<u64>(N) << 20));
        std::normal_distribution<f32> gauss(0.0f, 1.0f);
        auto host = HostTensor{{N, k}};
        host.alloc();
        for (auto& value : host.v)
            value = chost{gauss(rng), gauss(rng)};
        auto* device_ptr = arena_upload(*arena_, host);
        omegas_.emplace(key, device_ptr);
        return device_ptr;
    }

    auto reduce(
        CuMatrixConstBatched input,
        int k,
        CuMatrixBatched q_out,
        CuMatrixBatched r_out,
        int dim_batch
    ) -> void
    {
        batched_rangefinder(
            *linalg_,
            {
                .input = input,
                .k = k,
                .omega = omega(input.cols(), k),
                .q_out = q_out,
                .r_out = r_out,
                .dim_batch = dim_batch,
                .sketch = sketch_,
                .proj = proj_,
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
    BumpArena* arena_{};

    std::vector<std::vector<cf*>> mpo_{};
    std::vector<std::vector<HostTensor>> mpo_host_{};
    std::vector<HostEnvRow> dlenv_host_{};
    std::vector<cf*> ket_row0_{};
    std::vector<HostTensor> ket_row0_host_{};
    std::map<std::pair<int, int>, cf*> omegas_{};

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
    CuArray proj_{};
    CuArray rfactor_next_{};
    CuArray gram_{};
    cf** gram_ptrs_{};
    cf** sketch_ptrs_{};
    cf** tmp_a_ptrs_{};
    cf** tmp_b_ptrs_{};
    cf** dl_unit_ptrs_{};
    std::vector<cf**> envu_ptrs_{};
    std::vector<cf**> ket_row0_ptrs_{};
    std::vector<std::vector<cf**>> mpo_ptrs_{};
    std::vector<std::vector<cf**>> dlenv_env_ptrs_{};
    std::vector<std::vector<cf**>> dlenv_sigma_ptrs_{};
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
