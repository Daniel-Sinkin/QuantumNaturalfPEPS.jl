#ifndef QNPEPS_CTX_CUH
#define QNPEPS_CTX_CUH

#include "capi/qnpeps.h"
#include "sampler/state.cuh"

#include <cuda_runtime.h>
#include <map>
#include <random>
#include <vector>

namespace qnpeps::dlenv
{
inline constexpr u64 k_rf_seed{777};
}

using namespace qnpeps;

struct qnpeps_ctx
{
    QnpepsConfig cfg{};
    cudaStream_t stream{};
    bool stream_owned{};
    Linalg linalg{};

    struct DlBuild
    {
        bool allocated{};
        cf* peps_buf{};
        char* arena{};
        ArenaCursor known{};
        ArenaCursor rolling_r{};
        ArenaCursor scratch{};
        int* fail{};
        f64* scales_all{};
        cf* triv{};
        cf* scalar_r{};
        bool warmed{};
        bool capturing{};
        std::map<i64, cf*> rf_omega{};
        std::mt19937_64 rf_rng{qnpeps::dlenv::k_rf_seed};
    } dl{};

    struct DlEnv
    {
        void* buf[2]{};
        cf* views[2]{};
        int active{};
        bool valid[2]{};
        int build_count{};
        bool views_allocated{};
        i64 views_elems{};
        std::vector<std::vector<i64>> env_off{};
        std::vector<std::vector<i64>> sigma_off{};
        std::vector<int32_t> dims{};
        bool header_written[2]{};
        cudaGraphExec_t graph[2]{};
        std::vector<cf*> ptr_host{};
    } dlenv{};

    struct SamplerState
    {
        Sampler samp{};

        struct Allocation
        {
            char* base{};
            ArenaCursor cursor{};
            bool owned{};
            cf* unit{};
            cf** ptr_region{};
            u64* device_seed{};
            int dim_batch_capacity{};
            bool allocated{};
        } allocation{};

        struct HostStaging
        {
            u8* h_samples{};
            f64* h_logpc{};
            f64* h_lognorm{};
            u64 h_seed{};
            std::vector<u8> all_samples{};
            std::vector<f64> all_logpc{};
            std::vector<f64> all_lognorm{};
        } staging{};

        struct Execution
        {
            cudaGraphExec_t graph{};
            bool warmed{};
            int refresh_gen{-1};
            int dim_batch{};
        } execution{};

        [[nodiscard]] auto ready() const noexcept -> bool
        {
            return allocation.allocated and allocation.base and allocation.unit
                   and allocation.ptr_region and allocation.device_seed
                   and allocation.dim_batch_capacity >= execution.dim_batch
                   and execution.dim_batch > 0 and staging.h_samples and staging.h_logpc
                   and staging.h_lognorm;
        }
    } sampler{};

    bool use_graph{true};
};

namespace qnpeps::dlenv
{
auto rf_omega(qnpeps_ctx& ctx, int n, int k) -> cf*;
auto set_dl_capturing(qnpeps_ctx& ctx, bool on) -> void;
auto ensure_dlenv_views(qnpeps_ctx& ctx) -> void;
auto materialize_dlenv_views(qnpeps_ctx& ctx, const cf* raw_values, cf* views_out) -> void;
auto dl_free(qnpeps_ctx& ctx) -> void;
auto build_dlenv(qnpeps_ctx& ctx, const void* device_peps, f64* cumulative_row_logs) -> int;
}

namespace qnpeps::sampler
{
auto ctx_sampler_setup(qnpeps_ctx& ctx, const DlEnvView* dlenv, void* scratch, usize scratch_bytes)
    -> void;
auto ctx_sample_refresh(qnpeps_ctx& ctx, const void* device_peps, PepsLayout layout) -> void;
auto ctx_sample_run(qnpeps_ctx& ctx, const std::vector<int>& batch_ids) -> void;
auto ctx_sampler_free(qnpeps_ctx& ctx) -> void;

struct CtxSampleArgs
{
    u8* output;
    f64* logpc_out;
    f64* lognorm_out;
    uint64_t n_samples;
    u64 batch_base;
};

auto ctx_sample(qnpeps_ctx& ctx, const CtxSampleArgs& args) -> qnpeps_status;
}

#endif
