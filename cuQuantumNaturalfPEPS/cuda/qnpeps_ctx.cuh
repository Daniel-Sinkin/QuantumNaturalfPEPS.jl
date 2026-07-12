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
        Carver known{};
        Carver rolling_r{};
        Carver scratch{};
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
        char* arena{};
        Carver arena_view{};
        bool arena_owned{};
        cf* unit{};
        cf** ptr_region{};
        u8* h_samples{};
        f64* h_logpc{};
        f64* h_lognorm{};
        u64* device_seed{};
        u64 h_seed{};
        cudaGraphExec_t graph{};
        bool warmed{};
        int refresh_gen{-1};
        int dim_batch{};
        std::vector<u8> all_samples{};
        std::vector<f64> all_logpc{};
        std::vector<f64> all_lognorm{};
        bool allocated{};
    } sampler{};

    std::vector<std::vector<HostTensor>> host_peps{};
    bool use_graph{true};
};

namespace qnpeps::dlenv
{
auto rf_omega(qnpeps_ctx& ctx, int n, int k) -> cf*;
auto set_dl_capturing(qnpeps_ctx& ctx, bool on) -> void;
auto ensure_dlenv_views(qnpeps_ctx& ctx) -> void;
auto materialize_dlenv_views(qnpeps_ctx& ctx, const cf* raw_values, cf* views_out) -> void;
auto dl_free(qnpeps_ctx& ctx) -> void;
auto ctx_build_dlenv(qnpeps_ctx& ctx, const void* device_peps, f64* cumulative_row_logs) -> int;
}

namespace qnpeps::sampler
{
auto ctx_sampler_setup(qnpeps_ctx& ctx, const DlEnvView* dlenv, void* scratch, usize scratch_bytes)
    -> void;
auto ctx_sample_refresh(qnpeps_ctx& ctx) -> void;
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
