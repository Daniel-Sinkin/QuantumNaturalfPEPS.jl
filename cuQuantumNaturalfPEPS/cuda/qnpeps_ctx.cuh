#ifndef QNPEPS_CTX_CUH
#define QNPEPS_CTX_CUH

#include "capi/qnpeps.h"
#include "dlenv/state.cuh"
#include "sampler/state.cuh"

#include <array>
#include <cassert>
#include <cuda_runtime.h>
#include <memory>
#include <utility>
#include <vector>

namespace qnpeps::dlenv
{
inline constexpr usize k_buffer_lane_count{2};
inline constexpr usize k_sampling_layout_count{2};
}

using namespace qnpeps;

struct qnpeps_ctx
{
    qnpeps_ctx(
        QnpepsConfig config,
        cudaStream_t stream_handle,
        bool own_stream,
        std::unique_ptr<Linalg> linalg
    )
        : cfg(config), stream(stream_handle), owns_stream(own_stream), linalg_(std::move(linalg))
    {
        assert(linalg_);
    }

    [[nodiscard]] auto linalg() noexcept -> Linalg& { return *linalg_; }

    QnpepsConfig cfg{};
    cudaStream_t stream{};
    bool owns_stream{};

    qnpeps::dlenv::BuildState dl{};

    struct DlEnv
    {
        struct BufferLane
        {
            void* packed{};
            cuFloatComplex* sampling{};
            bool valid{};
            bool header_written{};
            cudaGraphExec_t graph{};
        };

        std::array<BufferLane, qnpeps::dlenv::k_buffer_lane_count> lanes{};
        usize active_lane{};
        int build_count{};
        i64 sampling_elements{};
        std::vector<std::vector<i64>> env_off{};
        std::vector<std::vector<i64>> sigma_off{};
        std::vector<int32_t> dims{};
        std::vector<cuFloatComplex*> ptr_host{};
    } dlenv{};

    struct SamplerState
    {
        Sampler samp{};

        struct Allocation
        {
            char* base{};
            ArenaCursor cursor{};
            bool owned{};
            cuFloatComplex* unit{};
            cuFloatComplex** ptr_region{};
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
            const auto arena_ready = allocation.allocated and allocation.base and allocation.unit;
            const auto device_pointers_ready = allocation.ptr_region and allocation.device_seed;
            const auto batch_configured = execution.dim_batch > 0;
            const auto batch_fits = allocation.dim_batch_capacity >= execution.dim_batch;
            const auto batch_ready = batch_configured and batch_fits;
            const auto staging_ready = staging.h_samples and staging.h_logpc and staging.h_lognorm;
            return arena_ready and device_pointers_ready and batch_ready and staging_ready;
        }
    } sampler{};

    bool use_graph{true};

  private:
    std::unique_ptr<Linalg> linalg_{};
};

namespace qnpeps::dlenv
{
auto set_dl_capturing(qnpeps_ctx& ctx, bool on) -> void;
auto ensure_sampling_buffers(qnpeps_ctx& ctx) -> void;
auto materialize_sampling_buffer(
    qnpeps_ctx& ctx, const cuFloatComplex* raw_values, cuFloatComplex* sampling_out
) -> void;
auto dl_free(qnpeps_ctx& ctx) -> void;
auto build_dlenv(qnpeps_ctx& ctx, const void* device_peps, f64* cumulative_row_logs) -> int;
}

namespace qnpeps::sampler
{
enum class SampleOutputLocation;

auto ctx_sampler_setup(qnpeps_ctx& ctx, const DlEnvView* dlenv, void* scratch, usize scratch_bytes)
    -> void;
auto ctx_sample_refresh(qnpeps_ctx& ctx, const void* device_peps, PepsLayout layout) -> void;
struct HostSampleOutput
{
    u8* samples{};
    f64* logpc{};
    f64* lognorm{};
    u64 n_samples{};
};

auto ctx_sample_run(
    qnpeps_ctx& ctx,
    const std::vector<int>& batch_ids,
    HostSampleOutput* host_output = nullptr
) -> void;
auto ctx_sampler_free(qnpeps_ctx& ctx) -> void;

struct CtxSampleArgs
{
    u8* output{};
    f64* logpc_out{};
    f64* lognorm_out{};
    u64 n_samples{};
    u64 batch_base{};
    u64 dim_batch{};
    SampleOutputLocation output_location;
};

auto ctx_sample(qnpeps_ctx& ctx, const CtxSampleArgs& args) -> qnpeps_status;
}

#endif
