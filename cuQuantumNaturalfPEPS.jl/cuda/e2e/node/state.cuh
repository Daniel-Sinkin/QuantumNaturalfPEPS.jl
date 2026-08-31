#ifndef QNPEPS_E2E_NODE_STATE_CUH
#define QNPEPS_E2E_NODE_STATE_CUH

#include "capi/qnpeps.h"
#include "core/session.cuh"
#include "e2e/common.cuh"
#include "eo/dans_qnpeps_eloc.h"
#include "e2e/mg_common.cuh"
#include "e2e/update.cuh"
#include "minsr/distributed.cuh"

#include <condition_variable>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <vector>

class NodeWorkers;
class NodeSolve;

struct GpuCtx
{
    [[nodiscard]] auto stream() const -> cudaStream_t { return session->stream(); }
    [[nodiscard]] auto linalg() -> qnpeps::Linalg& { return session->linalg(); }

    int device{};
    std::unique_ptr<qnpeps::Session> session{};
    u8* device_peps_slots[2]{};
    u8* device_dlenv_slots[2]{};
    f64* device_row_logs{};
    u8* device_draw_samples{};
    f64* device_draw_log_proposals{};
    f64* device_draw_log_gauges{};
    u8* device_eo_samples{};
    qn_e2e::cf* device_eo_rows{};
    f64* device_eo_log_amplitudes{};
    f64* device_eo_local_energies{};
};

enum class WorkerMode
{
    Idle,
    Sample,
    Eo,
    Gram,
    Exit
};

struct WorkerCmd
{
    WorkerMode mode{WorkerMode::Idle};
    i64 first_batch{};
    i64 nbatch{};
    i64 ring_off{};
    qn_e2e::mg::Shard shard{};
    i64 epoch{};
    const qnpeps::minsr::DistributedGramArgs* gram{};
};

class NodeState
{
  public:
    NodeState();
    ~NodeState();
    NodeState(const NodeState&) = delete;
    auto operator=(const NodeState&) -> NodeState& = delete;
    NodeState(NodeState&&) = delete;
    auto operator=(NodeState&&) -> NodeState& = delete;

    [[nodiscard]] auto gpu(int device) -> GpuCtx& { return ctx[static_cast<usize>(device)]; }
    [[nodiscard]] auto stream0() const -> cudaStream_t { return ctx[0].stream(); }
    [[nodiscard]] auto linalg0() -> qnpeps::Linalg& { return ctx[0].linalg(); }
    [[nodiscard]] auto status() const -> qnpeps_e2e_status;
    auto fail(qnpeps_e2e_status value) -> void;

    QnpepsE2eConfig cfg{};
    QnpepsConfig scfg{};
    QnpepsElocConfig lcfg{};
    int gpus{};
    int sampler_gpu{};
    i64 ns_capacity{};
    i64 ns_ahead{};
    i64 dim_batch{};
    i64 draw_cap{};
    i64 shard_capacity{};
    i64 host_tile_bytes{};
    i64 peer_tile_bytes{};
    i64 active_samples{};
    const char* step_stage{"unknown"};

    i64 sites{};
    i64 compact{};
    i64 dense{};
    i64 peps_bytes{};
    i64 dlenv_bytes{};

    std::vector<QnpepsElocDiagBond> terms_diag{};
    std::vector<QnpepsElocFlipTerm> terms_flip{};
    QnpepsElocTermTable terms{};

    qnpeps_e2e_status status_value{QNPEPS_E2E_OK};
    mutable std::mutex status_m;
    std::mutex sampler_m;

    i64 epoch_current{-1};
    std::mutex epoch_m;
    std::condition_variable epoch_cv;
    i64 epoch_ready{-1};

    u8* ring_cfg{};
    f64* ring_logq{};
    f64* ring_lgauge{};
    std::vector<i64> ring_epoch{};
    i64 ring_count{};
    i64 next_batch{};
    bool predraw_pending{};

    u8* device_samples{};
    f64* device_log_proposals{};
    f64* device_log_gauges{};
    f64* device_log_amplitudes{};
    f64* device_local_energies{};
    qn_e2e::cf* device_gram{};
    qn_e2e::cf* device_theta{};
    qn_e2e::UpdateSite* device_update_sites{};
    qn_e2e::cf* device_stage_a{};
    qn_e2e::cf* device_stage_b{};
    qn_e2e::cf* device_tile{};
    qn_e2e::cf* host_rows{};
    int inject_oom_gpu{-1};
    int inject_delay_gpu{-1};
    int inject_delay_ms{};

    bool use_p2p{};
    bool use_distributed_minsr{};
    std::vector<i32> slot_sites{};
    std::vector<u8> host_peps{};
    std::vector<u8> host_dlenv{};
    std::vector<u8> host_samples{};
    std::vector<std::vector<f64>> host_log_amplitudes{};
    std::vector<std::vector<f64>> host_local_energies{};
    std::vector<GpuCtx> ctx{};
    std::unique_ptr<NodeWorkers> workers_{};
    std::unique_ptr<NodeSolve> solve_{};
    int caller_device{};
};

struct qnpeps_e2e_node
{
    NodeState state_{};
};

#endif
