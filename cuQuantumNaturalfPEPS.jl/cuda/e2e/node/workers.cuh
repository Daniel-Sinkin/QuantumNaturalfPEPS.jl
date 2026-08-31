#ifndef QNPEPS_E2E_NODE_WORKERS_CUH
#define QNPEPS_E2E_NODE_WORKERS_CUH

#include "e2e/node/state.cuh"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

struct WorkerSync
{
    std::mutex mutex;
    std::condition_variable command_ready;
    std::condition_variable command_done;
    i64 command_generation{};
    i64 done_generation{};
    WorkerCmd command{};
    qnpeps_e2e_status status{QNPEPS_E2E_OK};
};

class NodeWorkers
{
  public:
    explicit NodeWorkers(NodeState& state) noexcept;
    ~NodeWorkers();
    NodeWorkers(const NodeWorkers&) = delete;
    auto operator=(const NodeWorkers&) -> NodeWorkers& = delete;
    NodeWorkers(NodeWorkers&&) = delete;
    auto operator=(NodeWorkers&&) -> NodeWorkers& = delete;

    auto start() -> void;
    auto stop() noexcept -> void;
    auto issue(int device, const WorkerCmd& command) -> void;
    [[nodiscard]] auto wait(int device) -> qnpeps_e2e_status;
    auto run_dlbuild(i64 epoch) -> void;

  private:
    auto worker_main(int device) -> void;
    auto dlbuild_main() -> void;

    NodeState& state_;
    std::vector<std::thread> workers_{};
    std::vector<std::unique_ptr<WorkerSync>> sync_{};
    std::thread dlbuild_{};
    std::mutex build_mutex_;
    std::condition_variable build_ready_;
    i64 build_request_{};
    i64 build_epoch_{};
    bool build_exit_{};
};

auto wait_epoch_ready(NodeState* state, i64 epoch) -> void;
[[nodiscard]] auto draw_fill(
    NodeState* state, GpuCtx& gpu, i64 first_batch, i64 batch_count, i64 ring_offset, i64 epoch
) -> qnpeps_e2e_status;
[[nodiscard]] auto draw_fill_multigpu(
    NodeState* state, i64 first_batch, i64 batch_count, i64 ring_offset, i64 epoch
) -> qnpeps_e2e_status;
[[nodiscard]] auto eo_run_shard(
    NodeState* state, int device, const qn_e2e::mg::Shard& shard, i64 epoch
) -> qnpeps_e2e_status;

#endif
