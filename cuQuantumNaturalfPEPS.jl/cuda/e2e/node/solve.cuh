#ifndef QNPEPS_E2E_NODE_SOLVE_CUH
#define QNPEPS_E2E_NODE_SOLVE_CUH

#include "e2e/node/state.cuh"

#include <vector>

namespace qnpeps::minsr
{
class MinsrSolveContext;
}

class NodeSolve
{
  public:
    explicit NodeSolve(NodeState& state) noexcept;
    ~NodeSolve();

    NodeSolve(const NodeSolve&) = delete;
    auto operator=(const NodeSolve&) -> NodeSolve& = delete;

    auto solve(
        i64 sample_count,
        const std::vector<qn_e2e::mg::Shard>& shards,
        f64 relative_cut,
        f64 absolute_cut,
        qn_e2e::cf* theta_output,
        f64* energy_mean_output,
        f64* energy_variance_output,
        f64* ess_output
    ) -> void;
    auto solve_distributed(
        i64 sample_count,
        const std::vector<qn_e2e::mg::Shard>& shards,
        f64 relative_cut,
        f64 absolute_cut,
        qn_e2e::cf* theta_output,
        f64* energy_mean_output,
        f64* energy_variance_output,
        f64* ess_output
    ) -> void;
    [[nodiscard]] auto host_pinned(usize bytes) -> void*;

  private:
    [[nodiscard]] auto cached_solve(i64 sample_count) -> qnpeps::minsr::MinsrSolveContext*;

    NodeState& state_;
    qnpeps::minsr::MinsrSolveContext* solve_cache_{};
    i64 solve_cache_samples_{};
};

#endif
