#include "e2e/node/solve.cuh"
#include "e2e/node/state.cuh"
#include "e2e/node/workers.cuh"

#include <memory>
#include <mutex>

NodeState::NodeState()
    : workers_{std::make_unique<NodeWorkers>(*this)}, solve_{std::make_unique<NodeSolve>(*this)}
{
}

NodeState::~NodeState() = default;

auto NodeState::status() const -> qnpeps_e2e_status
{
    std::lock_guard<std::mutex> lock{status_m};
    return status_value;
}

auto NodeState::fail(qnpeps_e2e_status value) -> void
{
    if (value == QNPEPS_E2E_OK) return;
    std::lock_guard<std::mutex> lock{status_m};
    if (status_value == QNPEPS_E2E_OK) status_value = value;
}
