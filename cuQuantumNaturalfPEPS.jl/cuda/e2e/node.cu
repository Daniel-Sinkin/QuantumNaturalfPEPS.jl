#include "e2e/node/solve.cuh"
#include "e2e/node/state.cuh"
#include "e2e/node/workers.cuh"

#include <memory>
#include <mutex>
#include <string>

NodeState::NodeState()
    : workers_{std::make_unique<NodeWorkers>(*this)}, solve_{std::make_unique<NodeSolve>(*this)}
{
}

NodeState::~NodeState() = default;

auto NodeState::status() const -> qnpeps_status
{
    std::lock_guard<std::mutex> lock{status_m};
    return status_value;
}

auto NodeState::fail(qnpeps_status value, std::string message) -> void
{
    if (value == QNPEPS_OK) return;
    std::lock_guard<std::mutex> lock{status_m};
    if (status_value != QNPEPS_OK) return;
    status_value = value;
    failure_message_value = std::move(message);
}

auto NodeState::failure_message() const -> std::string
{
    std::lock_guard<std::mutex> lock{status_m};
    return failure_message_value;
}

auto NodeState::raise_failure() const -> void
{
    std::lock_guard<std::mutex> lock{status_m};
    qnpeps::relay(static_cast<Status>(status_value), failure_message_value);
}

auto NodeState::raise_if_failed() const -> void
{
    if (status() != QNPEPS_OK) raise_failure();
}
