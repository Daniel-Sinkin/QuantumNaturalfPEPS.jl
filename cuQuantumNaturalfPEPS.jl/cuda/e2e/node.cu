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

auto NodeState::status() const -> qnpeps_e2e_status
{
    std::lock_guard<std::mutex> lock{status_m};
    return status_value;
}

auto NodeState::fail(qnpeps_e2e_status value) -> void
{
    if (value == QNPEPS_E2E_OK) return;
    std::lock_guard<std::mutex> lock{status_m};
    if (status_value != QNPEPS_E2E_OK) return;
    status_value = value;
    const auto* file = qnpeps::err_file();
    const auto has_location = file != nullptr and file[0] != '\0';
    failure_location_value =
        has_location ? std::string{file} + ":" + std::to_string(qnpeps::err_line()) : "unknown";
    const auto* message = qnpeps::err_message();
    const auto* backend = qnpeps::err_backend();
    const auto has_message = message != nullptr and message[0] != '\0';
    const auto has_backend = backend != nullptr and backend[0] != '\0';
    if (has_message)
        failure_message_value = message;
    else if (has_backend)
        failure_message_value =
            std::string{backend} + " code " + std::to_string(qnpeps::err_backend_code());
    else
        failure_message_value.clear();
}

auto NodeState::failure_message() const -> const char*
{
    std::lock_guard<std::mutex> lock{status_m};
    return failure_message_value.c_str();
}

auto NodeState::failure_location() const -> const char*
{
    std::lock_guard<std::mutex> lock{status_m};
    return failure_location_value.c_str();
}
