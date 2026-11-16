#include "e2e/node/step_api.cuh"

#include "core/arena_cursor.cuh"
#include "core/defer.cuh"
#include "e2e/common.cuh"
#include "e2e/mg_common.cuh"

#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <format>
#include <vector>

using qn_e2e::cf;
using qn_e2e::mg::build_shards;
using qn_e2e::mg::Shard;

namespace
{
struct StepOutputs
{
    cf* theta;
    f64* energy_mean;
    f64* energy_variance;
    f64* ess;
    u8* samples;
    f64* logq;
    f64* log_gauge;
    f64* logpsi;
    f64* e_loc;
    qnpeps_e2e_cbuf* host_rows;
    i64* epoch;
};

auto caller_device() -> int
{
    int device{0};
    unchecked(cudaGetDevice(&device));
    return device;
}

auto submit_theta(NodeState& state, const void* device_peps) -> void
{
    state.raise_if_failed();
    const auto caller = caller_device();
    const qnpeps::Defer restore_device{[&] { unchecked(cudaSetDevice(caller)); }};
    const i64 next_epoch{state.epoch_current + 1};
    const int slot{static_cast<int>(next_epoch & 1)};

    state.recorded(
        "submit",
        [&]
        {
            unchecked(cudaSetDevice(0));
            check(cudaMemcpy(
                state.ctx[0].device_peps_slots[slot],
                device_peps,
                static_cast<usize>(state.peps_bytes),
                cudaMemcpyDeviceToDevice
            ));
            check(cudaDeviceSynchronize());
        }
    );

    state.epoch_current = next_epoch;
    state.workers_->run_dlbuild(next_epoch);
}

auto draw_into_ring(NodeState& state, i64 sample_count, i64 epoch) -> void
{
    const i64 batch_size{state.dim_batch};
    const i64 needed_samples{sample_count - state.ring_count};
    const i64 batch_count{(needed_samples + batch_size - 1) / batch_size};
    const i64 ring_offset{state.ring_count};
    const usize ring_offset_u{static_cast<usize>(ring_offset)};
    const usize drawn_sample_count{static_cast<usize>(batch_count * batch_size)};
    require(
        ring_offset + batch_count * batch_size <= state.ns_capacity,
        Status::bad_config,
        "drawing {} batches at ring offset {} exceeds the node capacity {}",
        batch_count,
        ring_offset,
        state.ns_capacity
    );
    const i64 first_batch{state.next_batch};
    state.next_batch += batch_count;
    for (auto sample_index = 0_uz; sample_index < drawn_sample_count; ++sample_index)
        state.ring_epoch[ring_offset_u + sample_index] = epoch;
    state.ring_count += batch_count * batch_size;
    if (state.gpus >= 2 and state.ns_ahead == 0)
    {
        state.recorded(
            "sampling",
            [&] { draw_fill_multigpu(&state, first_batch, batch_count, ring_offset, epoch); }
        );
    }
    else if (state.gpus >= 2)
    {
        state.workers_->issue(
            state.sampler_gpu,
            WorkerCmd{WorkerMode::Sample, first_batch, batch_count, ring_offset, {}, epoch}
        );
        const auto outcome = state.workers_->wait(state.sampler_gpu);
        if (outcome != QNPEPS_OK)
        {
            state.fail(outcome, std::string{});
            state.raise_failure();
        }
    }
    else
    {
        state.recorded(
            "sampling",
            [&] { draw_fill(&state, state.ctx[0], first_batch, batch_count, ring_offset, epoch); }
        );
    }
}

auto consume_ring(NodeState& state, i64 sample_count, i64* epoch_out) -> void
{
    const usize sample_count_u{static_cast<usize>(sample_count)};
    const usize sample_site_count{static_cast<usize>(sample_count * state.sites)};
    state.recorded(
        "sampling",
        [&]
        {
            check(cudaMemcpy(
                state.device_samples, state.ring_cfg, sample_site_count, cudaMemcpyHostToDevice
            ));
            check(cudaMemcpy(
                state.device_log_proposals,
                state.ring_logq,
                sizeof(f64) * sample_count_u,
                cudaMemcpyHostToDevice
            ));
            check(cudaMemcpy(
                state.device_log_gauges,
                state.ring_lgauge,
                sizeof(f64) * sample_count_u,
                cudaMemcpyHostToDevice
            ));
        }
    );
    if (epoch_out) std::memcpy(epoch_out, state.ring_epoch.data(), sizeof(i64) * sample_count_u);
    const i64 left{state.ring_count - sample_count};
    if (left > 0)
    {
        const usize left_u{static_cast<usize>(left)};
        std::memmove(
            state.ring_cfg,
            state.ring_cfg + sample_count * state.sites,
            left_u * static_cast<usize>(state.sites)
        );
        std::memmove(state.ring_logq, state.ring_logq + sample_count, sizeof(f64) * left_u);
        std::memmove(state.ring_lgauge, state.ring_lgauge + sample_count, sizeof(f64) * left_u);
        std::memmove(
            state.ring_epoch.data(), state.ring_epoch.data() + sample_count, sizeof(i64) * left_u
        );
    }
    state.ring_count = left;
}

auto evaluate_shards(NodeState& state, const std::vector<Shard>& shards, i64 epoch) -> void
{
    for (int gpu{1}; gpu < state.gpus; ++gpu)
    {
        const usize gpu_index{static_cast<usize>(gpu)};
        state.workers_->issue(gpu, WorkerCmd{WorkerMode::Eo, 0, 0, 0, shards[gpu_index], epoch});
    }
    const auto device_zero_outcome =
        guarded_labelled("eo", 0, [&] { eo_run_shard(&state, 0, shards[0], epoch); });
    if (device_zero_outcome != QNPEPS_OK)
        state.fail(device_zero_outcome, std::move(qnpeps::last_error_message()));
    for (int gpu{1}; gpu < state.gpus; ++gpu)
    {
        const auto outcome = state.workers_->wait(gpu);
        if (outcome != QNPEPS_OK) state.fail(outcome, std::string{});
    }
}

auto gather_shard_scalars(NodeState& state, const std::vector<Shard>& shards) -> void
{
    unchecked(cudaSetDevice(0));
    for (int gpu{1}; gpu < state.gpus; ++gpu)
    {
        const usize gpu_index{static_cast<usize>(gpu)};
        const Shard shard{shards[gpu_index]};
        if (shard.count <= 0) continue;
        const usize scalar_bytes{sizeof(f64) * static_cast<usize>(2 * shard.count)};
        check(cudaMemcpy(
            state.device_log_amplitudes + 2 * shard.base,
            state.host_log_amplitudes[gpu_index].data(),
            scalar_bytes,
            cudaMemcpyHostToDevice
        ));
        check(cudaMemcpy(
            state.device_local_energies + 2 * shard.base,
            state.host_local_energies[gpu_index].data(),
            scalar_bytes,
            cudaMemcpyHostToDevice
        ));
    }
}

auto predraw(NodeState& state, i64 batch_count, i64 epoch) -> void
{
    const i64 batch_size{state.dim_batch};
    const i64 ring_offset{state.ring_count};
    const usize ring_offset_u{static_cast<usize>(ring_offset)};
    const i64 first_batch{state.next_batch};
    state.next_batch += batch_count;
    const usize predraw_sample_count{static_cast<usize>(batch_count * batch_size)};
    for (auto sample_index = 0_uz; sample_index < predraw_sample_count; ++sample_index)
        state.ring_epoch[ring_offset_u + sample_index] = epoch;
    state.ring_count += batch_count * batch_size;
    if (state.gpus >= 2)
    {
        state.workers_->issue(
            state.sampler_gpu,
            WorkerCmd{WorkerMode::Sample, first_batch, batch_count, ring_offset, {}, epoch}
        );
        state.predraw_pending = true;
        return;
    }
    const auto outcome = guarded_labelled(
        "sampling",
        [&] { draw_fill(&state, state.ctx[0], first_batch, batch_count, ring_offset, epoch); }
    );
    if (outcome != QNPEPS_OK) state.fail(outcome, std::move(qnpeps::last_error_message()));
}

auto copy_step_outputs(NodeState& state, i64 sample_count, const StepOutputs& outputs) -> void
{
    const usize sample_count_u{static_cast<usize>(sample_count)};
    const usize sample_site_count{static_cast<usize>(sample_count * state.sites)};
    const usize doubled_sample_count{2 * sample_count_u};
    const auto copy = [&](void* destination, const void* source, usize bytes)
    {
        if (not destination) return;
        check(
            cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, state.stream0())
        );
    };
    copy(outputs.samples, state.device_samples, sample_site_count);
    copy(outputs.logq, state.device_log_proposals, sizeof(f64) * sample_count_u);
    copy(outputs.log_gauge, state.device_log_gauges, sizeof(f64) * sample_count_u);
    copy(outputs.logpsi, state.device_log_amplitudes, sizeof(f64) * doubled_sample_count);
    copy(outputs.e_loc, state.device_local_energies, sizeof(f64) * doubled_sample_count);
    check(cudaStreamSynchronize(state.stream0()));
    if (outputs.host_rows)
    {
        std::memcpy(
            reinterpret_cast<cf*>(outputs.host_rows),
            state.host_rows,
            sizeof(cf) * sample_count_u * static_cast<usize>(state.compact)
        );
    }
}

auto finish_step(
    NodeState& state,
    i64 sample_count,
    const std::vector<Shard>& shards,
    f64 relative_cut,
    f64 absolute_cut,
    i64 predraw_batch_count,
    i64 epoch,
    const StepOutputs& outputs
) -> void
{
    if (not state.use_p2p and state.gpus >= 2) gather_shard_scalars(state, shards);
    state.step_stage = "sampling";
    if (predraw_batch_count > 0) predraw(state, predraw_batch_count, epoch);
    state.step_stage = "minsr";
    unchecked(cudaSetDevice(0));
    if (state.use_distributed_minsr)
    {
        state.solve_->solve_distributed(
            sample_count,
            shards,
            relative_cut,
            absolute_cut,
            outputs.theta,
            outputs.energy_mean,
            outputs.energy_variance,
            outputs.ess
        );
    }
    else
    {
        state.solve_->solve(
            sample_count,
            shards,
            relative_cut,
            absolute_cut,
            outputs.theta,
            outputs.energy_mean,
            outputs.energy_variance,
            outputs.ess
        );
    }
    if (state.status() == QNPEPS_OK) copy_step_outputs(state, sample_count, outputs);
}

auto node_step(
    qnpeps_e2e_node* node,
    i64 sample_count,
    f64 relative_cut,
    f64 absolute_cut,
    const StepOutputs& outputs
) -> void
{
    const auto has_outputs =
        node and outputs.theta and outputs.energy_mean and outputs.energy_variance and outputs.ess;
    require(has_outputs, Status::null_argument, "node step misses the node or a required output");
    auto& state = node->state_;
    state.raise_if_failed();
    require(
        state.epoch_current >= 0, Status::bad_config, "node step before any theta was submitted"
    );
    const auto valid_count =
        sample_count >= 0 and sample_count != 1 and sample_count <= state.ns_capacity;
    require(
        valid_count,
        Status::bad_config,
        "node step sample count {} outside 0 or 2 to {}",
        sample_count,
        state.ns_capacity
    );

    const auto caller = caller_device();
    const qnpeps::Defer restore_device{[&] { unchecked(cudaSetDevice(caller)); }};
    unchecked(cudaSetDevice(0));
    state.step_stage = "minsr";

    if (sample_count == 0)
    {
        outputs.energy_mean[0] = 0.0;
        outputs.energy_mean[1] = 0.0;
        outputs.energy_variance[0] = 0.0;
        outputs.ess[0] = 0.0;
        state.recorded(
            "minsr",
            [&]
            {
                check(cudaMemsetAsync(
                    outputs.theta, 0, sizeof(cf) * static_cast<usize>(state.dense), state.stream0()
                ));
                check(cudaStreamSynchronize(state.stream0()));
            }
        );
        return;
    }

    const i64 epoch{state.epoch_current};
    const i64 batch_size{state.dim_batch};
    state.active_samples = sample_count;

    state.step_stage = "sampling";
    if (state.predraw_pending)
    {
        if (state.gpus >= 2) static_cast<void>(state.workers_->wait(state.sampler_gpu));
        state.predraw_pending = false;
    }
    state.step_stage = "dlenv";
    wait_epoch_ready(&state, epoch);
    state.raise_if_failed();

    state.step_stage = "sampling";
    if (state.ring_count < sample_count) draw_into_ring(state, sample_count, epoch);
    consume_ring(state, sample_count, outputs.epoch);

    const i64 predraw_batch_count{
        state.ns_ahead > 0 ? (state.ns_ahead + batch_size - 1) / batch_size : 0
    };
    const auto predraw_exceeds_capacity =
        state.ring_count + predraw_batch_count * batch_size > state.ns_capacity;
    require(
        predraw_batch_count <= 0 or not predraw_exceeds_capacity,
        Status::bad_config,
        "predrawing {} batches on {} held samples exceeds the node capacity {}",
        predraw_batch_count,
        state.ring_count,
        state.ns_capacity
    );

    if (not state.use_p2p and state.gpus >= 2)
    {
        const usize sample_site_count{static_cast<usize>(sample_count * state.sites)};
        state.host_samples.resize(sample_site_count);
        state.recorded(
            "eo",
            [&]
            {
                check(cudaMemcpy(
                    state.host_samples.data(),
                    state.device_samples,
                    sample_site_count,
                    cudaMemcpyDeviceToHost
                ));
            }
        );
    }

    state.step_stage = "eo";
    const std::vector<Shard> shards{build_shards(sample_count, state.gpus, state.cfg.meo)};
    evaluate_shards(state, shards, epoch);
    state.raise_if_failed();

    const auto late_outcome = guarded(
        [&]
        {
            finish_step(
                state,
                sample_count,
                shards,
                relative_cut,
                absolute_cut,
                predraw_batch_count,
                epoch,
                outputs
            );
        }
    );
    if (late_outcome != QNPEPS_OK)
    {
        state.fail(
            late_outcome, std::format("[{}] {}", state.step_stage, qnpeps::last_error_message())
        );
    }
    if (state.status() == QNPEPS_OK) return;
    qnpeps::relay(Status::internal, state.failure_message());
}

auto node_step_euler(qnpeps_e2e_node* node, const QnpepsE2eEulerStepArgs* args) -> void
{
    require(
        node != nullptr and args != nullptr,
        Status::null_argument,
        "Euler step needs a node and arguments"
    );
    require_struct_size(*args, "Euler step arguments");
    const auto has_outputs = args->peps_f32_io and args->energy_mean_output
                             and args->energy_variance_output and args->ess_output;
    require(has_outputs, Status::null_argument, "Euler step misses a required output");
    auto& state = node->state_;
    state.raise_if_failed();
    require(
        std::isfinite(args->learning_rate),
        Status::bad_config,
        "Euler step learning rate {} is not finite",
        args->learning_rate
    );

    const u64 peps_bytes{static_cast<u64>(state.peps_bytes)};
    const u64 state_f64_bytes{static_cast<u64>(state.dense) * sizeof(qn_e2e::zd)};
    const u64 theta_bytes{static_cast<u64>(state.dense) * sizeof(cf)};
    require(
        args->peps_f32_bytes == peps_bytes,
        Status::bad_config,
        "Euler step PEPS buffer holds {} bytes, expected {}",
        args->peps_f32_bytes,
        peps_bytes
    );
    const auto invalid_theta_bytes = (args->theta_output and args->theta_dot_bytes != theta_bytes)
                                     or (not args->theta_output and args->theta_dot_bytes != 0);
    require(
        not invalid_theta_bytes,
        Status::bad_config,
        "Euler step theta buffer holds {} bytes, expected {}",
        args->theta_dot_bytes,
        args->theta_output ? theta_bytes : 0
    );
    require(
        args->theta_output != args->peps_f32_io,
        Status::bad_config,
        "Euler step theta output aliases the PEPS buffer"
    );

    if (args->precision == QNPEPS_E2E_UPDATE_F32)
    {
        const auto has_master = args->state_f64_io or args->state_f64_bytes != 0;
        require(
            not has_master,
            Status::bad_config,
            "Euler step in f32 precision passes an f64 master state"
        );
    }
    else if (args->precision == QNPEPS_E2E_UPDATE_F64_MASTER)
    {
        require(
            args->state_f64_io != nullptr,
            Status::null_argument,
            "Euler step f64 master state is null"
        );
        require(
            args->state_f64_bytes == state_f64_bytes,
            Status::bad_config,
            "Euler step master state holds {} bytes, expected {}",
            args->state_f64_bytes,
            state_f64_bytes
        );
        const auto aliased_state = args->state_f64_io == static_cast<void*>(args->peps_f32_io)
                                   or args->state_f64_io == static_cast<void*>(args->theta_output);
        require(
            not aliased_state, Status::bad_config, "Euler step master state aliases another buffer"
        );
    }
    else
        fail(Status::bad_config, "Euler step precision {} is unknown", args->precision);

    auto theta{args->theta_output ? reinterpret_cast<cf*>(args->theta_output) : state.device_theta};
    node_step(
        node,
        args->n_samples,
        args->relative_cut,
        args->absolute_cut,
        {
            .theta = theta,
            .energy_mean = args->energy_mean_output,
            .energy_variance = args->energy_variance_output,
            .ess = args->ess_output,
            .samples = args->samples_out,
            .logq = args->logq_out,
            .log_gauge = args->log_gauge_out,
            .logpsi = args->logpsi_out,
            .e_loc = args->e_loc_out,
            .host_rows = args->host_rows,
            .epoch = args->epoch_out,
        }
    );

    int caller{};
    const auto get_status = cudaGetDevice(&caller);
    require(
        get_status == cudaSuccess,
        Status::cuda,
        "Euler step device query failed, {}",
        qnpeps::cuda_failure_text(get_status)
    );
    const auto set_status = cudaSetDevice(0);
    require(
        set_status == cudaSuccess,
        Status::cuda,
        "Euler step cannot select GPU 0, {}",
        qnpeps::cuda_failure_text(set_status)
    );
    const qnpeps::Defer restore_device{[&] { unchecked(cudaSetDevice(caller)); }};

    auto* peps{reinterpret_cast<cf*>(args->peps_f32_io)};
    state.recorded(
        "update",
        [&]
        {
            if (args->precision == QNPEPS_E2E_UPDATE_F32)
            {
                const int current_slot{static_cast<int>(state.epoch_current & 1)};
                const cudaError_t copy_status{cudaMemcpyAsync(
                    peps,
                    state.ctx[0].device_peps_slots[current_slot],
                    static_cast<usize>(state.peps_bytes),
                    cudaMemcpyDeviceToDevice,
                    state.stream0()
                )};
                require(
                    copy_status == cudaSuccess,
                    Status::cuda,
                    "Euler step PEPS copy failed, {}",
                    qnpeps::cuda_failure_text(copy_status)
                );
                qn_e2e::launch_update_f32(
                    state.device_update_sites,
                    static_cast<int>(state.sites),
                    peps,
                    theta,
                    args->learning_rate,
                    state.stream0()
                );
            }
            else
            {
                qn_e2e::launch_update_f64(
                    state.device_update_sites,
                    static_cast<int>(state.sites),
                    static_cast<qn_e2e::zd*>(args->state_f64_io),
                    theta,
                    peps,
                    args->learning_rate,
                    state.stream0()
                );
            }
            const auto sync_status = cudaStreamSynchronize(state.stream0());
            require(
                sync_status == cudaSuccess,
                Status::cuda,
                "Euler step update synchronize failed, {}",
                qnpeps::cuda_failure_text(sync_status)
            );
        }
    );

    submit_theta(state, peps);
}

auto wait_environment(qnpeps_e2e_node* node) -> void
{
    require(node != nullptr, Status::null_argument, "environment wait needs a node");
    auto& state = node->state_;
    state.raise_if_failed();
    if (state.epoch_current < 0) return;
    wait_epoch_ready(&state, state.epoch_current);
    state.raise_if_failed();
}

auto footprint_bytes(
    const QnpepsE2eConfig* cfg,
    int gpus,
    int64_t ns_capacity,
    int64_t dim_batch,
    const void* terms,
    uint64_t* out_bytes
) -> void
{
    qn_e2e::mg::validate_node_config(cfg);
    require(out_bytes != nullptr, Status::null_argument, "node footprint output is null");
    qn_e2e::mg::validate_node_terms(*cfg, terms);
    require(
        gpus >= 1 and ns_capacity >= 2 and dim_batch >= 1,
        Status::bad_config,
        "node footprint rejects {} GPUs, capacity {}, batch {}",
        gpus,
        ns_capacity,
        dim_batch
    );
    *out_bytes = static_cast<u64>(qnpeps::arena_reservation_bytes());
}
}

extern "C" qnpeps_status qnpeps_e2e_node_submit_theta(
    qnpeps_e2e_node* node, const void* device_peps
)
{
    return guarded(
        [&]
        {
            require(
                node != nullptr and device_peps != nullptr,
                Status::null_argument,
                "theta submission needs a node and a PEPS"
            );
            submit_theta(node->state_, device_peps);
        }
    );
}

extern "C" qnpeps_status qnpeps_e2e_node_step(
    qnpeps_e2e_node* node,
    int64_t sample_count,
    double relative_cut,
    double absolute_cut,
    qnpeps_e2e_cbuf* theta_dot_out,
    double* e_mean_out,
    double* e_var_out,
    double* ess_out,
    uint8_t* samples_out,
    double* logq_out,
    double* log_gauge_out,
    double* logpsi_out,
    double* e_loc_out,
    qnpeps_e2e_cbuf* o_rows_host,
    int64_t* epoch_out
)
{
    return guarded(
        [&]
        {
            node_step(
                node,
                sample_count,
                relative_cut,
                absolute_cut,
                {
                    .theta = reinterpret_cast<cf*>(theta_dot_out),
                    .energy_mean = e_mean_out,
                    .energy_variance = e_var_out,
                    .ess = ess_out,
                    .samples = samples_out,
                    .logq = logq_out,
                    .log_gauge = log_gauge_out,
                    .logpsi = logpsi_out,
                    .e_loc = e_loc_out,
                    .host_rows = o_rows_host,
                    .epoch = epoch_out,
                }
            );
        }
    );
}

extern "C" qnpeps_status qnpeps_e2e_node_step_euler(
    qnpeps_e2e_node* node, const QnpepsE2eEulerStepArgs* args
)
{
    return guarded([&] { node_step_euler(node, args); });
}

extern "C" qnpeps_status qnpeps_e2e_node_wait_environment(qnpeps_e2e_node* node)
{
    return guarded([&] { wait_environment(node); });
}

extern "C" qnpeps_status qnpeps_e2e_node_footprint_bytes(
    const QnpepsE2eConfig* cfg,
    int gpus,
    int64_t ns_capacity,
    [[maybe_unused]] int64_t ns_ahead,
    int64_t dim_batch,
    const void* terms,
    uint64_t* out_bytes
)
{
    return guarded([&] { footprint_bytes(cfg, gpus, ns_capacity, dim_batch, terms, out_bytes); });
}
