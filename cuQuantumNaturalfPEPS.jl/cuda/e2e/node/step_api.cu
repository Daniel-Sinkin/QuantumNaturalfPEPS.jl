#include "e2e/node/step_api.cuh"

#include "core/arena_cursor.cuh"
#include "e2e/common.cuh"
#include "e2e/mg_common.cuh"

#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

using qn_e2e::cf;
using qn_e2e::err_state;
using qn_e2e::mg::build_shards;
using qn_e2e::mg::Shard;

extern "C" qnpeps_e2e_status qnpeps_e2e_node_submit_theta(
    qnpeps_e2e_node* node, const void* device_peps
)
{
    if (not node or not device_peps) return QNPEPS_E2E_ERR_NULL_ARG;
    if (node->state_.status() != QNPEPS_E2E_OK) return QNPEPS_E2E_ERR_INTERNAL;

    int caller_device{0};
    cudaGetDevice(&caller_device);
    qn_e2e::clear_err();
    const i64 next_epoch{node->state_.epoch_current + 1};
    const int slot{static_cast<int>(next_epoch & 1)};

    cudaSetDevice(0);
    QN_E2E_CUDA_CHECK(cudaMemcpy(
        node->state_.ctx[0].device_peps_slots[slot],
        device_peps,
        static_cast<usize>(node->state_.peps_bytes),
        cudaMemcpyDeviceToDevice
    ));
    QN_E2E_CUDA_CHECK(cudaDeviceSynchronize());
    cudaSetDevice(caller_device);
    if (err_state() != QNPEPS_E2E_OK)
    {
        node->state_.fail(err_state());
        return QNPEPS_E2E_ERR_INTERNAL;
    }

    node->state_.epoch_current = next_epoch;
    node->state_.workers_->run_dlbuild(next_epoch);
    return QNPEPS_E2E_OK;
}

extern "C" qnpeps_e2e_status qnpeps_e2e_node_step(
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
    if (not node or not theta_dot_out or not e_mean_out or not e_var_out or not ess_out)
        return QNPEPS_E2E_ERR_NULL_ARG;
    if (node->state_.status() != QNPEPS_E2E_OK) return QNPEPS_E2E_ERR_INTERNAL;
    if (node->state_.epoch_current < 0) return QNPEPS_E2E_ERR_BAD_CONFIG;
    if (sample_count < 0 or sample_count == 1 or sample_count > node->state_.ns_capacity)
        return QNPEPS_E2E_ERR_BAD_CONFIG;
    const usize sample_count_u{static_cast<usize>(sample_count)};
    const usize sample_site_count{static_cast<usize>(sample_count * node->state_.sites)};
    const usize doubled_sample_count{2 * sample_count_u};

    int caller_device{0};
    cudaGetDevice(&caller_device);
    cudaSetDevice(0);
    qn_e2e::clear_err();
    auto theta{reinterpret_cast<cf*>(theta_dot_out)};
    node->state_.step_stage = "minsr";

    if (sample_count == 0)
    {
        QN_E2E_CUDA_CHECK(cudaMemsetAsync(
            theta, 0, sizeof(cf) * static_cast<usize>(node->state_.dense), node->state_.stream0()
        ));
        QN_E2E_CUDA_CHECK(cudaStreamSynchronize(node->state_.stream0()));
        e_mean_out[0] = 0.0;
        e_mean_out[1] = 0.0;
        e_var_out[0] = 0.0;
        ess_out[0] = 0.0;
        cudaSetDevice(caller_device);
        if (err_state() != QNPEPS_E2E_OK)
        {
            node->state_.fail(err_state());
            return QNPEPS_E2E_ERR_INTERNAL;
        }
        return QNPEPS_E2E_OK;
    }

    const i64 epoch{node->state_.epoch_current};
    const i64 batch_size{node->state_.dim_batch};
    node->state_.active_samples = sample_count;

    node->state_.step_stage = "sampling";
    if (node->state_.predraw_pending)
    {
        if (node->state_.gpus >= 2)
            static_cast<void>(node->state_.workers_->wait(node->state_.sampler_gpu));
        node->state_.predraw_pending = false;
    }
    node->state_.step_stage = "dlenv";
    wait_epoch_ready(&node->state_, epoch);
    if (node->state_.status() != QNPEPS_E2E_OK)
    {
        cudaSetDevice(caller_device);
        return node->state_.status();
    }

    node->state_.step_stage = "sampling";
    if (node->state_.ring_count < sample_count)
    {
        const i64 needed_samples{sample_count - node->state_.ring_count};
        const i64 batch_count{(needed_samples + batch_size - 1) / batch_size};
        const i64 ring_offset{node->state_.ring_count};
        const usize ring_offset_u{static_cast<usize>(ring_offset)};
        const usize drawn_sample_count{static_cast<usize>(batch_count * batch_size)};
        if (ring_offset + batch_count * batch_size > node->state_.ns_capacity)
        {
            cudaSetDevice(caller_device);
            return QNPEPS_E2E_ERR_BAD_CONFIG;
        }
        const i64 first_batch{node->state_.next_batch};
        node->state_.next_batch += batch_count;
        for (auto sample_index = 0_uz; sample_index < drawn_sample_count; ++sample_index)
            node->state_.ring_epoch[ring_offset_u + sample_index] = epoch;
        node->state_.ring_count += batch_count * batch_size;
        qnpeps_e2e_status draw_status{QNPEPS_E2E_OK};
        if (node->state_.gpus >= 2 and node->state_.ns_ahead == 0)
            draw_status =
                draw_fill_multigpu(&node->state_, first_batch, batch_count, ring_offset, epoch);
        else if (node->state_.gpus >= 2)
        {
            node->state_.workers_->issue(
                node->state_.sampler_gpu,
                WorkerCmd{WorkerMode::Sample, first_batch, batch_count, ring_offset, {}, epoch}
            );
            draw_status = node->state_.workers_->wait(node->state_.sampler_gpu);
        }
        else
            draw_status = draw_fill(
                &node->state_, node->state_.ctx[0], first_batch, batch_count, ring_offset, epoch
            );
        if (draw_status != QNPEPS_E2E_OK)
        {
            node->state_.fail(draw_status);
            cudaSetDevice(caller_device);
            return node->state_.status();
        }
    }

    QN_E2E_CUDA_CHECK(cudaMemcpy(
        node->state_.device_samples,
        node->state_.ring_cfg,
        sample_site_count,
        cudaMemcpyHostToDevice
    ));
    QN_E2E_CUDA_CHECK(cudaMemcpy(
        node->state_.device_log_proposals,
        node->state_.ring_logq,
        sizeof(f64) * sample_count_u,
        cudaMemcpyHostToDevice
    ));
    QN_E2E_CUDA_CHECK(cudaMemcpy(
        node->state_.device_log_gauges,
        node->state_.ring_lgauge,
        sizeof(f64) * sample_count_u,
        cudaMemcpyHostToDevice
    ));
    if (epoch_out)
        std::memcpy(epoch_out, node->state_.ring_epoch.data(), sizeof(i64) * sample_count_u);
    if (err_state() != QNPEPS_E2E_OK)
    {
        node->state_.fail(err_state());
        cudaSetDevice(caller_device);
        return node->state_.status();
    }
    const i64 left{node->state_.ring_count - sample_count};
    if (left > 0)
    {
        const usize left_u{static_cast<usize>(left)};
        std::memmove(
            node->state_.ring_cfg,
            node->state_.ring_cfg + sample_count * node->state_.sites,
            left_u * static_cast<usize>(node->state_.sites)
        );
        std::memmove(
            node->state_.ring_logq, node->state_.ring_logq + sample_count, sizeof(f64) * left_u
        );
        std::memmove(
            node->state_.ring_lgauge, node->state_.ring_lgauge + sample_count, sizeof(f64) * left_u
        );
        std::memmove(
            node->state_.ring_epoch.data(),
            node->state_.ring_epoch.data() + sample_count,
            sizeof(i64) * left_u
        );
    }
    node->state_.ring_count = left;

    const i64 predraw_batch_count{
        node->state_.ns_ahead > 0 ? (node->state_.ns_ahead + batch_size - 1) / batch_size : 0
    };
    const auto predraw_exceeds_capacity =
        node->state_.ring_count + predraw_batch_count * batch_size > node->state_.ns_capacity;
    if (predraw_batch_count > 0 and predraw_exceeds_capacity)
    {
        cudaSetDevice(caller_device);
        return QNPEPS_E2E_ERR_BAD_CONFIG;
    }

    if (not node->state_.use_p2p and node->state_.gpus >= 2)
    {
        node->state_.host_samples.resize(sample_site_count);
        QN_E2E_CUDA_CHECK(cudaMemcpy(
            node->state_.host_samples.data(),
            node->state_.device_samples,
            sample_site_count,
            cudaMemcpyDeviceToHost
        ));
    }

    node->state_.step_stage = "eo";
    const std::vector<Shard> shards{
        build_shards(sample_count, node->state_.gpus, node->state_.cfg.meo)
    };
    for (int gpu{1}; gpu < node->state_.gpus; ++gpu)
    {
        const usize gpu_index{static_cast<usize>(gpu)};
        node->state_.workers_->issue(
            gpu, WorkerCmd{WorkerMode::Eo, 0, 0, 0, shards[gpu_index], epoch}
        );
    }
    {
        const qnpeps_e2e_status device_zero_status{
            eo_run_shard(&node->state_, 0, shards[0], epoch)
        };
        if (device_zero_status != QNPEPS_E2E_OK) node->state_.fail(device_zero_status);
    }
    for (int gpu{1}; gpu < node->state_.gpus; ++gpu)
    {
        const qnpeps_e2e_status worker_status{node->state_.workers_->wait(gpu)};
        if (worker_status != QNPEPS_E2E_OK) node->state_.fail(worker_status);
    }
    if (node->state_.status() == QNPEPS_E2E_OK and not node->state_.use_p2p)
    {
        cudaSetDevice(0);
        for (int gpu{1}; gpu < node->state_.gpus; ++gpu)
        {
            const usize gpu_index{static_cast<usize>(gpu)};
            const Shard shard{shards[gpu_index]};
            if (shard.count <= 0) continue;
            const usize scalar_bytes{sizeof(f64) * static_cast<usize>(2 * shard.count)};
            QN_E2E_CUDA_CHECK(cudaMemcpy(
                node->state_.device_log_amplitudes + 2 * shard.base,
                node->state_.host_log_amplitudes[gpu_index].data(),
                scalar_bytes,
                cudaMemcpyHostToDevice
            ));
            QN_E2E_CUDA_CHECK(cudaMemcpy(
                node->state_.device_local_energies + 2 * shard.base,
                node->state_.host_local_energies[gpu_index].data(),
                scalar_bytes,
                cudaMemcpyHostToDevice
            ));
        }
    }
    if (node->state_.status() != QNPEPS_E2E_OK)
    {
        cudaSetDevice(caller_device);
        return node->state_.status();
    }

    node->state_.step_stage = "sampling";
    if (predraw_batch_count > 0)
    {
        const i64 ring_offset{node->state_.ring_count};
        const usize ring_offset_u{static_cast<usize>(ring_offset)};
        const i64 first_batch{node->state_.next_batch};
        node->state_.next_batch += predraw_batch_count;
        const usize predraw_sample_count{static_cast<usize>(predraw_batch_count * batch_size)};
        for (auto sample_index = 0_uz; sample_index < predraw_sample_count; ++sample_index)
            node->state_.ring_epoch[ring_offset_u + sample_index] = epoch;
        node->state_.ring_count += predraw_batch_count * batch_size;
        if (node->state_.gpus >= 2)
        {
            node->state_.workers_->issue(
                node->state_.sampler_gpu,
                WorkerCmd{
                    WorkerMode::Sample, first_batch, predraw_batch_count, ring_offset, {}, epoch
                }
            );
            node->state_.predraw_pending = true;
        }
        else
        {
            const qnpeps_e2e_status draw_status{draw_fill(
                &node->state_,
                node->state_.ctx[0],
                first_batch,
                predraw_batch_count,
                ring_offset,
                epoch
            )};
            if (draw_status != QNPEPS_E2E_OK) node->state_.fail(draw_status);
        }
    }
    node->state_.step_stage = "minsr";
    cudaSetDevice(0);
    if (node->state_.use_distributed_minsr)
        node->state_.solve_->solve_distributed(
            sample_count, shards, relative_cut, absolute_cut, theta, e_mean_out, e_var_out, ess_out
        );
    else
        node->state_.solve_->solve(
            sample_count, shards, relative_cut, absolute_cut, theta, e_mean_out, e_var_out, ess_out
        );
    if (err_state() != QNPEPS_E2E_OK) node->state_.fail(err_state());

    if (node->state_.status() == QNPEPS_E2E_OK)
    {
        if (samples_out)
        {
            QN_E2E_CUDA_CHECK(cudaMemcpyAsync(
                samples_out,
                node->state_.device_samples,
                sample_site_count,
                cudaMemcpyDeviceToDevice,
                node->state_.stream0()
            ));
        }
        if (logq_out)
        {
            QN_E2E_CUDA_CHECK(cudaMemcpyAsync(
                logq_out,
                node->state_.device_log_proposals,
                sizeof(f64) * sample_count_u,
                cudaMemcpyDeviceToDevice,
                node->state_.stream0()
            ));
        }
        if (log_gauge_out)
        {
            QN_E2E_CUDA_CHECK(cudaMemcpyAsync(
                log_gauge_out,
                node->state_.device_log_gauges,
                sizeof(f64) * sample_count_u,
                cudaMemcpyDeviceToDevice,
                node->state_.stream0()
            ));
        }
        if (logpsi_out)
        {
            QN_E2E_CUDA_CHECK(cudaMemcpyAsync(
                logpsi_out,
                node->state_.device_log_amplitudes,
                sizeof(f64) * doubled_sample_count,
                cudaMemcpyDeviceToDevice,
                node->state_.stream0()
            ));
        }
        if (e_loc_out)
        {
            QN_E2E_CUDA_CHECK(cudaMemcpyAsync(
                e_loc_out,
                node->state_.device_local_energies,
                sizeof(f64) * doubled_sample_count,
                cudaMemcpyDeviceToDevice,
                node->state_.stream0()
            ));
        }
        QN_E2E_CUDA_CHECK(cudaStreamSynchronize(node->state_.stream0()));
        if (o_rows_host)
        {
            std::memcpy(
                reinterpret_cast<cf*>(o_rows_host),
                node->state_.host_rows,
                sizeof(cf) * sample_count_u * static_cast<usize>(node->state_.compact)
            );
        }
    }
    cudaSetDevice(caller_device);
    if (err_state() != QNPEPS_E2E_OK) node->state_.fail(err_state());
    return node->state_.status() == QNPEPS_E2E_OK ? QNPEPS_E2E_OK : QNPEPS_E2E_ERR_INTERNAL;
}

extern "C" __attribute__((visibility("default"))) const char* qnpeps_e2e_node_error_stage(
    const qnpeps_e2e_node* node
)
{
    return node ? node->state_.step_stage : "unknown";
}

extern "C" qnpeps_e2e_status qnpeps_e2e_node_step_euler(
    qnpeps_e2e_node* node, const QnpepsE2eEulerStepArgs* args
)
{
    if (not node or not args) return QNPEPS_E2E_ERR_NULL_ARG;
    if (args->struct_size != sizeof(QnpepsE2eEulerStepArgs)) return QNPEPS_E2E_ERR_BAD_VERSION;
    const auto missing_output = not args->peps_f32_io or not args->energy_mean_output
                                or not args->energy_variance_output or not args->ess_output;
    if (missing_output) return QNPEPS_E2E_ERR_NULL_ARG;
    if (node->state_.status() != QNPEPS_E2E_OK) return QNPEPS_E2E_ERR_INTERNAL;
    if (not std::isfinite(args->learning_rate)) return QNPEPS_E2E_ERR_BAD_CONFIG;

    const u64 peps_bytes{static_cast<u64>(node->state_.peps_bytes)};
    const u64 state_f64_bytes{static_cast<u64>(node->state_.dense) * sizeof(qn_e2e::zd)};
    const u64 theta_bytes{static_cast<u64>(node->state_.dense) * sizeof(cf)};
    if (args->peps_f32_bytes != peps_bytes) return QNPEPS_E2E_ERR_BAD_CONFIG;
    const auto invalid_theta_bytes = (args->theta_output and args->theta_dot_bytes != theta_bytes)
                                     or (not args->theta_output and args->theta_dot_bytes != 0);
    if (invalid_theta_bytes) return QNPEPS_E2E_ERR_BAD_CONFIG;
    if (args->theta_output == args->peps_f32_io) return QNPEPS_E2E_ERR_BAD_CONFIG;

    if (args->precision == QNPEPS_E2E_UPDATE_F32)
    {
        if (args->state_f64_io or args->state_f64_bytes != 0) return QNPEPS_E2E_ERR_BAD_CONFIG;
    }
    else if (args->precision == QNPEPS_E2E_UPDATE_F64_MASTER)
    {
        if (not args->state_f64_io) return QNPEPS_E2E_ERR_NULL_ARG;
        if (args->state_f64_bytes != state_f64_bytes) return QNPEPS_E2E_ERR_BAD_CONFIG;
        const auto aliased_state = args->state_f64_io == static_cast<void*>(args->peps_f32_io)
                                   or args->state_f64_io == static_cast<void*>(args->theta_output);
        if (aliased_state) return QNPEPS_E2E_ERR_BAD_CONFIG;
    }
    else
        return QNPEPS_E2E_ERR_BAD_CONFIG;

    auto theta{
        args->theta_output ? reinterpret_cast<cf*>(args->theta_output) : node->state_.device_theta
    };
    qnpeps_e2e_status status{qnpeps_e2e_node_step(
        node,
        args->n_samples,
        args->relative_cut,
        args->absolute_cut,
        reinterpret_cast<qnpeps_e2e_cbuf*>(theta),
        args->energy_mean_output,
        args->energy_variance_output,
        args->ess_output,
        args->samples_out,
        args->logq_out,
        args->log_gauge_out,
        args->logpsi_out,
        args->e_loc_out,
        args->host_rows,
        args->epoch_out
    )};
    if (status != QNPEPS_E2E_OK) return status;

    int caller_device{};
    if (cudaGetDevice(&caller_device) != cudaSuccess) return QNPEPS_E2E_ERR_CUDA;
    if (cudaSetDevice(0) != cudaSuccess) return QNPEPS_E2E_ERR_CUDA;

    auto* peps{reinterpret_cast<cf*>(args->peps_f32_io)};
    if (args->precision == QNPEPS_E2E_UPDATE_F32)
    {
        const int current_slot{static_cast<int>(node->state_.epoch_current & 1)};
        const cudaError_t copy_status{cudaMemcpyAsync(
            peps,
            node->state_.ctx[0].device_peps_slots[current_slot],
            static_cast<usize>(node->state_.peps_bytes),
            cudaMemcpyDeviceToDevice,
            node->state_.stream0()
        )};
        status = copy_status == cudaSuccess ? qn_e2e::launch_update_f32(
                                                  node->state_.device_update_sites,
                                                  static_cast<int>(node->state_.sites),
                                                  peps,
                                                  theta,
                                                  args->learning_rate,
                                                  node->state_.stream0()
                                              )
                                            : QNPEPS_E2E_ERR_CUDA;
    }
    else
        status = qn_e2e::launch_update_f64(
            node->state_.device_update_sites,
            static_cast<int>(node->state_.sites),
            static_cast<qn_e2e::zd*>(args->state_f64_io),
            theta,
            peps,
            args->learning_rate,
            node->state_.stream0()
        );

    if (status == QNPEPS_E2E_OK and cudaStreamSynchronize(node->state_.stream0()) != cudaSuccess)
        status = QNPEPS_E2E_ERR_CUDA;
    if (status != QNPEPS_E2E_OK)
    {
        node->state_.fail(status);
        cudaSetDevice(caller_device);
        return status;
    }

    status = qnpeps_e2e_node_submit_theta(node, peps);
    cudaSetDevice(caller_device);
    return status;
}

extern "C" qnpeps_e2e_status qnpeps_e2e_node_footprint_bytes(
    const QnpepsE2eConfig* cfg,
    int gpus,
    int64_t ns_capacity,
    int64_t ns_ahead,
    int64_t dim_batch,
    const void* terms,
    uint64_t* out_bytes
)
{
    const qnpeps_e2e_status v{qn_e2e::mg::_config_check(cfg)};
    if (v != QNPEPS_E2E_OK) return v;
    if (not terms or not out_bytes) return QNPEPS_E2E_ERR_NULL_ARG;
    if (gpus < 1 or ns_capacity < 2 or dim_batch < 1) return QNPEPS_E2E_ERR_BAD_CONFIG;
    (void) ns_ahead;
    const auto reservation = qnpeps::arena_reservation_bytes();
    if (err_state() != QNPEPS_E2E_OK) return err_state();
    *out_bytes = static_cast<u64>(reservation);
    return QNPEPS_E2E_OK;
}
