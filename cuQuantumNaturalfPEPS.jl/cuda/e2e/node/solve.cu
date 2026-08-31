#include "e2e/node/solve.cuh"

#include "e2e/common.cuh"
#include "e2e/node/workers.cuh"
#include "minsr/distributed.cuh"
#include "minsr/solve.cuh"

#include <cuda_runtime.h>

using qn_e2e::cf;
using qn_e2e::err_state;
using qn_e2e::set_err;
using qn_e2e::mg::map_local_energy_status;
using qn_e2e::mg::Shard;

NodeSolve::NodeSolve(NodeState& state) noexcept : state_{state} {}

NodeSolve::~NodeSolve()
{
    if (solve_cache_) qnpeps::minsr::prepared_solve_destroy(solve_cache_);
}

auto NodeSolve::cached_solve(i64 sample_count) -> qnpeps::minsr::MinsrSolveContext*
{
    if (solve_cache_ and solve_cache_samples_ == sample_count) return solve_cache_;
    if (solve_cache_)
    {
        qnpeps::minsr::prepared_solve_destroy(solve_cache_);
        solve_cache_ = nullptr;
        solve_cache_samples_ = 0;
    }
    auto* const created = qnpeps::minsr::prepared_solve_create(sample_count, state_.linalg0());
    if (not created) return nullptr;
    if (err_state() != QNPEPS_E2E_OK)
    {
        qnpeps::minsr::prepared_solve_destroy(created);
        return nullptr;
    }
    solve_cache_ = created;
    solve_cache_samples_ = sample_count;
    return solve_cache_;
}

auto NodeSolve::solve(
    i64 ns,
    const std::vector<Shard>& shards,
    f64 relative_cut,
    f64 absolute_cut,
    cf* theta_dot_out,
    f64* e_mean_out,
    f64* e_var_out,
    f64* ess_out
) -> void
{
    auto* n{&state_};
    const i64 compact{n->compact};
    QN_E2E_CUDA_CHECK(
        cudaMemsetAsync(n->device_gram, 0, sizeof(cf) * static_cast<usize>(ns * ns), n->stream0())
    );
    for (auto i = 0_uz; i < shards.size() and err_state() == QNPEPS_E2E_OK; ++i)
    {
        const Shard si{shards[i]};
        if (si.count <= 0) continue;
        QN_E2E_CUDA_CHECK(cudaMemcpyAsync(
            n->device_stage_a,
            n->host_rows + si.base * compact,
            sizeof(cf) * static_cast<usize>(si.count * compact),
            cudaMemcpyHostToDevice,
            n->stream0()
        ));
        for (auto j = 0_uz; j < shards.size() and err_state() == QNPEPS_E2E_OK; ++j)
        {
            const Shard sj{shards[j]};
            if (sj.count <= 0) continue;
            const usize shard_sample_count{static_cast<usize>(sj.count)};
            auto rows_b{n->device_stage_a};
            auto samp_b{n->device_samples + si.base * n->sites};
            if (j != i)
            {
                QN_E2E_CUDA_CHECK(cudaMemcpyAsync(
                    n->device_stage_b,
                    n->host_rows + sj.base * compact,
                    sizeof(cf) * shard_sample_count * static_cast<usize>(compact),
                    cudaMemcpyHostToDevice,
                    n->stream0()
                ));
                rows_b = n->device_stage_b;
                samp_b = n->device_samples + sj.base * n->sites;
            }
            map_local_energy_status(qnpeps_eloc_gram_tile(
                &n->lcfg,
                reinterpret_cast<const qnpeps_eloc_cbuf*>(n->device_stage_a),
                n->device_samples + si.base * n->sites,
                si.count,
                reinterpret_cast<const qnpeps_eloc_cbuf*>(rows_b),
                samp_b,
                sj.count,
                reinterpret_cast<qnpeps_eloc_cbuf*>(n->device_tile),
                n->stream0()
            ));
            if (err_state() == QNPEPS_E2E_OK)
            {
                QN_E2E_CUDA_CHECK(cudaMemcpy2DAsync(
                    n->device_gram + si.base * ns + sj.base,
                    sizeof(cf) * static_cast<usize>(ns),
                    n->device_tile,
                    sizeof(cf) * shard_sample_count,
                    sizeof(cf) * shard_sample_count,
                    static_cast<usize>(si.count),
                    cudaMemcpyDeviceToDevice,
                    n->stream0()
                ));
            }
        }
    }
    QN_E2E_CUDA_CHECK(cudaStreamSynchronize(n->stream0()));
    if (err_state() != QNPEPS_E2E_OK) return;

    qnpeps::minsr::qn_e2e_minsr_impl(
        n->cfg,
        ns,
        n->device_samples,
        n->device_log_amplitudes,
        n->device_local_energies,
        n->device_log_proposals,
        n->device_gram,
        nullptr,
        n->host_rows,
        n->host_tile_bytes,
        relative_cut,
        absolute_cut,
        theta_dot_out,
        e_mean_out,
        e_var_out,
        ess_out,
        n->linalg0(),
        cached_solve(ns)
    );
}

auto NodeSolve::solve_distributed(
    i64 ns,
    const std::vector<Shard>& shards,
    f64 relative_cut,
    f64 absolute_cut,
    cf* theta_dot_out,
    f64* e_mean_out,
    f64* e_var_out,
    f64* ess_out
) -> void
{
    auto* n{&state_};
    const usize gpu_count{static_cast<usize>(n->gpus)};
    std::vector<qnpeps::minsr::DistributedLane> lanes{gpu_count};
    for (int g{}; g < n->gpus; ++g)
    {
        const usize gpu_index{static_cast<usize>(g)};
        const auto shard = shards[gpu_index];
        lanes[gpu_index] = {
            .device = g,
            .base = shard.base,
            .count = shard.count,
            .samples = g == 0 ? n->device_samples : n->ctx[gpu_index].device_eo_samples,
            .rows = n->ctx[gpu_index].device_eo_rows,
            .linalg = &n->ctx[gpu_index].linalg()
        };
    }

    std::vector<qnpeps::minsr::DistributedGramArgs> arguments{gpu_count};
    for (int g{}; g < n->gpus; ++g)
    {
        const usize gpu_index{static_cast<usize>(g)};
        arguments[gpu_index] = {
            .config = &n->lcfg,
            .n_samples = ns,
            .sites = n->sites,
            .compact = n->compact,
            .peer_tile_bytes = n->peer_tile_bytes,
            .destination = &lanes[gpu_index],
            .lanes = lanes,
            .gram_device0 = n->device_gram
        };
    }

    for (int g{1}; g < n->gpus; ++g)
    {
        const usize gpu_index{static_cast<usize>(g)};
        n->workers_->issue(g, WorkerCmd{.mode = WorkerMode::Gram, .gram = &arguments[gpu_index]});
    }
    qnpeps::minsr::build_gram_blockrow(arguments[0]);
    if (err_state() != QNPEPS_E2E_OK) n->fail(err_state());
    for (int g{1}; g < n->gpus; ++g)
    {
        const auto status = n->workers_->wait(g);
        if (status != QNPEPS_E2E_OK) n->fail(status);
    }
    cudaSetDevice(0);
    if (n->status() != QNPEPS_E2E_OK) return;

    std::vector<qn_e2e::cd> coefficients{};
    bool did_solve{false};
    qnpeps::minsr::minsr_solve(
        ns,
        n->device_log_amplitudes,
        n->device_local_energies,
        n->device_log_proposals,
        n->device_gram,
        relative_cut,
        absolute_cut,
        e_mean_out,
        e_var_out,
        ess_out,
        coefficients,
        did_solve,
        n->linalg0(),
        cached_solve(ns)
    );
    if (err_state() != QNPEPS_E2E_OK) return;

    if (did_solve)
    {
        qnpeps::minsr::scatter_ring(
            qnpeps::minsr::DistributedScatterArgs{
                .n_samples = ns,
                .sites = n->sites,
                .compact = n->compact,
                .dense = n->dense,
                .dim_phys = n->cfg.dim_phys,
                .lanes = lanes,
                .coefficients = coefficients,
                .slot_site = n->slot_sites,
                .theta_device0 = theta_dot_out
            }
        );
    }
    else
    {
        QN_E2E_CUDA_CHECK(cudaMemsetAsync(
            theta_dot_out, 0, sizeof(cf) * static_cast<usize>(n->dense), n->stream0()
        ));
        QN_E2E_CUDA_CHECK(cudaStreamSynchronize(n->stream0()));
    }
}

auto NodeSolve::host_pinned(usize bytes) -> void*
{
    if (err_state() != QNPEPS_E2E_OK) return nullptr;
    void* p{};
    if (cudaHostAlloc(&p, bytes < 1 ? 1 : bytes, cudaHostAllocPortable) != cudaSuccess)
    {
        set_err(QNPEPS_E2E_ERR_OOM);
        return nullptr;
    }
    return p;
}
