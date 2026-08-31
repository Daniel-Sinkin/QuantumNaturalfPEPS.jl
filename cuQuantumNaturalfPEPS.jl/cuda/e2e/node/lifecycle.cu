#include "e2e/node/lifecycle.cuh"

#include "e2e/common.cuh"
#include "e2e/layout.cuh"
#include "e2e/mg_common.cuh"
#include "eo/env_build.cuh"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

using qn_e2e::cf;
using qn_e2e::compact_count;
using qn_e2e::err_state;
using qn_e2e::set_err;
using qn_e2e::mg::build_shards;
using qn_e2e::mg::device_allocate;
using qn_e2e::mg::local_energy_config;
using qn_e2e::mg::probe_peer_access_cached;
using qn_e2e::mg::sampler_config;
using qn_e2e::mg::Shard;

extern "C" qnpeps_e2e_status qnpeps_e2e_node_create(
    const QnpepsE2eConfig* cfg,
    int gpus,
    int64_t ns_capacity,
    int64_t ns_ahead,
    int64_t dim_batch,
    int64_t host_tile_bytes,
    const void* terms,
    qnpeps_e2e_node** node_out
)
{
    const qnpeps_e2e_status v{qn_e2e::mg::_config_check(cfg)};
    if (v != QNPEPS_E2E_OK) return v;
    if (not terms or not node_out) return QNPEPS_E2E_ERR_NULL_ARG;
    *node_out = nullptr;
    if (ns_capacity < 2 or ns_ahead < 0 or ns_ahead > ns_capacity) return QNPEPS_E2E_ERR_BAD_CONFIG;
    if (dim_batch < 1 or dim_batch > qnpeps::k_max_batch_size or dim_batch > ns_capacity)
        return QNPEPS_E2E_ERR_BAD_CONFIG;

    int dev_count{0};
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess) return QNPEPS_E2E_ERR_CUDA;
    if (gpus < 1 or gpus > dev_count) return QNPEPS_E2E_ERR_BAD_CONFIG;
    const usize gpu_count{static_cast<usize>(gpus)};
    const usize capacity{static_cast<usize>(ns_capacity)};

    int caller_device{0};
    cudaGetDevice(&caller_device);
    qn_e2e::clear_err();

    auto* n{new qnpeps_e2e_node()};
    n->state_.cfg = *cfg;
    n->state_.scfg = sampler_config(*cfg);
    n->state_.lcfg = local_energy_config(*cfg);
    n->state_.gpus = gpus;
    n->state_.sampler_gpu = gpus >= 2 ? 1 : 0;
    n->state_.ns_capacity = ns_capacity;
    n->state_.ns_ahead = ns_ahead;
    n->state_.dim_batch = dim_batch;
    n->state_.draw_cap = ((ns_capacity + dim_batch - 1) / dim_batch) * dim_batch;
    const std::vector<Shard> capacity_shards{build_shards(ns_capacity, gpus, cfg->meo)};
    for (const Shard& shard : capacity_shards)
        if (shard.count > n->state_.shard_capacity) n->state_.shard_capacity = shard.count;
    n->state_.host_tile_bytes = host_tile_bytes;
    n->state_.caller_device = caller_device;

    n->state_.sites = static_cast<i64>(cfg->lx) * cfg->ly;
    n->state_.compact = compact_count(*cfg);
    n->state_.dense = static_cast<i64>(cfg->dim_phys) * n->state_.compact;
    n->state_.peps_bytes = qnpeps_peps_bytes(&n->state_.scfg);
    n->state_.dlenv_bytes = qnpeps_dlenv_bytes(&n->state_.scfg);
    if (n->state_.peps_bytes < 0 or n->state_.dlenv_bytes < 0)
    {
        delete n;
        cudaSetDevice(caller_device);
        return QNPEPS_E2E_ERR_BAD_CONFIG;
    }

    const auto* tt{static_cast<const QnpepsElocTermTable*>(terms)};
    n->state_.terms_diag.assign(tt->diag, tt->diag + tt->n_diag);
    n->state_.terms_flip.assign(tt->flip, tt->flip + tt->n_flip);
    n->state_.terms.n_diag = tt->n_diag;
    n->state_.terms.diag = n->state_.terms_diag.empty() ? nullptr : n->state_.terms_diag.data();
    n->state_.terms.n_flip = tt->n_flip;
    n->state_.terms.flip = n->state_.terms_flip.empty() ? nullptr : n->state_.terms_flip.data();

    if (auto e{std::getenv("QNPEPS_E2E_INJECT_OOM_GPU")}; e and e[0])
        n->state_.inject_oom_gpu = std::atoi(e);
    if (auto e{std::getenv("QNPEPS_E2E_INJECT_DELAY_GPU")}; e and e[0])
    {
        n->state_.inject_delay_gpu = std::atoi(e);
        if (auto colon{std::strchr(e, ':')}) n->state_.inject_delay_ms = std::atoi(colon + 1);
    }

    bool force_dist{false};
    if (auto fd{std::getenv("QNPEPS_E2E_FORCE_DIST")}; fd and fd[0] == '1') force_dist = true;
    bool force_staged{false};
    if (auto fs{std::getenv("QNPEPS_E2E_FORCE_STAGED")}; fs and fs[0] == '1') force_staged = true;
    n->state_.use_p2p = not force_staged;
    if (n->state_.use_p2p and gpus >= 2)
    {
        n->state_.use_p2p = probe_peer_access_cached(gpus);
        cudaSetDevice(0);
    }
    if (force_dist and not force_staged and gpus >= 2) n->state_.use_p2p = true;
    n->state_.use_distributed_minsr = ns_ahead == 0 and gpus >= 2 and n->state_.use_p2p;
    n->state_.host_log_amplitudes.assign(gpu_count, {});
    n->state_.host_local_energies.assign(gpu_count, {});

    n->state_.ctx.resize(gpu_count);
    const i64 nc{ns_capacity};
    const i64 sc{n->state_.shard_capacity};
    const i64 comp{n->state_.compact};
    for (int g{0}; g < gpus and err_state() == QNPEPS_E2E_OK; ++g)
    {
        const usize gpu_index{static_cast<usize>(g)};
        auto& c{n->state_.ctx[gpu_index]};
        c.device = g;
        if (cudaSetDevice(g) != cudaSuccess)
        {
            set_err(QNPEPS_E2E_ERR_CUDA);
            break;
        }
        c.session = qnpeps::make_session(nullptr);
        if (not c.session)
        {
            set_err(err_state());
            break;
        }
        c.device_peps_slots[0] = device_allocate<u8>(n->state_.peps_bytes);
        c.device_peps_slots[1] = device_allocate<u8>(n->state_.peps_bytes);
        c.device_dlenv_slots[0] = device_allocate<u8>(n->state_.dlenv_bytes);
        c.device_dlenv_slots[1] = device_allocate<u8>(n->state_.dlenv_bytes);
        c.device_eo_samples = device_allocate<u8>(nc * n->state_.sites);
        c.device_eo_rows = device_allocate<cf>(sc * comp);
        c.device_eo_log_amplitudes = device_allocate<f64>(2 * sc);
        c.device_eo_local_energies = device_allocate<f64>(2 * sc);
        if (g == 0) c.device_row_logs = device_allocate<f64>(cfg->lx - 1);
        if (g == n->state_.sampler_gpu)
        {
            c.device_draw_samples = device_allocate<u8>(n->state_.draw_cap * n->state_.sites);
            c.device_draw_log_proposals = device_allocate<f64>(n->state_.draw_cap);
            c.device_draw_log_gauges = device_allocate<f64>(n->state_.draw_cap);
        }
    }

    if (err_state() == QNPEPS_E2E_OK)
    {
        cudaSetDevice(0);
        n->state_.device_samples = device_allocate<u8>(nc * n->state_.sites);
        n->state_.device_log_proposals = device_allocate<f64>(nc);
        n->state_.device_log_gauges = device_allocate<f64>(nc);
        n->state_.device_log_amplitudes = device_allocate<f64>(2 * nc);
        n->state_.device_local_energies = device_allocate<f64>(2 * nc);
        n->state_.device_gram = device_allocate<cf>(nc * nc);
        n->state_.device_theta = device_allocate<cf>(n->state_.dense);
        const std::vector<qn_e2e::UpdateSite> device_update_sites{qn_e2e::build_update_sites(*cfg)};
        if (device_update_sites.size() != static_cast<usize>(n->state_.sites))
            set_err(QNPEPS_E2E_ERR_BAD_CONFIG);
        if (err_state() == QNPEPS_E2E_OK)
        {
            n->state_.device_update_sites = device_allocate<qn_e2e::UpdateSite>(n->state_.sites);
            QN_E2E_CUDA_CHECK(cudaMemcpy(
                n->state_.device_update_sites,
                device_update_sites.data(),
                device_update_sites.size() * sizeof(qn_e2e::UpdateSite),
                cudaMemcpyHostToDevice
            ));
        }
        n->state_.device_stage_a = device_allocate<cf>(sc * comp);
        n->state_.device_stage_b = device_allocate<cf>(sc * comp);
        n->state_.device_tile = device_allocate<cf>(sc * sc);
        n->state_.slot_sites = qn_e2e::build_slot_site(*cfg, n->state_.compact);
    }

    n->state_.ring_cfg = static_cast<u8*>(
        n->state_.solve_->host_pinned(capacity * static_cast<usize>(n->state_.sites))
    );
    n->state_.ring_logq = static_cast<f64*>(n->state_.solve_->host_pinned(sizeof(f64) * capacity));
    n->state_.ring_lgauge =
        static_cast<f64*>(n->state_.solve_->host_pinned(sizeof(f64) * capacity));
    n->state_.host_rows =
        static_cast<cf*>(n->state_.solve_->host_pinned(sizeof(cf) * static_cast<usize>(nc * comp)));
    n->state_.ring_epoch.assign(capacity, 0);

    if (err_state() != QNPEPS_E2E_OK)
    {
        const qnpeps_e2e_status st{err_state()};
        n->state_.status_value = QNPEPS_E2E_ERR_INTERNAL;
        qnpeps_e2e_node_destroy(n);
        cudaSetDevice(caller_device);
        return st;
    }

    n->state_.workers_->start();

    cudaSetDevice(caller_device);
    *node_out = n;
    return QNPEPS_E2E_OK;
}

extern "C" qnpeps_e2e_status qnpeps_e2e_node_destroy(qnpeps_e2e_node* node)
{
    if (not node) return QNPEPS_E2E_OK;
    int caller_device{0};
    cudaGetDevice(&caller_device);

    node->state_.workers_->stop();

    for (int gpu{0}; gpu < node->state_.gpus; ++gpu)
    {
        const usize gpu_index{static_cast<usize>(gpu)};
        auto& context{node->state_.ctx[gpu_index]};
        cudaSetDevice(gpu);
        cudaFree(context.device_peps_slots[0]);
        cudaFree(context.device_peps_slots[1]);
        cudaFree(context.device_dlenv_slots[0]);
        cudaFree(context.device_dlenv_slots[1]);
        cudaFree(context.device_eo_samples);
        cudaFree(context.device_eo_rows);
        cudaFree(context.device_eo_log_amplitudes);
        cudaFree(context.device_eo_local_energies);
        cudaFree(context.device_row_logs);
        cudaFree(context.device_draw_samples);
        cudaFree(context.device_draw_log_proposals);
        cudaFree(context.device_draw_log_gauges);
        if (context.session)
        {
            CUDA_NOCHECK(cudaStreamSynchronize(context.stream()));
            context.session.reset();
        }
    }
    cudaSetDevice(0);
    cudaFree(node->state_.device_samples);
    cudaFree(node->state_.device_log_proposals);
    cudaFree(node->state_.device_log_gauges);
    cudaFree(node->state_.device_log_amplitudes);
    cudaFree(node->state_.device_local_energies);
    cudaFree(node->state_.device_gram);
    cudaFree(node->state_.device_theta);
    cudaFree(node->state_.device_update_sites);
    cudaFree(node->state_.device_stage_a);
    cudaFree(node->state_.device_stage_b);
    cudaFree(node->state_.device_tile);
    if (node->state_.ring_cfg) cudaFreeHost(node->state_.ring_cfg);
    if (node->state_.ring_logq) cudaFreeHost(node->state_.ring_logq);
    if (node->state_.ring_lgauge) cudaFreeHost(node->state_.ring_lgauge);
    if (node->state_.host_rows) cudaFreeHost(node->state_.host_rows);

    cudaSetDevice(caller_device);
    delete node;
    return QNPEPS_E2E_OK;
}
