#include "e2e/node/lifecycle.cuh"

#include "core/arena_cursor.cuh"
#include "core/defer.cuh"
#include "e2e/common.cuh"
#include "e2e/layout.cuh"
#include "e2e/mg_common.cuh"
#include "eo/env_build.cuh"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <string_view>
#include <vector>

using qn_e2e::cf;
using qn_e2e::compact_count;
using qn_e2e::mg::build_shards;
using qn_e2e::mg::local_energy_config;
using qn_e2e::mg::probe_peer_access_cached;
using qn_e2e::mg::sampler_config;
using qn_e2e::mg::Shard;

namespace
{
template <class T>
[[nodiscard]] auto lease_buffer(GpuCtx& gpu, std::string_view name, i64 count) -> T*
{
    const auto element_count = static_cast<usize>(std::max(count, i64{1}));
    require(
        element_count <= std::numeric_limits<usize>::max() / sizeof(T),
        Status::out_of_memory,
        "node buffer {} of {} elements overflows the size range",
        name,
        element_count
    );
    const auto bytes = sizeof(T) * element_count;
    const auto largest_free = gpu.buffer_lease->remaining();
    if (bytes > largest_free)
    {
        const auto arena = qnpeps::shared_device_arena(gpu.device)->reading();
        fail(
            Status::out_of_memory,
            "node buffer {} of {} bytes does not fit the arena of GPU {}, {} of {} bytes free, "
            "largest free block {}",
            name,
            bytes,
            gpu.device,
            arena.capacity - arena.in_use,
            arena.capacity,
            largest_free
        );
    }
    return static_cast<T*>(gpu.buffer_lease->take(bytes));
}

auto destroy_node(qnpeps_e2e_node* node) -> void
{
    if (not node) return;
    int caller_device{0};
    unchecked(cudaGetDevice(&caller_device));

    node->state_.workers_->stop();
    node->state_.solve_.reset();

    for (auto& context : node->state_.ctx)
    {
        if (not context.session) continue;
        unchecked(cudaSetDevice(context.device));
        unchecked(cudaDeviceSynchronize());
    }
    for (auto& context : node->state_.ctx)
    {
        if (not context.session) continue;
        unchecked(cudaSetDevice(context.device));
        context.buffer_lease.reset();
        context.session.reset();
    }
    qnpeps::maybe_free_host(node->state_.ring_cfg);
    qnpeps::maybe_free_host(node->state_.ring_logq);
    qnpeps::maybe_free_host(node->state_.ring_lgauge);
    qnpeps::maybe_free_host(node->state_.host_rows);

    unchecked(cudaSetDevice(caller_device));
    delete node;
}

auto create_node(
    const QnpepsE2eConfig* cfg,
    int gpus,
    int64_t ns_capacity,
    int64_t ns_ahead,
    int64_t dim_batch,
    int64_t host_tile_bytes,
    const void* terms,
    qnpeps_e2e_node** node_out
) -> void
{
    qn_e2e::mg::validate_node_config(cfg);
    require(node_out != nullptr, Status::null_argument, "node output is null");
    *node_out = nullptr;
    qn_e2e::mg::validate_node_terms(*cfg, terms);
    const auto valid_capacity = ns_capacity >= 2 and ns_ahead >= 0 and ns_ahead <= ns_capacity;
    require(
        valid_capacity,
        Status::bad_config,
        "node sample capacity {} with lookahead {} is not valid",
        ns_capacity,
        ns_ahead
    );
    const auto valid_batch =
        dim_batch >= 1 and dim_batch <= qnpeps::k_max_batch_size and dim_batch <= ns_capacity;
    require(
        valid_batch,
        Status::bad_config,
        "node batch {} outside 1 to {} or above the capacity {}",
        dim_batch,
        qnpeps::k_max_batch_size,
        ns_capacity
    );

    int dev_count{0};
    const auto count_status = cudaGetDeviceCount(&dev_count);
    require(
        count_status == cudaSuccess,
        Status::cuda,
        "node device count query failed, {}",
        qnpeps::cuda_failure_text(count_status)
    );
    require(
        gpus >= 1 and gpus <= dev_count,
        Status::bad_config,
        "node asks for {} of {} GPUs",
        gpus,
        dev_count
    );
    const usize gpu_count{static_cast<usize>(gpus)};
    const usize capacity{static_cast<usize>(ns_capacity)};

    int caller_device{0};
    unchecked(cudaGetDevice(&caller_device));
    const qnpeps::Defer restore_device{[&] { unchecked(cudaSetDevice(caller_device)); }};

    auto* n{new qnpeps_e2e_node()};
    auto created = false;
    const qnpeps::Defer destroy_after_failure{[&]
                                              {
                                                  if (created) return;
                                                  n->state_.status_value = QNPEPS_ERR_INTERNAL;
                                                  destroy_node(n);
                                              }};
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
    require(
        n->state_.peps_bytes >= 0 and n->state_.dlenv_bytes >= 0,
        Status::bad_config,
        "node sampler config yields PEPS bytes {} and environment bytes {}",
        n->state_.peps_bytes,
        n->state_.dlenv_bytes
    );

    n->state_.owned_terms =
        qnpeps::eo::OwnedTermTable{*static_cast<const QnpepsElocTermTable*>(terms)};
    n->state_.terms = n->state_.owned_terms.view();

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
        unchecked(cudaSetDevice(0));
    }
    if (force_dist and not force_staged and gpus >= 2) n->state_.use_p2p = true;
    n->state_.use_distributed_minsr = ns_ahead == 0 and gpus >= 2 and n->state_.use_p2p;
    n->state_.host_log_amplitudes.assign(gpu_count, {});
    n->state_.host_local_energies.assign(gpu_count, {});

    n->state_.ctx.resize(gpu_count);
    const i64 nc{ns_capacity};
    const i64 sc{n->state_.shard_capacity};
    const i64 comp{n->state_.compact};
    for (int g{0}; g < gpus; ++g)
    {
        const usize gpu_index{static_cast<usize>(g)};
        auto& c{n->state_.ctx[gpu_index]};
        c.device = g;
        const auto device_status = cudaSetDevice(g);
        require(
            device_status == cudaSuccess,
            Status::cuda,
            "node cannot select GPU {}, {}",
            g,
            qnpeps::cuda_failure_text(device_status)
        );
        c.session = qnpeps::make_session(nullptr);
        c.buffer_lease = qnpeps::shared_device_arena(g)->acquire_lease(c.stream());
        c.device_peps_slots[0] = lease_buffer<u8>(c, "peps_slot", n->state_.peps_bytes);
        c.device_peps_slots[1] = lease_buffer<u8>(c, "peps_slot", n->state_.peps_bytes);
        c.device_dlenv_slots[0] = lease_buffer<u8>(c, "dlenv_slot", n->state_.dlenv_bytes);
        c.device_dlenv_slots[1] = lease_buffer<u8>(c, "dlenv_slot", n->state_.dlenv_bytes);
        c.device_eo_samples = lease_buffer<u8>(c, "eo_samples", nc * n->state_.sites);
        c.device_eo_rows = lease_buffer<cf>(c, "eo_rows", sc * comp);
        c.device_eo_log_amplitudes = lease_buffer<f64>(c, "eo_log_amplitudes", 2 * sc);
        c.device_eo_local_energies = lease_buffer<f64>(c, "eo_local_energies", 2 * sc);
        if (g == 0) c.device_row_logs = lease_buffer<f64>(c, "row_logs", cfg->lx - 1);
        if (g == n->state_.sampler_gpu)
        {
            const auto draw_cap = n->state_.draw_cap;
            c.device_draw_samples = lease_buffer<u8>(c, "draw_samples", draw_cap * n->state_.sites);
            c.device_draw_log_proposals = lease_buffer<f64>(c, "draw_log_proposals", draw_cap);
            c.device_draw_log_gauges = lease_buffer<f64>(c, "draw_log_gauges", draw_cap);
        }
    }

    {
        unchecked(cudaSetDevice(0));
        auto& c0{n->state_.ctx[0]};
        n->state_.device_samples = lease_buffer<u8>(c0, "samples", nc * n->state_.sites);
        n->state_.device_log_proposals = lease_buffer<f64>(c0, "log_proposals", nc);
        n->state_.device_log_gauges = lease_buffer<f64>(c0, "log_gauges", nc);
        n->state_.device_log_amplitudes = lease_buffer<f64>(c0, "log_amplitudes", 2 * nc);
        n->state_.device_local_energies = lease_buffer<f64>(c0, "local_energies", 2 * nc);
        n->state_.device_gram = lease_buffer<cf>(c0, "gram", nc * nc);
        n->state_.device_theta = lease_buffer<cf>(c0, "theta", n->state_.dense);
        const std::vector<qn_e2e::UpdateSite> device_update_sites{qn_e2e::build_update_sites(*cfg)};
        require(
            device_update_sites.size() == static_cast<usize>(n->state_.sites),
            Status::bad_config,
            "node update layout covers {} of {} sites",
            device_update_sites.size(),
            n->state_.sites
        );
        {
            n->state_.device_update_sites =
                lease_buffer<qn_e2e::UpdateSite>(c0, "update_sites", n->state_.sites);
            check(cudaMemcpy(
                n->state_.device_update_sites,
                device_update_sites.data(),
                device_update_sites.size() * sizeof(qn_e2e::UpdateSite),
                cudaMemcpyHostToDevice
            ));
        }
        n->state_.device_stage_a = lease_buffer<cf>(c0, "stage_a", sc * comp);
        n->state_.device_stage_b = lease_buffer<cf>(c0, "stage_b", sc * comp);
        n->state_.device_tile = lease_buffer<cf>(c0, "tile", sc * sc);
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

    n->state_.workers_->start();
    *node_out = n;
    created = true;
}
}

extern "C" qnpeps_status qnpeps_e2e_node_create(
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
    return guarded(
        [&]
        {
            create_node(
                cfg, gpus, ns_capacity, ns_ahead, dim_batch, host_tile_bytes, terms, node_out
            );
        }
    );
}

extern "C" qnpeps_status qnpeps_e2e_node_destroy(qnpeps_e2e_node* node)
{
    return guarded([&] { destroy_node(node); });
}
