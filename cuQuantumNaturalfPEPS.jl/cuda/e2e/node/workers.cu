#include "e2e/node/workers.cuh"

#include "dlenv/build.cuh"
#include "eo/shards.cuh"
#include "sampler/draw.cuh"

#include <chrono>
#include <cuda_runtime.h>

using qn_e2e::cf;
using qn_e2e::err_state;
using qn_e2e::mg::map_sampler_status;
using qn_e2e::mg::pin_thread_to_gpu_numa;
using qn_e2e::mg::Shard;

namespace
{
auto copy_in(
    NodeState* n,
    void* destination,
    int destination_device,
    const void* source,
    int source_device,
    usize bytes
) -> void
{
    if (n->use_p2p)
        QN_E2E_CUDA_CHECK(
            cudaMemcpyPeer(destination, destination_device, source, source_device, bytes)
        );
    else
    {
        std::vector<u8> h{};
        h.resize(bytes);
        QN_E2E_CUDA_CHECK(cudaMemcpy(h.data(), source, bytes, cudaMemcpyDeviceToHost));
        const int save{[]
                       {
                           int d{};
                           cudaGetDevice(&d);
                           return d;
                       }()};
        cudaSetDevice(destination_device);
        QN_E2E_CUDA_CHECK(cudaMemcpy(destination, h.data(), bytes, cudaMemcpyHostToDevice));
        cudaSetDevice(save);
    }
}
}

auto wait_epoch_ready(NodeState* n, i64 e) -> void
{
    std::unique_lock<std::mutex> lock{n->epoch_m};
    n->epoch_cv.wait(lock, [&] { return n->epoch_ready >= e or n->status() != QNPEPS_E2E_OK; });
}

auto draw_fill(NodeState* n, GpuCtx& g, i64 first_batch, i64 nbatch, i64 ring_off, i64 epoch)
    -> qnpeps_e2e_status
{
    if (nbatch <= 0) return QNPEPS_E2E_OK;
    wait_epoch_ready(n, epoch);
    if (n->status() != QNPEPS_E2E_OK) return QNPEPS_E2E_ERR_INTERNAL;

    const i64 count{nbatch * n->dim_batch};
    const usize count_u{static_cast<usize>(count)};
    qnpeps_status ss{QNPEPS_OK};
    {
        std::lock_guard<std::mutex> lk{n->sampler_m};
        const qnpeps::sampler::SampleArgs sample_args{
            .device_peps = g.device_peps_slots[epoch & 1],
            .device_dlenv = g.device_dlenv_slots[epoch & 1],
            .scratch = nullptr,
            .scratch_bytes = 0,
            .output = g.device_draw_samples,
            .logpc_out = g.device_draw_log_proposals,
            .lognorm_out = g.device_draw_log_gauges,
            .n_samples = static_cast<u64>(count),
            .batch_base = static_cast<u64>(first_batch),
            .dim_batch = static_cast<u64>(n->dim_batch),
            .stream = g.stream(),
            .output_location = qnpeps::sampler::SampleOutputLocation::device
        };
        ss = qnpeps::sampler::sample(n->scfg, sample_args, g.linalg());
        if (ss == QNPEPS_OK) cudaStreamSynchronize(g.stream());
    }
    if (ss != QNPEPS_OK)
    {
        return ss == QNPEPS_ERR_OOM    ? QNPEPS_E2E_ERR_OOM
               : ss == QNPEPS_ERR_CUDA ? QNPEPS_E2E_ERR_CUDA
                                       : QNPEPS_E2E_ERR_INTERNAL;
    }

    cudaError_t e{cudaSuccess};
    e = cudaMemcpy(
        n->ring_cfg + ring_off * n->sites,
        g.device_draw_samples,
        count_u * static_cast<usize>(n->sites),
        cudaMemcpyDeviceToHost
    );
    if (e == cudaSuccess)
    {
        e = cudaMemcpy(
            n->ring_logq + ring_off,
            g.device_draw_log_proposals,
            sizeof(f64) * count_u,
            cudaMemcpyDeviceToHost
        );
    }
    if (e == cudaSuccess)
    {
        e = cudaMemcpy(
            n->ring_lgauge + ring_off,
            g.device_draw_log_gauges,
            sizeof(f64) * count_u,
            cudaMemcpyDeviceToHost
        );
    }
    return e == cudaSuccess ? QNPEPS_E2E_OK : QNPEPS_E2E_ERR_CUDA;
}

auto draw_fill_multigpu(NodeState* n, i64 first_batch, i64 nbatch, i64 ring_off, i64 epoch)
    -> qnpeps_e2e_status
{
    if (nbatch <= 0) return QNPEPS_E2E_OK;
    wait_epoch_ready(n, epoch);
    if (n->status() != QNPEPS_E2E_OK) return QNPEPS_E2E_ERR_INTERNAL;

    const usize gpu_count{static_cast<usize>(n->gpus)};
    std::vector<qnpeps::Linalg*> linalg{gpu_count};
    for (int g{}; g < n->gpus; ++g)
    {
        const usize gpu_index{static_cast<usize>(g)};
        linalg[gpu_index] = &n->ctx[gpu_index].linalg();
    }

    const i64 count{nbatch * n->dim_batch};
    qnpeps_status status{QNPEPS_OK};
    {
        std::lock_guard<std::mutex> lock{n->sampler_m};
        status = qnpeps::sampler::sample_multigpu(
            n->scfg,
            qnpeps::sampler::SampleMultigpuArgs{
                .device_peps = n->ctx[0].device_peps_slots[epoch & 1],
                .device_dlenv = n->ctx[0].device_dlenv_slots[epoch & 1],
                .gpus = n->gpus,
                .output = n->ring_cfg + ring_off * n->sites,
                .logpc_out = n->ring_logq + ring_off,
                .lognorm_out = n->ring_lgauge + ring_off,
                .n_samples = static_cast<u64>(count),
                .batch_base = static_cast<u64>(first_batch),
                .dim_batch = static_cast<u64>(n->dim_batch),
                .output_location = qnpeps::sampler::SampleOutputLocation::host
            },
            linalg
        );
    }
    return status == QNPEPS_OK         ? QNPEPS_E2E_OK
           : status == QNPEPS_ERR_OOM  ? QNPEPS_E2E_ERR_OOM
           : status == QNPEPS_ERR_CUDA ? QNPEPS_E2E_ERR_CUDA
                                       : QNPEPS_E2E_ERR_INTERNAL;
}

auto eo_run_shard(NodeState* n, int g, const Shard& sh, i64 epoch) -> qnpeps_e2e_status
{
    if (sh.count <= 0) return QNPEPS_E2E_OK;
    if (g == n->inject_oom_gpu) return QNPEPS_E2E_ERR_OOM;
    const usize gpu_index{static_cast<usize>(g)};
    auto& c{n->ctx[gpu_index]};
    const bool is0{g == 0};
    const u8* samples{};
    if (is0)
        samples = n->device_samples + sh.base * n->sites;
    else
    {
        const i64 copy_base{n->use_distributed_minsr ? 0 : sh.base};
        const i64 copy_count{n->use_distributed_minsr ? n->active_samples : sh.count};
        if (n->use_p2p)
        {
            QN_E2E_CUDA_CHECK(cudaMemcpyPeer(
                c.device_eo_samples,
                g,
                n->device_samples + copy_base * n->sites,
                0,
                static_cast<usize>(copy_count * n->sites)
            ));
        }
        else
            QN_E2E_CUDA_CHECK(cudaMemcpy(
                c.device_eo_samples,
                n->host_samples.data() + copy_base * n->sites,
                static_cast<usize>(copy_count * n->sites),
                cudaMemcpyHostToDevice
            ));
        samples = c.device_eo_samples + (sh.base - copy_base) * n->sites;
    }
    if (err_state() != QNPEPS_E2E_OK) return err_state();

    auto lp{is0 ? n->device_log_amplitudes + 2 * sh.base : c.device_eo_log_amplitudes};
    auto el{is0 ? n->device_local_energies + 2 * sh.base : c.device_eo_local_energies};
    qnpeps::eo::run_shard(
        qnpeps::eo::ShardArgs{
            .config = &n->lcfg,
            .device_peps = reinterpret_cast<const cf*>(c.device_peps_slots[epoch & 1]),
            .samples = samples,
            .n_samples = sh.count,
            .terms = &n->terms,
            .logpsi = lp,
            .e_loc = el,
            .o_rows_device = c.device_eo_rows,
            .o_rows_host = n->host_rows + sh.base * n->compact,
            .linalg = &c.linalg()
        }
    );
    if (err_state() != QNPEPS_E2E_OK) return err_state();

    if (not is0)
    {
        const usize scalar_count{static_cast<usize>(2 * sh.count)};
        const usize scalar_bytes{sizeof(f64) * scalar_count};
        if (n->use_p2p)
        {
            QN_E2E_CUDA_CHECK(cudaMemcpyPeer(
                n->device_log_amplitudes + 2 * sh.base,
                0,
                c.device_eo_log_amplitudes,
                g,
                scalar_bytes
            ));
            QN_E2E_CUDA_CHECK(cudaMemcpyPeer(
                n->device_local_energies + 2 * sh.base,
                0,
                c.device_eo_local_energies,
                g,
                scalar_bytes
            ));
        }
        else
        {
            auto& hl{n->host_log_amplitudes[gpu_index]};
            auto& he{n->host_local_energies[gpu_index]};
            hl.resize(scalar_count);
            he.resize(scalar_count);
            QN_E2E_CUDA_CHECK(cudaMemcpy(
                hl.data(), c.device_eo_log_amplitudes, scalar_bytes, cudaMemcpyDeviceToHost
            ));
            QN_E2E_CUDA_CHECK(cudaMemcpy(
                he.data(), c.device_eo_local_energies, scalar_bytes, cudaMemcpyDeviceToHost
            ));
        }
    }
    QN_E2E_CUDA_CHECK(cudaStreamSynchronize(c.stream()));
    return err_state();
}

NodeWorkers::NodeWorkers(NodeState& state) noexcept : state_{state} {}

NodeWorkers::~NodeWorkers()
{
    stop();
}

auto NodeWorkers::start() -> void
{
    const usize gpu_count{static_cast<usize>(state_.gpus)};
    sync_.resize(gpu_count);
    for (int device{1}; device < state_.gpus; ++device)
        sync_[static_cast<usize>(device)] = std::make_unique<WorkerSync>();
    workers_.resize(gpu_count);
    for (int device{1}; device < state_.gpus; ++device)
        workers_[static_cast<usize>(device)] = std::thread{&NodeWorkers::worker_main, this, device};
    dlbuild_ = std::thread{&NodeWorkers::dlbuild_main, this};
}

auto NodeWorkers::stop() noexcept -> void
{
    const int worker_count{static_cast<int>(sync_.size())};
    const auto has_pending_worker = state_.predraw_pending and state_.gpus >= 2
                                    and state_.sampler_gpu < worker_count
                                    and sync_[static_cast<usize>(state_.sampler_gpu)];
    if (has_pending_worker)
    {
        static_cast<void>(wait(state_.sampler_gpu));
        state_.predraw_pending = false;
    }
    for (int device{1}; device < worker_count; ++device)
    {
        const usize device_index{static_cast<usize>(device)};
        if (sync_[device_index]) issue(device, WorkerCmd{WorkerMode::Exit});
    }
    for (int device{1}; device < static_cast<int>(workers_.size()); ++device)
    {
        const usize device_index{static_cast<usize>(device)};
        if (workers_[device_index].joinable()) workers_[device_index].join();
    }
    if (dlbuild_.joinable())
    {
        {
            std::lock_guard<std::mutex> lock{build_mutex_};
            build_exit_ = true;
        }
        build_ready_.notify_all();
        dlbuild_.join();
    }
    for (int device{1}; device < worker_count; ++device)
        sync_[static_cast<usize>(device)].reset();
}

auto NodeWorkers::worker_main(int g) -> void
{
    auto* n{&state_};
    const usize gpu_index{static_cast<usize>(g)};
    auto& s{*sync_[gpu_index]};
    pin_thread_to_gpu_numa(g);
    cudaSetDevice(g);
    i64 seen{0};
    while (true)
    {
        WorkerCmd cmd{};
        {
            std::unique_lock<std::mutex> lock{s.mutex};
            s.command_ready.wait(lock, [&] { return s.command_generation > seen; });
            cmd = s.command;
            seen = s.command_generation;
        }
        if (cmd.mode == WorkerMode::Exit) break;
        if (g == n->inject_delay_gpu and n->inject_delay_ms > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(n->inject_delay_ms));
        qn_e2e::clear_err();
        qnpeps_e2e_status st{QNPEPS_E2E_OK};
        if (cmd.mode == WorkerMode::Sample)
        {
            st = draw_fill(
                n, n->ctx[gpu_index], cmd.first_batch, cmd.nbatch, cmd.ring_off, cmd.epoch
            );
        }
        else if (cmd.mode == WorkerMode::Eo)
            st = eo_run_shard(n, g, cmd.shard, cmd.epoch);
        else if (cmd.mode == WorkerMode::Gram)
            st = static_cast<qnpeps_e2e_status>(qnpeps::minsr::build_gram_blockrow(*cmd.gram));
        if (st != QNPEPS_E2E_OK) n->fail(st);
        {
            std::lock_guard<std::mutex> lock{s.mutex};
            s.status = st;
            s.done_generation = seen;
        }
        s.command_done.notify_all();
    }
}

auto NodeWorkers::issue(int g, const WorkerCmd& cmd) -> void
{
    auto& s{*sync_[static_cast<usize>(g)]};
    {
        std::lock_guard<std::mutex> lock{s.mutex};
        s.command = cmd;
        ++s.command_generation;
    }
    s.command_ready.notify_all();
}

auto NodeWorkers::wait(int g) -> qnpeps_e2e_status
{
    auto& s{*sync_[static_cast<usize>(g)]};
    std::unique_lock<std::mutex> lock{s.mutex};
    s.command_done.wait(lock, [&] { return s.done_generation == s.command_generation; });
    return s.status;
}

auto NodeWorkers::run_dlbuild(i64 epoch) -> void
{
    {
        std::lock_guard<std::mutex> lock{build_mutex_};
        build_epoch_ = epoch;
        ++build_request_;
    }
    build_ready_.notify_all();
}

auto NodeWorkers::dlbuild_main() -> void
{
    auto* n{&state_};
    pin_thread_to_gpu_numa(0);
    cudaSetDevice(0);
    i64 seen{0};
    while (true)
    {
        i64 e{};
        {
            std::unique_lock<std::mutex> lock{build_mutex_};
            build_ready_.wait(lock, [&] { return build_request_ > seen or build_exit_; });
            if (build_exit_ and build_request_ <= seen) break;
            e = build_epoch_;
            seen = build_request_;
        }
        qn_e2e::clear_err();
        const int slot{static_cast<int>(e & 1)};
        auto& g0{n->ctx[0]};
        {
            std::lock_guard<std::mutex> lk{n->sampler_m};
            const qnpeps_status dlenv_status{qnpeps::dlenv::build_dlenv_packed(
                n->scfg,
                g0.device_peps_slots[slot],
                g0.device_dlenv_slots[slot],
                g0.device_row_logs,
                g0.linalg()
            )};
            map_sampler_status(dlenv_status);
            if (err_state() == QNPEPS_E2E_OK) cudaStreamSynchronize(g0.stream());
        }
        if (err_state() == QNPEPS_E2E_OK)
        {
            for (int w{1}; w < n->gpus; ++w)
            {
                auto& gw{n->ctx[static_cast<usize>(w)]};
                copy_in(
                    n,
                    gw.device_peps_slots[slot],
                    w,
                    g0.device_peps_slots[slot],
                    0,
                    static_cast<usize>(n->peps_bytes)
                );
                copy_in(
                    n,
                    gw.device_dlenv_slots[slot],
                    w,
                    g0.device_dlenv_slots[slot],
                    0,
                    static_cast<usize>(n->dlenv_bytes)
                );
            }
        }
        cudaSetDevice(0);
        if (err_state() != QNPEPS_E2E_OK) n->fail(err_state());
        {
            std::lock_guard<std::mutex> lk{n->epoch_m};
            n->epoch_ready = e;
        }
        n->epoch_cv.notify_all();
    }
}
