#include "core/defer.cuh"
#include "core/qnpeps_ctx.cuh"
#include "dlenv/build.cuh"
#include "linalg/transfer.cuh"
#include "sampler/draw.cuh"

#include <algorithm>
#include <cstdint>
#include <format>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace qnpeps
{
[[nodiscard]] static auto dlenv_dims_fit_sampler(
    const std::vector<int>& dims, int num_sites, int dim_bond, int bond_cap
) -> bool
{
    for (auto site = 0; site < num_sites; ++site)
    {
        const auto base = static_cast<usize>(site) * k_dl_axis_count;
        const auto bond_left = dims[base + k_dl_bond_left];
        const auto ket = dims[base + k_dl_ket];
        const auto bra = dims[base + k_dl_bra];
        const auto bond_right = dims[base + k_dl_bond_right];
        if (ket != dim_bond or bra != dim_bond) return false;
        if (bond_left < 1 or bond_left > bond_cap) return false;
        if (bond_right < 1 or bond_right > bond_cap) return false;
    }
    return true;
}
namespace sampler
{
auto ctx_sample(qnpeps_ctx& ctx, const CtxSampleArgs& args) -> void
{
    auto* const output = args.output;
    auto* const logpc_out = args.logpc_out;
    auto* const lognorm_out = args.lognorm_out;
    const auto n_samples = args.n_samples;
    const auto batch_base = args.batch_base;

    const auto& active_lane = ctx.dlenv.lanes[ctx.dlenv.active_lane];
    require(
        active_lane.valid,
        Status::bad_config,
        "sampling before a double-layer environment was built"
    );
    if (n_samples == 0) return;

    const auto& config = ctx.cfg;
    const auto lx = config.lx;
    const auto ly = config.ly;
    const auto dim_bond = config.dim_bond;
    const auto chi_dl = std::min(config.chi_dl, dim_bond * dim_bond);

    const auto n_samples_i = static_cast<i64>(n_samples);
    const auto dim_batch = static_cast<i64>(args.dim_batch);
    if (ctx.sampler.allocation.allocated and dim_batch > ctx.sampler.allocation.dim_batch_capacity)
    {
        check(cudaStreamSynchronize(ctx.linalg().stream()));
        ctx_sampler_free(ctx);
        ctx.sampler = {};
    }
    ctx.sampler.execution.dim_batch = static_cast<int>(dim_batch);
    if (ctx.sampler.allocation.dim_batch_capacity == 0)
        ctx.sampler.allocation.dim_batch_capacity = static_cast<int>(dim_batch);
    const auto batches = static_cast<int>(ceil_div(n_samples_i, dim_batch));

    const auto num_sites = (lx - 1) * ly;
    require(
        dlenv_dims_fit_sampler(ctx.dlenv.dims, num_sites, dim_bond, chi_dl),
        Status::bad_config,
        "double-layer environment header does not fit bond dimension {} and cap {}",
        dim_bond,
        chi_dl
    );
    const usize header_bytes{ctx.dlenv.dims.size() * sizeof(int)};
    DlEnvView dlenv_view{};
    dlenv_view.dims = ctx.dlenv.dims.data();
    dlenv_view.values = byte_offset<cuFloatComplex>(active_lane.packed, header_bytes);

    ctx_sampler_setup(ctx, &dlenv_view, nullptr, 0);
    ctx.sampler.samp.cfg().batch_base = batch_base;

    if (ctx.sampler.execution.refresh_gen != ctx.dlenv.build_count)
    {
        ctx_sample_refresh(ctx, ctx.dl.peps_buf, PepsLayout::reverse_packed);
        ctx.sampler.execution.refresh_gen = ctx.dlenv.build_count;
    }

    std::vector<int> all_batch_ids{};
    all_batch_ids.resize(static_cast<usize>(batches));
    std::iota(all_batch_ids.begin(), all_batch_ids.end(), 0);
    HostSampleOutput host_output{
        .samples = output,
        .logpc = logpc_out,
        .lognorm = lognorm_out,
        .n_samples = n_samples,
    };
    ctx_sample_run(
        ctx,
        all_batch_ids,
        args.output_location == SampleOutputLocation::host ? &host_output : nullptr
    );
    check(cudaStreamSynchronize(ctx.linalg().stream()));
    if (args.output_location == SampleOutputLocation::host) return;

    const usize total_sample_count{
        static_cast<usize>(n_samples) * static_cast<usize>(lx) * static_cast<usize>(ly)
    };
    require(
        ctx.sampler.staging.all_samples.size() >= total_sample_count,
        "sampler staged {} sample sites, expected {}",
        ctx.sampler.staging.all_samples.size(),
        total_sample_count
    );
    upload(output, ctx.sampler.staging.all_samples.data(), total_sample_count);

    if (logpc_out)
    {
        require(
            ctx.sampler.staging.all_logpc.size() >= n_samples,
            "sampler staged {} proposal logs, expected {}",
            ctx.sampler.staging.all_logpc.size(),
            n_samples
        );
        upload(logpc_out, ctx.sampler.staging.all_logpc.data(), n_samples);
    }
    if (lognorm_out)
    {
        require(
            ctx.sampler.staging.all_lognorm.size() >= n_samples,
            "sampler staged {} gauge logs, expected {}",
            ctx.sampler.staging.all_lognorm.size(),
            n_samples
        );
        upload(lognorm_out, ctx.sampler.staging.all_lognorm.data(), n_samples);
    }
}

auto sample(const QnpepsConfig& config, const SampleArgs& args, Linalg& linalg) -> void
{
    const auto* device_peps = args.device_peps;
    const auto* device_dlenv = args.device_dlenv;
    auto* const scratch = args.scratch;
    const auto scratch_bytes = args.scratch_bytes;
    auto* const output = args.output;
    auto* const logpc_out = args.logpc_out;
    auto* const lognorm_out = args.lognorm_out;
    const auto n_samples = args.n_samples;
    const auto batch_base = args.batch_base;
    const auto dim_batch = static_cast<i64>(args.dim_batch);
    const auto lx = config.lx;
    const auto ly = config.ly;
    const auto dim_bond = config.dim_bond;
    const auto chi_dl = std::min(config.chi_dl, dim_bond * dim_bond);

    const auto foreign_stream =
        args.stream and static_cast<cudaStream_t>(args.stream) != linalg.stream();
    require(
        not foreign_stream, Status::bad_config, "sampling stream differs from the execution stream"
    );
    qnpeps_ctx ctx{config, linalg};

    const auto n_samples_i = static_cast<i64>(n_samples);
    ctx.sampler.execution.dim_batch = static_cast<int>(dim_batch);
    ctx.sampler.allocation.dim_batch_capacity = static_cast<int>(dim_batch);
    const auto batches = static_cast<int>(ceil_div(n_samples_i, dim_batch));

    const auto num_sites = (lx - 1) * ly;
    ctx.dlenv.dims.resize(static_cast<usize>(num_sites) * k_dl_axis_count);
    const auto header_elements = ctx.dlenv.dims.size();
    const usize header_bytes{header_elements * sizeof(int)};
    download(ctx.dlenv.dims.data(), static_cast<const int*>(device_dlenv), header_elements);
    require(
        dlenv_dims_fit_sampler(ctx.dlenv.dims, num_sites, dim_bond, chi_dl),
        Status::bad_config,
        "double-layer environment header does not fit bond dimension {} and cap {}",
        dim_bond,
        chi_dl
    );
    DlEnvView dlenv_view{};
    dlenv_view.dims = ctx.dlenv.dims.data();
    dlenv_view.values = byte_offset<cuFloatComplex>(device_dlenv, header_bytes);
    auto& dlenv_lane = ctx.dlenv.lanes.front();
    dlenv_lane.packed = const_cast<void*>(device_dlenv);
    dlenv_lane.valid = true;
    ctx.dlenv.active_lane = 0;

    DEFER(
        [&]
        {
            unchecked(cudaStreamSynchronize(ctx.stream()));
            ctx_sampler_free(ctx);
        }
    );

    ctx_sampler_setup(ctx, &dlenv_view, scratch, scratch_bytes);
    ctx.sampler.samp.cfg().batch_base = batch_base;

    ctx_sample_refresh(ctx, device_peps, PepsLayout::canonical);

    std::vector<int> all_batch_ids{};
    all_batch_ids.resize(static_cast<usize>(batches));
    std::iota(all_batch_ids.begin(), all_batch_ids.end(), 0);
    HostSampleOutput host_output{
        .samples = output,
        .logpc = logpc_out,
        .lognorm = lognorm_out,
        .n_samples = n_samples,
    };
    ctx_sample_run(
        ctx,
        all_batch_ids,
        args.output_location == SampleOutputLocation::host ? &host_output : nullptr
    );
    check(cudaStreamSynchronize(ctx.linalg().stream()));
    if (args.output_location == SampleOutputLocation::host) return;

    const usize total_sample_count{
        static_cast<usize>(n_samples) * static_cast<usize>(lx) * static_cast<usize>(ly)
    };
    require(
        ctx.sampler.staging.all_samples.size() >= total_sample_count,
        "sampler staged {} sample sites, expected {}",
        ctx.sampler.staging.all_samples.size(),
        total_sample_count
    );
    upload(output, ctx.sampler.staging.all_samples.data(), total_sample_count);

    if (logpc_out)
    {
        require(
            ctx.sampler.staging.all_logpc.size() >= n_samples,
            "sampler staged {} proposal logs, expected {}",
            ctx.sampler.staging.all_logpc.size(),
            n_samples
        );
        upload(logpc_out, ctx.sampler.staging.all_logpc.data(), n_samples);
    }
    if (lognorm_out)
    {
        require(
            ctx.sampler.staging.all_lognorm.size() >= n_samples,
            "sampler staged {} gauge logs, expected {}",
            ctx.sampler.staging.all_lognorm.size(),
            n_samples
        );
        upload(lognorm_out, ctx.sampler.staging.all_lognorm.data(), n_samples);
    }
}

auto sample(const QnpepsConfig& config, const SampleArgs& args) -> void
{
    auto session = make_session(static_cast<cudaStream_t>(args.stream));
    sample(config, args, session->linalg());
}

auto sample_multigpu(
    const QnpepsConfig& config, const SampleMultigpuArgs& args, std::span<Linalg* const> linalg
) -> void
{
    const auto* device_peps = args.device_peps;
    const auto* device_dlenv = args.device_dlenv;
    const auto gpus = args.gpus;
    auto* const output = args.output;
    auto* const logpc_out = args.logpc_out;
    auto* const lognorm_out = args.lognorm_out;
    const auto n_samples = args.n_samples;
    const auto batch_base = args.batch_base;
    const auto dim_batch = static_cast<i64>(args.dim_batch);

    require(
        gpus >= 1 and linalg.size() == static_cast<usize>(gpus),
        Status::bad_config,
        "multi-GPU sampling over {} GPUs with {} execution contexts",
        gpus,
        linalg.size()
    );

    int caller_device{0};
    check(cudaGetDevice(&caller_device));
    DEFER([&] { unchecked(cudaSetDevice(caller_device)); });

    const auto lx = config.lx;
    const auto ly = config.ly;
    const auto dim_phys = config.dim_phys;
    const auto dim_bond = config.dim_bond;

    const auto chi_dl = std::min(config.chi_dl, dim_bond * dim_bond);

    const auto n_samples_i = static_cast<i64>(n_samples);
    const auto dim_batch_i = static_cast<int>(dim_batch);
    const auto batches = static_cast<int>(ceil_div(n_samples_i, dim_batch));

    const auto num_sites = (lx - 1) * ly;
    std::vector<int> dims{};
    dims.resize(static_cast<usize>(num_sites) * k_dl_axis_count);
    const auto header_elements = dims.size();
    const usize header_bytes{header_elements * sizeof(int)};
    download(dims.data(), static_cast<const int*>(device_dlenv), header_elements);
    require(
        dlenv_dims_fit_sampler(dims, num_sites, dim_bond, chi_dl),
        Status::bad_config,
        "double-layer environment header does not fit bond dimension {} and cap {}",
        dim_bond,
        chi_dl
    );

    i64 dlenv_nvals{0};
    for (auto site = 0; site < num_sites; ++site)
    {
        const auto base = static_cast<usize>(site) * k_dl_axis_count;
        const auto bond_left = static_cast<i64>(dims[base + k_dl_bond_left]);
        const auto ket = static_cast<i64>(dims[base + k_dl_ket]);
        const auto bra = static_cast<i64>(dims[base + k_dl_bra]);
        const auto bond_right = static_cast<i64>(dims[base + k_dl_bond_right]);
        dlenv_nvals += bond_left * ket * bra * bond_right;
    }
    std::vector<cuFloatComplex> host_dlenv{};
    host_dlenv.resize(static_cast<usize>(dlenv_nvals));
    download(
        host_dlenv.data(),
        byte_offset<cuFloatComplex>(device_dlenv, header_bytes),
        static_cast<usize>(dlenv_nvals)
    );

    i64 peps_elems{};
    for (auto row = 0; row < lx; ++row)
    {
        for (auto col = 0; col < ly; ++col)
        {
            const auto bond_left = bond_dim(ly, col, dim_bond);
            const auto bond_right = bond_dim(ly, col + 1, dim_bond);
            const auto bond_up = bond_dim(lx, row, dim_bond);
            const auto bond_down = bond_dim(lx, row + 1, dim_bond);
            peps_elems += static_cast<i64>(bond_left) * bond_down * bond_right * bond_up * dim_phys;
        }
    }
    std::vector<cuFloatComplex> peps_staging{};
    peps_staging.resize(static_cast<usize>(peps_elems));
    download(
        peps_staging.data(),
        static_cast<const cuFloatComplex*>(device_peps),
        static_cast<usize>(peps_elems)
    );

    struct GpuShard
    {
        std::vector<u8> samples{};
        std::vector<f64> logpc{};
        std::vector<f64> lognorm{};
        qnpeps_status status{QNPEPS_OK};
        std::string message{};
    };
    using BatchIds = std::vector<int>;

    const auto gpu_count = static_cast<usize>(gpus);

    std::vector<BatchIds> batches_of_gpu{};
    batches_of_gpu.resize(gpu_count);
    for (auto b = 0; b < batches; ++b)
        batches_of_gpu[static_cast<usize>(b % gpus)].push_back(b);

    std::vector<GpuShard> shards{};
    shards.resize(gpu_count);
    HostSampleOutput host_output{
        .samples = output,
        .logpc = logpc_out,
        .lognorm = lognorm_out,
        .n_samples = n_samples,
    };

    const auto dlenv_values_bytes = sizeof(cuFloatComplex) * static_cast<usize>(dlenv_nvals);
    const auto peps_bytes = sizeof(cuFloatComplex) * static_cast<usize>(peps_elems);
    const auto sample_on_gpu = [&](usize g)
    {
        check(cudaSetDevice(static_cast<int>(g)));
        void* device_dlenv_copy{};
        check_allocation(cudaMalloc(&device_dlenv_copy, header_bytes + dlenv_values_bytes));
        auto dlenv_copy_adopted = false;
        const Defer free_dlenv_copy{[&]
                                    {
                                        if (not dlenv_copy_adopted)
                                            unchecked(cudaFree(device_dlenv_copy));
                                    }};
        cuFloatComplex* device_peps_copy{};
        check_allocation(cudaMalloc(&device_peps_copy, peps_bytes));
        const Defer free_peps_copy{[&] { unchecked(cudaFree(device_peps_copy)); }};
        check(cudaMemcpy(device_dlenv_copy, dims.data(), header_bytes, cudaMemcpyHostToDevice));
        check(cudaMemcpy(
            byte_offset<cuFloatComplex>(device_dlenv_copy, header_bytes),
            host_dlenv.data(),
            dlenv_values_bytes,
            cudaMemcpyHostToDevice
        ));
        check(
            cudaMemcpy(device_peps_copy, peps_staging.data(), peps_bytes, cudaMemcpyHostToDevice)
        );

        auto* execution = linalg[g];
        require(
            execution != nullptr and execution->device() == static_cast<int>(g),
            Status::bad_config,
            "GPU {} has no execution context on its own device",
            g
        );
        qnpeps_ctx ctx{config, *execution};
        ctx.sampler.execution.dim_batch = dim_batch_i;
        ctx.sampler.allocation.dim_batch_capacity = dim_batch_i;
        ctx.dlenv.dims = dims;
        auto& dlenv_lane = ctx.dlenv.lanes.front();
        dlenv_lane.packed = device_dlenv_copy;
        dlenv_copy_adopted = true;
        dlenv_lane.valid = true;
        ctx.dlenv.active_lane = 0;
        const Defer release_context{[&]
                                    {
                                        unchecked(cudaStreamSynchronize(ctx.linalg().stream()));
                                        ctx_sampler_free(ctx);
                                        dlenv::dl_free(ctx);
                                    }};

        DlEnvView view{};
        view.dims = ctx.dlenv.dims.data();
        view.values = byte_offset<cuFloatComplex>(dlenv_lane.packed, header_bytes);

        ctx_sampler_setup(ctx, &view, nullptr, 0);
        ctx.sampler.samp.cfg().batch_base = batch_base;
        ctx_sample_refresh(ctx, device_peps_copy, PepsLayout::canonical);
        ctx_sample_run(
            ctx,
            batches_of_gpu[g],
            args.output_location == SampleOutputLocation::host ? &host_output : nullptr
        );
        check(cudaStreamSynchronize(ctx.linalg().stream()));
        if (args.output_location == SampleOutputLocation::device)
        {
            shards[g].samples = std::move(ctx.sampler.staging.all_samples);
            shards[g].logpc = std::move(ctx.sampler.staging.all_logpc);
            shards[g].lognorm = std::move(ctx.sampler.staging.all_lognorm);
        }
    };
    const auto run_gpu = [&](usize g)
    {
        auto& shard = shards[g];
        shard.status = guarded([&] { sample_on_gpu(g); });
        if (shard.status != QNPEPS_OK) shard.message = std::move(last_error_message());
    };

    {
        std::vector<std::thread> threads{};
        const Defer join_threads{[&]
                                 {
                                     for (auto& thread : threads)
                                     {
                                         if (thread.joinable()) thread.join();
                                     }
                                 }};
        for (auto g = 1_uz; g < gpu_count; ++g)
            threads.emplace_back(run_gpu, g);
        run_gpu(0);
    }

    check(cudaSetDevice(caller_device));
    for (auto g = 0_uz; g < gpu_count; ++g)
    {
        const auto& shard = shards[g];
        if (shard.status != QNPEPS_OK)
            relay(static_cast<Status>(shard.status), std::format("[GPU {}] {}", g, shard.message));
    }
    if (args.output_location == SampleOutputLocation::host) return;

    const i64 sample_len{static_cast<i64>(lx) * ly};
    const auto sample_len_u = static_cast<usize>(sample_len);
    const auto n_samples_u = static_cast<usize>(n_samples);
    const auto dim_batch_u64 = static_cast<uint64_t>(dim_batch);
    std::vector<u8> out_samples{};
    out_samples.resize(n_samples_u * sample_len_u);
    std::vector<f64> out_logpc{};
    out_logpc.resize(n_samples_u);
    std::vector<f64> out_lognorm{};
    out_lognorm.resize(n_samples_u);
    for (auto sample = 0_u64; sample < n_samples; ++sample)
    {
        const auto batch_id = static_cast<int>(sample / dim_batch_u64);
        const auto slot_in_batch = static_cast<usize>(sample % dim_batch_u64);
        const auto& shard = shards[static_cast<usize>(batch_id % gpus)];

        const auto batches_before = static_cast<usize>(batch_id / gpus);
        const auto batch_offset = batches_before * static_cast<usize>(dim_batch);
        const auto shard_sample = batch_offset + slot_in_batch;
        require(
            (shard_sample + 1) * sample_len_u <= shard.samples.size(),
            "sample {} reads past the {} sites staged by its GPU",
            sample,
            shard.samples.size()
        );

        const auto destination_offset = sample * sample_len_u;
        const auto source_offset = shard_sample * sample_len_u;
        for (auto element = 0_i64; element < sample_len; ++element)
        {
            out_samples[destination_offset + static_cast<usize>(element)] =
                shard.samples[source_offset + static_cast<usize>(element)];
        }

        out_logpc[sample] = shard.logpc[shard_sample];
        out_lognorm[sample] = shard.lognorm[shard_sample];
    }

    upload(output, out_samples.data(), n_samples_u * sample_len_u);
    if (logpc_out)
    {
        upload(logpc_out, out_logpc.data(), n_samples_u);
    }
    if (lognorm_out)
    {
        upload(lognorm_out, out_lognorm.data(), n_samples_u);
    }
}

auto sample_multigpu(const QnpepsConfig& config, const SampleMultigpuArgs& args) -> void
{
    require(args.gpus >= 1, Status::bad_config, "multi-GPU sampling over {} GPUs", args.gpus);
    int caller_device{};
    check(cudaGetDevice(&caller_device));
    std::vector<std::unique_ptr<Session>> sessions{static_cast<usize>(args.gpus)};
    std::vector<Linalg*> linalg{static_cast<usize>(args.gpus)};
    {
        const Defer restore_device{[&] { unchecked(cudaSetDevice(caller_device)); }};
        for (int device{}; device < args.gpus; ++device)
        {
            check(cudaSetDevice(device));
            sessions[static_cast<usize>(device)] = make_session(nullptr);
            linalg[static_cast<usize>(device)] = &sessions[static_cast<usize>(device)]->linalg();
        }
    }
    sample_multigpu(config, args, linalg);
}
}
}
