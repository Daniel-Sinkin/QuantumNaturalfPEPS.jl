#include "defer.cuh"
#include "dlenv/build.cuh"
#include "qnpeps_ctx.cuh"
#include "sampler/draw.cuh"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace qnpeps
{
[[nodiscard]] static auto
dlenv_dims_fit_sampler(const std::vector<int>& dims, int num_sites, int dim_bond, int bond_cap)
    -> bool
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
auto ctx_sample(qnpeps_ctx& ctx, const CtxSampleArgs& args) -> qnpeps_status
{
    auto* const output = args.output;
    auto* const logpc_out = args.logpc_out;
    auto* const lognorm_out = args.lognorm_out;
    const auto n_samples = args.n_samples;
    const auto batch_base = args.batch_base;

    if (not ctx.dlenv.valid[ctx.dlenv.active]) return set_err(QNPEPS_ERR_BAD_CONFIG);
    if (n_samples == 0) return QNPEPS_OK;

    const auto& config = ctx.cfg;
    const auto lx = config.lx;
    const auto ly = config.ly;
    const auto dim_bond = config.dim_bond;
    const auto chi_dl = std::min(config.chi_dl, dim_bond * dim_bond);

    const auto n_samples_i = static_cast<i64>(n_samples);
    auto dim_batch = std::min<i64>(n_samples_i, k_max_batch_size);
    if (dim_batch < 1) dim_batch = 1;
    if (ctx.sampler.allocation.allocated and dim_batch > ctx.sampler.allocation.dim_batch_capacity)
    {
        CUDA_CHECK(cudaStreamSynchronize(ctx.linalg().stream()));
        ctx_sampler_free(ctx);
        ctx.sampler = {};
    }
    ctx.sampler.execution.dim_batch = static_cast<int>(dim_batch);
    if (ctx.sampler.allocation.dim_batch_capacity == 0)
        ctx.sampler.allocation.dim_batch_capacity = static_cast<int>(dim_batch);
    const auto batches = static_cast<int>(ceil_div(n_samples_i, dim_batch));

    const auto num_sites = (lx - 1) * ly;
    if (not dlenv_dims_fit_sampler(ctx.dlenv.dims, num_sites, dim_bond, chi_dl))
    {
        return set_err(QNPEPS_ERR_BAD_CONFIG);
    }
    const usize header_bytes{ctx.dlenv.dims.size() * sizeof(int)};
    DlEnvView dlenv_view{};
    dlenv_view.dims = ctx.dlenv.dims.data();
    dlenv_view.values = byte_offset<cuFloatComplex>(ctx.dlenv.buf[ctx.dlenv.active], header_bytes);

    ctx_sampler_setup(ctx, &dlenv_view, nullptr, 0);
    if (err_state() != QNPEPS_OK) return err_state();
    ctx.sampler.samp.cfg().batch_base = batch_base;

    if (ctx.sampler.execution.refresh_gen != ctx.dlenv.build_count)
    {
        ctx_sample_refresh(ctx, ctx.dl.peps_buf, PepsLayout::reverse_packed);
        ctx.sampler.execution.refresh_gen = ctx.dlenv.build_count;
    }

    std::vector<int> all_batch_ids{};
    all_batch_ids.resize(static_cast<usize>(batches));
    std::iota(all_batch_ids.begin(), all_batch_ids.end(), 0);
    ctx_sample_run(ctx, all_batch_ids);
    CUDA_CHECK(cudaStreamSynchronize(ctx.linalg().stream()));

    const usize total_sample_count{
        static_cast<usize>(n_samples) * static_cast<usize>(lx) * static_cast<usize>(ly)
    };
    if (ctx.sampler.staging.all_samples.size() < total_sample_count)
        return set_err(QNPEPS_ERR_INTERNAL);
    CUDA_CHECK(cudaMemcpy(
        output,
        ctx.sampler.staging.all_samples.data(),
        total_sample_count * sizeof(u8),
        cudaMemcpyHostToDevice
    ));

    if (logpc_out)
    {
        if (ctx.sampler.staging.all_logpc.size() < n_samples) return set_err(QNPEPS_ERR_INTERNAL);
        CUDA_CHECK(cudaMemcpy(
            logpc_out,
            ctx.sampler.staging.all_logpc.data(),
            n_samples * sizeof(f64),
            cudaMemcpyHostToDevice
        ));
    }
    if (lognorm_out)
    {
        if (ctx.sampler.staging.all_lognorm.size() < n_samples) return set_err(QNPEPS_ERR_INTERNAL);
        CUDA_CHECK(cudaMemcpy(
            lognorm_out,
            ctx.sampler.staging.all_lognorm.data(),
            n_samples * sizeof(f64),
            cudaMemcpyHostToDevice
        ));
    }
    return err_state();
}

auto sample(const QnpepsConfig& config, const SampleArgs& args) -> qnpeps_status
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
    auto* const stream = args.stream;

    const auto lx = config.lx;
    const auto ly = config.ly;
    const auto dim_bond = config.dim_bond;
    const auto chi_dl = std::min(config.chi_dl, dim_bond * dim_bond);

    cudaStream_t stream_use{};
    bool stream_owned{};
    if (stream)
    {
        stream_use = static_cast<cudaStream_t>(stream);
    }
    else
    {
        const auto stream_status = cudaStreamCreate(&stream_use);
        if (stream_status != cudaSuccess) return set_cuda_err(stream_status);
        stream_owned = true;
    }
    DEFER(
        [&]
        {
            if (stream_owned) CUDA_NOCHECK(cudaStreamDestroy(stream_use));
        }
    );

    auto linalg = make_linalg(stream_use);
    if (not linalg) return err_state();
    qnpeps_ctx ctx{config, stream_use, stream_owned, std::move(linalg)};

    const auto n_samples_i = static_cast<i64>(n_samples);
    ctx.sampler.execution.dim_batch = static_cast<int>(dim_batch);
    ctx.sampler.allocation.dim_batch_capacity = static_cast<int>(dim_batch);
    const auto batches = static_cast<int>(ceil_div(n_samples_i, dim_batch));

    const auto num_sites = (lx - 1) * ly;
    ctx.dlenv.dims.resize(static_cast<usize>(num_sites) * k_dl_axis_count);
    const usize header_bytes{ctx.dlenv.dims.size() * sizeof(int)};
    CUDA_CHECK(
        cudaMemcpy(ctx.dlenv.dims.data(), device_dlenv, header_bytes, cudaMemcpyDeviceToHost)
    );
    if (not dlenv_dims_fit_sampler(ctx.dlenv.dims, num_sites, dim_bond, chi_dl))
    {
        return set_err(QNPEPS_ERR_BAD_CONFIG);
    }
    DlEnvView dlenv_view{};
    dlenv_view.dims = ctx.dlenv.dims.data();
    dlenv_view.values = byte_offset<cuFloatComplex>(device_dlenv, header_bytes);
    ctx.dlenv.buf[0] = const_cast<void*>(device_dlenv);
    ctx.dlenv.active = 0;
    ctx.dlenv.valid[0] = true;

    DEFER(
        [&]
        {
            if (ctx.stream) CUDA_NOCHECK(cudaStreamSynchronize(ctx.stream));
            ctx_sampler_free(ctx);
        }
    );

    ctx_sampler_setup(ctx, &dlenv_view, scratch, scratch_bytes);
    if (err_state() != QNPEPS_OK) return err_state();
    ctx.sampler.samp.cfg().batch_base = batch_base;

    ctx_sample_refresh(ctx, device_peps, PepsLayout::canonical);

    std::vector<int> all_batch_ids{};
    all_batch_ids.resize(static_cast<usize>(batches));
    std::iota(all_batch_ids.begin(), all_batch_ids.end(), 0);
    ctx_sample_run(ctx, all_batch_ids);
    CUDA_CHECK(cudaStreamSynchronize(ctx.linalg().stream()));

    const usize total_sample_count{
        static_cast<usize>(n_samples) * static_cast<usize>(lx) * static_cast<usize>(ly)
    };
    if (ctx.sampler.staging.all_samples.size() < total_sample_count)
        return set_err(QNPEPS_ERR_INTERNAL);
    CUDA_CHECK(cudaMemcpy(
        output,
        ctx.sampler.staging.all_samples.data(),
        total_sample_count * sizeof(u8),
        cudaMemcpyHostToDevice
    ));

    if (logpc_out)
    {
        if (ctx.sampler.staging.all_logpc.size() < n_samples) return set_err(QNPEPS_ERR_INTERNAL);
        CUDA_CHECK(cudaMemcpy(
            logpc_out,
            ctx.sampler.staging.all_logpc.data(),
            n_samples * sizeof(f64),
            cudaMemcpyHostToDevice
        ));
    }
    if (lognorm_out)
    {
        if (ctx.sampler.staging.all_lognorm.size() < n_samples) return set_err(QNPEPS_ERR_INTERNAL);
        CUDA_CHECK(cudaMemcpy(
            lognorm_out,
            ctx.sampler.staging.all_lognorm.data(),
            n_samples * sizeof(f64),
            cudaMemcpyHostToDevice
        ));
    }
    return err_state();
}

auto sample_multigpu(const QnpepsConfig& config, const SampleMultigpuArgs& args) -> qnpeps_status
{
    const auto* device_peps = args.device_peps;
    const auto* device_dlenv = args.device_dlenv;
    const auto gpus = args.gpus;
    auto* const output = args.output;
    auto* const logpc_out = args.logpc_out;
    auto* const lognorm_out = args.lognorm_out;
    const auto n_samples = args.n_samples;
    const auto dim_batch = static_cast<i64>(args.dim_batch);

    if (gpus < 1) return set_err(QNPEPS_ERR_BAD_CONFIG);

    int caller_device{0};
    CUDA_CHECK(cudaGetDevice(&caller_device));
    if (err_state() != QNPEPS_OK) return err_state();
    DEFER([&] { CUDA_NOCHECK(cudaSetDevice(caller_device)); });

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
    const usize header_bytes{dims.size() * sizeof(int)};
    CUDA_CHECK(cudaMemcpy(dims.data(), device_dlenv, header_bytes, cudaMemcpyDeviceToHost));
    if (not dlenv_dims_fit_sampler(dims, num_sites, dim_bond, chi_dl))
    {
        return set_err(QNPEPS_ERR_BAD_CONFIG);
    }

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
    CUDA_CHECK(cudaMemcpy(
        host_dlenv.data(),
        byte_offset<cuFloatComplex>(device_dlenv, header_bytes),
        static_cast<usize>(dlenv_nvals) * sizeof(cuFloatComplex),
        cudaMemcpyDeviceToHost
    ));

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
    CUDA_CHECK(cudaMemcpy(
        peps_staging.data(),
        device_peps,
        sizeof(cuFloatComplex) * static_cast<usize>(peps_elems),
        cudaMemcpyDeviceToHost
    ));

    struct GpuShard
    {
        std::vector<u8> samples{};
        std::vector<f64> logpc{};
        std::vector<f64> lognorm{};
        qnpeps_status err{QNPEPS_OK};
        const char* err_file{};
        int32_t err_line{};
        std::string err_message{};
    };
    using BatchIds = std::vector<int>;

    const auto gpu_count = static_cast<usize>(gpus);

    std::vector<BatchIds> batches_of_gpu{};
    batches_of_gpu.resize(gpu_count);
    for (auto b = 0; b < batches; ++b)
        batches_of_gpu[static_cast<usize>(b % gpus)].push_back(b);

    std::vector<GpuShard> shards{};
    shards.resize(gpu_count);

    const auto dlenv_values_bytes = sizeof(cuFloatComplex) * static_cast<usize>(dlenv_nvals);
    const auto peps_bytes = sizeof(cuFloatComplex) * static_cast<usize>(peps_elems);
    auto run_gpu = [&](usize g)
    {
        reset_err();
        GpuShard& shard = shards[g];
        const auto capture_error = [&]
        {
            shard.err = err_state();
            shard.err_file = err_file();
            shard.err_line = err_line();
            if (const auto* message = err_message()) shard.err_message = message;
        };
        const auto set_device_status = cudaSetDevice(static_cast<int>(g));
        if (set_device_status != cudaSuccess)
        {
            set_cuda_err(set_device_status);
            capture_error();
            return;
        }
        void* device_dlenv_copy{};
        const auto dlenv_total_bytes = header_bytes + dlenv_values_bytes;
        const auto dlenv_alloc_status = cudaMalloc(&device_dlenv_copy, dlenv_total_bytes);
        if (dlenv_alloc_status != cudaSuccess)
        {
            set_cuda_err(dlenv_alloc_status, QNPEPS_ERR_OOM);
            capture_error();
            return;
        }
        cuFloatComplex* device_peps_copy{};
        const auto peps_alloc_status = cudaMalloc(&device_peps_copy, peps_bytes);
        if (peps_alloc_status != cudaSuccess)
        {
            CUDA_NOCHECK(cudaFree(device_dlenv_copy));
            set_cuda_err(peps_alloc_status, QNPEPS_ERR_OOM);
            capture_error();
            return;
        }
        const auto header_status =
            cudaMemcpy(device_dlenv_copy, dims.data(), header_bytes, cudaMemcpyHostToDevice);
        if (header_status != cudaSuccess)
        {
            CUDA_NOCHECK(cudaFree(device_dlenv_copy));
            CUDA_NOCHECK(cudaFree(device_peps_copy));
            set_cuda_err(header_status);
            capture_error();
            return;
        }
        const auto values_status = cudaMemcpy(
            byte_offset<cuFloatComplex>(device_dlenv_copy, header_bytes),
            host_dlenv.data(),
            dlenv_values_bytes,
            cudaMemcpyHostToDevice
        );
        if (values_status != cudaSuccess)
        {
            CUDA_NOCHECK(cudaFree(device_dlenv_copy));
            CUDA_NOCHECK(cudaFree(device_peps_copy));
            set_cuda_err(values_status);
            capture_error();
            return;
        }
        const auto peps_status =
            cudaMemcpy(device_peps_copy, peps_staging.data(), peps_bytes, cudaMemcpyHostToDevice);
        if (peps_status != cudaSuccess)
        {
            CUDA_NOCHECK(cudaFree(device_dlenv_copy));
            CUDA_NOCHECK(cudaFree(device_peps_copy));
            set_cuda_err(peps_status);
            capture_error();
            return;
        }

        cudaStream_t stream_use{};
        const auto stream_status = cudaStreamCreate(&stream_use);
        if (stream_status != cudaSuccess)
        {
            CUDA_NOCHECK(cudaFree(device_dlenv_copy));
            CUDA_NOCHECK(cudaFree(device_peps_copy));
            set_cuda_err(stream_status);
            capture_error();
            return;
        }
        DEFER([&] { CUDA_NOCHECK(cudaStreamDestroy(stream_use)); });

        auto linalg = make_linalg(stream_use);
        if (not linalg)
        {
            CUDA_NOCHECK(cudaFree(device_dlenv_copy));
            CUDA_NOCHECK(cudaFree(device_peps_copy));
            capture_error();
            return;
        }
        qnpeps_ctx ctx{config, stream_use, true, std::move(linalg)};
        ctx.sampler.execution.dim_batch = dim_batch_i;
        ctx.sampler.allocation.dim_batch_capacity = dim_batch_i;
        ctx.dlenv.dims = dims;
        ctx.dlenv.buf[0] = device_dlenv_copy;
        ctx.dlenv.active = 0;
        ctx.dlenv.valid[0] = true;

        DlEnvView view{};
        view.dims = ctx.dlenv.dims.data();
        view.values = byte_offset<cuFloatComplex>(ctx.dlenv.buf[0], header_bytes);

        ctx_sampler_setup(ctx, &view, nullptr, 0);
        if (err_state() == QNPEPS_OK)
        {
            ctx_sample_refresh(ctx, device_peps_copy, PepsLayout::canonical);
            ctx_sample_run(ctx, batches_of_gpu[g]);
        }
        CUDA_CHECK(cudaStreamSynchronize(ctx.linalg().stream()));
        shard.samples = std::move(ctx.sampler.staging.all_samples);
        shard.logpc = std::move(ctx.sampler.staging.all_logpc);
        shard.lognorm = std::move(ctx.sampler.staging.all_lognorm);
        capture_error();

        ctx_sampler_free(ctx);
        dlenv::dl_free(ctx);
        CUDA_NOCHECK(cudaFree(device_peps_copy));
    };

    std::vector<std::thread> threads{};
    for (auto g = 1_uz; g < gpu_count; ++g)
        threads.emplace_back(run_gpu, g);
    run_gpu(0);
    for (auto& t : threads)
        t.join();

    CUDA_CHECK(cudaSetDevice(caller_device));
    for (const auto& shard : shards)
    {
        if (shard.err != QNPEPS_OK)
        {
            const auto* message = shard.err_message.empty() ? nullptr : shard.err_message.c_str();
            return set_err_at(shard.err, shard.err_file, shard.err_line, message);
        }
    }

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
        const GpuShard& shard = shards[static_cast<usize>(batch_id % gpus)];

        const auto batch_offset =
            static_cast<usize>(batch_id / gpus) * static_cast<usize>(dim_batch);
        const auto shard_sample = batch_offset + slot_in_batch;
        if ((shard_sample + 1) * sample_len_u > shard.samples.size())
        {
            return set_err(QNPEPS_ERR_INTERNAL);
        }

        const auto destination_offset = sample * sample_len_u;
        const auto source_offset = shard_sample * sample_len_u;
        for (auto element = 0_i64; element < sample_len; ++element)
            out_samples[destination_offset + static_cast<usize>(element)] =
                shard.samples[source_offset + static_cast<usize>(element)];

        out_logpc[sample] = shard.logpc[shard_sample];
        out_lognorm[sample] = shard.lognorm[shard_sample];
    }

    CUDA_CHECK(cudaMemcpy(
        output, out_samples.data(), n_samples_u * sample_len_u * sizeof(u8), cudaMemcpyHostToDevice
    ));
    if (logpc_out)
    {
        CUDA_CHECK(
            cudaMemcpy(logpc_out, out_logpc.data(), n_samples * sizeof(f64), cudaMemcpyHostToDevice)
        );
    }
    if (lognorm_out)
    {
        CUDA_CHECK(cudaMemcpy(
            lognorm_out, out_lognorm.data(), n_samples * sizeof(f64), cudaMemcpyHostToDevice
        ));
    }
    return err_state();
}
}
}
