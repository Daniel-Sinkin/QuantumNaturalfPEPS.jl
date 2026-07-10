#include "defer.cuh"
#include "dlenv/build.cuh"
#include "qnpeps_ctx.cuh"
#include "sampler/draw.cuh"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <thread>
#include <vector>

namespace qnpeps
{
[[nodiscard]] static auto
dlenv_dims_fit_sampler(const std::vector<int>& dims, int sites, int dim_bond, int bond_cap) -> bool
{
    for (auto s = 0; s < sites; ++s)
    {
        const usize base{static_cast<usize>(s) * k_dl_axis_count};
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
    ctx.sampler.dim_batch = static_cast<int>(dim_batch);
    const auto batches = static_cast<int>(ceil_div(n_samples_i, dim_batch));

    const int sites{(lx - 1) * ly};
    if (not dlenv_dims_fit_sampler(ctx.dlenv.dims, sites, dim_bond, chi_dl))
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

    if (ctx.sampler.refresh_gen != ctx.dlenv.build_count)
    {
        ctx_sample_refresh(ctx);
        ctx.sampler.refresh_gen = ctx.dlenv.build_count;
    }

    std::vector<int> all_batch_ids{};
    all_batch_ids.resize(static_cast<usize>(batches));
    std::iota(all_batch_ids.begin(), all_batch_ids.end(), 0);
    ctx_sample_run(ctx, all_batch_ids);
    CUDA_CHECK(cudaStreamSynchronize(ctx.linalg.stream()));

    const usize total_sample_count{
        static_cast<usize>(n_samples) * static_cast<usize>(lx) * static_cast<usize>(ly)
    };
    if (ctx.sampler.all_samples.size() < total_sample_count) return set_err(QNPEPS_ERR_INTERNAL);
    CUDA_CHECK(cudaMemcpy(
        output,
        ctx.sampler.all_samples.data(),
        total_sample_count * sizeof(u8),
        cudaMemcpyHostToDevice
    ));

    if (logpc_out)
    {
        if (ctx.sampler.all_logpc.size() < n_samples) return set_err(QNPEPS_ERR_INTERNAL);
        CUDA_CHECK(cudaMemcpy(
            logpc_out, ctx.sampler.all_logpc.data(), n_samples * sizeof(f64), cudaMemcpyHostToDevice
        ));
    }
    if (lognorm_out)
    {
        if (ctx.sampler.all_lognorm.size() < n_samples) return set_err(QNPEPS_ERR_INTERNAL);
        CUDA_CHECK(cudaMemcpy(
            lognorm_out,
            ctx.sampler.all_lognorm.data(),
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
    const auto dim_batch_pin = args.dim_batch_pin;
    auto* const stream = args.stream;

    const auto lx = config.lx;
    const auto ly = config.ly;
    const auto dim_phys = config.dim_phys;
    const auto dim_bond = config.dim_bond;
    const auto chi_dl = std::min(config.chi_dl, dim_bond * dim_bond);

    qnpeps_ctx ctx{};
    ctx.cfg = config;

    const auto n_samples_i = static_cast<i64>(n_samples);
    auto dim_batch = dim_batch_pin != 0 ? static_cast<i64>(dim_batch_pin)
                                        : std::min<i64>(n_samples_i, k_max_batch_size);
    if (dim_batch < 1) dim_batch = 1;
    ctx.sampler.dim_batch = static_cast<int>(dim_batch);
    const auto batches = static_cast<int>(ceil_div(n_samples_i, dim_batch));

    const int sites{(lx - 1) * ly};
    ctx.dlenv.dims.resize(static_cast<usize>(sites) * k_dl_axis_count);
    const usize header_bytes{ctx.dlenv.dims.size() * sizeof(int)};
    CUDA_CHECK(
        cudaMemcpy(ctx.dlenv.dims.data(), device_dlenv, header_bytes, cudaMemcpyDeviceToHost)
    );
    if (not dlenv_dims_fit_sampler(ctx.dlenv.dims, sites, dim_bond, chi_dl))
    {
        return set_err(QNPEPS_ERR_BAD_CONFIG);
    }
    DlEnvView dlenv_view{};
    dlenv_view.dims = ctx.dlenv.dims.data();
    dlenv_view.values = byte_offset<cuFloatComplex>(device_dlenv, header_bytes);
    ctx.dlenv.buf[0] = const_cast<void*>(device_dlenv);
    ctx.dlenv.active = 0;
    ctx.dlenv.valid[0] = true;

    i64 total{0};
    for (auto row = 0; row < lx; ++row)
    {
        for (auto col = 0; col < ly; ++col)
        {
            const auto bond_left = bond_dim(ly, col, dim_bond);
            const auto bond_right = bond_dim(ly, col + 1, dim_bond);
            const auto bond_up = bond_dim(lx, row, dim_bond);
            const auto bond_down = bond_dim(lx, row + 1, dim_bond);
            total += static_cast<i64>(bond_left) * bond_down * bond_right * bond_up * dim_phys;
        }
    }
    std::vector<chost> host_peps_buf{};
    host_peps_buf.resize(static_cast<usize>(total));
    CUDA_CHECK(cudaMemcpy(
        host_peps_buf.data(),
        device_peps,
        static_cast<usize>(total) * sizeof(chost),
        cudaMemcpyDeviceToHost
    ));
    const auto num_rows = static_cast<usize>(lx);
    const auto num_cols = static_cast<usize>(ly);
    ctx.host_peps.resize(num_rows);
    for (auto& peps_row : ctx.host_peps)
        peps_row.resize(num_cols);
    i64 off{0};
    for (auto row = 0_uz; row < num_rows; ++row)
    {
        const auto row_i = static_cast<int>(row);
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const auto col_i = static_cast<int>(col);
            const auto bond_left = bond_dim(ly, col_i, dim_bond);
            const auto bond_right = bond_dim(ly, col_i + 1, dim_bond);
            const auto bond_up = bond_dim(lx, row_i, dim_bond);
            const auto bond_down = bond_dim(lx, row_i + 1, dim_bond);
            HostTensor& h = ctx.host_peps[row][col];
            h.dim = {bond_left, bond_down, bond_right, bond_up, dim_phys};
            h.alloc();
            std::copy(
                host_peps_buf.begin() + off, host_peps_buf.begin() + off + h.n(), h.v.begin()
            );
            off += h.n();
        }
    }

    cudaStream_t stream_use{};
    if (stream)
    {
        stream_use = static_cast<cudaStream_t>(stream);
        ctx.stream_owned = false;
    }
    else
    {
        CUDA_CHECK(cudaStreamCreate(&stream_use));
        ctx.stream_owned = true;
    }
    ctx.stream = stream_use;
    ctx.linalg.create(stream_use);
    DEFER(
        [&]
        {
            if (ctx.stream) CUDA_NOCHECK(cudaStreamSynchronize(ctx.stream));
            ctx_sampler_free(ctx);
            ctx.linalg.destroy();
            if (ctx.stream_owned and ctx.stream) CUDA_NOCHECK(cudaStreamDestroy(ctx.stream));
        }
    );

    ctx_sampler_setup(ctx, &dlenv_view, scratch, scratch_bytes);
    if (err_state() != QNPEPS_OK) return err_state();
    ctx.sampler.samp.cfg().batch_base = batch_base;

    ctx_sample_refresh(ctx);

    std::vector<int> all_batch_ids{};
    all_batch_ids.resize(static_cast<usize>(batches));
    std::iota(all_batch_ids.begin(), all_batch_ids.end(), 0);
    ctx_sample_run(ctx, all_batch_ids);
    CUDA_CHECK(cudaStreamSynchronize(ctx.linalg.stream()));

    const usize total_sample_count{
        static_cast<usize>(n_samples) * static_cast<usize>(lx) * static_cast<usize>(ly)
    };
    if (ctx.sampler.all_samples.size() < total_sample_count) return set_err(QNPEPS_ERR_INTERNAL);
    CUDA_CHECK(cudaMemcpy(
        output,
        ctx.sampler.all_samples.data(),
        total_sample_count * sizeof(u8),
        cudaMemcpyHostToDevice
    ));

    if (logpc_out)
    {
        if (ctx.sampler.all_logpc.size() < n_samples) return set_err(QNPEPS_ERR_INTERNAL);
        CUDA_CHECK(cudaMemcpy(
            logpc_out, ctx.sampler.all_logpc.data(), n_samples * sizeof(f64), cudaMemcpyHostToDevice
        ));
    }
    if (lognorm_out)
    {
        if (ctx.sampler.all_lognorm.size() < n_samples) return set_err(QNPEPS_ERR_INTERNAL);
        CUDA_CHECK(cudaMemcpy(
            lognorm_out,
            ctx.sampler.all_lognorm.data(),
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

    if (gpus < 1) return set_err(QNPEPS_ERR_BAD_CONFIG);

    int caller_device{0};
    CUDA_NOCHECK(cudaGetDevice(&caller_device));
    DEFER([&] { CUDA_NOCHECK(cudaSetDevice(caller_device)); });

    const auto lx = config.lx;
    const auto ly = config.ly;
    const auto dim_phys = config.dim_phys;
    const auto dim_bond = config.dim_bond;

    const auto chi_dl = std::min(config.chi_dl, dim_bond * dim_bond);

    const auto n_samples_i = static_cast<i64>(n_samples);
    auto dim_batch = std::min<i64>(n_samples_i, k_max_batch_size);
    if (dim_batch < 1) dim_batch = 1;
    const auto dim_batch_i = static_cast<int>(dim_batch);
    const auto batches = static_cast<int>(ceil_div(n_samples_i, dim_batch));

    const int sites{(lx - 1) * ly};
    std::vector<int> dims{};
    dims.resize(static_cast<usize>(sites) * k_dl_axis_count);
    const usize header_bytes{dims.size() * sizeof(int)};
    CUDA_CHECK(cudaMemcpy(dims.data(), device_dlenv, header_bytes, cudaMemcpyDeviceToHost));
    if (not dlenv_dims_fit_sampler(dims, sites, dim_bond, chi_dl))
    {
        return set_err(QNPEPS_ERR_BAD_CONFIG);
    }

    i64 dlenv_nvals{0};
    for (auto s = 0; s < sites; ++s)
    {
        const usize base{static_cast<usize>(s) * k_dl_axis_count};
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

    i64 total{0};
    for (auto row = 0; row < lx; ++row)
    {
        for (auto col = 0; col < ly; ++col)
        {
            const auto bond_left = bond_dim(ly, col, dim_bond);
            const auto bond_right = bond_dim(ly, col + 1, dim_bond);
            const auto bond_up = bond_dim(lx, row, dim_bond);
            const auto bond_down = bond_dim(lx, row + 1, dim_bond);
            total += static_cast<i64>(bond_left) * bond_down * bond_right * bond_up * dim_phys;
        }
    }
    std::vector<chost> host_peps_buf{};
    host_peps_buf.resize(static_cast<usize>(total));
    CUDA_CHECK(cudaMemcpy(
        host_peps_buf.data(),
        device_peps,
        static_cast<usize>(total) * sizeof(chost),
        cudaMemcpyDeviceToHost
    ));
    const auto num_rows = static_cast<usize>(lx);
    const auto num_cols = static_cast<usize>(ly);
    std::vector<std::vector<HostTensor>> peps{};
    peps.resize(num_rows);
    for (auto& peps_row : peps)
        peps_row.resize(num_cols);
    i64 off{0};
    for (auto row = 0_uz; row < num_rows; ++row)
    {
        const auto row_i = static_cast<int>(row);
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const auto col_i = static_cast<int>(col);
            const auto bond_left = bond_dim(ly, col_i, dim_bond);
            const auto bond_right = bond_dim(ly, col_i + 1, dim_bond);
            const auto bond_up = bond_dim(lx, row_i, dim_bond);
            const auto bond_down = bond_dim(lx, row_i + 1, dim_bond);
            HostTensor& h = peps[row][col];
            h.dim = {bond_left, bond_down, bond_right, bond_up, dim_phys};
            h.alloc();
            std::copy(
                host_peps_buf.begin() + off, host_peps_buf.begin() + off + h.n(), h.v.begin()
            );
            off += h.n();
        }
    }

    struct GpuShard
    {
        std::vector<u8> samples{};
        std::vector<f64> logpc{};
        std::vector<f64> lognorm{};
        qnpeps_status err{QNPEPS_OK};
    };
    using BatchIds = std::vector<int>;

    const auto gpu_count = static_cast<usize>(gpus);

    std::vector<BatchIds> batches_of_gpu{};
    batches_of_gpu.resize(gpu_count);
    for (auto b = 0; b < batches; ++b)
        batches_of_gpu[static_cast<usize>(b % gpus)].push_back(b);

    std::vector<GpuShard> shards{};
    shards.resize(gpu_count);

    const auto dlenv_values_bytes = static_cast<usize>(dlenv_nvals) * sizeof(cuFloatComplex);
    auto run_gpu = [&](usize g)
    {
        err_state() = QNPEPS_OK;
        GpuShard& shard = shards[g];
        if (cudaSetDevice(static_cast<int>(g)) != cudaSuccess)
        {
            shard.err = QNPEPS_ERR_CUDA;
            return;
        }
        void* device_dlenv_copy{};
        const auto dlenv_total_bytes = header_bytes + dlenv_values_bytes;
        if (cudaMalloc(&device_dlenv_copy, dlenv_total_bytes) != cudaSuccess)
        {
            shard.err = QNPEPS_ERR_OOM;
            return;
        }
        const cudaError_t header_rc{
            cudaMemcpy(device_dlenv_copy, dims.data(), header_bytes, cudaMemcpyHostToDevice)
        };
        const cudaError_t values_rc{cudaMemcpy(
            byte_offset<cuFloatComplex>(device_dlenv_copy, header_bytes),
            host_dlenv.data(),
            dlenv_values_bytes,
            cudaMemcpyHostToDevice
        )};
        if (header_rc != cudaSuccess or values_rc != cudaSuccess)
        {
            CUDA_NOCHECK(cudaFree(device_dlenv_copy));
            shard.err = QNPEPS_ERR_CUDA;
            return;
        }

        qnpeps_ctx ctx{};
        ctx.cfg = config;
        ctx.sampler.dim_batch = dim_batch_i;
        ctx.host_peps = peps;
        ctx.dlenv.dims = dims;
        ctx.dlenv.buf[0] = device_dlenv_copy;
        ctx.dlenv.active = 0;
        ctx.dlenv.valid[0] = true;

        cudaStream_t stream_use{};
        if (cudaStreamCreate(&stream_use) != cudaSuccess)
        {
            cudaFree(device_dlenv_copy);
            shard.err = QNPEPS_ERR_CUDA;
            return;
        }
        ctx.stream = stream_use;
        ctx.stream_owned = true;
        ctx.linalg.create(stream_use);

        DlEnvView view{};
        view.dims = ctx.dlenv.dims.data();
        view.values = byte_offset<cuFloatComplex>(ctx.dlenv.buf[0], header_bytes);

        ctx_sampler_setup(ctx, &view, nullptr, 0);
        if (err_state() == QNPEPS_OK)
        {
            ctx_sample_refresh(ctx);
            ctx_sample_run(ctx, batches_of_gpu[g]);
        }
        cudaStreamSynchronize(ctx.linalg.stream());
        shard.samples = std::move(ctx.sampler.all_samples);
        shard.logpc = std::move(ctx.sampler.all_logpc);
        shard.lognorm = std::move(ctx.sampler.all_lognorm);
        shard.err = err_state();

        ctx_sampler_free(ctx);
        dlenv::dl_free(ctx);
        ctx.linalg.destroy();
        if (ctx.stream_owned and ctx.stream) CUDA_NOCHECK(cudaStreamDestroy(ctx.stream));
    };

    std::vector<std::thread> threads{};
    for (auto g = 1_uz; g < gpu_count; ++g)
        threads.emplace_back(run_gpu, g);
    run_gpu(0);
    for (auto& t : threads)
        t.join();

    CUDA_CHECK(cudaSetDevice(0));
    for (const auto& shard : shards)
        if (shard.err != QNPEPS_OK) return set_err(shard.err);

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
    for (auto s = 0_u64; s < n_samples; ++s)
    {
        const auto batch_id = static_cast<int>(s / dim_batch_u64);
        const auto slot_in_batch = static_cast<usize>(s % dim_batch_u64);
        const GpuShard& shard = shards[static_cast<usize>(batch_id % gpus)];

        const auto batch_offset =
            static_cast<usize>(batch_id / gpus) * static_cast<usize>(dim_batch);
        const auto base = batch_offset + slot_in_batch;
        if ((base + 1) * sample_len_u > shard.samples.size())
        {
            return set_err(QNPEPS_ERR_INTERNAL);
        }

        const auto dst_base = s * sample_len_u;
        const auto src_base = base * sample_len_u;
        for (auto e = 0_i64; e < sample_len; ++e)
            out_samples[dst_base + static_cast<usize>(e)] =
                shard.samples[src_base + static_cast<usize>(e)];

        out_logpc[s] = shard.logpc[base];
        out_lognorm[s] = shard.lognorm[base];
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
