#include "contraction.cuh"
#include "cuda_utils.cuh"
#include "defer.cuh"
#include "dlenv/build.cuh"
#include "linalg.cuh"
#include "peps.cuh"
#include "permutation.cuh"
#include "qnpeps_ctx.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <span>
#include <vector>

namespace qnpeps::dlenv
{
struct ZipupContext
{
    QnpepsConfig config{};
    int maxdim{};
    Dims dims{};
    cudaStream_t stream{};
    bool owns_stream{};
    std::unique_ptr<Linalg> linalg{};
    BuildState dl{};
    usize scale_count{};
    bool active{};
};

[[nodiscard]] static auto zipup_context(qnpeps_zipup_ctx& context) noexcept -> ZipupContext&
{
    return *reinterpret_cast<ZipupContext*>(&context);
}

[[nodiscard]] static auto zipup_context(qnpeps_zipup_ctx* context) noexcept -> ZipupContext*
{
    return reinterpret_cast<ZipupContext*>(context);
}

using PepsRow = std::vector<DeviceTensor>;
using DlEnvRow = std::vector<DeviceTensor>;

static auto init_dl_units(
    Linalg& la, cuFloatComplex* unit_environment, cuFloatComplex* initial_factor
) -> void
{
    constexpr cuFloatComplex one{1.0f, 0.0f};
    cu_set_constant<<<1, 1, 0, la.stream()>>>(unit_environment, one);
    CUDA_CHECK(cudaGetLastError());
    cu_set_constant<<<1, 1, 0, la.stream()>>>(initial_factor, one);
    CUDA_CHECK(cudaGetLastError());
}

struct DlSiteDims
{
    int bond_left{};
    int ket{};
    int bra{};
    int bond_right{};

    [[nodiscard]] auto num_elems() const noexcept -> i64
    {
        return static_cast<i64>(bond_left) * ket * bra * bond_right;
    }
};

[[nodiscard]] static auto read_site_dims(const int32_t* header, usize site) -> DlSiteDims
{
    const usize base{site * k_dl_axis_count};
    return DlSiteDims{
        .bond_left = header[base + k_dl_bond_left],
        .ket = header[base + k_dl_ket],
        .bra = header[base + k_dl_bra],
        .bond_right = header[base + k_dl_bond_right],
    };
}

static auto write_site_dims(int32_t* header, usize site, const Shape& dim) -> void
{
    const usize base{site * k_dl_axis_count};
    header[base + k_dl_bond_left] = dim[k_dl_bond_left];
    header[base + k_dl_ket] = dim[k_dl_ket];
    header[base + k_dl_bra] = dim[k_dl_bra];
    header[base + k_dl_bond_right] = dim[k_dl_bond_right];
}

[[nodiscard]] static auto peps_site_shape(const Dims& dims, int row_up, int row_down, int col)
    -> Shape
{
    const int bond_left{bond_dim(dims.ly, col, dims.dim_bond)};
    const int bond_right{bond_dim(dims.ly, col + 1, dims.dim_bond)};
    const int bond_up{bond_dim(dims.lx, row_up, dims.dim_bond)};
    const int bond_down{bond_dim(dims.lx, row_down, dims.dim_bond)};
    return Shape{bond_left, bond_down, bond_right, bond_up, dims.dim_phys};
}

[[nodiscard]] static auto peps_row_elems(const Dims& dims, int row_up, int row_down) -> i64
{
    i64 total{};
    for (auto col = 0; col < dims.ly; ++col)
        total += static_cast<i64>(peps_site_shape(dims, row_up, row_down, col).num_elems());
    return total;
}

static auto pack_peps_row(
    const Dims& dims,
    int row_up,
    int row_down,
    const cuFloatComplex* source_base,
    i64& source_offset,
    cuFloatComplex* packed_base,
    i64& packed_offset,
    std::vector<DeviceTensor>& output_row,
    cudaStream_t stream
) -> void
{
    const auto reversed = Permutation::reverse(k_peps_site_rank);
    for (auto col = 0; col < dims.ly; ++col)
    {
        const auto source_shape = peps_site_shape(dims, row_up, row_down, col);
        const auto site_elements = static_cast<i64>(source_shape.num_elems());
        const DeviceTensor source{
            source_shape, const_cast<cuFloatComplex*>(source_base + source_offset)
        };
        permute_axes(source, reversed, false, packed_base + packed_offset, stream);
        output_row[static_cast<usize>(col)] =
            DeviceTensor{reversed.apply(source_shape), packed_base + packed_offset};
        source_offset += site_elements;
        packed_offset += site_elements;
    }
}

auto set_dl_capturing(qnpeps_ctx& ctx, bool on) -> void
{
    ctx.dl.capturing = on;
}

auto ensure_sampling_buffers(qnpeps_ctx& ctx) -> void
{
    auto buffers_ready = true;
    for (const auto& lane : ctx.dlenv.lanes)
        buffers_ready = buffers_ready and lane.sampling;
    if (buffers_ready) return;

    const auto num_env_rows = static_cast<usize>(ctx.cfg.lx - 1);
    const auto num_cols = static_cast<usize>(ctx.cfg.ly);
    ctx.dlenv.env_off.assign(num_env_rows, std::vector<i64>(num_cols, 0));
    ctx.dlenv.sigma_off.assign(num_env_rows, std::vector<i64>(num_cols, 0));
    i64 cursor{};
    for (auto row = 0_uz; row < num_env_rows; ++row)
    {
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const auto site = read_site_dims(ctx.dlenv.dims.data(), row * num_cols + col);
            ctx.dlenv.env_off[row][col] = cursor;
            cursor += site.num_elems();
        }
    }
    const i64 total_env{cursor};
    for (auto row = 0_uz; row < num_env_rows; ++row)
        for (auto col = 0_uz; col < num_cols; ++col)
            ctx.dlenv.sigma_off[row][col] = total_env + ctx.dlenv.env_off[row][col];
    const auto layout_count = static_cast<i64>(k_sampling_layout_count);
    ctx.dlenv.sampling_elements = layout_count * total_env;
    const auto sampling_elements = static_cast<usize>(ctx.dlenv.sampling_elements);
    const auto sampling_bytes = sampling_elements * sizeof(cuFloatComplex);
    for (auto& lane : ctx.dlenv.lanes)
    {
        if (lane.sampling) continue;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&lane.sampling), sampling_bytes));
        if (err_state() != QNPEPS_OK) return;
    }
}

auto materialize_sampling_buffer(
    qnpeps_ctx& ctx, const cuFloatComplex* raw_values, cuFloatComplex* sampling_out
) -> void
{
    const auto num_env_rows = static_cast<usize>(ctx.cfg.lx - 1);
    const auto num_cols = static_cast<usize>(ctx.cfg.ly);
    const auto stream = ctx.linalg().stream();
    const auto* device_raw_values = raw_values;
    i64 raw_offset{};
    for (auto row = 0_uz; row < num_env_rows; ++row)
    {
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const auto site_dims = read_site_dims(ctx.dlenv.dims.data(), row * num_cols + col);
            DeviceTensor site{
                {site_dims.bond_left, site_dims.ket, site_dims.bra, site_dims.bond_right},
                const_cast<cuFloatComplex*>(device_raw_values + raw_offset)
            };
            permute_axes(
                site, {1, 3, 2, 0}, false, sampling_out + ctx.dlenv.env_off[row][col], stream
            );
            permute_axes(
                site, {1, 0, 2, 3}, false, sampling_out + ctx.dlenv.sigma_off[row][col], stream
            );
            raw_offset += static_cast<i64>(site.num_elems());
        }
    }
}

struct BuildEnvRowArgs
{
    Dims dims{};
    const std::vector<DeviceTensor>* row_ket{};
    const std::vector<DeviceTensor>* env_below{};
    int maxdim{};
    f64* row_log_scale{};
    int* fail_flag{};
    int build_step{};
    bool defer_scales{};
};

static auto build_env_row(
    BuildState& dl, Linalg& la, const Arenas& ar, const BuildEnvRowArgs& args
) -> std::vector<DeviceTensor>
{
    const auto& row_ket = *args.row_ket;
    const auto num_cols = static_cast<usize>(args.dims.ly);
    const DeviceTensor unit_environment{{1, 1, 1, 1}, dl.unit_environment};
    zipup::State state{
        .initial_factor = dl.initial_factor,
        .device_scales =
            dl.scales_all + static_cast<usize>(args.build_step) * num_cols,
        .fail_flag = args.fail_flag,
        .omegas = &dl.omegas,
        .rangefinder_rng = &dl.rangefinder_rng,
    };
    auto grouped_output = zipup::fused_peps_row(
        state,
        la,
        ar,
        {
            .row_ket = &row_ket,
            .environment = args.env_below,
            .unit_environment = unit_environment,
            .maxdim = args.maxdim,
            .log_scale = args.row_log_scale,
            .defer_scales = args.defer_scales,
        }
    );
    if (err_state() != QNPEPS_OK) return std::vector<DeviceTensor>(num_cols);

    std::vector<DeviceTensor> output(num_cols);
    for (auto col = 0_uz; col < num_cols; ++col)
    {
        const auto vertical = row_ket[col].dim[1];
        output[col] = DeviceTensor{
            {grouped_output[col].dim[0], vertical, vertical, grouped_output[col].dim[2]},
            grouped_output[col].d
        };
    }
    return output;
}

static auto build_env_rows(
    BuildState& dl,
    Linalg& la,
    const Arenas& ar,
    const Dims& dims,
    const std::vector<PepsRow>& peps,
    int maxdim,
    int* fail_flag
) -> std::vector<DlEnvRow>
{
    const auto num_env_rows = static_cast<usize>(dims.lx - 1);
    std::vector<DlEnvRow> env_rows{};
    env_rows.resize(num_env_rows);

    f64 ignored{};
    const auto last_env = num_env_rows - 1;
    env_rows[last_env] = build_env_row(
        dl,
        la,
        ar,
        {
            .dims = dims,
            .row_ket = &peps[num_env_rows],
            .env_below = nullptr,
            .maxdim = maxdim,
            .row_log_scale = &ignored,
            .fail_flag = fail_flag,
            .build_step = 0,
            .defer_scales = true,
        }
    );

    for (usize row{num_env_rows}; row >= 2; --row)
    {
        if (err_state() != QNPEPS_OK) return env_rows;
        const int build_step{static_cast<int>(num_env_rows - row + 1)};
        env_rows[row - 2] = build_env_row(
            dl,
            la,
            ar,
            {
                .dims = dims,
                .row_ket = &peps[row - 1],
                .env_below = &env_rows[row - 1],
                .maxdim = maxdim,
                .row_log_scale = &ignored,
                .fail_flag = fail_flag,
                .build_step = build_step,
                .defer_scales = true,
            }
        );
    }

    return env_rows;
}

struct DlSizes
{
    usize known{};
    usize rolling_r{};
    usize scratch{};
    [[nodiscard]] constexpr auto total() const noexcept -> usize
    {
        return known + rolling_r + scratch;
    }
};
static auto plan_dl_arena(Linalg& la, const Dims& dims, int maxdim, usize retained_rows) -> DlSizes
{
    constexpr usize arena_tail_pad{64 * 256};

    const auto dim_bond = static_cast<usize>(dims.dim_bond);
    const auto dim_bond2 = dim_bond * dim_bond;

    const auto chi = std::min(static_cast<usize>(maxdim), dim_bond2);
    const auto chi2 = chi * chi;

    const auto dim_phys = static_cast<usize>(dims.dim_phys);
    const auto num_cols = static_cast<usize>(dims.ly);

    const auto out_slot = device_align(sizeof(cuFloatComplex) * chi2 * dim_bond2);

    const auto per_row = [&]
    {
        usize out{0};
        out += device_align(sizeof(cuFloatComplex));
        out += device_align(sizeof(f32));
        out += device_align(sizeof(f64) * num_cols);
        out += 3 * out_slot;
        out += device_align(sizeof(int));
        return out;
    }();
    const auto known = retained_rows * (num_cols * out_slot + per_row);

    const auto rolling_r = out_slot;

    const auto site_peak = chi2 * dim_bond2 * dim_bond2;
    const auto scratch_elems = (4 + 3 * dim_phys) * site_peak + 4 * chi2 * dim_bond2;
    auto scratch = device_align(sizeof(cuFloatComplex) * scratch_elems);

    const auto qr_rows = static_cast<int>(chi * dim_bond2);
    const auto qr_cols = static_cast<int>(chi);
    scratch += la.qr_scratch(qr_rows, qr_cols).total();

    scratch += arena_tail_pad;

    return DlSizes{known, rolling_r, scratch};
}

static auto carve_dl_arena(
    BuildState& dl, const DlSizes& sz, usize scales_count, ArenaCursor arena
) -> ArenaCursor
{
    dl.fail = arena.take<int>(1);
    dl.scales_all = arena.take<f64>(scales_count);
    dl.unit_environment = arena.take<cuFloatComplex>(1);
    dl.initial_factor = arena.take<cuFloatComplex>(1);
    dl.known = arena.take_subarena(sz.known);
    dl.rolling_r = arena.take_subarena(sz.rolling_r);
    dl.scratch = arena.take_subarena(sz.scratch);
    return arena;
}

static auto dl_ensure_allocated(qnpeps_ctx& ctx, Linalg& la) -> void
{
    if (ctx.dl.allocated) return;

    const auto& cfg = ctx.cfg;
    const int lx{cfg.lx};
    const int ly{cfg.ly};
    const auto dim_bond = cfg.dim_bond;
    const auto dim_phys = cfg.dim_phys;

    const Dims dims{lx, ly, dim_phys, dim_bond};
    i64 peps_total{};
    for (auto row = 0; row < lx; ++row)
        peps_total += peps_row_elems(dims, row, row + 1);
    if (not ctx.dl.peps_buf)
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&ctx.dl.peps_buf),
            static_cast<usize>(peps_total) * sizeof(cuFloatComplex)
        ));
    if (err_state() != QNPEPS_OK) return;

    const int chi_c{std::min(cfg.chi_dl, dim_bond * dim_bond)};
    const auto num_env_rows = static_cast<usize>(lx - 1);
    const DlSizes sz = plan_dl_arena(la, dims, chi_c, num_env_rows);

    const auto num_cols = static_cast<usize>(ly);
    const usize scales_count{num_env_rows * num_cols};
    const auto measured = carve_dl_arena(ctx.dl, sz, scales_count, ArenaCursor::measure());

    if (not ctx.dl.arena)
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx.dl.arena), measured.total()));
    if (err_state() != QNPEPS_OK) return;

    carve_dl_arena(ctx.dl, sz, scales_count, ArenaCursor::carve(ctx.dl.arena, measured.total()));

    const auto dlenv_bytes = qnpeps_dlenv_bytes(&cfg);
    for (auto& lane : ctx.dlenv.lanes)
    {
        if (lane.packed) continue;
        CUDA_CHECK(cudaMalloc(&lane.packed, static_cast<usize>(dlenv_bytes)));
        if (err_state() != QNPEPS_OK) return;
    }

    init_dl_units(la, ctx.dl.unit_environment, ctx.dl.initial_factor);

    ctx.dl.allocated = true;
}

static auto free_dl_build(BuildState& dl) -> void
{
    if (dl.peps_buf)
    {
        CUDA_NOCHECK(cudaFree(dl.peps_buf));
        dl.peps_buf = nullptr;
    }
    if (dl.arena)
    {
        CUDA_NOCHECK(cudaFree(dl.arena));
        dl.arena = nullptr;
    }
    for (auto& entry : dl.omegas)
    {
        if (entry.second) CUDA_NOCHECK(cudaFree(entry.second));
    }
    dl.omegas.clear();
}

auto dl_free(qnpeps_ctx& ctx) -> void
{
    free_dl_build(ctx.dl);
    for (auto& lane : ctx.dlenv.lanes)
    {
        if (lane.packed)
        {
            CUDA_NOCHECK(cudaFree(lane.packed));
            lane.packed = nullptr;
        }
        if (lane.graph)
        {
            CUDA_NOCHECK(cudaGraphExecDestroy(lane.graph));
            lane.graph = nullptr;
        }
    }
}
}

namespace qnpeps::dlenv
{
auto build_dlenv(qnpeps_ctx& ctx, const void* device_peps, f64* cumulative_row_logs) -> int
{
    auto& la = ctx.linalg();
    const auto& cfg = ctx.cfg;
    const int lx{cfg.lx};
    const int ly{cfg.ly};
    const auto dim_bond = cfg.dim_bond;
    const auto dim_phys = cfg.dim_phys;
    const auto num_rows = static_cast<usize>(lx);
    const auto num_cols = static_cast<usize>(ly);

    dl_ensure_allocated(ctx, la);
    if (err_state() != QNPEPS_OK) return err_state();

    const Dims dims{lx, ly, dim_phys, dim_bond};

    std::vector<PepsRow> device_peps_grid{};
    device_peps_grid.resize(num_rows);

    const auto device_peps_base = reinterpret_cast<const cuFloatComplex*>(device_peps);
    i64 source_offset{};
    i64 packed_offset{};
    for (auto row = 0; row < lx; ++row)
    {
        const auto row_u = static_cast<usize>(row);
        device_peps_grid[row_u].resize(num_cols);
        pack_peps_row(
            dims,
            row,
            row + 1,
            device_peps_base,
            source_offset,
            ctx.dl.peps_buf,
            packed_offset,
            device_peps_grid[row_u],
            la.stream()
        );
    }

    ctx.dl.known.rewind();
    ctx.dl.rolling_r.rewind();
    ctx.dl.scratch.rewind();
    CUDA_CHECK(cudaMemsetAsync(ctx.dl.fail, 0, sizeof(int), la.stream()));

    const int chi_c{std::min(cfg.chi_dl, dim_bond * dim_bond)};
    const Arenas ar{ctx.dl.known, ctx.dl.rolling_r, ctx.dl.scratch};

    const auto target = static_cast<usize>(ctx.dlenv.build_count) % ctx.dlenv.lanes.size();
    auto& target_lane = ctx.dlenv.lanes[target];
    const auto num_sites = (num_rows - 1) * num_cols;
    auto* device_header = reinterpret_cast<int32_t*>(target_lane.packed);
    const auto header_elements = num_sites * k_dl_axis_count;
    auto* device_values = reinterpret_cast<cuFloatComplex*>(device_header + header_elements);

    const auto build_region = [&]
    {
        auto env_rows = build_env_rows(ctx.dl, la, ar, dims, device_peps_grid, chi_c, ctx.dl.fail);
        if (err_state() != QNPEPS_OK) return;

        if (not target_lane.header_written)
        {
            ctx.dlenv.dims.resize(num_sites * k_dl_axis_count);
            for (auto row = 0_uz; row < num_rows - 1; ++row)
                for (auto col = 0_uz; col < num_cols; ++col)
                    write_site_dims(
                        ctx.dlenv.dims.data(), row * num_cols + col, env_rows[row][col].dim
                    );
            copy_h2d_async(
                device_header, ctx.dlenv.dims.data(), ctx.dlenv.dims.size(), la.stream()
            );
            target_lane.header_written = true;
        }

        i64 values_offset{};
        for (auto row = 0_uz; row < num_rows - 1; ++row)
        {
            for (auto col = 0_uz; col < num_cols; ++col)
            {
                const DeviceTensor& site = env_rows[row][col];
                CUDA_CHECK(cudaMemcpyAsync(
                    device_values + values_offset,
                    site.d,
                    site.num_elems() * sizeof(cuFloatComplex),
                    cudaMemcpyDeviceToDevice,
                    la.stream()
                ));
                values_offset += static_cast<i64>(site.num_elems());
            }
        }
    };

    if (ctx.use_graph and target_lane.graph)
    {
        if (std::getenv("QNPEPS_GRAPH_LOG"))
            std::fprintf(stderr, "[qnpeps] dl_graph replayed lane=%zu\n", target);
        CUDA_CHECK(cudaGraphLaunch(target_lane.graph, la.stream()));
    }
    else if (ctx.use_graph and ctx.dl.warmed)
    {
        cudaGraph_t graph{};
        set_dl_capturing(ctx, true);
        CUDA_CHECK(cudaStreamBeginCapture(la.stream(), cudaStreamCaptureModeThreadLocal));
        build_region();
        const auto capture_status = cudaStreamEndCapture(la.stream(), &graph);
        set_dl_capturing(ctx, false);
        if (capture_status == cudaSuccess and err_state() == QNPEPS_OK
            and instantiate_graph(target_lane.graph, graph) == cudaSuccess)
        {
            CUDA_CHECK(cudaGraphDestroy(graph));
            if (std::getenv("QNPEPS_GRAPH_LOG"))
                std::fprintf(stderr, "[qnpeps] dl_graph captured lane=%zu\n", target);
            CUDA_CHECK(cudaGraphLaunch(target_lane.graph, la.stream()));
        }
        else
        {
            cudaGetLastError();
            target_lane.graph = nullptr;
            if (graph) CUDA_NOCHECK(cudaGraphDestroy(graph));
            ctx.dl.known.rewind();
            ctx.dl.rolling_r.rewind();
            ctx.dl.scratch.rewind();
            CUDA_CHECK(cudaMemsetAsync(ctx.dl.fail, 0, sizeof(int), la.stream()));
            build_region();
        }
    }
    else
    {
        build_region();
        ctx.dl.warmed = true;
    }
    if (err_state() != QNPEPS_OK) return err_state();

    CUDA_CHECK(cudaStreamSynchronize(la.stream()));

    const auto num_env_rows = num_rows - 1;
    std::vector<f64> scales_host{};
    scales_host.resize(num_env_rows * num_cols);
    CUDA_CHECK(cudaMemcpy(
        scales_host.data(),
        ctx.dl.scales_all,
        scales_host.size() * sizeof(f64),
        cudaMemcpyDeviceToHost
    ));

    std::vector<f64> row_logs{};
    row_logs.assign(num_env_rows, 0.0);
    f64 total_log{0.0};
    for (auto step = 0_uz; step < num_env_rows; ++step)
    {
        bool aborted{false};
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const auto scale = scales_host[step * num_cols + col];
            if (not std::isfinite(scale))
            {
                set_err(QNPEPS_ERR_INTERNAL);
                aborted = true;
                break;
            }
            if (scale > 0.0) total_log += std::log(scale);
        }
        row_logs[num_env_rows - 1 - step] = total_log;
        if (aborted) break;
    }

    if (cumulative_row_logs and not row_logs.empty())
    {
        CUDA_CHECK(cudaMemcpy(
            cumulative_row_logs,
            row_logs.data(),
            row_logs.size() * sizeof(f64),
            cudaMemcpyHostToDevice
        ));
    }

    int fail_host{};
    CUDA_CHECK(cudaMemcpy(&fail_host, ctx.dl.fail, sizeof(int), cudaMemcpyDeviceToHost));
    if (fail_host != 0) set_err(QNPEPS_ERR_CUDA);
    if (err_state() != QNPEPS_OK) return err_state();

    if (ctx.sampler.allocation.allocated)
    {
        ensure_sampling_buffers(ctx);
        if (err_state() != QNPEPS_OK) return err_state();
        materialize_sampling_buffer(ctx, device_values, target_lane.sampling);
    }

    target_lane.valid = true;
    ctx.dlenv.active_lane = target;
    ctx.dlenv.build_count += 1;
    return err_state();
}

static auto allocate_zipup_context(ZipupContext& context) -> void
{
    auto max_row_elements = 0_i64;
    for (auto row = 2; row <= context.config.lx; ++row)
        max_row_elements = std::max(
            max_row_elements, peps_row_elems(context.dims, row - 1, row)
        );
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&context.dl.peps_buf),
        static_cast<usize>(max_row_elements) * sizeof(cuFloatComplex)
    ));
    if (err_state() != QNPEPS_OK) return;

    const auto dim_bond = context.config.dim_bond;
    const auto chi_dl = std::min(context.maxdim, dim_bond * dim_bond);
    const auto sizes = plan_dl_arena(*context.linalg, context.dims, chi_dl, 1);
    const auto num_env_rows = static_cast<usize>(context.config.lx - 1);
    const auto num_cols = static_cast<usize>(context.config.ly);
    const auto scales_count = num_env_rows * num_cols;
    const auto measured = carve_dl_arena(
        context.dl, sizes, scales_count, ArenaCursor::measure()
    );
    if (err_state() != QNPEPS_OK) return;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&context.dl.arena), measured.total()));
    if (err_state() != QNPEPS_OK) return;
    carve_dl_arena(
        context.dl,
        sizes,
        scales_count,
        ArenaCursor::carve(context.dl.arena, measured.total())
    );
    if (err_state() != QNPEPS_OK) return;

    init_dl_units(
        *context.linalg, context.dl.unit_environment, context.dl.initial_factor
    );
    context.dl.allocated = err_state() == QNPEPS_OK;
}

inline constexpr usize k_grouped_mps_axis_count{3};
inline constexpr usize k_grouped_mps_left{0};
inline constexpr usize k_grouped_mps_physical{1};
inline constexpr usize k_grouped_mps_right{2};

struct GroupedMpsDims
{
    int left{};
    int physical{};
    int right{};
};

[[nodiscard]] static auto read_grouped_mps_dims(const int32_t* dims, usize site)
    -> GroupedMpsDims
{
    const auto offset = site * k_grouped_mps_axis_count;
    return GroupedMpsDims{
        .left = dims[offset + k_grouped_mps_left],
        .physical = dims[offset + k_grouped_mps_physical],
        .right = dims[offset + k_grouped_mps_right],
    };
}

static auto write_grouped_mps_dims(int32_t* dims, usize site, const Shape& shape) -> void
{
    const auto offset = site * k_grouped_mps_axis_count;
    dims[offset + k_grouped_mps_left] = shape[k_grouped_mps_left];
    dims[offset + k_grouped_mps_physical] = shape[k_grouped_mps_physical];
    dims[offset + k_grouped_mps_right] = shape[k_grouped_mps_right];
}

[[nodiscard]] static auto append_product(u64& total, std::initializer_list<u64> factors) -> bool
{
    auto product = 1_u64;
    for (const auto factor : factors)
    {
        if (factor != 0 and product > std::numeric_limits<u64>::max() / factor) return false;
        product *= factor;
    }
    if (total > std::numeric_limits<u64>::max() - product) return false;
    total += product;
    return true;
}

[[nodiscard]] auto zipup_peps_row_bytes(const QnpepsConfig& config, int maxdim) -> i64
{
    if (maxdim < 1) return -1;
    const auto ly = static_cast<u64>(config.ly);
    const auto dim_bond = static_cast<u64>(config.dim_bond);
    if (dim_bond != 0 and dim_bond > std::numeric_limits<u64>::max() / dim_bond) return -1;
    const auto bond_pair = dim_bond * dim_bond;
    const auto capped_dim = std::min(static_cast<u64>(maxdim), bond_pair);
    auto elements = 0_u64;
    if (not append_product(elements, {ly, capped_dim, capped_dim, bond_pair})) return -1;
    if (elements > static_cast<u64>(std::numeric_limits<i64>::max()) / sizeof(cuFloatComplex))
        return -1;
    return static_cast<i64>(elements * sizeof(cuFloatComplex));
}

[[nodiscard]] static auto make_environment_views(
    const std::vector<DeviceTensor>& row_ket,
    const QnpepsZipupPepsRowArgs& args,
    std::vector<DeviceTensor>& environment
) -> qnpeps_status
{
    const auto has_dims = args.mps_dims != nullptr;
    const auto has_values = args.mps_values != nullptr;
    if (has_dims != has_values) return set_err(QNPEPS_ERR_NULL_ARG);
    if (not has_dims)
    {
        if (args.mps_bytes != 0) return set_err(QNPEPS_ERR_BAD_CONFIG);
        for (const auto& ket : row_ket)
            if (ket.dim[3] != 1) return set_err(QNPEPS_ERR_BAD_CONFIG);
        return QNPEPS_OK;
    }

    environment.resize(row_ket.size());
    const auto* values = reinterpret_cast<const cuFloatComplex*>(args.mps_values);
    auto previous_right = 1;
    auto value_elements = 0_u64;
    for (auto site = 0_uz; site < row_ket.size(); ++site)
    {
        const auto dims = read_grouped_mps_dims(args.mps_dims, site);
        if (dims.left < 1 or dims.physical < 1 or dims.right < 1)
            return set_err(QNPEPS_ERR_BAD_CONFIG);
        if (dims.left != previous_right) return set_err(QNPEPS_ERR_BAD_CONFIG);
        const auto vertical = row_ket[site].dim[3];
        if (static_cast<i64>(dims.physical) != static_cast<i64>(vertical) * vertical)
            return set_err(QNPEPS_ERR_BAD_CONFIG);
        const auto offset = value_elements;
        if (not append_product(
                value_elements,
                {
                    static_cast<u64>(dims.left),
                    static_cast<u64>(dims.physical),
                    static_cast<u64>(dims.right),
                }
            ))
        {
            return set_err(QNPEPS_ERR_BAD_CONFIG);
        }
        environment[site] = DeviceTensor{
            {dims.left, vertical, vertical, dims.right},
            const_cast<cuFloatComplex*>(values + static_cast<usize>(offset))
        };
        previous_right = dims.right;
    }
    if (previous_right != 1) return set_err(QNPEPS_ERR_BAD_CONFIG);
    if (value_elements > std::numeric_limits<u64>::max() / sizeof(cuFloatComplex))
        return set_err(QNPEPS_ERR_BAD_CONFIG);
    if (args.mps_bytes != value_elements * sizeof(cuFloatComplex))
        return set_err(QNPEPS_ERR_BAD_CONFIG);
    return QNPEPS_OK;
}

[[nodiscard]] static auto copy_grouped_output(
    const std::vector<DeviceTensor>& output,
    const QnpepsZipupPepsRowArgs& args,
    cudaStream_t stream
) -> qnpeps_status
{
    auto value_elements = 0_u64;
    for (auto site = 0_uz; site < output.size(); ++site)
    {
        if (output[site].dim.rank() != k_grouped_mps_axis_count)
            return set_err(QNPEPS_ERR_INTERNAL);
        write_grouped_mps_dims(args.output_dims, site, output[site].dim);
        if (not append_product(
                value_elements,
                {
                    static_cast<u64>(output[site].dim[0]),
                    static_cast<u64>(output[site].dim[1]),
                    static_cast<u64>(output[site].dim[2]),
                }
            ))
        {
            return set_err(QNPEPS_ERR_INTERNAL);
        }
    }
    if (value_elements > std::numeric_limits<u64>::max() / sizeof(cuFloatComplex))
        return set_err(QNPEPS_ERR_INTERNAL);
    const auto value_bytes = value_elements * sizeof(cuFloatComplex);
    if (value_bytes > args.output_bytes) return set_err(QNPEPS_ERR_BAD_CONFIG);

    auto* destination = reinterpret_cast<cuFloatComplex*>(args.output_values);
    auto offset = 0_uz;
    for (const auto& site : output)
    {
        CUDA_CHECK(cudaMemcpyAsync(
            destination + offset,
            site.d,
            site.num_elems() * sizeof(cuFloatComplex),
            cudaMemcpyDeviceToDevice,
            stream
        ));
        offset += site.num_elems();
    }
    return err_state();
}

auto create_zipup_context(const QnpepsConfig& config, int maxdim, cudaStream_t stream)
    -> qnpeps_zipup_ctx*
{
    auto stream_use = stream;
    auto owns_stream = false;
    if (not stream_use)
    {
        const auto stream_status = cudaStreamCreateWithFlags(&stream_use, cudaStreamNonBlocking);
        if (stream_status != cudaSuccess)
        {
            set_cuda_err(stream_status);
            return nullptr;
        }
        owns_stream = true;
    }

    auto linalg = make_linalg(stream_use);
    if (not linalg)
    {
        if (owns_stream) CUDA_NOCHECK(cudaStreamDestroy(stream_use));
        return nullptr;
    }

    auto context = std::unique_ptr<ZipupContext>{new (std::nothrow) ZipupContext{}};
    if (not context)
    {
        linalg.reset();
        if (owns_stream) CUDA_NOCHECK(cudaStreamDestroy(stream_use));
        set_err(QNPEPS_ERR_OOM);
        return nullptr;
    }

    context->config = config;
    context->maxdim = maxdim;
    context->dims = Dims{config.lx, config.ly, config.dim_phys, config.dim_bond};
    context->stream = stream_use;
    context->owns_stream = owns_stream;
    context->linalg = std::move(linalg);
    allocate_zipup_context(*context);
    if (err_state() != QNPEPS_OK)
    {
        free_dl_build(context->dl);
        context->linalg.reset();
        context.reset();
        if (owns_stream) CUDA_NOCHECK(cudaStreamDestroy(stream_use));
        return nullptr;
    }
    return reinterpret_cast<qnpeps_zipup_ctx*>(context.release());
}

auto destroy_zipup_context(qnpeps_zipup_ctx* opaque_context) -> void
{
    auto* context = zipup_context(opaque_context);
    if (not context) return;
    CUDA_NOCHECK(cudaStreamSynchronize(context->stream));
    free_dl_build(context->dl);
    context->linalg.reset();
    const auto stream = context->stream;
    const auto owns_stream = context->owns_stream;
    delete context;
    if (owns_stream) CUDA_NOCHECK(cudaStreamDestroy(stream));
}

auto begin_zipup_context(qnpeps_zipup_ctx& opaque_context) -> qnpeps_status
{
    auto& context = zipup_context(opaque_context);
    if (context.active) return set_err(QNPEPS_ERR_BAD_CONFIG);
    context.scale_count = 0;
    CUDA_CHECK(cudaMemsetAsync(context.dl.fail, 0, sizeof(int), context.stream));
    if (err_state() != QNPEPS_OK) return err_state();
    context.active = true;
    return QNPEPS_OK;
}

auto enqueue_peps_row(qnpeps_zipup_ctx& opaque_context, const QnpepsZipupPepsRowArgs& args)
    -> qnpeps_status
{
    auto& context = zipup_context(opaque_context);
    if (not context.active) return set_err(QNPEPS_ERR_BAD_CONFIG);
    if (args.row < 2 or args.row > context.config.lx)
        return set_err(QNPEPS_ERR_BAD_CONFIG);

    const auto row_elements = peps_row_elems(context.dims, args.row - 1, args.row);
    const auto expected_peps_bytes = static_cast<u64>(row_elements) * sizeof(cuFloatComplex);
    const auto required_output_bytes = zipup_peps_row_bytes(context.config, context.maxdim);
    if (args.peps_row_bytes != expected_peps_bytes or required_output_bytes < 0
        or args.output_bytes < static_cast<u64>(required_output_bytes))
    {
        return set_err(QNPEPS_ERR_BAD_CONFIG);
    }

    const auto num_cols = static_cast<usize>(context.config.ly);
    std::vector<DeviceTensor> row_ket(num_cols);
    auto source_offset = 0_i64;
    auto packed_offset = 0_i64;
    pack_peps_row(
        context.dims,
        args.row - 1,
        args.row,
        reinterpret_cast<const cuFloatComplex*>(args.peps_row),
        source_offset,
        context.dl.peps_buf,
        packed_offset,
        row_ket,
        context.stream
    );
    if (source_offset != row_elements or packed_offset != row_elements)
        return set_err(QNPEPS_ERR_INTERNAL);

    std::vector<DeviceTensor> environment{};
    const auto environment_status = make_environment_views(row_ket, args, environment);
    if (environment_status != QNPEPS_OK) return environment_status;

    context.dl.known.rewind();
    context.dl.rolling_r.rewind();
    context.dl.scratch.rewind();
    const Arenas arenas{context.dl.known, context.dl.rolling_r, context.dl.scratch};
    const auto scale_capacity = static_cast<usize>(context.config.lx - 1) * num_cols;
    if (context.scale_count + num_cols > scale_capacity)
        return set_err(QNPEPS_ERR_BAD_CONFIG);
    zipup::State state{
        .initial_factor = context.dl.initial_factor,
        .device_scales = context.dl.scales_all + context.scale_count,
        .fail_flag = context.dl.fail,
        .omegas = &context.dl.omegas,
        .rangefinder_rng = &context.dl.rangefinder_rng,
    };
    const DeviceTensor unit_environment{{1, 1, 1, 1}, context.dl.unit_environment};
    auto output = zipup::fused_peps_row(
        state,
        *context.linalg,
        arenas,
        {
            .row_ket = &row_ket,
            .environment = environment.empty() ? nullptr : &environment,
            .unit_environment = unit_environment,
            .maxdim = std::min(
                context.maxdim, context.config.dim_bond * context.config.dim_bond
            ),
            .log_scale = nullptr,
            .defer_scales = true,
        }
    );
    if (err_state() != QNPEPS_OK) return err_state();
    const auto output_status = copy_grouped_output(output, args, context.stream);
    if (output_status != QNPEPS_OK) return output_status;
    context.scale_count += num_cols;
    return QNPEPS_OK;
}

auto finish_zipup_context(qnpeps_zipup_ctx& opaque_context, f64* scales, usize count)
    -> qnpeps_status
{
    auto& context = zipup_context(opaque_context);
    if (not context.active or count != context.scale_count)
        return set_err(QNPEPS_ERR_BAD_CONFIG);
    if (not scales) return set_err(QNPEPS_ERR_NULL_ARG);

    auto fail_host = 0;
    copy_d2h_async(scales, context.dl.scales_all, count, context.stream);
    copy_d2h_async(&fail_host, context.dl.fail, 1, context.stream);
    CUDA_CHECK(cudaStreamSynchronize(context.stream));
    context.active = false;
    if (err_state() != QNPEPS_OK) return err_state();
    if (fail_host != 0) return set_err(QNPEPS_ERR_CUDA);
    for (auto index = 0_uz; index < count; ++index)
        if (not std::isfinite(scales[index])) return set_err(QNPEPS_ERR_INTERNAL);
    return QNPEPS_OK;
}
}
