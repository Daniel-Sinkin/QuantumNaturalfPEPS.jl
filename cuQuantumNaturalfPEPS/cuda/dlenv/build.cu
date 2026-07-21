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
#include <map>
#include <span>
#include <vector>

namespace qnpeps::dlenv
{
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

auto rangefinder_omega(qnpeps_ctx::DlBuild& dl, int cols, int rank) -> cuFloatComplex*
{
    const auto key = std::pair{cols, rank};
    if (const auto it = dl.omegas.find(key); it != dl.omegas.end())
    {
        return it->second;
    }
    std::vector<cuFloatComplex> host_omega{};
    host_omega.resize(static_cast<usize>(cols) * static_cast<usize>(rank));
    dl.rangefinder_rng.fill_complex_normal(std::span{host_omega});
    cuFloatComplex* device_omega{};
    CUDA_CHECK(cudaMalloc(&device_omega, host_omega.size() * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMemcpy(
        device_omega,
        host_omega.data(),
        host_omega.size() * sizeof(cuFloatComplex),
        cudaMemcpyHostToDevice
    ));
    dl.omegas[key] = device_omega;
    return device_omega;
}

auto set_dl_capturing(qnpeps_ctx& ctx, bool on) -> void
{
    ctx.dl.capturing = on;
}

auto ensure_dlenv_views(qnpeps_ctx& ctx) -> void
{
    auto& dlenv = ctx.dlenv;
    if (dlenv.views_allocated) return;
    const auto num_env_rows = static_cast<usize>(ctx.cfg.lx - 1);
    const auto num_cols = static_cast<usize>(ctx.cfg.ly);
    dlenv.env_off.assign(num_env_rows, std::vector<i64>(num_cols, 0));
    dlenv.sigma_off.assign(num_env_rows, std::vector<i64>(num_cols, 0));
    const auto total_env = [&]
    {
        i64 out{};
        for (auto row = 0_uz; row < num_env_rows; ++row)
        {
            for (auto col = 0_uz; col < num_cols; ++col)
            {
                const auto site = read_site_dims(dlenv.dims.data(), row * num_cols + col);
                dlenv.env_off[row][col] = out;
                out += site.num_elems();
            }
        }
        return out;
    }();

    for (auto row = 0_uz; row < num_env_rows; ++row)
        for (auto col = 0_uz; col < num_cols; ++col)
            dlenv.sigma_off[row][col] = total_env + dlenv.env_off[row][col];

    dlenv.views_elems = static_cast<i64>(k_sampling_layout_count) * total_env;
    for (auto& view : dlenv.views)
    {
        if (view) continue;
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&view),
            static_cast<usize>(dlenv.views_elems) * sizeof(cuFloatComplex)
        ));
        if (err_state() != QNPEPS_OK) return;
    }
    dlenv.views_allocated = true;
}

auto materialize_dlenv_views(
    qnpeps_ctx& ctx, const cuFloatComplex* raw_values, cuFloatComplex* views_out
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
                site, {1, 3, 2, 0}, false, views_out + ctx.dlenv.env_off[row][col], stream
            );
            permute_axes(
                site, {1, 0, 2, 3}, false, views_out + ctx.dlenv.sigma_off[row][col], stream
            );
            raw_offset += static_cast<i64>(site.num_elems());
        }
    }
}

__global__ auto cu_or_status(const int* status, int* flag) -> void
{
    if (*status != 0) atomicOr(flag, 1);
}

struct DlRangefinderArgs
{
    DeviceTensor input{};
    int rows{};
    int cols{};
    int maxdim{};
    DeviceTensor* q{};
    int* rank_out{};
    DeviceTensor* r{};
    int* fail_flag{};
};

static auto rangefinder(
    qnpeps_ctx::DlBuild& dl, Linalg& la, const Arenas& ar, const DlRangefinderArgs& args
) -> void
{
    const auto& input = args.input;
    const int rows{args.rows};
    const int cols{args.cols};
    const int maxdim{args.maxdim};
    DeviceTensor& q = *args.q;
    DeviceTensor& r = *args.r;
    int* fail_flag = args.fail_flag;

    int rank{maxdim < rows ? maxdim : rows};
    if (rank > cols) rank = cols;
    if (rank < 1) rank = 1;
    *args.rank_out = rank;
    const CuMatrix input_matrix{input.d, rows, cols};

    auto sketch = alloc(ar.scratch, {rows, rank});
    DEFER([&] { free(sketch); });
    auto projection = alloc(ar.scratch, {cols, rank});
    DEFER([&] { free(projection); });
    auto* omega = rangefinder_omega(dl, cols, rank);
    const CuMatrix omega_matrix{omega, cols, rank};
    const CuMatrix sketch_matrix{sketch.d, rows, rank};
    const CuMatrix projection_matrix{projection.d, cols, rank};
    la.matmul(input_matrix, omega_matrix, sketch_matrix);
    for (auto iteration = 0; iteration < 2; ++iteration)
    {
        la.matmul_left_adj(input_matrix, sketch_matrix, projection_matrix);
        la.matmul(input_matrix, projection_matrix, sketch_matrix);
    }

    const auto qr_layout = la.qr_scratch(sketch_matrix);
    void* qr_scratch{ar.scratch.take<char>(qr_layout.total())};
    la.qr(sketch_matrix, qr_scratch, qr_layout);
    if (fail_flag)
    {
        const auto* qr_status = byte_offset<int>(qr_scratch, qr_layout.reflector_bytes);
        cu_or_status<<<1, 1, 0, la.stream()>>>(qr_status, fail_flag);
        CUDA_CHECK(cudaGetLastError());
    }

    q = alloc(ar.known, {rows, rank});
    CUDA_CHECK(cudaMemcpyAsync(
        q.d,
        sketch.d,
        static_cast<usize>(rows) * static_cast<usize>(rank) * sizeof(cuFloatComplex),
        cudaMemcpyDeviceToDevice,
        la.stream()
    ));
    r = alloc(ar.scratch, {rank, cols});
    la.matmul_left_adj(sketch_matrix, input_matrix, CuMatrix{r.d, rank, cols});
}

__global__ auto cu_absmax_inverse(
    const cuFloatComplex* factor, i64 element_count, f32* device_inverse_scale, f64* device_scale
) -> void
{
    __shared__ f64 shared_max[k_tree_reduce_threads];
    f64 local_max{0.0};

    for (auto index = threadIdx.x; index < element_count; index += blockDim.x)
    {
        const auto component_abs_sum =
            fabs(static_cast<f64>(factor[index].x)) + fabs(static_cast<f64>(factor[index].y));
        if (component_abs_sum > local_max) local_max = component_abs_sum;
    }
    shared_max[threadIdx.x] = local_max;
    __syncthreads();

    for (auto offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset and shared_max[threadIdx.x + offset] > shared_max[threadIdx.x])
        {
            shared_max[threadIdx.x] = shared_max[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        const auto scale = shared_max[0];
        *device_scale = scale;
        const auto valid_scale = (scale > 0.0) and isfinite(scale);
        *device_inverse_scale = valid_scale ? static_cast<f32>(1.0 / scale) : 1.0f;
    }
}

__global__ auto cu_apply_inverse_scale(
    cuFloatComplex* factor, i64 element_count, const f32* device_inverse_scale
) -> void
{
    const auto inverse_scale = *device_inverse_scale;
    for (auto index = global_lane(); index < element_count; index += grid_stride())
    {
        factor[index].x *= inverse_scale;
        factor[index].y *= inverse_scale;
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
    qnpeps_ctx::DlBuild& dl, Linalg& la, const Arenas& ar, const BuildEnvRowArgs& args
) -> std::vector<DeviceTensor>
{
    const std::vector<DeviceTensor>& row_ket = *args.row_ket;
    const std::vector<DeviceTensor>* env_below = args.env_below;
    const int maxdim{args.maxdim};
    f64& row_log_scale = *args.row_log_scale;
    int* fail_flag = args.fail_flag;
    const int build_step{args.build_step};
    const bool defer_scales{args.defer_scales};

    const int ly{args.dims.ly};
    const auto num_cols = static_cast<usize>(ly);
    std::vector<DeviceTensor> out{};
    out.resize(num_cols);

    f64* device_scales = dl.scales_all + static_cast<usize>(build_step) * num_cols;

    auto carried_factor = DeviceTensor{{1, 1, 1, 1}, dl.initial_factor};
    const auto unit_environment = DeviceTensor{{1, 1, 1, 1}, dl.unit_environment};

    for (auto col = 0_uz; col < num_cols; ++col)
    {
        if (err_state() != QNPEPS_OK) return out;
        ArenaCursor column_scratch{ar.scratch};
        const Arenas column_arenas{ar.known, ar.rolling_r, column_scratch};

        const DeviceTensor& ket = row_ket[col];
        const auto& environment = env_below ? (*env_below)[col] : unit_environment;

        DeviceTensor left_environment{};
        if (not contract(
                column_scratch,
                la,
                {
                    .dims_a = carried_factor.dim,
                    .contracted_a = {3},
                    .dims_b = environment.dim,
                    .contracted_b = {0},
                },
                carried_factor,
                environment,
                left_environment
            ))
            return out;
        DEFER([&] { free(left_environment); });
        DeviceTensor left_environment_ket{};
        if (not contract(
                column_scratch,
                la,
                {
                    .dims_a = left_environment.dim,
                    .contracted_a = {1, 3},
                    .dims_b = ket.dim,
                    .contracted_b = {4, 3},
                },
                left_environment,
                ket,
                left_environment_ket
            ))
            return out;
        DEFER([&] { free(left_environment_ket); });
        DeviceTensor column_tensor{};
        if (not contract(
                column_scratch,
                la,
                {
                    .dims_a = left_environment_ket.dim,
                    .contracted_a = {1, 2, 4},
                    .dims_b = ket.dim,
                    .contracted_b = {4, 3, 0},
                    .transforms = {.conj_b = true},
                },
                left_environment_ket,
                ket,
                column_tensor
            ))
            return out;
        DEFER([&] { free(column_tensor); });

        auto column_matrix = permute_axes(
            column_scratch, column_tensor, {0, 2, 4, 1, 3, 5}, false, la.stream()
        );
        DEFER([&] { free(column_matrix); });
        const int bond_left{column_tensor.dim[0]};
        const int bond_right{column_tensor.dim[1]};
        const int ket_vertical{column_tensor.dim[2]};
        const int ket_horizontal{column_tensor.dim[3]};
        const int bra_vertical{column_tensor.dim[4]};
        const int bra_horizontal{column_tensor.dim[5]};
        const int rows{bond_left * ket_vertical * bra_vertical};
        const int cols{bond_right * ket_horizontal * bra_horizontal};

        DeviceTensor q{};
        DeviceTensor r_factor{};
        int rank{};
        rangefinder(
            dl,
            la,
            column_arenas,
            {
                .input = column_matrix,
                .rows = rows,
                .cols = cols,
                .maxdim = maxdim,
                .q = &q,
                .rank_out = &rank,
                .r = &r_factor,
                .fail_flag = fail_flag,
            }
        );

        {
            const auto r_factor_elems = static_cast<i64>(rank) * cols;

            auto* device_inverse_scale = column_scratch.take<f32>(1);
            cu_absmax_inverse<<<1, k_tree_reduce_threads, 0, la.stream()>>>(
                r_factor.d, r_factor_elems, device_inverse_scale, device_scales + col
            );
            CUDA_CHECK(cudaGetLastError());

            const auto blocks = static_cast<u32>(ceil_div(r_factor_elems, k_tree_reduce_threads));
            cu_apply_inverse_scale<<<blocks, k_tree_reduce_threads, 0, la.stream()>>>(
                r_factor.d, r_factor_elems, device_inverse_scale
            );
            CUDA_CHECK(cudaGetLastError());
        }

        out[col] = DeviceTensor{{bond_left, ket_vertical, bra_vertical, rank}, q.d};

        const auto r_reshaped =
            DeviceTensor{{rank, bond_right, ket_horizontal, bra_horizontal}, r_factor.d};
        ar.rolling_r.rewind();
        carried_factor =
            permute_axes(ar.rolling_r, r_reshaped, {0, 2, 3, 1}, false, la.stream());
    }

    if (not defer_scales)
    {
        std::vector<f64> scales{};
        scales.resize(num_cols);
        CUDA_CHECK(
            cudaMemcpy(scales.data(), device_scales, num_cols * sizeof(f64), cudaMemcpyDeviceToHost)
        );
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const auto scale = scales[col];
            if (not std::isfinite(scale))
            {
                set_err(QNPEPS_ERR_INTERNAL);
                break;
            }
            if (scale > 0.0) row_log_scale += std::log(scale);
        }
    }

    DeviceTensor& last = out[num_cols - 1];
    {
        DeviceTensor folded{};
        if (not contract(
                ar.known,
                la,
                {
                    .dims_a = last.dim,
                    .contracted_a = {3},
                    .dims_b = carried_factor.dim,
                    .contracted_b = {0},
                },
                last,
                carried_factor,
                folded
            ))
            return out;
        last = DeviceTensor{{folded.dim[0], folded.dim[1], folded.dim[2], 1}, folded.d};
    }
    return out;
}

static auto build_env_rows(
    qnpeps_ctx::DlBuild& dl,
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
static auto plan_dl_arena(Linalg& la, const Dims& dims, int maxdim) -> DlSizes
{
    constexpr usize arena_tail_pad{64 * 256};

    const auto dim_bond = static_cast<usize>(dims.dim_bond);
    const auto dim_bond2 = dim_bond * dim_bond;

    const auto chi = std::min(static_cast<usize>(maxdim), dim_bond2);
    const auto chi2 = chi * chi;

    const auto dim_phys = static_cast<usize>(dims.dim_phys);
    const auto num_env_rows = static_cast<usize>(dims.lx - 1);
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
    const auto known = num_env_rows * (num_cols * out_slot + per_row);

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
    qnpeps_ctx::DlBuild& dl, const DlSizes& sz, usize scales_count, ArenaCursor arena
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
    const DlSizes sz = plan_dl_arena(la, dims, chi_c);

    const auto num_env_rows = static_cast<usize>(lx - 1);
    const auto num_cols = static_cast<usize>(ly);
    const usize scales_count{num_env_rows * num_cols};
    const auto measured = carve_dl_arena(ctx.dl, sz, scales_count, ArenaCursor::measure());

    if (not ctx.dl.arena)
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx.dl.arena), measured.total()));
    if (err_state() != QNPEPS_OK) return;

    carve_dl_arena(ctx.dl, sz, scales_count, ArenaCursor::carve(ctx.dl.arena, measured.total()));

    const auto dlenv_bytes = qnpeps_dlenv_bytes(&cfg);
    if (not ctx.dlenv.buf[0])
        CUDA_CHECK(cudaMalloc(&ctx.dlenv.buf[0], static_cast<usize>(dlenv_bytes)));
    if (not ctx.dlenv.buf[1])
        CUDA_CHECK(cudaMalloc(&ctx.dlenv.buf[1], static_cast<usize>(dlenv_bytes)));
    if (err_state() != QNPEPS_OK) return;

    init_dl_units(la, ctx.dl.unit_environment, ctx.dl.initial_factor);

    ctx.dl.allocated = true;
}

static auto free_dl_build(qnpeps_ctx::DlBuild& dl) -> void
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
    for (auto& buf : ctx.dlenv.buf)
    {
        if (buf)
        {
            CUDA_NOCHECK(cudaFree(buf));
            buf = nullptr;
        }
    }
    for (auto& graph : ctx.dlenv.graph)
    {
        if (graph)
        {
            CUDA_NOCHECK(cudaGraphExecDestroy(graph));
            graph = nullptr;
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

    const auto target = static_cast<usize>(ctx.dlenv.build_count) % ctx.dlenv.buf.size();
    const auto num_sites = (num_rows - 1) * num_cols;
    auto* device_header = reinterpret_cast<int32_t*>(ctx.dlenv.buf[target]);
    auto* device_values =
        reinterpret_cast<cuFloatComplex*>(device_header + num_sites * k_dl_axis_count);

    const auto build_region = [&]
    {
        auto env_rows = build_env_rows(ctx.dl, la, ar, dims, device_peps_grid, chi_c, ctx.dl.fail);
        if (err_state() != QNPEPS_OK) return;

        if (not ctx.dlenv.header_written[target])
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
            ctx.dlenv.header_written[target] = true;
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

    if (ctx.use_graph and ctx.dlenv.graph[target])
    {
        if (std::getenv("QNPEPS_GRAPH_LOG"))
            std::fprintf(stderr, "[qnpeps] dl_graph replayed buf=%zu\n", target);
        CUDA_CHECK(cudaGraphLaunch(ctx.dlenv.graph[target], la.stream()));
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
            and instantiate_graph(ctx.dlenv.graph[target], graph) == cudaSuccess)
        {
            CUDA_CHECK(cudaGraphDestroy(graph));
            if (std::getenv("QNPEPS_GRAPH_LOG"))
                std::fprintf(stderr, "[qnpeps] dl_graph captured buf=%zu\n", target);
            CUDA_CHECK(cudaGraphLaunch(ctx.dlenv.graph[target], la.stream()));
        }
        else
        {
            cudaGetLastError();
            ctx.dlenv.graph[target] = nullptr;
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
        ensure_dlenv_views(ctx);
        if (err_state() != QNPEPS_OK) return err_state();
        materialize_dlenv_views(ctx, device_values, ctx.dlenv.views[target]);
    }

    ctx.dlenv.valid[target] = true;
    ctx.dlenv.active = target;
    ctx.dlenv.build_count += 1;
    return err_state();
}

auto build_dlenv_row(
    Linalg& la,
    const QnpepsConfig& cfg,
    int row,
    int maxdim,
    const void* device_peps_row,
    const void* device_env_below,
    void* dlenv_row_out,
    f64* row_log_out
) -> int
{
    const int lx{cfg.lx};
    const int ly{cfg.ly};
    const auto dim_bond = cfg.dim_bond;
    const auto dim_phys = cfg.dim_phys;
    const auto num_cols = static_cast<usize>(ly);
    const Dims dims{lx, ly, dim_phys, dim_bond};

    const i64 row_total{peps_row_elems(dims, row - 1, row)};

    cuFloatComplex* peps_buf{};
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&peps_buf), static_cast<usize>(row_total) * sizeof(cuFloatComplex)
    ));
    if (err_state() != QNPEPS_OK) return err_state();
    DEFER([&] { CUDA_NOCHECK(cudaFree(peps_buf)); });

    std::vector<DeviceTensor> row_ket{};
    row_ket.resize(num_cols);

    {
        const auto device_peps_base = reinterpret_cast<const cuFloatComplex*>(device_peps_row);
        i64 source_offset{};
        i64 packed_offset{};
        pack_peps_row(
            dims,
            row - 1,
            row,
            device_peps_base,
            source_offset,
            peps_buf,
            packed_offset,
            row_ket,
            la.stream()
        );
    }

    std::vector<DeviceTensor> env_below{};
    if (device_env_below)
    {
        env_below.resize(num_cols);
        std::vector<int32_t> environment_dims{};
        environment_dims.resize(num_cols * k_dl_axis_count);
        const auto header_bytes = environment_dims.size() * sizeof(int32_t);
        CUDA_CHECK(cudaMemcpy(
            environment_dims.data(), device_env_below, header_bytes, cudaMemcpyDeviceToHost
        ));
        const auto device_values = byte_offset<cuFloatComplex>(device_env_below, header_bytes);
        i64 values_offset{};
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const auto site_dims = read_site_dims(environment_dims.data(), col);
            env_below[col] = DeviceTensor{
                {site_dims.bond_left, site_dims.ket, site_dims.bra, site_dims.bond_right},
                const_cast<cuFloatComplex*>(device_values + values_offset)
            };
            values_offset += site_dims.num_elems();
        }
    }

    const int chi_dl{std::min(maxdim, dim_bond * dim_bond)};
    const DlSizes sz = plan_dl_arena(la, dims, chi_dl);
    char* arena_base{};
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&arena_base), sz.total()));
    if (err_state() != QNPEPS_OK) return err_state();
    DEFER([&] { CUDA_NOCHECK(cudaFree(arena_base)); });
    auto known = ArenaCursor::carve(arena_base, sz.known);
    auto rolling_r = ArenaCursor::carve(arena_base + sz.known, sz.rolling_r);
    auto scratch = ArenaCursor::carve(arena_base + sz.known + sz.rolling_r, sz.scratch);
    const Arenas ar{known, rolling_r, scratch};

    qnpeps_ctx::DlBuild dl{};
    DEFER([&] { free_dl_build(dl); });
    dl.fail = known.take<int>(1);
    dl.scales_all = known.take<f64>(num_cols);
    dl.unit_environment = known.take<cuFloatComplex>(1);
    dl.initial_factor = known.take<cuFloatComplex>(1);
    init_dl_units(la, dl.unit_environment, dl.initial_factor);
    CUDA_CHECK(cudaMemsetAsync(dl.fail, 0, sizeof(int), la.stream()));

    f64 row_log{0.0};
    auto env_row = build_env_row(
        dl,
        la,
        ar,
        {
            .dims = dims,
            .row_ket = &row_ket,
            .env_below = device_env_below ? &env_below : nullptr,
            .maxdim = chi_dl,
            .row_log_scale = &row_log,
            .fail_flag = dl.fail,
            .build_step = 0,
            .defer_scales = false,
        }
    );
    if (err_state() != QNPEPS_OK) return err_state();

    {
        auto* device_header = reinterpret_cast<int32_t*>(dlenv_row_out);
        std::vector<int32_t> header{};
        header.resize(num_cols * k_dl_axis_count);
        auto* device_values = reinterpret_cast<cuFloatComplex*>(device_header + header.size());
        i64 values_offset{};
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const DeviceTensor& site = env_row[col];
            write_site_dims(header.data(), col, site.dim);
            CUDA_CHECK(cudaMemcpyAsync(
                device_values + values_offset,
                site.d,
                site.num_elems() * sizeof(cuFloatComplex),
                cudaMemcpyDeviceToDevice,
                la.stream()
            ));
            values_offset += static_cast<i64>(site.num_elems());
        }
        copy_h2d_async(device_header, header.data(), header.size(), la.stream());
        CUDA_CHECK(cudaStreamSynchronize(la.stream()));
        int fail_host{};
        CUDA_CHECK(cudaMemcpy(&fail_host, dl.fail, sizeof(int), cudaMemcpyDeviceToHost));
        if (fail_host != 0) set_err(QNPEPS_ERR_CUDA);
        if (row_log_out) *row_log_out = row_log;
    }
    return err_state();
}
}
