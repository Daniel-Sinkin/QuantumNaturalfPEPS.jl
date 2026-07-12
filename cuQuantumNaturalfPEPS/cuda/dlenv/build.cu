#include "cuda_utils.cuh"
#include "defer.cuh"
#include "dlenv/build.cuh"
#include "dtensor.cuh"
#include "linalg.cuh"
#include "qnpeps_ctx.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <map>
#include <random>
#include <vector>

namespace qnpeps::dlenv
{
using PepsRow = std::vector<DeviceTensor>;
using DlEnvRow = std::vector<DeviceTensor>;

__global__ auto cu_dl_set_one(cuFloatComplex* p) -> void
{
    p->x = 1.0f;
    p->y = 0.0f;
}

static auto init_dl_units(Linalg& la, cf* triv, cf* scalar_r) -> void
{
    cu_dl_set_one<<<1, 1, 0, la.stream()>>>(cu_cast(triv));
    CUDA_CHECK(cudaGetLastError());
    cu_dl_set_one<<<1, 1, 0, la.stream()>>>(cu_cast(scalar_r));
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
    const cuFloatComplex* src_base,
    i64& src_off,
    cuFloatComplex* packed_base,
    i64& packed_off,
    std::vector<DeviceTensor>& out_row
) -> void
{
    const auto reversed = Permutation::reverse(5);
    for (auto col = 0; col < dims.ly; ++col)
    {
        const auto src_shape = peps_site_shape(dims, row_up, row_down, col);
        const auto nsite = static_cast<i64>(src_shape.num_elems());
        const DeviceTensor src{src_shape, const_cast<cuFloatComplex*>(src_base + src_off)};
        permute_axes(src, reversed, false, packed_base + packed_off);
        out_row[static_cast<usize>(col)] =
            DeviceTensor{reversed.apply(src_shape), packed_base + packed_off};
        src_off += nsite;
        packed_off += nsite;
    }
}

auto rf_omega(qnpeps_ctx& ctx, int n, int k) -> cf*
{
    const auto key = static_cast<i64>(n) * 1000000 + k;
    auto it = ctx.dl.rf_omega.find(key);
    if (it != ctx.dl.rf_omega.end()) return it->second;
    std::normal_distribution<f32> gauss(0.0f, 1.0f);
    std::vector<cf> host{};
    host.resize(static_cast<usize>(n) * static_cast<usize>(k));
    for (auto& z : host)
        z = cf{gauss(ctx.dl.rf_rng), gauss(ctx.dl.rf_rng)};
    cf* d{};
    CUDA_CHECK(cudaMalloc(&d, host.size() * sizeof(cf)));
    CUDA_CHECK(cudaMemcpy(d, host.data(), host.size() * sizeof(cf), cudaMemcpyHostToDevice));
    ctx.dl.rf_omega[key] = d;
    return d;
}

auto set_dl_capturing(qnpeps_ctx& ctx, bool on) -> void
{
    ctx.dl.capturing = on;
}

auto ensure_dlenv_views(qnpeps_ctx& ctx) -> void
{
    if (ctx.dlenv.views_allocated) return;
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
    ctx.dlenv.views_elems = 2 * total_env;
    if (not ctx.dlenv.views[0])
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&ctx.dlenv.views[0]),
            static_cast<usize>(ctx.dlenv.views_elems) * sizeof(cf)
        ));
    if (err_state() != QNPEPS_OK) return;
    if (not ctx.dlenv.views[1])
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&ctx.dlenv.views[1]),
            static_cast<usize>(ctx.dlenv.views_elems) * sizeof(cf)
        ));
    if (err_state() != QNPEPS_OK) return;
    ctx.dlenv.views_allocated = true;
}

auto materialize_dlenv_views(qnpeps_ctx& ctx, const cf* raw_values, cf* views_out) -> void
{
    const auto num_env_rows = static_cast<usize>(ctx.cfg.lx - 1);
    const auto num_cols = static_cast<usize>(ctx.cfg.ly);
    set_stream(ctx.linalg.stream());
    const auto* base = cu_cast(raw_values);
    i64 raw_off{};
    for (auto row = 0_uz; row < num_env_rows; ++row)
    {
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const auto d = read_site_dims(ctx.dlenv.dims.data(), row * num_cols + col);
            DeviceTensor site{
                {d.bond_left, d.ket, d.bra, d.bond_right},
                const_cast<cuFloatComplex*>(base + raw_off)
            };
            permute_axes(
                site, {1, 3, 2, 0}, false, cu_cast(views_out + ctx.dlenv.env_off[row][col])
            );
            permute_axes(
                site, {1, 0, 2, 3}, false, cu_cast(views_out + ctx.dlenv.sigma_off[row][col])
            );
            raw_off += static_cast<i64>(site.num_elems());
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

static auto
rangefinder(qnpeps_ctx& ctx, Linalg& la, const Arenas& ar, const DlRangefinderArgs& args) -> void
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
    const CuMatrix m_ac{cf_cast(input.d), rows, cols};

    auto Y = alloc(ar.scratch, {rows, rank});
    DEFER([&] { free(Y); });
    auto Z = alloc(ar.scratch, {cols, rank});
    DEFER([&] { free(Z); });
    auto* omega = rf_omega(ctx, cols, rank);
    const CuMatrix m_omega{omega, cols, rank};
    const CuMatrix m_y{cf_cast(Y.d), rows, rank};
    const CuMatrix m_z{cf_cast(Z.d), cols, rank};
    la.matmul(m_ac, m_omega, m_y);
    for (auto it = 0; it < 2; ++it)
    {
        la.matmul_adj_none(m_ac, m_y, m_z);
        la.matmul(m_ac, m_z, m_y);
    }

    const auto qr_layout = qr_scratch(la, m_y);
    void* qr_scratch_mem{ar.scratch.take<char>(qr_layout.total())};
    qr(la, m_y, qr_scratch_mem, qr_layout);
    if (fail_flag)
    {
        const auto* qr_status = byte_offset<int>(qr_scratch_mem, qr_layout.reflector_bytes);
        cu_or_status<<<1, 1, 0, stream()>>>(qr_status, fail_flag);
        CUDA_CHECK(cudaGetLastError());
    }

    q = alloc(ar.known, {rows, rank});
    CUDA_CHECK(cudaMemcpyAsync(
        q.d,
        Y.d,
        static_cast<usize>(rows) * static_cast<usize>(rank) * sizeof(cuFloatComplex),
        cudaMemcpyDeviceToDevice,
        la.stream()
    ));
    r = alloc(ar.scratch, {rank, cols});
    la.matmul_adj_none(m_y, m_ac, CuMatrix{cf_cast(r.d), rank, cols});
}

__global__ auto
cu_absmax_inverse(const cuFloatComplex* r, i64 n, f32* device_inverse_scale, f64* device_scale)
    -> void
{
    __shared__ f64 sh[k_tree_reduce_threads];
    f64 local{0.0};

    // There are k_tree_reduce_threads many threads, the first (threadidx.x = 0) thread
    // is responsible for r[0], r[256], r[512], ...
    // and locally computes
    //   local = 0.0;
    //   local = max(local, r[0]), local = max(local, r[256]), ...
    // so that at the end local = max({256 * i : 0 <= i, 256 * i  < n}) and that value is
    // writte into sh[0] <- shared memory visible to all threads
    // Similiarly for the jth thread (threadidx.x = j)
    // it computes max({j + 256 * i : 0 <= i, j + 256 * i< n}) and that value is
    // written into sh[j]
    for (auto i = threadIdx.x; i < n; i += blockDim.x)
    {
        // i = threadIdx.x + iteration * blockDim.x
        // i = j, j + 256, j + 512, ...
        const f64 abs{fabs(static_cast<f64>(r[i].x)) + fabs(static_cast<f64>(r[i].y))};
        if (abs > local) local = abs;
    }
    sh[threadIdx.x] = local;  // Recall that all 256 threads are runnign in parallel
    __syncthreads();          // Blocks until all threads have reached this point

    // Tree Reduce the 256 values into one
    // [0, 1, ..., 127, 128, ..., 255], 256 = blockDim.x
    // s = blockDim.x / 2 = 128, threadIdx.x < s makes sure only the first half of the
    // threads do any work in the first iteration. The jth thread (0 <= j <= 127) is
    // responsible for the values sh[j], sh[j + s] = sh[j + 128], meaning the threads look at
    // {0, 128}, {1, 129}, {2, 130}, ..., {127, 255} and check if the right value is larger.

    // Then they copy it over into the left value:
    // r[0] = max(r[0], r[128]) <- thread 0
    // r[1] = max(r[1], r[128]) <- thread 1
    // ...
    // r[127] = max(r[127], r[255]) <- thread 127

    // Recall that

    for (auto s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s and sh[threadIdx.x + s] > sh[threadIdx.x])
        {
            sh[threadIdx.x] = sh[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        const f64 s{sh[0]};
        *device_scale = s;
        const bool ok{(s > 0.0) and isfinite(s)};
        *device_inverse_scale = ok ? static_cast<f32>(1.0 / s) : 1.0f;
    }
}

__global__ auto cu_scale_inv(cuFloatComplex* r, i64 n, const f32* device_inverse_scale) -> void
{
    const f32 inv{*device_inverse_scale};
    for (i64 i{global_lane()}; i < n; i += grid_stride())
    {
        r[i].x *= inv;
        r[i].y *= inv;
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

static auto
build_env_row(qnpeps_ctx& ctx, Linalg& la, const Arenas& ar, const BuildEnvRowArgs& args)
    -> std::vector<DeviceTensor>
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

    f64* device_scales = ctx.dl.scales_all + static_cast<usize>(build_step) * num_cols;

    auto carried_r = view(cu_cast(ctx.dl.scalar_r), {1, 1, 1, 1});
    auto triv = view(cu_cast(ctx.dl.triv), {1, 1, 1, 1});

    for (auto col = 0_uz; col < num_cols; ++col)
    {
        if (err_state() != QNPEPS_OK) return out;
        Carver col_scratch{ar.scratch};
        const Arenas col_ar{ar.known, ar.rolling_r, col_scratch};

        const DeviceTensor& ket = row_ket[col];
        const DeviceTensor& env = env_below ? (*env_below)[col] : triv;

        auto r_env = contract(col_scratch, la, carried_r, {3}, env, {0}, {});
        DEFER([&] { free(r_env); });
        auto r_env_ket = contract(col_scratch, la, r_env, {1, 3}, ket, {4, 3}, {});
        DEFER([&] { free(r_env_ket); });
        auto rab =
            contract(col_scratch, la, r_env_ket, {1, 2, 4}, ket, {4, 3, 0}, {.conj_b = true});
        DEFER([&] { free(rab); });

        auto rab_mat = permute_axes(col_scratch, rab, {0, 2, 4, 1, 3, 5}, false);
        DEFER([&] { free(rab_mat); });
        const int bond_left{rab.dim[0]};
        const int bond_right{rab.dim[1]};
        const int ket_vertical{rab.dim[2]};
        const int ket_horizontal{rab.dim[3]};
        const int bra_vertical{rab.dim[4]};
        const int bra_horizontal{rab.dim[5]};
        const int rows{bond_left * ket_vertical * bra_vertical};
        const int cols{bond_right * ket_horizontal * bra_horizontal};

        DeviceTensor q{};
        DeviceTensor r_factor{};
        int rank{};
        rangefinder(
            ctx,
            la,
            col_ar,
            {
                .input = rab_mat,
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

            auto* device_inverse_scale = col_scratch.take<f32>(1);
            cu_absmax_inverse<<<1, k_tree_reduce_threads, 0, stream()>>>(
                r_factor.d, r_factor_elems, device_inverse_scale, device_scales + col
            );
            CUDA_CHECK(cudaGetLastError());

            const auto blocks = static_cast<u32>(ceil_div(r_factor_elems, k_tree_reduce_threads));
            cu_scale_inv<<<blocks, k_tree_reduce_threads, 0, stream()>>>(
                r_factor.d, r_factor_elems, device_inverse_scale
            );
            CUDA_CHECK(cudaGetLastError());
        }

        out[col] = DeviceTensor{{bond_left, ket_vertical, bra_vertical, rank}, q.d};

        auto r_reshaped = view(r_factor.d, {rank, bond_right, ket_horizontal, bra_horizontal});
        ar.rolling_r.offset() = 0;
        carried_r = permute_axes(ar.rolling_r, r_reshaped, {0, 2, 3, 1}, false);
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
            const auto s = scales[col];
            if (not std::isfinite(s))
            {
                set_err(QNPEPS_ERR_INTERNAL);
                break;
            }
            if (s > 0.0) row_log_scale += std::log(s);
        }
    }

    DeviceTensor& last = out[num_cols - 1];
    {
        auto folded = contract(ar.known, la, last, {3}, carried_r, {0}, {});
        last = DeviceTensor{{folded.dim[0], folded.dim[1], folded.dim[2], 1}, folded.d};
    }
    return out;
}

static auto build_env_rows(
    qnpeps_ctx& ctx,
    Linalg& la,
    const Arenas& ar,
    const Dims& dims,
    const std::vector<PepsRow>& peps,
    int maxdim,
    int* fail_flag
) -> std::vector<DlEnvRow>
{
    set_stream(la.stream());

    const auto num_env_rows = static_cast<usize>(dims.lx - 1);
    std::vector<DlEnvRow> env_rows{};
    env_rows.resize(num_env_rows);

    f64 ignored{};
    const auto last_env = num_env_rows - 1;
    env_rows[last_env] = build_env_row(
        ctx,
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
            ctx,
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
    const auto num_envs = static_cast<usize>(dims.lx - 1);
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
    const auto known = num_envs * (num_cols * out_slot + per_row);

    const auto rolling_r = out_slot;

    const auto site_peak = chi2 * dim_bond2 * dim_bond2;
    const auto scratch_elems = (4 + 3 * dim_phys) * site_peak + 4 * chi2 * dim_bond2;
    auto scratch = device_align(sizeof(cuFloatComplex) * scratch_elems);

    const auto qr_rows = static_cast<int>(chi * dim_bond2);
    const auto qr_cols = static_cast<int>(chi);
    scratch += qr_scratch(la, qr_rows, qr_cols).total();

    scratch += arena_tail_pad;

    return DlSizes{known, rolling_r, scratch};
}

static auto
carve_dl_arena(qnpeps_ctx::DlBuild& dl, const DlSizes& sz, usize scales_count, Carver carver)
    -> Carver
{
    dl.fail = carver.take<int>(1);
    dl.scales_all = carver.take<f64>(scales_count);
    dl.triv = carver.take<cf>(1);
    dl.scalar_r = carver.take<cf>(1);
    dl.known = Carver{carver.take<char>(sz.known), sz.known};
    dl.rolling_r = Carver{carver.take<char>(sz.rolling_r), sz.rolling_r};
    dl.scratch = Carver{carver.take<char>(sz.scratch), sz.scratch};
    return carver;
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
    const auto measured = carve_dl_arena(ctx.dl, sz, scales_count, Carver{});

    if (not ctx.dl.arena)
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx.dl.arena), measured.total()));
    if (err_state() != QNPEPS_OK) return;

    carve_dl_arena(ctx.dl, sz, scales_count, Carver{ctx.dl.arena, measured.total()});

    const auto dlenv_bytes = qnpeps_dlenv_bytes(&cfg);
    if (not ctx.dlenv.buf[0])
        CUDA_CHECK(cudaMalloc(&ctx.dlenv.buf[0], static_cast<usize>(dlenv_bytes)));
    if (not ctx.dlenv.buf[1])
        CUDA_CHECK(cudaMalloc(&ctx.dlenv.buf[1], static_cast<usize>(dlenv_bytes)));
    if (err_state() != QNPEPS_OK) return;

    init_dl_units(la, ctx.dl.triv, ctx.dl.scalar_r);

    ctx.dl.allocated = true;
}

auto dl_free(qnpeps_ctx& ctx) -> void
{
    if (ctx.dl.peps_buf)
    {
        CUDA_NOCHECK(cudaFree(ctx.dl.peps_buf));
        ctx.dl.peps_buf = nullptr;
    }
    if (ctx.dl.arena)
    {
        CUDA_NOCHECK(cudaFree(ctx.dl.arena));
        ctx.dl.arena = nullptr;
    }
    for (auto& buf : ctx.dlenv.buf)
    {
        if (buf)
        {
            CUDA_NOCHECK(cudaFree(buf));
            buf = nullptr;
        }
    }
    for (auto& entry : ctx.dl.rf_omega)
    {
        if (entry.second) CUDA_NOCHECK(cudaFree(entry.second));
    }
    ctx.dl.rf_omega.clear();
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
    Linalg& la = ctx.linalg;
    const auto& cfg = ctx.cfg;
    const int lx{cfg.lx};
    const int ly{cfg.ly};
    const auto dim_bond = cfg.dim_bond;
    const auto dim_phys = cfg.dim_phys;
    const auto num_rows = static_cast<usize>(lx);
    const auto num_cols = static_cast<usize>(ly);

    set_stream(la.stream());

    dl_ensure_allocated(ctx, la);
    if (err_state() != QNPEPS_OK) return err_state();

    const Dims dims{lx, ly, dim_phys, dim_bond};

    std::vector<PepsRow> device_peps_grid{};
    device_peps_grid.resize(num_rows);

    const auto peps_base = reinterpret_cast<const cuFloatComplex*>(device_peps);
    i64 off{};
    i64 boff{};
    for (auto row = 0; row < lx; ++row)
    {
        const auto row_u = static_cast<usize>(row);
        device_peps_grid[row_u].resize(num_cols);
        pack_peps_row(
            dims,
            row,
            row + 1,
            peps_base,
            off,
            cu_cast(ctx.dl.peps_buf),
            boff,
            device_peps_grid[row_u]
        );
    }

    ctx.dl.known.offset() = 0;
    ctx.dl.rolling_r.offset() = 0;
    ctx.dl.scratch.offset() = 0;
    CUDA_CHECK(cudaMemsetAsync(ctx.dl.fail, 0, sizeof(int), la.stream()));

    const int chi_c{std::min(cfg.chi_dl, dim_bond * dim_bond)};
    const Arenas ar{ctx.dl.known, ctx.dl.rolling_r, ctx.dl.scratch};

    const int target{ctx.dlenv.build_count % 2};
    const auto nsites = (num_rows - 1) * num_cols;
    auto* header_ptr = reinterpret_cast<int32_t*>(ctx.dlenv.buf[target]);
    auto* values_ptr = reinterpret_cast<cuFloatComplex*>(header_ptr + nsites * k_dl_axis_count);

    const auto build_region = [&]
    {
        auto env_rows = build_env_rows(ctx, la, ar, dims, device_peps_grid, chi_c, ctx.dl.fail);
        if (err_state() != QNPEPS_OK) return;

        if (not ctx.dlenv.header_written[target])
        {
            ctx.dlenv.dims.resize(nsites * k_dl_axis_count);
            for (auto row = 0_uz; row < num_rows - 1; ++row)
                for (auto col = 0_uz; col < num_cols; ++col)
                    write_site_dims(
                        ctx.dlenv.dims.data(), row * num_cols + col, env_rows[row][col].dim
                    );
            CUDA_CHECK(cudaMemcpyAsync(
                header_ptr,
                ctx.dlenv.dims.data(),
                ctx.dlenv.dims.size() * sizeof(int32_t),
                cudaMemcpyHostToDevice,
                la.stream()
            ));
            ctx.dlenv.header_written[target] = true;
        }

        i64 values_offset{};
        for (auto row = 0_uz; row < num_rows - 1; ++row)
        {
            for (auto col = 0_uz; col < num_cols; ++col)
            {
                const DeviceTensor& site = env_rows[row][col];
                CUDA_CHECK(cudaMemcpyAsync(
                    values_ptr + values_offset,
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
            std::fprintf(stderr, "[qnpeps] dl_graph replayed buf=%d\n", target);
        CUDA_CHECK(cudaGraphLaunch(ctx.dlenv.graph[target], la.stream()));
    }
    else if (ctx.use_graph and ctx.dl.warmed)
    {
        cudaGraph_t graph{};
        set_dl_capturing(ctx, true);
        CUDA_CHECK(cudaStreamBeginCapture(la.stream(), cudaStreamCaptureModeThreadLocal));
        build_region();
        const cudaError_t cap_rc = cudaStreamEndCapture(la.stream(), &graph);
        set_dl_capturing(ctx, false);
        if (cap_rc == cudaSuccess and err_state() == QNPEPS_OK
            and instantiate_graph(ctx.dlenv.graph[target], graph) == cudaSuccess)
        {
            CUDA_CHECK(cudaGraphDestroy(graph));
            if (std::getenv("QNPEPS_GRAPH_LOG"))
                std::fprintf(stderr, "[qnpeps] dl_graph captured buf=%d\n", target);
            CUDA_CHECK(cudaGraphLaunch(ctx.dlenv.graph[target], la.stream()));
        }
        else
        {
            cudaGetLastError();
            ctx.dlenv.graph[target] = nullptr;
            if (graph) CUDA_NOCHECK(cudaGraphDestroy(graph));
            ctx.dl.known.offset() = 0;
            ctx.dl.rolling_r.offset() = 0;
            ctx.dl.scratch.offset() = 0;
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

    if (ctx.sampler.allocated)
    {
        ensure_dlenv_views(ctx);
        if (err_state() != QNPEPS_OK) return err_state();
        materialize_dlenv_views(ctx, cf_cast(values_ptr), ctx.dlenv.views[target]);
    }

    ctx.dlenv.valid[target] = true;
    ctx.dlenv.active = target;
    ++ctx.dlenv.build_count;
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
    set_stream(la.stream());

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
        const auto peps_base = reinterpret_cast<const cuFloatComplex*>(device_peps_row);
        i64 off{};
        i64 boff{};
        pack_peps_row(dims, row - 1, row, peps_base, off, peps_buf, boff, row_ket);
    }

    std::vector<DeviceTensor> env_below{};
    if (device_env_below)
    {
        env_below.resize(num_cols);
        std::vector<int32_t> below_dims{};
        below_dims.resize(num_cols * k_dl_axis_count);
        const usize hb{below_dims.size() * sizeof(int32_t)};
        CUDA_CHECK(cudaMemcpy(below_dims.data(), device_env_below, hb, cudaMemcpyDeviceToHost));
        const auto values_base = byte_offset<cuFloatComplex>(device_env_below, hb);
        i64 values_offset{};
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const auto d = read_site_dims(below_dims.data(), col);
            env_below[col] = DeviceTensor{
                {d.bond_left, d.ket, d.bra, d.bond_right},
                const_cast<cuFloatComplex*>(values_base + values_offset)
            };
            values_offset += d.num_elems();
        }
    }

    const int chi_dl{std::min(maxdim, dim_bond * dim_bond)};
    const DlSizes sz = plan_dl_arena(la, dims, chi_dl);
    char* arena_base{};
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&arena_base), sz.total()));
    if (err_state() != QNPEPS_OK) return err_state();
    DEFER([&] { CUDA_NOCHECK(cudaFree(arena_base)); });
    Carver known{arena_base, sz.known};
    Carver rolling_r{arena_base + sz.known, sz.rolling_r};
    Carver scratch{arena_base + sz.known + sz.rolling_r, sz.scratch};
    const Arenas ar{known, rolling_r, scratch};

    qnpeps_ctx tmp{};
    DEFER([&] { dl_free(tmp); });
    tmp.cfg = cfg;
    tmp.dl.fail = known.take<int>(1);
    tmp.dl.scales_all = known.take<f64>(num_cols);
    tmp.dl.triv = known.take<cf>(1);
    tmp.dl.scalar_r = known.take<cf>(1);
    init_dl_units(la, tmp.dl.triv, tmp.dl.scalar_r);
    CUDA_CHECK(cudaMemsetAsync(tmp.dl.fail, 0, sizeof(int), la.stream()));

    f64 row_log{0.0};
    auto env_row = build_env_row(
        tmp,
        la,
        ar,
        {
            .dims = dims,
            .row_ket = &row_ket,
            .env_below = device_env_below ? &env_below : nullptr,
            .maxdim = chi_dl,
            .row_log_scale = &row_log,
            .fail_flag = tmp.dl.fail,
            .build_step = 0,
            .defer_scales = false,
        }
    );
    if (err_state() != QNPEPS_OK) return err_state();

    {  // Copy finished dlenv out
        auto* header_ptr = reinterpret_cast<int32_t*>(dlenv_row_out);
        std::vector<int32_t> header{};
        header.resize(num_cols);
        auto* values_ptr = reinterpret_cast<cuFloatComplex*>(header_ptr + header.size());
        i64 values_offset{};
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const DeviceTensor& site = env_row[col];
            write_site_dims(header.data(), col, site.dim);
            CUDA_CHECK(cudaMemcpyAsync(
                values_ptr + values_offset,
                site.d,
                site.num_elems() * sizeof(cuFloatComplex),
                cudaMemcpyDeviceToDevice,
                la.stream()
            ));
            values_offset += static_cast<i64>(site.num_elems());
        }
        CUDA_CHECK(cudaMemcpyAsync(
            header_ptr,
            header.data(),
            header.size() * sizeof(int32_t),
            cudaMemcpyHostToDevice,
            la.stream()
        ));
        CUDA_CHECK(cudaStreamSynchronize(la.stream()));
        int fail_host{};
        CUDA_CHECK(cudaMemcpy(&fail_host, tmp.dl.fail, sizeof(int), cudaMemcpyDeviceToHost));
        if (fail_host != 0) set_err(QNPEPS_ERR_CUDA);
        if (row_log_out) *row_log_out = row_log;
    }
    return err_state();
}
}
