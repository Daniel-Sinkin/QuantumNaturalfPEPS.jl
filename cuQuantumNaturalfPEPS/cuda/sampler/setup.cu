#include "dlenv/build.cuh"
#include "qnpeps_ctx.cuh"
#include "sampler/draw.cuh"

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace qnpeps
{
static auto upload_tensor(cf* device_ptr, const HostTensor& host_tensor) -> void
{
    const auto bytes = static_cast<usize>(host_tensor.n()) * sizeof(cf);
    CUDA_CHECK(cudaMemcpy(device_ptr, host_tensor.v.data(), bytes, cudaMemcpyHostToDevice));
}

auto arena_upload(Carver& carver, const HostTensor& host_tensor) -> cf*
{
    auto* device_ptr = carver.take<cf>(static_cast<usize>(host_tensor.n()));
    upload_tensor(device_ptr, host_tensor);
    return device_ptr;
}

[[nodiscard]] static auto
carve_sampler_arena(qnpeps_ctx::SamplerState& state, const SamplerConfig& cfg, Carver carver)
    -> Carver
{
    Sampler& samp = state.samp;

    const auto dim_bond = static_cast<i64>(cfg.dim_bond);
    const auto dim_phys = static_cast<i64>(cfg.dim_phys);
    const auto chi_s = static_cast<i64>(cfg.chi_s);
    const auto chi_dl = static_cast<i64>(cfg.chi_dl);
    const auto chi_env_max = std::max(chi_s, dim_bond);
    const auto chi_aux_bond = chi_env_max * dim_bond;
    const auto max_reduced_n = std::max(chi_aux_bond, chi_dl * dim_bond * dim_bond);
    const auto max_reduced_m = std::max(chi_s * dim_phys * dim_bond, chi_aux_bond);
    const auto max_tmp_env = chi_s * chi_dl * chi_s * dim_phys * dim_bond;

    samp.max_env_above_site() = chi_aux_bond * chi_env_max;
    samp.max_ket_site() = chi_s * dim_phys * dim_bond * chi_s;
    samp.max_env_unsampled() = chi_s * chi_dl * chi_s;
    samp.max_reduce_input() = max_reduced_m * max_reduced_n;
    samp.max_rfactor() = chi_env_max * max_reduced_n;
    samp.max_sketch() = max_reduced_m * chi_env_max;
    samp.max_tmp() = std::max(max_tmp_env, samp.max_reduce_input());
    samp.max_sigma() = chi_s * chi_dl * chi_s;
    samp.max_sigma_full() = dim_phys * dim_phys * samp.max_sigma();
    samp.max_rho() = dim_phys * dim_phys;

    const auto dim_batch = static_cast<usize>(cfg.dim_batch);
    const auto num_rows = static_cast<usize>(cfg.lx);
    const auto num_cols = static_cast<usize>(cfg.ly);

    const auto peps_site_elems = [&](int row, int col) -> usize
    {
        const auto bond_left = static_cast<usize>(bond_dim(cfg.ly, col, cfg.dim_bond));
        const auto bond_down = static_cast<usize>(bond_dim(cfg.lx, row + 1, cfg.dim_bond));
        const auto bond_right = static_cast<usize>(bond_dim(cfg.ly, col + 1, cfg.dim_bond));
        const auto bond_up = static_cast<usize>(bond_dim(cfg.lx, row, cfg.dim_bond));
        return bond_left * bond_down * bond_right * bond_up * static_cast<usize>(cfg.dim_phys);
    };

    samp.mpo().assign(num_rows, std::vector<cf*>(num_cols, nullptr));
    for (auto row = 0; row < cfg.lx; ++row)
        for (auto col = 0; col < cfg.ly; ++col)
            samp.mpo()[static_cast<usize>(row)][static_cast<usize>(col)] =
                carver.take<cf>(peps_site_elems(row, col));
    samp.ket_row0().assign(num_cols, nullptr);
    for (auto col = 0; col < cfg.ly; ++col)
        samp.ket_row0()[static_cast<usize>(col)] = carver.take<cf>(peps_site_elems(0, col));
    state.unit = carver.take<cf>(1);

    const auto take_array = [&](i64 stride)
    {
        CuArray buffer{};
        buffer.stride = stride;
        buffer.p = carver.take<cf>(static_cast<usize>(stride) * dim_batch);
        return buffer;
    };

    // clang-format off
    samp.env_above()[0]       = take_array(samp.max_env_above_site() * cfg.ly);
    samp.env_above()[1]       = take_array(samp.max_env_above_site() * cfg.ly);
    samp.ket()                = take_array(samp.max_ket_site() * cfg.ly);
    samp.env_unsampled()      = take_array(samp.max_env_unsampled() * (cfg.ly + 1));
    samp.sigma()              = take_array(samp.max_sigma());
    samp.sigma_full()         = take_array(samp.max_sigma_full());
    samp.sigma_full_scratch() = take_array(samp.max_sigma_full());
    samp.rho()                = take_array(samp.max_rho());
    samp.rfactor()            = take_array(samp.max_rfactor());
    samp.tmp_a()              = take_array(samp.max_tmp());
    samp.tmp_b()              = take_array(samp.max_tmp());
    samp.reduce_input()       = take_array(samp.max_reduce_input());
    samp.sketch()             = take_array(samp.max_sketch());
    samp.proj()               = take_array(samp.max_rfactor());
    samp.rfactor_next()       = take_array(samp.max_rfactor());
    samp.gram()               = take_array(chi_env_max * chi_env_max);
    // clang-format on

    samp.gram_ptrs() = carver.take<cf*>(dim_batch);
    samp.sketch_ptrs() = carver.take<cf*>(dim_batch);
    samp.info() = carver.take<int>(dim_batch);
    samp.fail() = carver.take<int>(1);

    const auto ptr_capacity = static_cast<usize>(k_max_batch_size);
    const auto num_env = num_rows - 1;
    const usize ptr_slots{
        2_uz + (num_cols + 1) + num_env * num_cols + num_cols + 1 + 2 * num_env * num_cols
    };
    state.ptr_region = carver.take<cf*>(ptr_slots * ptr_capacity);

    samp.drawn_spin() = carver.take<int>(dim_batch);
    samp.row_spins() = carver.take<int>(dim_batch * num_cols);
    samp.logpc() = carver.take<f64>(dim_batch);
    samp.lognorm() = carver.take<f64>(dim_batch);
    samp.samples() = carver.take<u8>(dim_batch * num_rows * num_cols);
    state.device_seed = carver.take<u64>(1);

    return carver;
}

[[nodiscard]] static auto omega_region_bytes(const SamplerConfig& cfg) -> usize
{
    const auto num_rows = static_cast<usize>(cfg.lx);
    const auto num_cols = static_cast<usize>(cfg.ly);
    usize total{};

    std::map<std::pair<int, int>, char> seen;
    std::vector<int> bond_above{};
    bond_above.assign(num_cols + 1, 1);
    for (auto col = 0_uz; col <= num_cols; ++col)
        bond_above[col] = bond_dim(cfg.ly, static_cast<int>(col), cfg.dim_bond);

    for (auto row = 1_uz; row < num_rows; ++row)
    {
        std::vector<int> ket_bonds{};
        ket_bonds.assign(num_cols + 1, 1);
        int k{1};
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const int bond_down{bond_dim(cfg.lx, static_cast<int>(row) + 1, cfg.dim_bond)};
            const int bond_right{bond_dim(cfg.ly, static_cast<int>(col) + 1, cfg.dim_bond)};
            const int reduce_rows{k * cfg.dim_phys * bond_down};
            const auto reduce_cols = bond_above[col + 1] * bond_right;
            const int k_next{std::max(1, std::min({cfg.chi_s, reduce_rows, reduce_cols}))};
            ket_bonds[col + 1] = k_next;
            k = k_next;
        }
        ket_bonds[num_cols] = 1;
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const int bond_right{bond_dim(cfg.ly, static_cast<int>(col) + 1, cfg.dim_bond)};
            const auto omega_rows = bond_above[col + 1] * bond_right;
            const auto omega_cols = ket_bonds[col + 1];
            if (seen.emplace(std::make_pair(omega_rows, omega_cols), 0).second)
            {
                total += device_align(
                    static_cast<usize>(omega_rows) * static_cast<usize>(omega_cols) * sizeof(cf)
                );
            }
        }
        bond_above = ket_bonds;
    }
    return total;
}
namespace sampler
{
auto ctx_sampler_setup(qnpeps_ctx& ctx, const DlEnvView* dlenv, void* scratch, usize scratch_bytes)
    -> void
{
    if (ctx.sampler.allocated) return;
    if (not dlenv or not dlenv->values)
    {
        set_err(QNPEPS_ERR_INTERNAL);
        return;
    }

    Sampler& samp = ctx.sampler.samp;

    SamplerConfig cfg{};
    cfg.lx = ctx.cfg.lx;
    cfg.ly = ctx.cfg.ly;
    cfg.dim_phys = ctx.cfg.dim_phys;
    cfg.dim_bond = ctx.cfg.dim_bond;
    cfg.chi_s = ctx.cfg.chi_s;
    cfg.chi_dl = std::min(ctx.cfg.chi_dl, ctx.cfg.dim_bond * ctx.cfg.dim_bond);
    cfg.fast_mode = true;
    cfg.seed = ctx.cfg.seed;
    cfg.batch_base = 0;
    cfg.dim_batch = ctx.sampler.dim_batch;
    samp.cfg() = cfg;

    SamplerConfig capacity_cfg{cfg};
    capacity_cfg.dim_batch = k_max_batch_size;
    const auto measured = carve_sampler_arena(ctx.sampler, capacity_cfg, Carver{});
    const usize total{measured.total() + omega_region_bytes(capacity_cfg)};

    char* base{};
    if (scratch)
    {
        if (total > scratch_bytes)
        {
            set_err(QNPEPS_ERR_OOM);
            return;
        }
        base = static_cast<char*>(scratch);
        ctx.sampler.arena_owned = false;
    }
    else
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&base), total));
        if (not base)
        {
            set_err(QNPEPS_ERR_OOM);
            return;
        }
        ctx.sampler.arena_owned = true;
    }
    ctx.sampler.arena = base;
    ctx.sampler.arena_view = carve_sampler_arena(ctx.sampler, capacity_cfg, Carver{base, total});
    samp.bind(ctx.linalg, ctx.sampler.arena_view);

    const auto dim_batch = static_cast<usize>(k_max_batch_size);
    const auto num_rows = static_cast<usize>(cfg.lx);
    const auto num_cols = static_cast<usize>(cfg.ly);
    samp.mpo_host().resize(num_rows);
    for (auto row = 0_uz; row < num_rows; ++row)
    {
        samp.mpo_host()[row].resize(num_cols);
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            auto permuted = hpermute(ctx.host_peps[row][col], {0, 3, 4, 1, 2});
            upload_tensor(samp.mpo()[row][col], permuted);
            samp.mpo_host()[row][col] = std::move(permuted);
        }
    }

    samp.ket_row0_host().resize(num_cols);
    for (auto col = 0_uz; col < num_cols; ++col)
    {
        auto permuted = hpermute(ctx.host_peps[0][col], {0, 4, 1, 2, 3});
        upload_tensor(samp.ket_row0()[col], permuted);
        samp.ket_row0_host()[col] = std::move(permuted);
    }

    const auto num_dl_envs = num_rows - 1;
    samp.dlenv_host().resize(num_dl_envs);
    usize dims_offset{};
    for (auto row = 0_uz; row < num_dl_envs; ++row)
    {
        auto& env_row = samp.dlenv_host()[row];
        env_row.site.resize(num_cols);
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const int bond_left{dlenv->dims[dims_offset + k_dl_bond_left]};
            const int ket_dim{dlenv->dims[dims_offset + k_dl_ket]};
            const int bra_dim{dlenv->dims[dims_offset + k_dl_bra]};
            const int bond_right{dlenv->dims[dims_offset + k_dl_bond_right]};
            dims_offset += k_dl_axis_count;
            env_row.site[col] = HostTensor{{bond_left, ket_dim, bra_dim, bond_right}};
        }
    }

    dlenv::ensure_dlenv_views(ctx);
    if (err_state() != QNPEPS_OK) return;
    dlenv::materialize_dlenv_views(ctx, cf_cast(dlenv->values), ctx.dlenv.views[ctx.dlenv.active]);

    HostTensor unit{};
    unit.dim = {1, 1, 1, 1};
    unit.alloc();
    unit.v[0] = chost{1.0f, 0.0f};
    upload_tensor(ctx.sampler.unit, unit);

    {
        std::vector<cf*> host_ptrs{};
        host_ptrs.resize(dim_batch);

        for (auto lane = 0_uz; lane < dim_batch; ++lane)
            host_ptrs[lane] = samp.gram().p + static_cast<i64>(lane) * samp.gram().stride;
        CUDA_CHECK(cudaMemcpy(
            samp.gram_ptrs(), host_ptrs.data(), dim_batch * sizeof(cf*), cudaMemcpyHostToDevice
        ));

        for (auto lane = 0_uz; lane < dim_batch; ++lane)
            host_ptrs[lane] = samp.sketch().p + static_cast<i64>(lane) * samp.sketch().stride;
        CUDA_CHECK(cudaMemcpy(
            samp.sketch_ptrs(), host_ptrs.data(), dim_batch * sizeof(cf*), cudaMemcpyHostToDevice
        ));
    }
    {
        const auto cap = dim_batch;
        const auto num_env = num_rows - 1;
        const usize fixed_slots{2_uz + (num_cols + 1) + num_env * num_cols + num_cols + 1};
        auto* region = ctx.sampler.ptr_region;

        std::vector<cf*> host{};
        host.resize(fixed_slots * cap);
        usize s{};
        const auto place = [&](cf**& dst) -> usize
        {
            dst = region + s * cap;
            return s++;
        };
        const auto fill_lane = [&](usize slot, cf* slot_base, i64 stride)
        {
            for (auto lane = 0_uz; lane < cap; ++lane)
                host[slot * cap + lane] = slot_base + static_cast<i64>(lane) * stride;
        };
        const auto fill_broadcast = [&](usize slot, cf* slot_base)
        {
            for (auto lane = 0_uz; lane < cap; ++lane)
                host[slot * cap + lane] = slot_base;
        };

        fill_lane(place(samp.tmp_a_ptrs()), samp.tmp_a().p, samp.tmp_a().stride);
        fill_lane(place(samp.tmp_b_ptrs()), samp.tmp_b().p, samp.tmp_b().stride);
        samp.envu_ptrs().resize(num_cols + 1);
        for (auto off = 0_uz; off <= num_cols; ++off)
        {
            const auto base_off = static_cast<i64>(off) * samp.max_env_unsampled();
            fill_lane(
                place(samp.envu_ptrs()[off]),
                samp.env_unsampled().p + base_off,
                samp.env_unsampled().stride
            );
        }
        samp.mpo_ptrs().assign(num_rows, std::vector<cf**>(num_cols, nullptr));
        for (auto row = 1_uz; row < num_rows; ++row)
            for (auto col = 0_uz; col < num_cols; ++col)
                fill_broadcast(place(samp.mpo_ptrs()[row][col]), samp.mpo()[row][col]);
        samp.ket_row0_ptrs().resize(num_cols);
        for (auto col = 0_uz; col < num_cols; ++col)
            fill_broadcast(place(samp.ket_row0_ptrs()[col]), samp.ket_row0()[col]);
        fill_broadcast(place(samp.dl_unit_ptrs()), ctx.sampler.unit);

        CUDA_CHECK(
            cudaMemcpy(region, host.data(), fixed_slots * cap * sizeof(cf*), cudaMemcpyHostToDevice)
        );

        samp.dlenv_env_ptrs().assign(num_env, std::vector<cf**>(num_cols, nullptr));
        for (auto row = 0_uz; row < num_env; ++row)
            for (auto col = 0_uz; col < num_cols; ++col)
                place(samp.dlenv_env_ptrs()[row][col]);
        samp.dlenv_sigma_ptrs().assign(num_env, std::vector<cf**>(num_cols, nullptr));
        for (auto row = 0_uz; row < num_env; ++row)
            for (auto col = 0_uz; col < num_cols; ++col)
                place(samp.dlenv_sigma_ptrs()[row][col]);
    }

    const i64 lane_samples{static_cast<i64>(dim_batch) * cfg.lx * cfg.ly};
    const auto lane_samples_u = static_cast<usize>(lane_samples);
    CUDA_CHECK(
        cudaHostAlloc(&ctx.sampler.h_samples, lane_samples_u * sizeof(u8), cudaHostAllocDefault)
    );
    CUDA_CHECK(cudaHostAlloc(&ctx.sampler.h_logpc, dim_batch * sizeof(f64), cudaHostAllocDefault));
    CUDA_CHECK(
        cudaHostAlloc(&ctx.sampler.h_lognorm, dim_batch * sizeof(f64), cudaHostAllocDefault)
    );
    ctx.use_graph = true;

    ctx.sampler.allocated = true;
}

auto ctx_sampler_free(qnpeps_ctx& ctx) -> void
{
    for (auto& view : ctx.dlenv.views)
    {
        if (not view) continue;
        CUDA_NOCHECK(cudaFree(view));
        view = nullptr;
    }
    ctx.dlenv.views_allocated = false;
    if (ctx.sampler.allocated) ctx.sampler.samp.permutation_cache().release();
    if (ctx.sampler.arena_owned and ctx.sampler.arena) CUDA_NOCHECK(cudaFree(ctx.sampler.arena));
    if (ctx.sampler.h_samples) CUDA_NOCHECK(cudaFreeHost(ctx.sampler.h_samples));
    if (ctx.sampler.h_logpc) CUDA_NOCHECK(cudaFreeHost(ctx.sampler.h_logpc));
    if (ctx.sampler.h_lognorm) CUDA_NOCHECK(cudaFreeHost(ctx.sampler.h_lognorm));
    if (ctx.sampler.graph) CUDA_NOCHECK(cudaGraphExecDestroy(ctx.sampler.graph));

    ctx.sampler.arena = nullptr;
    ctx.sampler.h_samples = nullptr;
    ctx.sampler.h_logpc = nullptr;
    ctx.sampler.h_lognorm = nullptr;
    ctx.sampler.graph = nullptr;
    ctx.sampler.allocated = false;
}

auto sample_arena_bytes(const QnpepsConfig& config, int max_dim_batch) -> int64_t
{
    SamplerConfig cfg{};
    cfg.lx = config.lx;
    cfg.ly = config.ly;
    cfg.dim_phys = config.dim_phys;
    cfg.dim_bond = config.dim_bond;
    cfg.chi_s = config.chi_s;
    cfg.chi_dl = std::min(config.chi_dl, config.dim_bond * config.dim_bond);
    cfg.dim_batch = std::max(max_dim_batch, 1);
    qnpeps_ctx::SamplerState state{};
    const auto measured = carve_sampler_arena(state, cfg, Carver{});
    return static_cast<int64_t>(measured.total() + omega_region_bytes(cfg));
}
}
}
