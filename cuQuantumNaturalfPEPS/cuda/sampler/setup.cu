#include "dlenv/build.cuh"
#include "qnpeps_ctx.cuh"
#include "sampler/draw.cuh"

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace qnpeps
{
auto arena_upload(BumpArena& arena, const HostTensor& host_tensor) -> cf*
{
    const auto bytes = static_cast<usize>(host_tensor.n()) * sizeof(cf);
    auto* device_ptr = static_cast<cf*>(arena.bump(bytes));
    CUDA_CHECK(cudaMemcpy(device_ptr, host_tensor.v.data(), bytes, cudaMemcpyHostToDevice));
    return device_ptr;
}

static auto bump_device_buffer(BumpArena& arena, usize stride, usize dim_batch) -> CuArray
{
    CuArray buffer{};
    buffer.stride = static_cast<i64>(stride);
    buffer.p = static_cast<cf*>(arena.bump(stride * dim_batch * sizeof(cf)));
    return buffer;
}

[[nodiscard]] static auto arena_bytes(const SamplerConfig& cfg) -> usize
{
    const auto dim_bond = static_cast<i64>(cfg.dim_bond);
    const auto dim_phys = static_cast<i64>(cfg.dim_phys);

    const auto chi_s = static_cast<i64>(cfg.chi_s);
    const auto chi_dl = static_cast<i64>(cfg.chi_dl);
    const auto chi_env_max = std::max(chi_s, dim_bond);
    const auto chi_aux_bond = chi_env_max * dim_bond;

    const auto max_env_above_site = chi_aux_bond * chi_env_max;
    const auto max_ket_site = chi_s * dim_phys * dim_bond * chi_s;
    const auto max_env_unsampled = chi_s * chi_dl * chi_s;

    const auto max_reduced_n = std::max(chi_aux_bond, chi_dl * dim_bond * dim_bond);
    const auto max_reduced_m = std::max(chi_s * dim_phys * dim_bond, chi_aux_bond);

    const auto max_reduce_input = max_reduced_m * max_reduced_n;
    const auto max_rfactor = chi_env_max * max_reduced_n;
    const auto max_sketch = max_reduced_m * chi_env_max;
    const auto max_tmp_env = chi_s * chi_dl * chi_s * dim_phys * dim_bond;
    const auto max_tmp = std::max(max_tmp_env, max_reduce_input);

    const auto max_sigma = chi_s * chi_dl * chi_s;
    const auto max_sigma_full = dim_phys * dim_phys * max_sigma;
    const auto max_rho = dim_phys * dim_phys;

    const usize csz{sizeof(cf)};
    const auto dim_batch = static_cast<usize>(cfg.dim_batch);
    const auto num_rows = static_cast<usize>(cfg.lx);
    const auto num_cols = static_cast<usize>(cfg.ly);
    usize total{};
    auto add_buf = [&](i64 stride)
    { total += device_align(static_cast<usize>(stride) * dim_batch * csz); };

    add_buf(max_env_above_site * cfg.ly);
    add_buf(max_env_above_site * cfg.ly);
    add_buf(max_ket_site * cfg.ly);
    add_buf(max_env_unsampled * (cfg.ly + 1));
    add_buf(max_sigma);
    add_buf(max_sigma_full);
    add_buf(max_sigma_full);
    add_buf(max_rho);
    add_buf(max_rfactor);
    add_buf(max_tmp);
    add_buf(max_tmp);
    add_buf(max_reduce_input);
    add_buf(max_sketch);
    add_buf(max_rfactor);
    add_buf(max_rfactor);
    add_buf(chi_env_max * chi_env_max);

    total += device_align(dim_batch * sizeof(cf*));
    total += device_align(dim_batch * sizeof(cf*));
    total += device_align(dim_batch * sizeof(int));
    total += device_align(sizeof(int));
    total += device_align(dim_batch * sizeof(int));
    total += device_align(dim_batch * num_cols * sizeof(int));
    total += device_align(dim_batch * sizeof(f64));
    total += device_align(dim_batch * sizeof(f64));
    total += device_align(dim_batch * num_rows * num_cols * sizeof(u8));
    total += device_align(sizeof(u64));

    const auto peps_site_elems = [&](int row, int col) -> usize
    {
        const auto bond_left = static_cast<usize>(bond_dim(cfg.ly, col, cfg.dim_bond));
        const auto bond_down = static_cast<usize>(bond_dim(cfg.lx, row + 1, cfg.dim_bond));
        const auto bond_right = static_cast<usize>(bond_dim(cfg.ly, col + 1, cfg.dim_bond));
        const auto bond_up = static_cast<usize>(bond_dim(cfg.lx, row, cfg.dim_bond));
        return bond_left * bond_down * bond_right * bond_up * static_cast<usize>(dim_phys);
    };
    for (auto row = 0; row < cfg.lx; ++row)
        for (auto col = 0; col < cfg.ly; ++col)
            total += device_align(peps_site_elems(row, col) * csz);
    for (auto col = 0; col < cfg.ly; ++col)
        total += device_align(peps_site_elems(0, col) * csz);
    total += device_align(csz);

    {
        const auto ptr_capacity = static_cast<usize>(k_max_batch_size);
        const auto num_env = static_cast<usize>(cfg.lx - 1);
        const usize ptr_slots{
            2_uz + (num_cols + 1) + num_env * num_cols + num_cols + 1 + 2 * num_env * num_cols
        };
        total += device_align(ptr_slots * ptr_capacity * sizeof(cf*));
    }

    {
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
                        static_cast<usize>(omega_rows) * static_cast<usize>(omega_cols) * csz
                    );
                }
            }
            bond_above = ket_bonds;
        }
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

    const auto dim_bond = static_cast<i64>(cfg.dim_bond);
    const auto dim_phys = static_cast<i64>(cfg.dim_phys);

    const auto chi_s = static_cast<i64>(cfg.chi_s);
    const auto chi_dl = static_cast<i64>(cfg.chi_dl);
    const auto chi_env_max = std::max(chi_s, dim_bond);
    const auto chi_aux_bond = chi_env_max * dim_bond;

    samp.max_env_above_site() = chi_aux_bond * chi_env_max;

    samp.max_ket_site() = chi_s * dim_phys * dim_bond * chi_s;
    samp.max_env_unsampled() = chi_s * chi_dl * chi_s;

    const auto max_reduced_n = std::max(chi_aux_bond, chi_dl * dim_bond * dim_bond);
    const auto max_reduced_m = std::max(chi_s * dim_phys * dim_bond, chi_aux_bond);

    samp.max_reduce_input() = max_reduced_m * max_reduced_n;
    samp.max_rfactor() = chi_env_max * max_reduced_n;
    samp.max_sketch() = max_reduced_m * chi_env_max;

    const auto max_tmp_env = chi_s * chi_dl * chi_s * dim_phys * dim_bond;
    samp.max_tmp() = std::max(max_tmp_env, max_reduced_m * max_reduced_n);

    samp.max_sigma() = chi_s * chi_dl * chi_s;
    samp.max_sigma_full() = dim_phys * dim_phys * samp.max_sigma();

    samp.max_rho() = dim_phys * dim_phys;

    SamplerConfig capacity_cfg{cfg};
    capacity_cfg.dim_batch = k_max_batch_size;
    const usize total{arena_bytes(capacity_cfg)};

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
    ctx.sampler.arena_view = BumpArena{base, total, 0};
    samp.bind(ctx.linalg, ctx.sampler.arena_view);

    const auto dim_batch = static_cast<usize>(k_max_batch_size);
    const auto num_rows = static_cast<usize>(cfg.lx);
    const auto num_cols = static_cast<usize>(cfg.ly);
    samp.mpo().resize(num_rows);
    samp.mpo_host().resize(num_rows);
    for (auto row = 0_uz; row < num_rows; ++row)
    {
        samp.mpo()[row].resize(num_cols);
        samp.mpo_host()[row].resize(num_cols);
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            auto permuted = hpermute(ctx.host_peps[row][col], {0, 3, 4, 1, 2});
            samp.mpo()[row][col] = arena_upload(ctx.sampler.arena_view, permuted);
            samp.mpo_host()[row][col] = std::move(permuted);
        }
    }

    samp.ket_row0().resize(num_cols);
    samp.ket_row0_host().resize(num_cols);
    for (auto col = 0_uz; col < num_cols; ++col)
    {
        auto permuted = hpermute(ctx.host_peps[0][col], {0, 4, 1, 2, 3});
        samp.ket_row0()[col] = arena_upload(ctx.sampler.arena_view, permuted);
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
    ctx.sampler.unit = arena_upload(ctx.sampler.arena_view, unit);

    const auto allocb = [&](i64 stride)
    { return bump_device_buffer(ctx.sampler.arena_view, static_cast<usize>(stride), dim_batch); };

    // clang-format off
    samp.env_above()[0]       = allocb(samp.max_env_above_site() * cfg.ly);
    samp.env_above()[1]       = allocb(samp.max_env_above_site() * cfg.ly);
    samp.ket()                = allocb(samp.max_ket_site() * cfg.ly);
    samp.env_unsampled()      = allocb(samp.max_env_unsampled() * (cfg.ly + 1));
    samp.sigma()              = allocb(samp.max_sigma());
    samp.sigma_full()         = allocb(samp.max_sigma_full());
    samp.sigma_full_scratch() = allocb(samp.max_sigma_full());
    samp.rho()                = allocb(samp.max_rho());
    samp.rfactor()            = allocb(samp.max_rfactor());
    samp.tmp_a()              = allocb(samp.max_tmp());
    samp.tmp_b()              = allocb(samp.max_tmp());
    samp.reduce_input()       = allocb(samp.max_reduce_input());
    samp.sketch()             = allocb(samp.max_sketch());
    samp.proj()               = allocb(samp.max_rfactor());
    samp.rfactor_next()       = allocb(samp.max_rfactor());
    samp.gram()               = allocb(chi_env_max * chi_env_max);
    // clang-format on

    samp.gram_ptrs() = static_cast<cf**>(ctx.samp_bump(dim_batch * sizeof(cf*)));
    samp.sketch_ptrs() = static_cast<cf**>(ctx.samp_bump(dim_batch * sizeof(cf*)));
    samp.info() = static_cast<int*>(ctx.samp_bump(dim_batch * sizeof(int)));
    samp.fail() = static_cast<int*>(ctx.samp_bump(sizeof(int)));
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
        const usize dlenv_slots{2 * num_env * num_cols};
        const usize region_bytes{(fixed_slots + dlenv_slots) * cap * sizeof(cf*)};
        auto* region = static_cast<cf**>(ctx.samp_bump(region_bytes));

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
    {
        // clang-format off
        const usize drawn_spin_bytes = dim_batch * sizeof(int);
        const usize row_spins_bytes  = dim_batch * num_cols * sizeof(int);
        const usize logpc_bytes      = dim_batch * sizeof(f64);
        const usize lognorm_bytes    = dim_batch * sizeof(f64);
        const usize samples_bytes    = dim_batch * num_rows * num_cols * sizeof(u8);
        const usize seed_bytes       = sizeof(u64);
        samp.drawn_spin()       = static_cast<int*>(ctx.samp_bump(drawn_spin_bytes));
        samp.row_spins()        = static_cast<int*>(ctx.samp_bump(row_spins_bytes));
        samp.logpc()            = static_cast<f64*>(ctx.samp_bump(logpc_bytes));
        samp.lognorm()          = static_cast<f64*>(ctx.samp_bump(lognorm_bytes));
        samp.samples()          = static_cast<u8*>(ctx.samp_bump(samples_bytes));
        ctx.sampler.device_seed = static_cast<u64*>(ctx.samp_bump(seed_bytes));
        // clang-format on
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
    return static_cast<int64_t>(arena_bytes(cfg));
}
}
}
