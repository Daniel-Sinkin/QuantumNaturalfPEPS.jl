#include "parts.cuh"

namespace qn_eloc::env
{

namespace
{
constexpr usize k_persistent_wave_budget_divisor{2};

template <typename T>
auto upload_pool(RunCache& state, T*& device, const std::vector<T>& host) -> void
{
    if (host.empty()) return;
    cache_alloc(state, device, host.size());
    usize bytes{};
    arena_product({sizeof(T), host.size()}, bytes);
    check(cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice));
}

auto upload_term_pools(RunCache& state) -> void
{
    const auto& plan = state.plan;
    upload_pool(state, state.pools.diagonal, plan.diagonal);
    upload_pool(state, state.pools.diagonal_values, plan.diagonal_values);
    upload_pool(state, state.pools.active, plan.active);
    upload_pool(state, state.pools.coefficients, plan.coefficients);
    upload_pool(state, state.pools.weights, plan.weights);
    upload_pool(state, state.pools.targets, plan.targets);
}

[[nodiscard]] auto aligned_bytes(usize element_bytes, usize count) -> usize
{
    usize bytes{};
    usize aligned{};
    arena_product({element_bytes, count}, bytes);
    arena_align(bytes, aligned);
    return aligned;
}

template <typename T>
[[nodiscard]] auto pool_bytes(const std::vector<T>& host) -> usize
{
    return aligned_bytes(sizeof(T), host.size());
}

[[nodiscard]] auto pools_after_workers_bytes(const RunCache& state) -> usize
{
    const auto& plan = state.plan;
    auto bytes = 0_uz;
    arena_accumulate(bytes, pool_bytes(plan.diagonal));
    arena_accumulate(bytes, pool_bytes(plan.diagonal_values));
    arena_accumulate(bytes, pool_bytes(plan.active));
    arena_accumulate(bytes, pool_bytes(plan.coefficients));
    arena_accumulate(bytes, pool_bytes(plan.weights));
    arena_accumulate(bytes, pool_bytes(plan.targets));
    const auto original_slots = static_cast<usize>(state.orig_active.slots);
    const auto transposed_slots = static_cast<usize>(state.tr_active.slots);
    arena_accumulate(bytes, aligned_bytes(sizeof(CuActiveSlot), original_slots));
    arena_accumulate(bytes, aligned_bytes(sizeof(CuActiveSlot), transposed_slots));
    return bytes;
}

[[nodiscard]] auto measured_worker_bytes(
    Linalg& linalg, const Shape& shape, bool want_h, bool want_fb
) -> usize
{
    Worker probe{linalg};
    auto cursor = qnpeps::ArenaCursor::measuring();
    return carve_worker(probe, shape, want_h, want_fb, cursor);
}

[[nodiscard]] auto wave_bytes(const RunCache& state, int lanes) -> usize
{
    const auto original = shape_for(state.cfg, lanes, false);
    const auto original_bytes =
        measured_worker_bytes(state.w.la, original, state.orig_h, state.orig_fb);
    if (state.plan.transposed.empty()) return original_bytes;
    const auto transposed = shape_for(state.cfg, lanes, true);
    const auto transposed_bytes = measured_worker_bytes(state.wt.la, transposed, true, state.tr_fb);
    usize total{};
    arena_sum(original_bytes, transposed_bytes, total);
    return total;
}

[[nodiscard]] auto planned_wave_bytes(const RunCache& state, i64 n_samples, int lanes) -> usize
{
    const auto full_wave_bytes = wave_bytes(state, lanes);
    const auto last_lanes = static_cast<int>(n_samples % lanes);
    if (last_lanes == 0) return full_wave_bytes;
    usize total{};
    arena_sum(full_wave_bytes, wave_bytes(state, last_lanes), total);
    return total;
}

[[nodiscard]] auto wave_budget(const RunCache& state, const ContextArena& context_arena) -> usize
{
    const auto free_bytes = context_arena.is_persistent()
                                ? context_arena.remaining() / k_persistent_wave_budget_divisor
                                : context_arena.remaining();
    const auto reserved_bytes = pools_after_workers_bytes(state);
    return free_bytes > reserved_bytes ? free_bytes - reserved_bytes : 0_uz;
}

[[nodiscard]] auto density_batched_wave_lanes(
    const RunCache& state, i64 n_samples, const ContextArena& context_arena
) -> int
{
    const auto budget = wave_budget(state, context_arena);
    auto lanes = static_cast<int>(std::min<i64>(state.cfg.meo, n_samples));
    while (lanes > 1 and planned_wave_bytes(state, n_samples, lanes) > budget)
        lanes = (lanes + 1) / 2;
    return lanes;
}
}

auto run_cache_teardown(RunCache& state) -> void
{
    if (state.stream) unchecked(cudaStreamSynchronize(state.stream));
    worker_teardown(state.w);
    worker_teardown(state.wt);
    worker_teardown(state.last_wave);
    worker_teardown(state.last_wave_transposed);
    state.orig_active = ActiveLists{};
    state.tr_active = ActiveLists{};
    state.plan = TermPlan{};
    state.pools = DeviceTermPools{};
    state.peps_t = nullptr;
    state.samples_t = nullptr;
    state.value_buf = nullptr;
    state.logpsi_tr = nullptr;
    state.gscratch = nullptr;
    state.d_boff = nullptr;
    state.d_bslice = nullptr;
    state.spins_i32 = nullptr;
    state.context_arena = nullptr;
    state.ready = false;
}

auto run_cache_setup(
    RunCache& state,
    const QnpepsElocConfig& cfg,
    i64 n_samples,
    const QnpepsElocTermTable& tt,
    uint32_t flags,
    ContextArena& context_arena
) -> void
{
    state.cfg = cfg;
    state.n_samples = n_samples;
    state.flags = flags;
    state.stream = state.w.la.stream();
    state.context_arena = &context_arena;

    int sites_int{};
    i64 sample_sites{};
    arena_int({static_cast<std::uint64_t>(cfg.lx), static_cast<std::uint64_t>(cfg.ly)}, sites_int);
    arena_slot(
        {static_cast<std::uint64_t>(n_samples), static_cast<std::uint64_t>(sites_int)}, sample_sites
    );
    const auto sites = i64{sites_int};
    const auto osh = Shape{cfg.lx, cfg.ly, cfg.dim_phys, cfg.dim_bond, cfg.chi_eo, 1};
    state.compact_count = ok_layout(osh, state.ok_off, state.ok_slice);
    if (state.compact_count < 1) return;
    require(
        plan_terms(cfg, tt, state.plan),
        Status::bad_config,
        "local energy term table cannot be planned on a {}x{} lattice",
        cfg.lx,
        cfg.ly
    );
    active_lists_setup(state.orig_active, state.plan.original, cfg.meo, context_arena);
    active_lists_setup(state.tr_active, state.plan.transposed, cfg.meo, context_arena);

    if (not state.plan.transposed.empty())
    {
        const auto offt =
            std::vector<i64>{site_offsets(cfg.ly, cfg.lx, cfg.dim_bond, cfg.dim_phys)};
        if (offt.size() <= static_cast<usize>(sites)) return;
        cache_alloc(state, state.peps_t, static_cast<usize>(offt[static_cast<usize>(sites)]));
        cache_alloc(state, state.samples_t, static_cast<usize>(sample_sites));
    }

    cache_alloc(state, state.value_buf, static_cast<usize>(cfg.meo));
    usize logpsi_count{};
    arena_product({2u, static_cast<std::uint64_t>(cfg.meo)}, logpsi_count);
    cache_alloc(state, state.logpsi_tr, logpsi_count);
    const auto want_o = bool{(flags & QNPEPS_ELOC_CTX_O_ROWS) != 0};
    const auto want_gram = bool{(flags & QNPEPS_ELOC_CTX_GRAM) != 0};
    if (want_o) cache_alloc(state, state.gscratch, static_cast<usize>(cfg.meo));
    if (want_gram)
    {
        const auto n_blocks = int{static_cast<int>(sites)};
        cache_alloc(state, state.d_boff, static_cast<usize>(n_blocks));
        cache_alloc(state, state.d_bslice, static_cast<usize>(n_blocks));
        cache_alloc(state, state.spins_i32, static_cast<usize>(sample_sites));
        usize block_bytes{};
        arena_product({sizeof(int), static_cast<std::uint64_t>(n_blocks)}, block_bytes);
        auto hoff = std::vector<int>(static_cast<usize>(n_blocks));
        for (auto b = int{0}; b < n_blocks; ++b)
            hoff[static_cast<usize>(b)] = static_cast<int>(state.ok_off[static_cast<usize>(b)]);
        check(cudaMemcpy(state.d_boff, hoff.data(), block_bytes, cudaMemcpyHostToDevice));
        check(
            cudaMemcpy(state.d_bslice, state.ok_slice.data(), block_bytes, cudaMemcpyHostToDevice)
        );
    }

    for (const auto& piece : state.plan.original)
        (piece.bucket == Bucket::fourbody ? state.orig_fb : state.orig_hterms) = true;
    for (const auto& piece : state.plan.transposed)
        if (piece.bucket == Bucket::fourbody) state.tr_fb = true;
    state.orig_h = want_o or state.orig_hterms;
    if (is_density_batched_route(cfg.truncation_route))
        state.cfg.meo = density_batched_wave_lanes(state, n_samples, context_arena);
    const auto density = cfg.truncation_route == QNPEPS_TRUNCATION_DENSITY;
    const auto lanes = density ? 1 : static_cast<int>(std::min<i64>(state.cfg.meo, n_samples));
    const auto wave_width = density ? 1 : state.cfg.meo;
    const auto sh = shape_for(cfg, lanes, false);
    worker_setup(
        state.w, sh, state.orig_h, state.orig_fb, nullptr, nullptr, state.stream, context_arena
    );
    if (not state.plan.transposed.empty())
    {
        const auto sht = shape_for(cfg, lanes, true);
        worker_setup(
            state.wt, sht, true, state.tr_fb, nullptr, nullptr, state.stream, context_arena
        );
    }
    upload_term_pools(state);
    bind_device_pools(state.plan.original, state.pools);
    bind_device_pools(state.plan.transposed, state.pools);
    upload_active_slots(state.orig_active, state.plan.original, state.pools.active, context_arena);
    upload_active_slots(state.tr_active, state.plan.transposed, state.pools.active, context_arena);

    for (auto done = i64{0}; done < n_samples; done += wave_width)
    {
        const auto wave_lanes = int{static_cast<int>(std::min<i64>(wave_width, n_samples - done))};
        state.wave_ranges.push_back({done, wave_lanes});
    }
    state.ready = true;
}

}
