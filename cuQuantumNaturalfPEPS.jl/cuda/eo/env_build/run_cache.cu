#include "parts.cuh"

namespace qn_eloc::env
{

auto run_cache_teardown(RunCache& state) -> void
{
    if (state.stream) CUDA_NOCHECK(cudaStreamSynchronize(state.stream));
    worker_teardown(state.w);
    worker_teardown(state.wt);
    active_lists_teardown(state.orig_active);
    active_lists_teardown(state.tr_active);
    state.peps_t = nullptr;
    state.samples_t = nullptr;
    state.d_a = nullptr;
    state.d_b = nullptr;
    state.d_j = nullptr;
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
    if (tt.n_diag > 0) state.diag_terms.assign(tt.diag, tt.diag + tt.n_diag);
    if (tt.n_flip > 0) state.flip_terms.assign(tt.flip, tt.flip + tt.n_flip);
    state.term_table.n_diag = static_cast<int32_t>(state.diag_terms.size());
    state.term_table.diag = state.diag_terms.empty() ? nullptr : state.diag_terms.data();
    state.term_table.n_flip = static_cast<int32_t>(state.flip_terms.size());
    state.term_table.flip = state.flip_terms.empty() ? nullptr : state.flip_terms.data();

    int sites_int{};
    i64 sample_sites{};
    if (not arena_int(
            {static_cast<std::uint64_t>(cfg.lx), static_cast<std::uint64_t>(cfg.ly)}, sites_int
        )
        or not arena_slot(
            {static_cast<std::uint64_t>(n_samples), static_cast<std::uint64_t>(sites_int)},
            sample_sites
        ))
        return;
    const auto sites = i64{sites_int};
    const auto osh = Shape{cfg.lx, cfg.ly, cfg.dim_phys, cfg.dim_bond, cfg.chi_eo, 1};
    state.compact_count = ok_layout(osh, state.ok_off, state.ok_slice);
    if (qn::err_state() != QNPEPS_ELOC_OK or state.compact_count < 1) return;
    for (const QnpepsElocFlipTerm& term : state.flip_terms)
    {
        FlipInst fi{};
        int pass{};
        if (not classify_route(cfg, term, fi, pass))
        {
            qn::set_err(QNPEPS_ELOC_ERR_BAD_CONFIG);
            return;
        }
        (pass == 0 ? state.orig_terms : state.tr_terms).push_back(fi);
    }
    state.active_compact = active_compact_requested_from_env();
    state.active_compact_min = active_compact_min_from_env();
    const auto orig_active_slots = int{assign_active_slots(state.orig_terms, state.active_compact)};
    const auto tr_active_slots = int{assign_active_slots(state.tr_terms, state.active_compact)};
    active_lists_setup(
        state.orig_active, state.orig_terms, orig_active_slots, cfg.meo, context_arena
    );
    active_lists_setup(state.tr_active, state.tr_terms, tr_active_slots, cfg.meo, context_arena);
    if (qn::err_state() != QNPEPS_ELOC_OK) return;

    if (not state.tr_terms.empty())
    {
        const auto offt =
            std::vector<i64>{site_offsets(cfg.ly, cfg.lx, cfg.dim_bond, cfg.dim_phys)};
        if (qn::err_state() != QNPEPS_ELOC_OK or offt.size() <= static_cast<usize>(sites)) return;
        cache_alloc(state, state.peps_t, static_cast<usize>(offt[static_cast<usize>(sites)]));
        cache_alloc(state, state.samples_t, static_cast<usize>(sample_sites));
    }

    if (not state.diag_terms.empty())
    {
        auto ha = std::vector<int>(state.diag_terms.size());
        auto hb = std::vector<int>(state.diag_terms.size());
        auto hj = std::vector<f64>(state.diag_terms.size());
        for (auto i = usize{0}; i < state.diag_terms.size(); ++i)
        {
            ha[i] = state.diag_terms[i].site_a;
            hb[i] = state.diag_terms[i].site_b;
            hj[i] = state.diag_terms[i].coeff;
        }
        cache_alloc(state, state.d_a, ha.size());
        cache_alloc(state, state.d_b, hb.size());
        cache_alloc(state, state.d_j, hj.size());
        if (qn::err_state() != QNPEPS_ELOC_OK) return;
        usize index_bytes{};
        usize coefficient_bytes{};
        if (not arena_product({sizeof(int), ha.size()}, index_bytes)
            or not arena_product({sizeof(f64), hj.size()}, coefficient_bytes))
            return;
        CUDA_CHECK(cudaMemcpy(state.d_a, ha.data(), index_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state.d_b, hb.data(), index_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state.d_j, hj.data(), coefficient_bytes, cudaMemcpyHostToDevice));
    }

    cache_alloc(state, state.value_buf, static_cast<usize>(cfg.meo));
    usize logpsi_count{};
    if (not arena_product({2u, static_cast<std::uint64_t>(cfg.meo)}, logpsi_count)) return;
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
        if (qn::err_state() != QNPEPS_ELOC_OK) return;
        usize block_bytes{};
        if (not arena_product({sizeof(int), static_cast<std::uint64_t>(n_blocks)}, block_bytes))
            return;
        auto hoff = std::vector<int>(static_cast<usize>(n_blocks));
        for (auto b = int{0}; b < n_blocks; ++b)
            hoff[static_cast<usize>(b)] = static_cast<int>(state.ok_off[static_cast<usize>(b)]);
        CUDA_CHECK(cudaMemcpy(state.d_boff, hoff.data(), block_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(state.d_bslice, state.ok_slice.data(), block_bytes, cudaMemcpyHostToDevice)
        );
    }
    if (qn::err_state() != QNPEPS_ELOC_OK) return;

    for (const FlipInst& term : state.orig_terms)
        (term.bucket == Bucket::fourbody ? state.orig_fb : state.orig_hterms) = true;
    for (const FlipInst& term : state.tr_terms)
        if (term.bucket == Bucket::fourbody) state.tr_fb = true;
    state.orig_h = want_o or state.orig_hterms;
    const auto density = bool{cfg.truncation_route == 6};
    const auto lanes = int{density ? 1 : static_cast<int>(std::min<i64>(cfg.meo, n_samples))};
    const auto sh = Shape{
        cfg.lx, cfg.ly, cfg.dim_phys, cfg.dim_bond, cfg.chi_eo, lanes, density, cfg.density_cutoff
    };
    worker_setup(
        state.w, sh, state.orig_h, state.orig_fb, nullptr, nullptr, state.stream, context_arena
    );
    if (not state.tr_terms.empty())
    {
        const auto sht = Shape{
            cfg.ly,
            cfg.lx,
            cfg.dim_phys,
            cfg.dim_bond,
            cfg.chi_eo,
            lanes,
            density,
            cfg.density_cutoff
        };
        worker_setup(
            state.wt, sht, true, state.tr_fb, nullptr, nullptr, state.stream, context_arena
        );
    }
    if (qn::err_state() != QNPEPS_ELOC_OK) return;

    const auto wave_width = int{density ? 1 : cfg.meo};
    for (auto done = i64{0}; done < n_samples; done += wave_width)
    {
        const auto wave_lanes = int{static_cast<int>(std::min<i64>(wave_width, n_samples - done))};
        state.wave_ranges.push_back({done, wave_lanes});
    }
    state.ready = true;
}

}
