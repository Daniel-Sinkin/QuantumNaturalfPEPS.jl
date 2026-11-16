#include "parts.cuh"

namespace qn_eloc::env
{

auto launch_fourbody(Worker& w, int mid, const cf* cl, const cf* cr, cf* out, int dim_batch) -> void
{
    launch_dot(w, mid, cl, mid, cr, mid, out, dim_batch);
}

auto project_corner_indexed(
    Worker& w,
    const TermPiece& piece,
    int r,
    int c,
    const int* active_indices,
    int active,
    cf* out,
    i64 out_stride
) -> BT
{
    const auto& sh = w.sh;
    const auto pd = proj_dims(w, r, c);
    const auto slab = bt_elems(pd);
    i64 total{};
    if (slab < 1 or active < 1) return {};
    arena_slot({static_cast<std::uint64_t>(slab), static_cast<std::uint64_t>(active)}, total);
    constexpr auto threads = 256;
    const auto blocks = static_cast<int>(std::min<i64>(4096, (total + threads - 1) / threads));
    const auto site_index = r * sh.ly + c;
    const auto position = support_position(piece, site_index);
    const auto site = w.peps + w.site_off[static_cast<usize>(site_index)];
    const auto lane_stride = static_cast<i64>(sh.lx) * sh.ly;
    if (position == k_keep_position or piece.is_one_hot)
    {
        const auto args = CuProjectGatherArgs{
            .site = site,
            .samples = w.samples,
            .lane_stride = lane_stride,
            .active_indices = active_indices,
            .site_index = site_index,
            .slab = slab,
            .local_state = piece.local_state,
            .support_position = position,
            .targets = piece.device_targets,
            .out = out,
            .out_stride = out_stride,
            .active = active,
        };
        cu_project_site_gather<<<blocks, threads, 0, w.la.stream()>>>(args);
    }
    else
    {
        const auto args = CuProjectWeightedArgs{
            .site = site,
            .samples = w.samples,
            .lane_stride = lane_stride,
            .active_indices = active_indices,
            .slab = slab,
            .local_state = piece.local_state,
            .support_position = position,
            .weights = piece.device_weights,
            .out = out,
            .out_stride = out_stride,
            .active = active,
        };
        cu_project_site_weighted<<<blocks, threads, 0, w.la.stream()>>>(args);
    }
    check(cudaGetLastError());
    return BT{out, out_stride, pd};
}

auto compact_tensor(Worker& w, const BT& src, const int* active_indices, int active, cf* out) -> BT
{
    const auto elems = i64{bt_elems(src.dim)};
    i64 total{};
    if (elems < 1 or active < 1) return {};
    arena_slot({static_cast<std::uint64_t>(elems), static_cast<std::uint64_t>(active)}, total);
    const auto threads = int{256};
    const auto blocks = int{static_cast<int>(std::min<i64>(4096, (total + threads - 1) / threads))};
    cu_compact_lanes<<<blocks, threads, 0, w.la.stream()>>>(
        out, elems, src.p, src.stride, active_indices, elems, active
    );
    check(cudaGetLastError());
    return BT{out, elems, src.dim};
}

auto chain_normalize_batch(Worker& w, cf* x, i64 n, i64 stride, f64* f_chain, int dim_batch) -> void
{
    int elements{};
    if (n < 1) return;
    arena_int({static_cast<std::uint64_t>(n)}, elements);
    cu_normalize_log<<<dim_batch, 256, 0, w.la.stream()>>>(
        x, elements, stride, f_chain, dim_batch, nullptr
    );
    check(cudaGetLastError());
}

namespace
{
struct ChainColumns
{
    int row{};
    int first{};
    int last{};
};

[[nodiscard]] auto chain_columns(const TermPiece& piece, int ly) noexcept -> ChainColumns
{
    auto columns = ChainColumns{.row = piece.moving[0] / ly, .first = ly, .last = -1};
    for (auto k = 0_uz; k < static_cast<usize>(piece.moving_count); ++k)
    {
        columns.first = std::min(columns.first, piece.moving[k] % ly);
        columns.last = std::max(columns.last, piece.moving[k] % ly);
    }
    return columns;
}
}

auto eval_horizontal_compact(
    Worker& w, const TermPiece& piece, const int* active_indices, int active, cf* value_out
) -> f64*
{
    const auto& sh = w.sh;
    const auto ly = sh.ly;
    const auto columns = chain_columns(piece, ly);
    const auto i = columns.row;
    const auto c0 = columns.first;
    const auto cend = columns.last;

    const auto left_src =
        BT{c0 > 0 ? BT{hl_site(w, i, c0 - 1), w.slot_h, hl_o_dims(w, i, c0)} : unit3(w)};
    const auto right_src =
        BT{cend < ly - 1 ? BT{hr_site(w, i, cend), w.slot_h, hr_o_dims(w, i, cend)} : unit3(w)};

    auto frame = Carver{w.scratch};
    auto chain_a{frame.take_product<cf>(
        {static_cast<std::uint64_t>(w.slot_h), static_cast<std::uint64_t>(active)}
    )};
    auto chain_b{frame.take_product<cf>(
        {static_cast<std::uint64_t>(w.slot_h), static_cast<std::uint64_t>(active)}
    )};
    auto right{frame.take_product<cf>(
        {static_cast<std::uint64_t>(w.slot_h), static_cast<std::uint64_t>(active)}
    )};
    auto flip{frame.take_product<cf>(
        {static_cast<std::uint64_t>(w.slot_proj), static_cast<std::uint64_t>(active)}
    )};
    auto compact_top{frame.take_product<cf>(
        {static_cast<std::uint64_t>(w.slot_env), static_cast<std::uint64_t>(active)}
    )};
    auto compact_bot{frame.take_product<cf>(
        {static_cast<std::uint64_t>(w.slot_env), static_cast<std::uint64_t>(active)}
    )};
    auto f_chain{frame.take<f64>(static_cast<usize>(active))};
    usize f_chain_bytes{};
    arena_product({sizeof(f64), static_cast<std::uint64_t>(active)}, f_chain_bytes);

    check(cudaMemsetAsync(f_chain, 0, f_chain_bytes, w.la.stream()));
    auto v = BT{compact_tensor(w, left_src, active_indices, active, chain_a)};
    auto next{chain_b};
    for (auto c = int{c0}; c <= cend; ++c)
    {
        const auto proj_elems = i64{bt_elems(proj_dims(w, i, c))};
        const auto projected =
            BT{project_corner_indexed(w, piece, i, c, active_indices, active, flip, proj_elems)};
        const auto top =
            BT{compact_tensor(w, top_bt(w, i, c), active_indices, active, compact_top)};
        const auto bot =
            BT{compact_tensor(w, bot_bt_h(w, i, c), active_indices, active, compact_bot)};
        const auto d =
            std::vector<int>{fold_left(w, frame, v, top, projected, bot, next, w.slot_h, active)};
        v = BT{next, w.slot_h, d};
        chain_normalize_batch(w, v.p, bt_elems(d), w.slot_h, f_chain, active);
        next = next == chain_a ? chain_b : chain_a;
    }

    const auto vend = BT{compact_tensor(w, right_src, active_indices, active, right)};
    launch_dot(
        w, static_cast<int>(bt_elems(v.dim)), v.p, v.stride, vend.p, vend.stride, value_out, active
    );
    return f_chain;
}

auto eval_fourbody_compact(
    Worker& w, const TermPiece& piece, const int* active_indices, int active, cf* value_out
) -> f64*
{
    const auto& sh = w.sh;
    const auto ly = int{sh.ly};
    const auto r0 = int{piece.moving[0] / ly};
    const auto c0 = int{piece.moving[0] % ly};
    const auto r1 = int{piece.moving[1] / ly};
    const auto c1 = int{piece.moving[1] % ly};
    const auto minr = int{std::min(r0, r1)};
    const auto maxr = int{std::max(r0, r1)};
    const auto miny = int{std::min(c0, c1)};
    const auto maxy = int{std::max(c0, c1)};
    const auto upper = int{minr};

    const auto piLd = std::vector<int>{proj_dims(w, minr, miny)};
    const auto pjLd = std::vector<int>{proj_dims(w, maxr, miny)};
    const auto piRd = std::vector<int>{proj_dims(w, minr, maxy)};
    const auto pjRd = std::vector<int>{proj_dims(w, maxr, maxy)};
    const auto piL_elems = i64{bt_elems(piLd)};
    const auto pjL_elems = i64{bt_elems(pjLd)};
    const auto piR_elems = i64{bt_elems(piRd)};
    const auto pjR_elems = i64{bt_elems(pjRd)};

    const auto topL = BT{top_bt(w, upper, miny)};
    const auto botL = BT{bot_bt_fb(w, upper, miny)};
    const auto topR = BT{top_bt(w, upper, maxy)};
    const auto botR = BT{bot_bt_fb(w, upper, maxy)};
    const auto fbL =
        BT{miny > 0 ? BT{fbl_site(w, upper, miny - 1), w.slot_fb, fb_left_dims(w, upper, miny - 1)}
                    : unit4(w)};
    const auto fbR =
        BT{maxy < ly - 1 ? BT{fbr_site(w, upper, maxy), w.slot_fb, fb_right_dims(w, upper, maxy)}
                         : unit4(w)};
    int mid{};
    arena_int(
        {static_cast<std::uint64_t>(topL.dim[2]),
         static_cast<std::uint64_t>(piLd[2]),
         static_cast<std::uint64_t>(pjLd[2]),
         static_cast<std::uint64_t>(botL.dim[2])},
        mid
    );

    auto frame = Carver{w.scratch};
    auto piL{frame.take_product<cf>(
        {static_cast<std::uint64_t>(piL_elems), static_cast<std::uint64_t>(active)}
    )};
    auto pjL{frame.take_product<cf>(
        {static_cast<std::uint64_t>(pjL_elems), static_cast<std::uint64_t>(active)}
    )};
    auto piR{frame.take_product<cf>(
        {static_cast<std::uint64_t>(piR_elems), static_cast<std::uint64_t>(active)}
    )};
    auto pjR{frame.take_product<cf>(
        {static_cast<std::uint64_t>(pjR_elems), static_cast<std::uint64_t>(active)}
    )};
    auto cl{frame.take_product<cf>(
        {static_cast<std::uint64_t>(mid), static_cast<std::uint64_t>(active)}
    )};
    auto cr{frame.take_product<cf>(
        {static_cast<std::uint64_t>(mid), static_cast<std::uint64_t>(active)}
    )};
    auto f_chain{frame.take<f64>(static_cast<usize>(active))};
    const auto fb_elems = i64{std::max(bt_elems(fbL.dim), bt_elems(fbR.dim))};
    const auto top_elems = i64{std::max(bt_elems(topL.dim), bt_elems(topR.dim))};
    const auto bot_elems = i64{std::max(bt_elems(botL.dim), bt_elems(botR.dim))};
    auto compact_fb{frame.take_product<cf>(
        {static_cast<std::uint64_t>(fb_elems), static_cast<std::uint64_t>(active)}
    )};
    auto compact_top{frame.take_product<cf>(
        {static_cast<std::uint64_t>(top_elems), static_cast<std::uint64_t>(active)}
    )};
    auto compact_bot{frame.take_product<cf>(
        {static_cast<std::uint64_t>(bot_elems), static_cast<std::uint64_t>(active)}
    )};

    const auto projected_pi_l =
        BT{project_corner_indexed(w, piece, minr, miny, active_indices, active, piL, piL_elems)};
    const auto projected_pj_l =
        BT{project_corner_indexed(w, piece, maxr, miny, active_indices, active, pjL, pjL_elems)};
    const auto projected_pi_r =
        BT{project_corner_indexed(w, piece, minr, maxy, active_indices, active, piR, piR_elems)};
    const auto projected_pj_r =
        BT{project_corner_indexed(w, piece, maxr, maxy, active_indices, active, pjR, pjR_elems)};
    usize f_chain_bytes{};
    arena_product({sizeof(f64), static_cast<std::uint64_t>(active)}, f_chain_bytes);
    check(cudaMemsetAsync(f_chain, 0, f_chain_bytes, w.la.stream()));

    const auto compact_fb_l = BT{compact_tensor(w, fbL, active_indices, active, compact_fb)};
    const auto compact_top_l = BT{compact_tensor(w, topL, active_indices, active, compact_top)};
    const auto compact_bot_l = BT{compact_tensor(w, botL, active_indices, active, compact_bot)};
    fold_left4(
        w,
        frame,
        compact_fb_l,
        compact_top_l,
        projected_pi_l,
        projected_pj_l,
        compact_bot_l,
        cl,
        static_cast<i64>(mid),
        active
    );

    const auto compact_fb_r = BT{compact_tensor(w, fbR, active_indices, active, compact_fb)};
    const auto compact_top_r = BT{compact_tensor(w, topR, active_indices, active, compact_top)};
    const auto compact_bot_r = BT{compact_tensor(w, botR, active_indices, active, compact_bot)};
    fold_right4(
        w,
        frame,
        compact_fb_r,
        compact_top_r,
        projected_pi_r,
        projected_pj_r,
        compact_bot_r,
        cr,
        static_cast<i64>(mid),
        active
    );
    chain_normalize_batch(w, cl, mid, mid, f_chain, active);
    chain_normalize_batch(w, cr, mid, mid, f_chain, active);
    launch_fourbody(w, mid, cl, cr, value_out, active);
    return f_chain;
}

auto eval_term_piece(
    Worker& w,
    const TermPiece& piece,
    cf* value_buf,
    f64* e_loc,
    const f64* logpsi,
    const int* active_indices,
    int active
) -> void
{
    const auto& sh = w.sh;
    const auto ly = int{sh.ly};
    const auto lanes = int{sh.lanes};
    const auto lx = int{sh.lx};
    if (active == 0) return;
    if (not active_indices)
    {
        active_indices = w.all_lanes;
        active = lanes;
    }
    f64* f_chain{};
    f64* f_top{};
    f64* f_down{};
    f64* f_rail_l{};
    f64* f_rail_r{};
    if (piece.bucket == Bucket::fourbody)
    {
        f_chain = eval_fourbody_compact(w, piece, active_indices, active, value_buf);
        if (f_chain == nullptr) return;
        const auto r0 = int{piece.moving[0] / ly};
        const auto r1 = int{piece.moving[1] / ly};
        const auto minr = int{std::min(r0, r1)};
        const auto maxr = int{std::max(r0, r1)};
        const auto c0 = int{piece.moving[0] % ly};
        const auto c1 = int{piece.moving[1] % ly};
        const auto miny = int{std::min(c0, c1)};
        const auto maxy = int{std::max(c0, c1)};
        f_top = minr > 0 ? w.f_top + static_cast<i64>(minr - 1) * lanes : nullptr;
        f_down = maxr < lx - 1 ? w.f_down + static_cast<i64>(lx - maxr - 2) * lanes : nullptr;
        f_rail_l = miny > 0 ? f_fbl_cut(w, minr, miny - 1) : nullptr;
        f_rail_r = maxy < ly - 1 ? f_fbr_cut(w, minr, maxy) : nullptr;
    }
    else
    {
        f_chain = eval_horizontal_compact(w, piece, active_indices, active, value_buf);
        if (f_chain == nullptr) return;
        const auto columns = chain_columns(piece, ly);
        const auto i = columns.row;
        const auto c0 = columns.first;
        const auto cend = columns.last;
        f_top = i > 0 ? w.f_top + static_cast<i64>(i - 1) * lanes : nullptr;
        f_down = i < lx - 1 ? w.f_down + static_cast<i64>(lx - i - 2) * lanes : nullptr;
        f_rail_l = c0 > 0 ? f_hl_cut(w, i, c0 - 1) : nullptr;
        f_rail_r = cend < ly - 1 ? f_hr_cut(w, i, cend) : nullptr;
    }
    reduce_term_indexed(
        w,
        piece,
        value_buf,
        f_chain,
        active_indices,
        active,
        f_top,
        f_down,
        f_rail_l,
        f_rail_r,
        e_loc,
        logpsi
    );
}

}
