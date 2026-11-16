#include "parts.cuh"

namespace qn_eloc::env
{
namespace
{
inline constexpr auto k_no_target{-1};

[[nodiscard]] auto to_device_complex(const QnpepsElocComplex& value) noexcept -> cuDoubleComplex
{
    return cuDoubleComplex{value.re, value.im};
}

[[nodiscard]] auto to_device_weight(const QnpepsElocComplex& value) noexcept -> cf
{
    return cf{static_cast<f32>(value.re), static_cast<f32>(value.im)};
}

[[nodiscard]] auto own_value(i64 state, usize position, const CuLocalState& local_state) noexcept
    -> int
{
    auto remaining = state;
    for (auto k = static_cast<usize>(local_state.site_count) - 1; k > position; --k)
        remaining /= local_state.dim_phys;
    return static_cast<int>(remaining % local_state.dim_phys);
}

[[nodiscard]] auto is_zero(const QnpepsElocComplex& value) noexcept -> bool
{
    return value.re == 0.0 and value.im == 0.0;
}

[[nodiscard]] auto is_exactly_one(const QnpepsElocComplex& value) noexcept -> bool
{
    return value.re == 1.0 and value.im == 0.0;
}

[[nodiscard]] auto single_nonzero(const QnpepsElocComplex* weights, usize dim_phys) noexcept -> int
{
    auto index = k_no_target;
    for (auto value = 0_uz; value < dim_phys; ++value)
    {
        if (is_zero(weights[value])) continue;
        if (index != k_no_target) return k_no_target;
        index = static_cast<int>(value);
    }
    return index;
}

[[nodiscard]] auto transposed_site(int site, int lx, int ly) noexcept -> int
{
    return (site % ly) * lx + site / ly;
}

struct PieceShape
{
    bool has_inactive{};
    std::vector<int> targets{};
};

struct PieceView
{
    const QnpepsElocTermTable& table;
    const QnpepsElocOffDiagonalPiece& source;
    usize site_count{};
    usize state_count{};
    usize dim_phys{};

    [[nodiscard]] auto is_active(usize state) const noexcept -> bool
    {
        return table.piece_active[static_cast<usize>(source.state_offset) + state] != 0;
    }

    [[nodiscard]] auto weights(usize state, usize position) const noexcept
        -> const QnpepsElocComplex*
    {
        const auto offset = (state * site_count + position) * dim_phys;
        return table.piece_weights + static_cast<usize>(source.weight_offset) + offset;
    }
};

auto mark_moving_positions(const PieceView& view, TermPiece& piece) -> void
{
    for (auto state = 0_uz; state < view.state_count; ++state)
    {
        if (not view.is_active(state)) continue;
        for (auto k = 0_uz; k < view.site_count; ++k)
        {
            const auto own = own_value(static_cast<i64>(state), k, piece.local_state);
            if (single_nonzero(view.weights(state, k), view.dim_phys) != own)
                piece.is_moving[k] = true;
        }
    }
    for (auto k = 0_uz; k < view.site_count; ++k)
    {
        if (not piece.is_moving[k]) continue;
        piece.moving[static_cast<usize>(piece.moving_count)] = piece.local_state.site[k];
        ++piece.moving_count;
    }
}

[[nodiscard]] auto folded_coefficient(const PieceView& view, const TermPiece& piece, usize state)
    -> cuDoubleComplex
{
    const auto& source =
        view.table.piece_coefficients[static_cast<usize>(view.source.state_offset) + state];
    auto coefficient = std::complex<f64>{source.re, source.im};
    for (auto k = 0_uz; k < view.site_count; ++k)
    {
        if (piece.is_moving[k]) continue;
        const auto own =
            static_cast<usize>(own_value(static_cast<i64>(state), k, piece.local_state));
        const auto& scale = view.weights(state, k)[own];
        if (not is_exactly_one(scale)) coefficient *= std::complex<f64>{scale.re, scale.im};
    }
    return cuDoubleComplex{coefficient.real(), coefficient.imag()};
}

[[nodiscard]] auto shape_piece(const PieceView& view, TermPiece& piece, TermPlan& plan)
    -> PieceShape
{
    auto shape = PieceShape{};
    mark_moving_positions(view, piece);
    piece.state_offset = static_cast<i64>(plan.active.size());
    piece.weight_offset = static_cast<i64>(plan.weights.size());
    shape.targets.assign(view.state_count * view.site_count, 0);
    piece.is_one_hot = true;
    for (auto state = 0_uz; state < view.state_count; ++state)
    {
        const auto is_active = view.is_active(state);
        shape.has_inactive = shape.has_inactive or not is_active;
        plan.active.push_back(is_active ? u8{1} : u8{0});
        plan.coefficients.push_back(
            is_active ? folded_coefficient(view, piece, state) : cuDoubleComplex{0.0, 0.0}
        );
        for (auto k = 0_uz; k < view.site_count; ++k)
        {
            const auto* weights = view.weights(state, k);
            for (auto value = 0_uz; value < view.dim_phys; ++value)
                plan.weights.push_back(to_device_weight(weights[value]));
            const auto slot = state * view.site_count + k;
            shape.targets[slot] = own_value(static_cast<i64>(state), k, piece.local_state);
            if (not is_active or not piece.is_moving[k]) continue;
            const auto target = single_nonzero(weights, view.dim_phys);
            const auto is_one_hot_target =
                target != k_no_target and is_exactly_one(weights[static_cast<usize>(target)]);
            if (not is_one_hot_target) piece.is_one_hot = false;
            if (target != k_no_target) shape.targets[slot] = target;
        }
    }
    return shape;
}
}

[[nodiscard]] auto support_position(const TermPiece& piece, int site) noexcept -> int
{
    for (auto k = 0_uz; k < static_cast<usize>(piece.local_state.site_count); ++k)
        if (piece.local_state.site[k] == site and piece.is_moving[k]) return static_cast<int>(k);
    return k_keep_position;
}

auto classify_route(const QnpepsElocConfig& cfg, TermPiece& piece) -> bool
{
    const auto ly = cfg.ly;
    const auto moving_count = static_cast<usize>(piece.moving_count);
    if (moving_count < 1) return false;
    auto min_row = piece.moving[0] / ly;
    auto max_row = min_row;
    auto min_column = piece.moving[0] % ly;
    auto max_column = min_column;
    for (auto k = 0_uz; k < moving_count; ++k)
    {
        min_row = std::min(min_row, piece.moving[k] / ly);
        max_row = std::max(max_row, piece.moving[k] / ly);
        min_column = std::min(min_column, piece.moving[k] % ly);
        max_column = std::max(max_column, piece.moving[k] % ly);
    }
    const auto row_span = max_row - min_row;
    const auto column_span = max_column - min_column;
    const auto is_plaquette_diagonal = row_span == 1 and column_span == 1 and moving_count == 2;
    if (moving_count == 1 or row_span == 0)
    {
        piece.pass = 0;
        piece.bucket = column_span <= 1 ? Bucket::horizontal : Bucket::longer_horizontal;
    }
    else if (column_span == 0)
    {
        piece.pass = 1;
        piece.bucket = row_span <= 1 ? Bucket::horizontal : Bucket::longer_horizontal;
    }
    else if (is_plaquette_diagonal)
    {
        piece.pass = 0;
        piece.bucket = Bucket::fourbody;
    }
    else
    {
        return false;
    }
    if (piece.pass == 0) return true;
    for (auto k = 0_uz; k < static_cast<usize>(piece.local_state.site_count); ++k)
        piece.local_state.site[k] = transposed_site(piece.local_state.site[k], cfg.lx, ly);
    for (auto k = 0_uz; k < moving_count; ++k)
        piece.moving[k] = transposed_site(piece.moving[k], cfg.lx, ly);
    return true;
}

auto plan_terms(const QnpepsElocConfig& cfg, const QnpepsElocTermTable& table, TermPlan& plan)
    -> bool
{
    plan = TermPlan{};
    const auto dim_phys = table.dim_phys;
    for (auto i = 0_uz; i < static_cast<usize>(table.diagonal_count); ++i)
    {
        const auto& source = table.diagonal[i];
        auto term = CuDiagonalTerm{.value_offset = source.value_offset};
        term.local_state.site_count = source.site_count;
        term.local_state.dim_phys = dim_phys;
        for (auto k = 0_uz; k < static_cast<usize>(source.site_count); ++k)
            term.local_state.site[k] = source.site[k];
        plan.diagonal.push_back(term);
    }
    for (auto i = 0_uz; i < static_cast<usize>(table.diagonal_value_count); ++i)
        plan.diagonal_values.push_back(to_device_complex(table.diagonal_values[i]));

    auto original_slots = 0;
    auto transposed_slots = 0;
    for (auto i = 0_uz; i < static_cast<usize>(table.piece_count); ++i)
    {
        const auto& source = table.piece[i];
        auto piece = TermPiece{};
        piece.local_state.site_count = source.site_count;
        piece.local_state.dim_phys = dim_phys;
        for (auto k = 0_uz; k < static_cast<usize>(source.site_count); ++k)
            piece.local_state.site[k] = source.site[k];
        const auto view = PieceView{
            .table = table,
            .source = source,
            .site_count = static_cast<usize>(source.site_count),
            .state_count = static_cast<usize>(source.state_count),
            .dim_phys = static_cast<usize>(dim_phys),
        };
        const auto shape = shape_piece(view, piece, plan);
        if (not classify_route(cfg, piece)) return false;
        if (piece.is_one_hot)
        {
            piece.target_offset = static_cast<i64>(plan.targets.size());
            for (const auto target : shape.targets)
                plan.targets.push_back(static_cast<u8>(target));
        }
        auto& slots = piece.pass == 0 ? original_slots : transposed_slots;
        if (shape.has_inactive) piece.active_slot = slots++;
        (piece.pass == 0 ? plan.original : plan.transposed).push_back(piece);
    }
    return true;
}

auto bind_device_pools(std::vector<TermPiece>& pieces, const DeviceTermPools& pools) -> void
{
    for (auto& piece : pieces)
    {
        piece.device_active = pools.active + piece.state_offset;
        piece.device_coefficients = pools.coefficients + piece.state_offset;
        piece.device_weights = piece.is_one_hot ? nullptr : pools.weights + piece.weight_offset;
        piece.device_targets = piece.is_one_hot ? pools.targets + piece.target_offset : nullptr;
    }
}

}

[[nodiscard]] auto qnpeps::eo::is_routable(
    i32 lx, i32 ly, const QnpepsElocTermTable& table
) noexcept -> bool
{
    try
    {
        auto config = QnpepsElocConfig{};
        config.lx = lx;
        config.ly = ly;
        auto plan = qn_eloc::env::TermPlan{};
        return qn_eloc::env::plan_terms(config, table, plan);
    }
    catch (...)
    {
        return false;
    }
}
