#include "slab.cuh"

namespace qn_eloc::env
{
namespace
{
inline constexpr u64 k_chain_buffers{2};
inline constexpr u64 k_fold_buffers{3};
inline constexpr u64 k_environment_buffers{2};
inline constexpr u64 k_alignment_tail_takes{16};
inline constexpr int k_projection_west{0};
inline constexpr int k_projection_north{3};
inline constexpr int k_environment_left{0};
inline constexpr int k_environment_vertical{1};
inline constexpr int k_state_top{0};
inline constexpr int k_closing_elements{1};

struct SlabExtents
{
    i64 chain{};
    i64 fold{};
    i64 environment{};
    i64 projection{};
};

struct FoldBuffers
{
    cf* current{};
    cf* spare{};
    i64 stride{};
};

[[nodiscard]] auto slab_extents(const Shape& shape, int rows) -> SlabExtents
{
    const auto chi = static_cast<u64>(shape.chi);
    const auto bond = static_cast<u64>(shape.dim_bond);
    const auto row_count = static_cast<usize>(rows);
    auto extents = SlabExtents{};
    usize chain{};
    arena_product({chi, chi}, chain);
    for (auto row = 0_uz; row < row_count; ++row)
        arena_product({chain, bond}, chain);
    arena_slot({chain}, extents.chain);
    arena_slot({chain, bond}, extents.fold);
    arena_slot({chi, bond, chi}, extents.environment);
    arena_slot({bond, bond, bond, bond}, extents.projection);
    return extents;
}

auto fold_slab_column(
    Worker& w,
    Carver& frame,
    const BT& state,
    const BT& top,
    const std::vector<BT>& rows,
    const BT& bot,
    FoldBuffers buffers,
    cf* out,
    i64 out_stride,
    int active
) -> std::vector<int>
{
    const auto vertical_axis = static_cast<int>(rows.size()) + 1;
    auto partial = bcontract(
        w,
        frame,
        state,
        {k_state_top},
        top,
        {k_environment_left},
        buffers.current,
        buffers.stride,
        active
    );
    for (const auto& row : rows)
    {
        std::swap(buffers.current, buffers.spare);
        partial = bcontract(
            w,
            frame,
            partial,
            {0, vertical_axis},
            row,
            {k_projection_west, k_projection_north},
            buffers.current,
            buffers.stride,
            active
        );
    }
    const auto closed = bcontract(
        w,
        frame,
        partial,
        {0, vertical_axis},
        bot,
        {k_environment_left, k_environment_vertical},
        out,
        out_stride,
        active
    );
    return closed.dim;
}
}

auto slab_scratch_bytes(const Shape& shape) -> usize
{
    if (shape.slab_rows < 1) return 0;
    const auto extents = slab_extents(shape, shape.slab_rows);
    const auto lanes = static_cast<u64>(shape.lanes);
    auto total = 0_uz;
    const auto reserve = [&](i64 elements, usize element_bytes, u64 count)
    {
        usize bytes{};
        usize aligned{};
        usize term{};
        arena_product({element_bytes, static_cast<u64>(elements), lanes}, bytes);
        arena_align(bytes, aligned);
        arena_product({count, aligned}, term);
        arena_accumulate(total, term);
    };
    reserve(extents.chain, sizeof(cf), k_chain_buffers);
    reserve(extents.fold, sizeof(cf), k_fold_buffers);
    reserve(extents.environment, sizeof(cf), k_environment_buffers);
    reserve(extents.projection, sizeof(cf), static_cast<u64>(shape.slab_rows));
    reserve(std::max(extents.environment, extents.projection), sizeof(cf), 1u);
    reserve(1, sizeof(f64), 1u);
    usize tail{};
    arena_product({k_alignment_tail_takes, CUDA_MALLOC_ALIGN}, tail);
    arena_accumulate(total, tail);
    return total;
}

auto eval_slab_compact(
    Worker& w, const TermPiece& piece, const int* active_indices, int active, cf* value_out
) -> f64*
{
    const auto& sh = w.sh;
    const auto bounds = moving_bounds(piece, sh.ly);
    const auto rows = bounds.bottom_row - bounds.top_row + 1;
    const auto row_count = static_cast<usize>(rows);
    const auto extents = slab_extents(sh, rows);
    const auto batch = static_cast<u64>(active);
    auto frame = Carver{w.scratch};
    const auto take = [&](i64 elements) -> cf*
    { return frame.take_product<cf>({static_cast<u64>(elements), batch}); };
    auto* const chain = take(extents.chain);
    auto* const chain_spare = take(extents.chain);
    const auto buffers = FoldBuffers{
        .current = take(extents.fold),
        .spare = take(extents.fold),
        .stride = extents.fold,
    };
    auto* const compact_top = take(extents.environment);
    auto* const compact_bot = take(extents.environment);
    auto projected = std::vector<cf*>(row_count);
    for (auto& slot : projected)
        slot = take(extents.projection);
    auto* const f_chain = frame.take<f64>(static_cast<usize>(active));
    usize f_chain_bytes{};
    arena_product({sizeof(f64), batch}, f_chain_bytes);
    check(cudaMemsetAsync(f_chain, 0, f_chain_bytes, w.la.stream()));

    auto state = BT{w.unit, 1, std::vector<int>(row_count + 2, 1)};
    const auto column_count = static_cast<usize>(sh.ly);
    for (auto column_index = 0_uz; column_index < column_count; ++column_index)
    {
        const auto column = static_cast<int>(column_index);
        const auto top_source = top_bt(w, bounds.top_row, column);
        const auto bot_source = bot_bt_h(w, bounds.bottom_row, column);
        const auto top = compact_tensor(w, top_source, active_indices, active, compact_top);
        const auto bot = compact_tensor(w, bot_source, active_indices, active, compact_bot);
        auto row_tensors = std::vector<BT>{};
        for (auto k = 0_uz; k < row_count; ++k)
        {
            const auto row = bounds.top_row + static_cast<int>(k);
            const auto elements = bt_elems(proj_dims(w, row, column));
            row_tensors.push_back(project_corner_indexed(
                w, piece, row, column, active_indices, active, projected[k], elements
            ));
        }
        auto* const next = state.p == chain ? chain_spare : chain;
        const auto dims = fold_slab_column(
            w, frame, state, top, row_tensors, bot, buffers, next, extents.chain, active
        );
        state = BT{next, extents.chain, dims};
        chain_normalize_batch(w, state.p, bt_elems(state.dim), state.stride, f_chain, active);
    }
    const auto closing = bt_elems(state.dim);
    require(
        closing == k_closing_elements,
        "slab route closes the lattice with {} elements per lane",
        closing
    );
    launch_dot(w, k_closing_elements, state.p, state.stride, w.unit, 1, value_out, active);
    return f_chain;
}

}
