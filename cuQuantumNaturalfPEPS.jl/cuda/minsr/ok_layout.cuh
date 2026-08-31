#ifndef QNPEPS_MINSR_OK_LAYOUT_CUH
#define QNPEPS_MINSR_OK_LAYOUT_CUH

#include "core/types.cuh"

#include <vector>

namespace qnpeps::minsr
{
[[nodiscard]] inline auto site_slice(const Dims& dims, int row, int column) noexcept -> i64
{
    return static_cast<i64>(bond_dim(dims.ly, column, dims.dim_bond))
           * bond_dim(dims.lx, row + 1, dims.dim_bond)
           * bond_dim(dims.ly, column + 1, dims.dim_bond) * bond_dim(dims.lx, row, dims.dim_bond);
}

[[nodiscard]] inline auto compact_count(const Dims& dims) noexcept -> i64
{
    i64 total{};
    const auto row_bound = static_cast<usize>(dims.lx);
    const auto column_bound = static_cast<usize>(dims.ly);
    for (auto row = 0_uz; row < row_bound; ++row)
    {
        const auto row_position = static_cast<int>(row);
        for (auto column = 0_uz; column < column_bound; ++column)
        {
            const auto column_position = static_cast<int>(column);
            total += site_slice(dims, row_position, column_position);
        }
    }
    return total;
}

[[nodiscard]] inline auto dense_count(const Dims& dims) noexcept -> i64
{
    return static_cast<i64>(dims.dim_phys) * compact_count(dims);
}

[[nodiscard]] inline auto build_slot_site(const Dims& dims, i64& out_compact) -> std::vector<i32>
{
    std::vector<i32> slot{};
    i64 total{};
    i32 site{};
    const auto row_bound = static_cast<usize>(dims.lx);
    const auto column_bound = static_cast<usize>(dims.ly);
    for (auto row = 0_uz; row < row_bound; ++row)
    {
        const auto row_position = static_cast<int>(row);
        for (auto column = 0_uz; column < column_bound; ++column)
        {
            const auto column_position = static_cast<int>(column);
            const auto slice = site_slice(dims, row_position, column_position);
            const auto slice_bound = static_cast<usize>(slice);
            for (auto index = 0_uz; index < slice_bound; ++index)
            {
                slot.push_back(site);
            }
            total += slice;
            ++site;
        }
    }
    out_compact = total;
    return slot;
}
}

#endif
