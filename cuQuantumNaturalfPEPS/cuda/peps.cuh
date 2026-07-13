#ifndef QNPEPS_PEPS_CUH
#define QNPEPS_PEPS_CUH

#include "dtensor.cuh"

namespace qnpeps
{
struct PepsDims
{
    i32 lx{};
    i32 ly{};
    i32 dim_phys{};
    i32 dim_bond{};
};

struct PepsSiteDims
{
    i32 bond_left{};
    i32 bond_down{};
    i32 bond_right{};
    i32 bond_up{};
    i32 dim_phys{};

    [[nodiscard]] constexpr auto num_elems() const noexcept -> i64
    {
        return static_cast<i64>(bond_left) * bond_down * bond_right * bond_up * dim_phys;
    }
};

[[nodiscard]] inline constexpr auto peps_site_dims(const PepsDims& dims, int row, int col) noexcept
    -> PepsSiteDims
{
    return PepsSiteDims{
        .bond_left = bond_dim(dims.ly, col, dims.dim_bond),
        .bond_down = bond_dim(dims.lx, row + 1, dims.dim_bond),
        .bond_right = bond_dim(dims.ly, col + 1, dims.dim_bond),
        .bond_up = bond_dim(dims.lx, row, dims.dim_bond),
        .dim_phys = dims.dim_phys,
    };
}

[[nodiscard]] inline auto peps_site_shape(const PepsDims& dims, int row, int col) -> Shape
{
    const auto site = peps_site_dims(dims, row, col);
    return Shape{site.bond_left, site.bond_down, site.bond_right, site.bond_up, site.dim_phys};
}

[[nodiscard]] inline constexpr auto peps_site_elems(const PepsDims& dims, int row, int col) noexcept
    -> i64
{
    return peps_site_dims(dims, row, col).num_elems();
}

[[nodiscard]] inline constexpr auto peps_row_elems(const PepsDims& dims, int row) noexcept -> i64
{
    i64 total{};
    for (auto col = 0; col < dims.ly; ++col)
        total += peps_site_elems(dims, row, col);
    return total;
}

[[nodiscard]] inline constexpr auto peps_elems(const PepsDims& dims) noexcept -> i64
{
    i64 total{};
    for (auto row = 0; row < dims.lx; ++row)
        total += peps_row_elems(dims, row);
    return total;
}
}

#endif
