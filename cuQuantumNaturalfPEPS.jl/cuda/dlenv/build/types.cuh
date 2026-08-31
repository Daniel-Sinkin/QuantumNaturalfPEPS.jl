#ifndef QNPEPS_DLENV_BUILD_TYPES_CUH
#define QNPEPS_DLENV_BUILD_TYPES_CUH

#include "dlenv/build/common.cuh"

namespace qnpeps::dlenv
{
struct ZipupContext
{
    QnpepsConfig config{};
    int maxdim{};
    Dims dims{};
    std::unique_ptr<Session> session{};
    BuildState dl{};
    usize scale_count{};
    f64 density_input_gauge{};
    bool active{};
};

[[nodiscard]] inline auto zipup_context(qnpeps_zipup_ctx& context) noexcept -> ZipupContext&
{
    return *reinterpret_cast<ZipupContext*>(&context);
}

[[nodiscard]] inline auto zipup_context(qnpeps_zipup_ctx* context) noexcept -> ZipupContext*
{
    return reinterpret_cast<ZipupContext*>(context);
}

using PepsRow = std::vector<DeviceTensor>;
using DlEnvRow = std::vector<DeviceTensor>;

inline auto init_dl_units(
    Linalg& la, cuFloatComplex* unit_environment, cuFloatComplex* initial_factor
) -> void
{
    constexpr cuFloatComplex one{1.0f, 0.0f};
    upload_async(la, unit_environment, &one, 1);
    upload_async(la, initial_factor, &one, 1);
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

[[nodiscard]] inline auto read_site_dims(const int32_t* header, usize site) -> DlSiteDims
{
    const usize base{site * k_dl_axis_count};
    return DlSiteDims{
        .bond_left = header[base + k_dl_bond_left],
        .ket = header[base + k_dl_ket],
        .bra = header[base + k_dl_bra],
        .bond_right = header[base + k_dl_bond_right],
    };
}

inline auto write_site_dims(int32_t* header, usize site, const Shape& dim) -> void
{
    const usize base{site * k_dl_axis_count};
    header[base + k_dl_bond_left] = dim[k_dl_bond_left];
    header[base + k_dl_ket] = dim[k_dl_ket];
    header[base + k_dl_bra] = dim[k_dl_bra];
    header[base + k_dl_bond_right] = dim[k_dl_bond_right];
}

[[nodiscard]] inline auto peps_site_shape(const Dims& dims, int row_up, int row_down, int col)
    -> Shape
{
    const int bond_left{bond_dim(dims.ly, col, dims.dim_bond)};
    const int bond_right{bond_dim(dims.ly, col + 1, dims.dim_bond)};
    const int bond_up{bond_dim(dims.lx, row_up, dims.dim_bond)};
    const int bond_down{bond_dim(dims.lx, row_down, dims.dim_bond)};
    return Shape{bond_left, bond_down, bond_right, bond_up, dims.dim_phys};
}

[[nodiscard]] inline auto peps_row_elems(const Dims& dims, int row_up, int row_down) -> i64
{
    i64 total{};
    for (auto col = 0; col < dims.ly; ++col)
        total += static_cast<i64>(peps_site_shape(dims, row_up, row_down, col).num_elems());
    return total;
}

inline auto pack_peps_row(
    const Dims& dims,
    int row_up,
    int row_down,
    const cuFloatComplex* source_base,
    i64& source_offset,
    cuFloatComplex* packed_base,
    i64& packed_offset,
    std::vector<DeviceTensor>& output_row,
    cudaStream_t stream
) -> void
{
    const auto reversed = Permutation::reverse(k_peps_site_rank);
    for (auto col = 0; col < dims.ly; ++col)
    {
        const auto source_shape = peps_site_shape(dims, row_up, row_down, col);
        const auto site_elements = static_cast<i64>(source_shape.num_elems());
        const auto source =
            DeviceTensor{source_shape, const_cast<cuFloatComplex*>(source_base + source_offset)};
        permute_axes(source, reversed, false, packed_base + packed_offset, stream);
        output_row[static_cast<usize>(col)] =
            DeviceTensor{reversed.apply(source_shape), packed_base + packed_offset};
        source_offset += site_elements;
        packed_offset += site_elements;
    }
}

}

#endif
