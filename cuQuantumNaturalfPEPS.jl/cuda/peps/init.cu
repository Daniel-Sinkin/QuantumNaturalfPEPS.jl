
#include "core/cuda_utils.cuh"
#include "core/defer.cuh"
#include "core/error.hpp"
#include "linalg/linalg.cuh"
#include "linalg/transfer.cuh"
#include "peps/init.cuh"
#include "peps/kernels.cuh"
#include "peps/peps.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda/std/cmath>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <limits>
#include <vector>

namespace qnpeps::peps
{
namespace
{
[[nodiscard]] auto site_layout(const PepsDims& dimensions, int row, int col, i64 output_offset)
    -> SiteLayout
{
    const auto site = peps_site_dims(dimensions, row, col);
    const auto incoming64 = static_cast<i64>(site.dim_phys) * site.bond_left * site.bond_up;
    const auto outgoing64 = static_cast<i64>(site.bond_right) * site.bond_down;
    const auto valid_positive_dimensions = incoming64 > 0 and outgoing64 > 0;
    const auto dimensions_fit = incoming64 <= std::numeric_limits<int>::max()
                                and outgoing64 <= std::numeric_limits<int>::max();
    require(
        valid_positive_dimensions and dimensions_fit,
        Status::bad_config,
        "PEPS site ({}, {}) matrix {}x{} is empty or exceeds the int range",
        row,
        col,
        incoming64,
        outgoing64
    );
    const auto incoming = static_cast<int>(incoming64);
    const auto outgoing = static_cast<int>(outgoing64);
    return {
        .bond_left = site.bond_left,
        .bond_down = site.bond_down,
        .bond_right = site.bond_right,
        .bond_up = site.bond_up,
        .dim_phys = site.dim_phys,
        .incoming = incoming,
        .outgoing = outgoing,
        .tall_rows = std::max(incoming, outgoing),
        .thin_cols = std::min(incoming, outgoing),
        .output_offset = output_offset,
    };
}

}

auto random_unitary(Linalg& linalg, const RandomUnitaryArgs& args) -> void
{
    const PepsDims dimensions{
        args.config.lx, args.config.ly, args.config.dim_phys, args.config.dim_bond
    };
    const auto required_elements = peps_elems(dimensions);
    const auto valid_required_elements =
        required_elements > 0
        and static_cast<u64>(required_elements)
                <= std::numeric_limits<usize>::max() / sizeof(cuFloatComplex);
    const auto required_bytes = valid_required_elements
                                    ? static_cast<usize>(required_elements) * sizeof(cuFloatComplex)
                                    : usize{};
    const auto valid_output = args.output and args.output_bytes >= required_bytes;
    const auto valid_alpha = std::isfinite(args.alpha) and args.alpha >= 0.0;
    require(
        valid_required_elements and valid_output and valid_alpha,
        Status::bad_config,
        "random PEPS rejects {} elements into {} output bytes with alpha {}",
        required_elements,
        args.output_bytes,
        args.alpha
    );

    const auto stream = linalg.stream();
    cuFloatComplex* device_matrix{};
    cuFloatComplex* device_tau{};
    cuFloatComplex* device_phases{};
    cuFloatComplex* device_workspace{};
    f32* device_spectrum{};
    int* device_info{};
    int* device_info_records{};
    int* device_failure{};
    DEFER(
        [&]
        {
            unchecked(cudaFree(device_failure));
            unchecked(cudaFree(device_info_records));
            unchecked(cudaFree(device_info));
            unchecked(cudaFree(device_spectrum));
            unchecked(cudaFree(device_workspace));
            unchecked(cudaFree(device_phases));
            unchecked(cudaFree(device_tau));
            unchecked(cudaFree(device_matrix));
        }
    );

    i64 max_matrix_elements{};
    int max_thin_cols{};
    int max_workspace_elements{1};
    i64 output_offset{};
    for (auto row = 0; row < dimensions.lx; ++row)
    {
        for (auto col = 0; col < dimensions.ly; ++col)
        {
            const auto layout = site_layout(dimensions, row, col, output_offset);
            max_matrix_elements = std::max(max_matrix_elements, layout.elements());
            max_thin_cols = std::max(max_thin_cols, layout.thin_cols);
            const auto workspace_count =
                linalg.qr_workspace_count(layout.tall_rows, layout.thin_cols);
            max_workspace_elements = std::max(max_workspace_elements, workspace_count);
            output_offset += layout.elements();
        }
    }
    require(
        output_offset == required_elements,
        "random PEPS site layouts cover {} elements, the PEPS holds {}",
        output_offset,
        required_elements
    );

    const auto site_count = static_cast<usize>(dimensions.lx) * static_cast<usize>(dimensions.ly);
    check(cudaMalloc(
        reinterpret_cast<void**>(&device_matrix),
        static_cast<usize>(max_matrix_elements) * sizeof(cuFloatComplex)
    ));
    check(cudaMalloc(
        reinterpret_cast<void**>(&device_tau),
        static_cast<usize>(max_thin_cols) * sizeof(cuFloatComplex)
    ));
    check(cudaMalloc(
        reinterpret_cast<void**>(&device_phases),
        static_cast<usize>(max_thin_cols) * sizeof(cuFloatComplex)
    ));
    check(cudaMalloc(
        reinterpret_cast<void**>(&device_workspace),
        static_cast<usize>(max_workspace_elements) * sizeof(cuFloatComplex)
    ));
    check(cudaMalloc(reinterpret_cast<void**>(&device_info), sizeof(int)));
    check(cudaMalloc(reinterpret_cast<void**>(&device_info_records), 2 * site_count * sizeof(int)));
    check(cudaMalloc(reinterpret_cast<void**>(&device_failure), sizeof(int)));
    zero_async(linalg, device_failure, 1);
    if (args.alpha != 0.0)
    {
        check(cudaMalloc(
            reinterpret_cast<void**>(&device_spectrum),
            static_cast<usize>(dimensions.dim_bond) * sizeof(f32)
        ));
        cu_fill_spectrum<<<
            grid_blocks_exact(dimensions.dim_bond),
            k_threads_per_block,
            0,
            stream>>>(device_spectrum, dimensions.dim_bond, -0.5 * args.alpha);
        check(cudaGetLastError());
    }

    output_offset = 0;
    usize site_index{};
    for (auto row = 0; row < dimensions.lx; ++row)
    {
        for (auto col = 0; col < dimensions.ly; ++col)
        {
            const auto layout = site_layout(dimensions, row, col, output_offset);
            const auto matrix_elements = layout.elements();
            const FillComplexNormalArgs normal_args{
                device_matrix, matrix_elements, args.seed, static_cast<u64>(output_offset)
            };
            cu_fill_complex_normal<<<
                grid_blocks_capped(matrix_elements),
                k_threads_per_block,
                0,
                stream>>>(normal_args);
            check(cudaGetLastError());
            const QrStageConfig qr_config{
                device_tau, device_workspace, max_workspace_elements, device_info
            };
            linalg.qr_factor(
                CuMatrixCF32{device_matrix, layout.tall_rows, layout.thin_cols}, qr_config
            );
            copy_device_async(linalg, device_info_records + 2 * site_index, device_info, 1);
            const ExtractPhasesArgs phase_args{
                device_matrix,
                layout.tall_rows,
                layout.thin_cols,
                device_phases,
                device_failure,
            };
            cu_extract_r_phases<<<
                grid_blocks_exact(layout.thin_cols),
                k_threads_per_block,
                0,
                stream>>>(phase_args);
            check(cudaGetLastError());
            linalg.qr_form(
                CuMatrixCF32{device_matrix, layout.tall_rows, layout.thin_cols}, qr_config
            );
            copy_device_async(linalg, device_info_records + 2 * site_index + 1, device_info, 1);
            const PackSiteArgs pack_args{
                device_matrix, device_phases, layout, device_spectrum, args.output
            };
            cu_pack_site<<<grid_blocks_capped(matrix_elements), k_threads_per_block, 0, stream>>>(
                pack_args
            );
            check(cudaGetLastError());
            output_offset += matrix_elements;
            ++site_index;
        }
    }

    check(cudaStreamSynchronize(stream));
    std::vector<int> host_info_records{};
    host_info_records.resize(2 * site_count);
    int host_failure{};
    download(host_info_records.data(), device_info_records, host_info_records.size());
    download(&host_failure, device_failure, 1);
    const auto failed_record = std::find_if(
        host_info_records.begin(), host_info_records.end(), [](int info) { return info != 0; }
    );
    if (failed_record != host_info_records.end())
    {
        const auto record = static_cast<usize>(failed_record - host_info_records.begin());
        const auto failed_site = record / 2;
        fail(
            Status::cuda,
            "{} info {} at PEPS site ({}, {})",
            record % 2 == 0 ? "geqrf" : "ungqr",
            *failed_record,
            failed_site / static_cast<usize>(dimensions.ly),
            failed_site % static_cast<usize>(dimensions.ly)
        );
    }
    require(
        host_failure == 0,
        Status::cuda,
        "random PEPS phase extraction reported failure flag {}",
        host_failure
    );
}
}
