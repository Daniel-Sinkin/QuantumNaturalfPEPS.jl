#include "density_rows.cuh"

namespace qn_eloc::env
{
namespace
{
using qnpeps::densitymatrix::BatchedRow;
using qnpeps::densitymatrix::BatchedSite;

[[nodiscard]] auto row_sites(
    Worker& w,
    int row,
    bool contract_up,
    const std::vector<int>* ks_adj,
    const std::function<cf*(int)>& adj_site,
    const std::function<cf*(int)>& out_site
) -> std::vector<BatchedSite>
{
    const auto& sh = w.sh;
    const auto columns = static_cast<usize>(sh.ly);
    auto sites = std::vector<BatchedSite>(columns);
    for (auto column = 0_uz; column < columns; ++column)
    {
        const auto col = static_cast<int>(column);
        const auto west = static_cast<i64>(bd(sh.ly, col, sh.dim_bond));
        const auto south = static_cast<i64>(bd(sh.lx, row + 1, sh.dim_bond));
        const auto east = static_cast<i64>(bd(sh.ly, col + 1, sh.dim_bond));
        const auto north = static_cast<i64>(bd(sh.lx, row, sh.dim_bond));
        const auto south_stride = west;
        const auto east_stride = west * south;
        const auto north_stride = east_stride * east;
        const auto* state = ks_adj ? adj_site(col) : nullptr;
        sites[column] = BatchedSite{
            .state_left = ks_adj ? static_cast<i64>((*ks_adj)[column]) : 1_i64,
            .physical_input = contract_up ? north : south,
            .state_right = ks_adj ? static_cast<i64>((*ks_adj)[column + 1]) : 1_i64,
            .operator_left = west,
            .physical_output = contract_up ? south : north,
            .operator_right = east,
            .operator_strides =
                {
                    .left = 1,
                    .input = contract_up ? north_stride : south_stride,
                    .output = contract_up ? south_stride : north_stride,
                    .right = east_stride,
                },
            .state = reinterpret_cast<const cuFloatComplex*>(state),
            .operator_values = reinterpret_cast<const cuFloatComplex*>(proj_site(w, col)),
            .output = reinterpret_cast<cuFloatComplex*>(out_site(col)),
        };
    }
    return sites;
}
}

auto carve_density_rows(Worker& w, const Shape& sh, Carver& cv) -> void
{
    w.density_rows = qnpeps::densitymatrix::take_batched_workspace(
        w.la,
        cv,
        {
            .num_sites = sh.ly,
            .state_bond = std::max(sh.chi, sh.dim_bond),
            .operator_bond = sh.dim_bond,
            .output_dimension = sh.dim_bond,
            .upper_bond = sh.chi,
            .lanes = sh.lanes,
        }
    );
}

auto build_density_row(
    Worker& w,
    int row,
    bool contract_up,
    const std::vector<int>* ks_adj,
    const std::function<cf*(int)>& adj_site,
    const std::function<cf*(int)>& out_site,
    f64* f_lane,
    std::vector<int>& ks_out
) -> void
{
    project_row(w, row);
    const auto sites = row_sites(w, row, contract_up, ks_adj, adj_site, out_site);
    const auto batched_row = BatchedRow{
        .sites = sites,
        .state_lane_stride = w.slot_env,
        .operator_lane_stride = w.slot_proj,
        .output_lane_stride = w.slot_env,
        .log_gauge = f_lane,
        .lanes = w.sh.lanes,
    };
    std::vector<i64> bonds{};
    qnpeps::collapse_to_internal(
        {Status::out_of_memory},
        [&]
        {
            bonds =
                ks_adj
                    ? qnpeps::densitymatrix::apply_batched(w.la, w.density_rows, batched_row)
                    : qnpeps::densitymatrix::normalize_batched(w.la, w.density_rows, batched_row);
        }
    );
    ks_out.resize(bonds.size());
    for (auto index = 0_uz; index < bonds.size(); ++index)
        ks_out[index] = static_cast<int>(bonds[index]);
}

auto clear_density_flags(Worker& w) -> void
{
    if (not w.sh.density_batched) return;
    qnpeps::densitymatrix::clear_lane_flags(w.la, w.density_rows);
}

auto report_density_failure(Worker& w) -> void
{
    if (not w.sh.density_batched) return;
    const auto flags = qnpeps::densitymatrix::read_lane_flags(w.la, w.density_rows, w.sh.lanes);
    for (auto lane = 0_uz; lane < flags.size(); ++lane)
    {
        require(
            flags[lane] == 0_u32,
            Status::cuda,
            "local energy density rows flagged lane {} of {} with {:#x}",
            lane,
            flags.size(),
            flags[lane]
        );
    }
}
}
