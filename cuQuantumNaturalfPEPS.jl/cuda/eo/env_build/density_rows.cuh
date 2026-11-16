#pragma once

#include "parts.cuh"

namespace qn_eloc::env
{
auto carve_density_rows(Worker& w, const Shape& sh, Carver& cv) -> void;

auto build_density_row(
    Worker& w,
    int row,
    bool contract_up,
    const std::vector<int>* ks_adj,
    const std::function<cf*(int)>& adj_site,
    const std::function<cf*(int)>& out_site,
    f64* f_lane,
    std::vector<int>& ks_out
) -> void;

auto clear_density_flags(Worker& w) -> void;

auto report_density_failure(Worker& w) -> void;
}
