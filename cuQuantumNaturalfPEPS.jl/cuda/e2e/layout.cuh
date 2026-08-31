#ifndef QNPEPS_E2E_LAYOUT_CUH
#define QNPEPS_E2E_LAYOUT_CUH

#include "common.cuh"

#include <vector>

namespace qn_e2e
{

inline auto bond_dim(int axis_len, int pos, int dim_bond) -> int
{
    if (pos <= 0 or pos >= axis_len) return 1;
    return dim_bond;
}

inline auto site_slice(const QnpepsE2eConfig& cfg, int i, int c) -> i64
{
    return static_cast<i64>(bond_dim(cfg.ly, c, cfg.dim_bond))
           * bond_dim(cfg.lx, i + 1, cfg.dim_bond) * bond_dim(cfg.ly, c + 1, cfg.dim_bond)
           * bond_dim(cfg.lx, i, cfg.dim_bond);
}

inline auto compact_count(const QnpepsE2eConfig& cfg) -> i64
{
    i64 acc{0};
    for (int i{0}; i < cfg.lx; ++i)
    {
        for (int c{0}; c < cfg.ly; ++c)
            acc += site_slice(cfg, i, c);
    }
    return acc;
}

inline auto dense_count(const QnpepsE2eConfig& cfg) -> i64
{
    return static_cast<i64>(cfg.dim_phys) * compact_count(cfg);
}

inline auto build_slot_site(const QnpepsE2eConfig& cfg, i64& out_compact) -> std::vector<i32>
{
    std::vector<i32> slot{};
    i64 acc{0};
    int site{0};
    for (int i{0}; i < cfg.lx; ++i)
    {
        for (int c{0}; c < cfg.ly; ++c)
        {
            const i64 slice{site_slice(cfg, i, c)};
            for (i64 k{0}; k < slice; ++k)
                slot.push_back(site);
            acc += slice;
            ++site;
        }
    }
    out_compact = acc;
    return slot;
}

}

#endif
