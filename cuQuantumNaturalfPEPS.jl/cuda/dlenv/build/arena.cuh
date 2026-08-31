#ifndef QNPEPS_DLENV_BUILD_ARENA_CUH
#define QNPEPS_DLENV_BUILD_ARENA_CUH

#include "dlenv/build/types.cuh"

namespace qnpeps::dlenv
{

class BuildAllocation
{
  public:
    explicit BuildAllocation(qnpeps_ctx& ctx) noexcept : ctx_(&ctx), dl_(ctx.dl) {}
    explicit BuildAllocation(BuildState& state) noexcept : dl_(state) {}

    auto carve(
        Linalg& la,
        const Dims& dims,
        int maxdim,
        usize retained_rows,
        bool density_route,
        usize scales_count,
        ArenaCursor& arena
    ) -> void
    {
        constexpr usize k_arena_tail_pad{64 * k_threads_per_block};
        constexpr usize k_density_arena_slack_allocations{16};

        const auto dim_bond = static_cast<usize>(dims.dim_bond);
        const auto dim_bond2 = dim_bond * dim_bond;

        const auto chi = std::min(static_cast<usize>(maxdim), dim_bond2);
        const auto chi2 = chi * chi;

        const auto dim_phys = static_cast<usize>(dims.dim_phys);
        const auto num_cols = static_cast<usize>(dims.ly);

        const auto out_slot = device_align(sizeof(cuFloatComplex) * chi2 * dim_bond2);

        const auto per_row = [&]
        {
            usize out{};
            out += device_align(sizeof(cuFloatComplex));
            out += device_align(sizeof(f32));
            out += device_align(sizeof(f64) * num_cols);
            out += 3 * out_slot;
            out += device_align(sizeof(int));
            return out;
        }();
        const auto known = retained_rows * (num_cols * out_slot + per_row);

        const auto rolling_r = out_slot;

        const auto site_peak = chi2 * dim_bond2 * dim_bond2;
        const auto scratch_elems = (4 + 3 * dim_phys) * site_peak + 4 * chi2 * dim_bond2;
        auto scratch = device_align(sizeof(cuFloatComplex) * scratch_elems);

        const auto qr_rows = static_cast<int>(chi * dim_bond2);
        const auto qr_cols = static_cast<int>(chi);
        scratch += la.qr_scratch(qr_rows, qr_cols).total();

        scratch += k_arena_tail_pad;

        dl_.fail = arena.take<int>(1);
        dl_.scales_all = arena.take<f64>(scales_count);
        dl_.unit_environment = arena.take<cuFloatComplex>(1);
        dl_.initial_factor = arena.take<cuFloatComplex>(1);
        dl_.known = arena.take_subarena(known);
        dl_.rolling_r = arena.take_subarena(rolling_r);
        dl_.scratch = arena.take_subarena(scratch);
        if (density_route)
        {
            const auto density_combined = chi * dim_bond2;
            const auto density_output = dim_bond2 * chi;
            const auto density_cuts = num_cols - 1;
            const auto density_left = density_cuts * density_combined * density_combined;
            const auto density_right = density_combined * density_combined * dim_bond2
                                       + density_combined * density_output + density_output * chi;
            const auto density_temporary = 2 * density_combined * density_output;
            const auto density_matrix = density_output * density_output;
            const auto density_state = num_cols * chi * dim_bond2 * chi;
            const auto density_bytes =
                sizeof(cuDoubleComplex)
                    * (density_left + density_right + density_temporary + density_matrix
                       + 2 * density_state)
                + sizeof(f64) * (2 * density_output + 2)
                + sizeof(i32) * (density_output + density_cuts + 1 + 3 * num_cols)
                + sizeof(QnpepsDensityRankRecord) * density_cuts
                + densitymatrix::eigen_workspace_bytes(la, static_cast<int>(density_output))
                + k_density_arena_slack_allocations * k_device_malloc_align;
            dl_.density = arena.take_subarena(density_bytes);
        }
    }

    auto ensure(Linalg& la) -> void
    {
        auto& ctx = *ctx_;
        if (ctx.dl.allocated) return;

        const auto& cfg = ctx.cfg;
        const int lx{cfg.lx};
        const int ly{cfg.ly};
        const auto dim_bond = cfg.dim_bond;
        const auto dim_phys = cfg.dim_phys;

        const Dims dims{lx, ly, dim_phys, dim_bond};
        const auto density_route = cfg.dlenv_truncation_route == QNPEPS_TRUNCATION_DENSITY;
        i64 peps_total{};
        for (auto row = 0; row < lx; ++row)
            peps_total += peps_row_elems(dims, row, row + 1);
        if (not ctx.dl.peps_buf)
        {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&ctx.dl.peps_buf),
                static_cast<usize>(peps_total) * sizeof(cuFloatComplex)
            ));
        }
        if (err_state() != QNPEPS_OK) return;

        const int chi_c{std::min(cfg.chi_dl, dim_bond * dim_bond)};
        const auto num_env_rows = static_cast<usize>(lx - 1);
        const auto num_cols = static_cast<usize>(ly);
        const usize scales_count{num_env_rows * num_cols};
        auto& context_arena = la.persistent_arena();
        carve(la, dims, chi_c, num_env_rows, density_route, scales_count, context_arena);

        const auto dlenv_bytes = qnpeps_dlenv_bytes(&cfg);
        for (auto& lane : ctx.dlenv.lanes)
        {
            if (lane.packed) continue;
            CUDA_CHECK(cudaMalloc(&lane.packed, static_cast<usize>(dlenv_bytes)));
            if (err_state() != QNPEPS_OK) return;
        }

        init_dl_units(la, ctx.dl.unit_environment, ctx.dl.initial_factor);

        ctx.dl.allocated = true;
    }

    auto release() noexcept -> void
    {
        if (dl_.peps_buf)
        {
            CUDA_NOCHECK(cudaFree(dl_.peps_buf));
            dl_.peps_buf = nullptr;
        }
        for (auto& entry : dl_.omegas)
        {
            if (entry.second) CUDA_NOCHECK(cudaFree(entry.second));
        }
        dl_.omegas.clear();
        if (not ctx_) return;
        for (auto& lane : ctx_->dlenv.lanes)
        {
            if (lane.packed)
            {
                CUDA_NOCHECK(cudaFree(lane.packed));
                lane.packed = nullptr;
            }
            if (lane.graph)
            {
                CUDA_NOCHECK(cudaGraphExecDestroy(lane.graph));
                lane.graph = nullptr;
            }
        }
    }

  private:
    qnpeps_ctx* ctx_{};
    BuildState& dl_;
};

auto dl_free(qnpeps_ctx& ctx) -> void;
}

#endif
