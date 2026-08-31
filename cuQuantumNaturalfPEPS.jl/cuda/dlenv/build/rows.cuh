#ifndef QNPEPS_DLENV_BUILD_ROWS_CUH
#define QNPEPS_DLENV_BUILD_ROWS_CUH

#include "dlenv/build/types.cuh"

namespace qnpeps::dlenv
{

struct BuildEnvRowArgs
{
    Dims dims{};
    const std::vector<DeviceTensor>* row_ket{};
    const std::vector<DeviceTensor>* env_below{};
    int maxdim{};
    int* fail_flag{};
    int build_step{};
};

class EnvironmentRowBuilder
{
  public:
    EnvironmentRowBuilder(BuildState& state, Linalg& linalg, const Arenas& arenas, const Dims& dims)
        : build_state_(state), linalg_(linalg), arenas_(arenas), dims_(dims)
    {
    }

    [[nodiscard]] auto build_row(const BuildEnvRowArgs& args) -> std::vector<DeviceTensor>
    {
        const auto& row_ket = *args.row_ket;
        const auto num_cols = static_cast<usize>(args.dims.ly);
        const DeviceTensor unit_environment{{1, 1, 1, 1}, build_state_.unit_environment};
        zipup::State state{
            .initial_factor = build_state_.initial_factor,
            .device_scales =
                build_state_.scales_all + static_cast<usize>(args.build_step) * num_cols,
            .fail_flag = args.fail_flag,
            .omegas = &build_state_.omegas,
            .rangefinder_rng = &build_state_.rangefinder_rng,
        };
        auto grouped_output = zipup::fused_peps_row(
            state,
            linalg_,
            arenas_,
            {
                .row_ket = &row_ket,
                .environment = args.env_below,
                .unit_environment = unit_environment,
                .maxdim = args.maxdim,
            }
        );
        if (err_state() != QNPEPS_OK) return std::vector<DeviceTensor>(num_cols);

        std::vector<DeviceTensor> output{};
        output.resize(num_cols);
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const auto vertical = row_ket[col].dim[1];
            output[col] = DeviceTensor{
                {grouped_output[col].dim[0], vertical, vertical, grouped_output[col].dim[2]},
                grouped_output[col].d
            };
        }
        return output;
    }

    [[nodiscard]] auto build_density_row(
        const std::vector<DeviceTensor>& row_ket,
        const std::vector<DeviceTensor>& environment,
        int maxdim,
        f64 cutoff,
        int build_step,
        f64& gauge
    ) -> std::vector<DeviceTensor>
    {
        return density_row(
            linalg_,
            build_state_.known,
            build_state_.density,
            {.row_ket = row_ket,
             .environment = environment,
             .maxdim = maxdim,
             .cutoff = cutoff,
             .input_gauge = gauge,
             .device_scales =
                 build_state_.scales_all + static_cast<usize>(build_step) * row_ket.size()},
            gauge
        );
    }

    [[nodiscard]] auto build_rows(
        const std::vector<PepsRow>& peps,
        int maxdim,
        int* fail_flag,
        int truncation_route,
        f64 density_cutoff
    ) -> std::vector<DlEnvRow>
    {
        const auto num_env_rows = static_cast<usize>(dims_.lx - 1);
        std::vector<DlEnvRow> env_rows{};
        env_rows.resize(num_env_rows);

        const auto last_env = num_env_rows - 1;
        env_rows[last_env] = build_row({
            .dims = dims_,
            .row_ket = &peps[num_env_rows],
            .env_below = nullptr,
            .maxdim = maxdim,
            .fail_flag = fail_flag,
            .build_step = 0,
        });

        auto density_gauge = 0.0;
        if (truncation_route == QNPEPS_TRUNCATION_DENSITY)
        {
            zipup::State boundary_state{
                .device_scales = build_state_.scales_all,
            };
            zipup::accumulate_log_scales(
                boundary_state, static_cast<usize>(dims_.ly), density_gauge
            );
        }

        for (usize row{num_env_rows}; row >= 2; --row)
        {
            if (err_state() != QNPEPS_OK) return env_rows;
            const int build_step{static_cast<int>(num_env_rows - row + 1)};
            if (truncation_route == QNPEPS_TRUNCATION_DENSITY)
            {
                env_rows[row - 2] = build_density_row(
                    peps[row - 1],
                    env_rows[row - 1],
                    maxdim,
                    density_cutoff,
                    build_step,
                    density_gauge
                );
            }
            else
                env_rows[row - 2] = build_row({
                    .dims = dims_,
                    .row_ket = &peps[row - 1],
                    .env_below = &env_rows[row - 1],
                    .maxdim = maxdim,
                    .fail_flag = fail_flag,
                    .build_step = build_step,
                });
        }

        return env_rows;
    }

  private:
    BuildState& build_state_;
    Linalg& linalg_;
    const Arenas& arenas_;
    const Dims& dims_;
};

}

#endif
