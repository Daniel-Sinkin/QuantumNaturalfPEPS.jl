#ifndef QNPEPS_ZIPUP_MPO_MPS_CUH
#define QNPEPS_ZIPUP_MPO_MPS_CUH

#include "arena_cursor.cuh"
#include "capi/qnpeps.h"
#include "contraction.cuh"
#include "linalg.cuh"
#include "permutation.cuh"
#include "rangefinder_rng.cuh"
#include "tensor.cuh"

#include <cmath>
#include <map>
#include <utility>
#include <vector>

namespace qnpeps::zipup
{
inline constexpr u64 k_rangefinder_seed{777};

struct Arenas
{
    ArenaCursor& known;
    ArenaCursor& rolling_r;
    ArenaCursor& scratch;
};

struct State
{
    cuFloatComplex* initial_factor{};
    f64* device_scales{};
    int* fail_flag{};
    std::map<std::pair<int, int>, cuFloatComplex*>* omegas{};
    RangefinderRng* rangefinder_rng{};
};

struct RangefinderArgs
{
    DeviceTensor input{};
    int rows{};
    int cols{};
    int maxdim{};
    DeviceTensor* q{};
    int* rank_out{};
    DeviceTensor* r{};
};

struct SweepArgs
{
    usize num_sites{};
    int maxdim{};
    f64* log_scale{};
    bool defer_scales{};
};

struct FusedPepsRowArgs
{
    const std::vector<DeviceTensor>* row_ket{};
    const std::vector<DeviceTensor>* environment{};
    DeviceTensor unit_environment{};
    int maxdim{};
    f64* log_scale{};
    bool defer_scales{};
};

auto rangefinder(State& state, Linalg& la, const Arenas& arenas, const RangefinderArgs& args)
    -> void;
auto normalize_factor(Linalg& la, ArenaCursor& scratch, DeviceTensor factor, f64* device_scale)
    -> void;
auto accumulate_log_scales(const State& state, usize count, f64& log_scale) -> void;
auto release_omegas(State& state) -> void;
auto fused_peps_row(
    State& state, Linalg& la, const Arenas& arenas, const FusedPepsRowArgs& args
) -> std::vector<DeviceTensor>;

template <typename PanelProvider>
auto sweep(
    State& state, Linalg& la, const Arenas& arenas, PanelProvider& provider, const SweepArgs& args
) -> std::vector<DeviceTensor>
{
    std::vector<DeviceTensor> output(args.num_sites);
    auto carried_factor = DeviceTensor{{1, 1, 1}, state.initial_factor};

    for (auto site = 0_uz; site < args.num_sites; ++site)
    {
        if (err_state() != QNPEPS_OK) return output;
        ArenaCursor column_scratch{arenas.scratch};
        const Arenas column_arenas{arenas.known, arenas.rolling_r, column_scratch};
        const auto panel = provider.make_panel(site, carried_factor, column_scratch, la);
        if (err_state() != QNPEPS_OK) return output;
        if (panel.dim.rank() != 4)
        {
            set_err(QNPEPS_ERR_INTERNAL);
            return output;
        }

        const int output_left{panel.dim[0]};
        const int physical_out{panel.dim[1]};
        const int mps_right{panel.dim[2]};
        const int mpo_right{panel.dim[3]};
        const int rows{output_left * physical_out};
        const int cols{mps_right * mpo_right};

        DeviceTensor q{};
        DeviceTensor r_factor{};
        int rank{};
        rangefinder(
            state,
            la,
            column_arenas,
            {
                .input = panel,
                .rows = rows,
                .cols = cols,
                .maxdim = args.maxdim,
                .q = &q,
                .rank_out = &rank,
                .r = &r_factor,
            }
        );
        if (err_state() != QNPEPS_OK) return output;

        normalize_factor(la, column_scratch, r_factor, state.device_scales + site);
        if (err_state() != QNPEPS_OK) return output;

        output[site] = DeviceTensor{{output_left, physical_out, rank}, q.d};
        const auto reshaped_r = DeviceTensor{{rank, mps_right, mpo_right}, r_factor.d};
        arenas.rolling_r.rewind();
        carried_factor = permute_axes(arenas.rolling_r, reshaped_r, {0, 2, 1}, false, la.stream());
    }

    if (not args.defer_scales)
    {
        if (not args.log_scale)
        {
            set_err(QNPEPS_ERR_INTERNAL);
            return output;
        }
        accumulate_log_scales(state, args.num_sites, *args.log_scale);
    }

    auto& last = output[args.num_sites - 1];
    DeviceTensor folded{};
    const auto folded_ok = contract(
        arenas.known,
        la,
        {
            .dims_a = last.dim,
            .contracted_a = {2},
            .dims_b = carried_factor.dim,
            .contracted_b = {0},
        },
        last,
        carried_factor,
        folded
    );
    if (not folded_ok) return output;
    last = DeviceTensor{{folded.dim[0], folded.dim[1], 1}, folded.d};
    return output;
}

auto output_bytes(const QnpepsZipupMpoMpsDesc* descriptor) -> i64;
auto execute(const QnpepsZipupMpoMpsDesc* descriptor, const QnpepsZipupMpoMpsArgs* args)
    -> qnpeps_status;
}

#endif
