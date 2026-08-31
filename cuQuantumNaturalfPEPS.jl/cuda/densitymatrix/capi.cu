#include "capi/qnpeps.h"
#include "densitymatrix/backend.cuh"
#include "densitymatrix/kernels.cuh"
#include "densitymatrix/types.cuh"
#include "core/qnpeps_ctx.cuh"

#include <cstdint>
#include <span>
#include <vector>

using namespace qnpeps;

namespace
{
template <typename T>
struct BufferView
{
    T* values{};
    usize count{};
};

template <typename T>
auto buffer(const QnpepsDeviceBuffer& input, bool optional = false) -> BufferView<T>
{
    const auto invalid_buffer = input.struct_size != sizeof(QnpepsDeviceBuffer)
                                or input.reserved != 0_u32 or input.bytes % sizeof(T) != 0;
    if (invalid_buffer)
    {
        set_err(QNPEPS_ERR_BAD_VERSION);
        return {};
    }
    if (input.values == 0_u64 and input.bytes != 0_u64)
    {
        set_err(QNPEPS_ERR_NULL_ARG);
        return {};
    }
    if (not optional and (input.values == 0_u64 or input.bytes == 0_u64))
    {
        set_err(QNPEPS_ERR_NULL_ARG);
        return {};
    }
    return {
        reinterpret_cast<T*>(static_cast<uintptr_t>(input.values)),
        static_cast<usize>(input.bytes / sizeof(T)),
    };
}

struct WorkspaceOutputs
{
    i32* active_ranks{};
    usize active_rank_count{};
    QnpepsDensityRankRecord* records{};
    usize record_count{};
};

auto workspace_outputs(const QnpepsDensityWorkspace& input) -> WorkspaceOutputs
{
    if (input.struct_size != sizeof(QnpepsDensityWorkspace) or input.reserved != 0_u32)
    {
        set_err(QNPEPS_ERR_BAD_VERSION);
        return {};
    }
    const auto ranks = buffer<i32>(input.active_ranks);
    const auto records = buffer<QnpepsDensityRankRecord>(input.truncation_records);
    return {
        .active_ranks = ranks.values,
        .active_rank_count = ranks.count,
        .records = records.values,
        .record_count = records.count,
    };
}

auto density_geometry(std::span<const densitymatrix::Site> planned_sites, i64 upper_bond)
    -> densitymatrix::Geometry
{
    auto combined = 1_i64;
    auto physical = 1_i64;
    for (const auto& site : planned_sites)
    {
        combined = std::max(
            combined,
            std::max(site.state_left * site.operator_left, site.state_right * site.operator_right)
        );
        physical = std::max(physical, site.physical_output);
    }
    return {
        .num_sites = static_cast<i64>(planned_sites.size()),
        .input_bond = combined,
        .operator_bond = 1,
        .output_dimension = physical,
        .upper_bond = upper_bond,
        .lanes = 1,
    };
}

auto trace_available(const QnpepsDensityTrace& input) -> bool
{
    if (input.struct_size != sizeof(QnpepsDensityTrace) or input.enabled > 1_u32)
    {
        set_err(QNPEPS_ERR_BAD_VERSION);
        return false;
    }
    if (input.enabled != 0_u32) return set_err(QNPEPS_ERR_BAD_CONFIG), false;
    return true;
}

auto sites(const QnpepsDensityApplyArgs& input) -> std::vector<densitymatrix::Site>
{
    std::vector<densitymatrix::Site> output{};
    const auto invalid_sites =
        input.num_sites < 2_u64 or input.sites == 0_u64
        or input.sites_bytes != input.num_sites * sizeof(QnpepsDensitySitePlan);
    if (invalid_sites)
    {
        set_err(QNPEPS_ERR_BAD_CONFIG);
        return output;
    }
    const auto* plans =
        reinterpret_cast<const QnpepsDensitySitePlan*>(static_cast<uintptr_t>(input.sites));
    output.reserve(static_cast<usize>(input.num_sites));
    for (auto index = 0_u64; index < input.num_sites; ++index)
    {
        const auto& plan = plans[index];
        const auto invalid_plan = plan.struct_size != sizeof(QnpepsDensitySitePlan)
                                  or plan.reserved != 0_u32
                                  or plan.operator_input != plan.physical_input;
        if (invalid_plan)
        {
            set_err(QNPEPS_ERR_BAD_VERSION);
            return {};
        }
        output.push_back(
            {.site = plan.site,
             .num_sites = plan.num_sites,
             .state_left = plan.state_left,
             .physical_input = plan.physical_input,
             .state_right = plan.state_right,
             .operator_left = plan.operator_left,
             .physical_output = plan.physical_output,
             .operator_right = plan.operator_right,
             .state_offset = static_cast<usize>(plan.state_offset),
             .operator_offset = static_cast<usize>(plan.operator_offset),
             .output_offset = static_cast<usize>(plan.output_offset)}
        );
    }
    return output;
}

auto bound_stream(qnpeps_ctx* ctx, void* stream) -> bool
{
    if (not ctx)
    {
        set_err(QNPEPS_ERR_NULL_ARG);
        return false;
    }
    if (stream and static_cast<cudaStream_t>(stream) != ctx->stream())
    {
        set_err(QNPEPS_ERR_BAD_CONFIG);
        return false;
    }
    return true;
}
}

extern "C" qnpeps_status qnpeps_density_workspace_sizes(
    qnpeps_ctx* ctx, const QnpepsDensityWorkspaceQuery* query
)
{
    reset_err();
    if (not ctx or not query) return set_err(QNPEPS_ERR_NULL_ARG);
    if (query->struct_size != sizeof(QnpepsDensityWorkspaceQuery) or query->reserved != 0_u32)
        return set_err(QNPEPS_ERR_BAD_VERSION);
    if (query->sizes_out == 0_u64) return set_err(QNPEPS_ERR_NULL_ARG);
    auto* output =
        reinterpret_cast<QnpepsDensityWorkspaceSizes*>(static_cast<uintptr_t>(query->sizes_out));
    if (output->struct_size != sizeof(QnpepsDensityWorkspaceSizes) or output->reserved != 0_u32)
        return set_err(QNPEPS_ERR_BAD_VERSION);
    const auto invalid_query = query->num_sites < 2 or query->input_bond < 1
                               or query->operator_bond < 1 or query->output_dimension < 1
                               or query->upper_bond < 1 or query->lanes < 1;
    if (invalid_query)
    {
        return set_err(QNPEPS_ERR_BAD_CONFIG);
    }
    const auto sites = static_cast<usize>(query->num_sites);
    const auto cuts = sites - 1;
    const auto input_bond = static_cast<usize>(query->input_bond);
    const auto operator_bond = static_cast<usize>(query->operator_bond);
    const auto output_dimension = static_cast<usize>(query->output_dimension);
    const auto upper_bond = static_cast<usize>(query->upper_bond);
    const auto lanes = static_cast<usize>(query->lanes);
    const auto combined = input_bond * operator_bond;
    const auto capped_output = output_dimension * upper_bond;
    const auto left = lanes * cuts * combined * combined;
    const auto products = lanes * combined * combined * output_dimension;
    const auto blocks = lanes * combined * capped_output;
    const auto bases = lanes * capped_output * upper_bond;
    const auto right = products + blocks + bases;
    const auto temporary = lanes * 2 * combined * capped_output;
    const auto density = lanes * capped_output * capped_output;
    const auto linear = lanes * capped_output;
    const auto ranks = lanes * cuts;
    const auto solver_bytes =
        densitymatrix::eigen_workspace_bytes(ctx->linalg(), static_cast<int>(capped_output));
    *output = {
        .struct_size = static_cast<u32>(sizeof(QnpepsDensityWorkspaceSizes)),
        .reserved = 0_u32,
        .combined_input = combined,
        .capped_output = capped_output,
        .left_environment_bytes = left * sizeof(cuDoubleComplex),
        .right_block_bytes = right * sizeof(cuDoubleComplex),
        .temporary_bytes = temporary * sizeof(cuDoubleComplex),
        .density_bytes = density * sizeof(cuDoubleComplex),
        .eigenvalues_bytes = linear * sizeof(f64),
        .sort_keys_bytes = linear * sizeof(f64),
        .sort_indices_bytes = linear * sizeof(i32),
        .active_ranks_bytes = ranks * sizeof(i32),
        .truncation_records_bytes = ranks * sizeof(QnpepsDensityRankRecord),
        .solver_workspace_bytes = solver_bytes,
        .solver_information_bytes = sizeof(i32),
        .arena_bytes = arena_reservation_bytes(),
    };
    return err_state();
}

extern "C" qnpeps_status qnpeps_densitymatrix_apply(
    qnpeps_ctx* ctx, const QnpepsDensityApplyArgs* input, void* stream
)
{
    reset_err();
    if (not input or not bound_stream(ctx, stream)) return err_state();
    if (input->struct_size != sizeof(QnpepsDensityApplyArgs) or input->normalize > 1_u32)
        return set_err(QNPEPS_ERR_BAD_VERSION);
    auto planned_sites = sites(*input);
    if (err_state() != QNPEPS_OK) return err_state();
    const auto state = buffer<const cuDoubleComplex>(input->state_values);
    const auto operator_values = buffer<const cuDoubleComplex>(input->operator_values);
    const auto dimensions = buffer<i32>(input->result_dimensions);
    const auto values = buffer<cuDoubleComplex>(input->result_values);
    const auto normalization = buffer<f64>(input->normalization_log);
    const auto gauge = buffer<f64>(input->output_gauge);
    const auto workspace = workspace_outputs(input->workspace);
    if (not trace_available(input->trace)) return err_state();
    if (err_state() != QNPEPS_OK) return err_state();
    const std::span<const densitymatrix::Site> site_span{planned_sites};
    const auto cuts = site_span.size() - 1;
    const auto upper_bond = static_cast<usize>(input->upper_bond);
    const auto result_required =
        planned_sites.back().output_offset
        + upper_bond * static_cast<usize>(planned_sites.back().physical_output) * upper_bond;
    const auto insufficient_buffers =
        state.count < 1 or operator_values.count < 1 or dimensions.count < 3 * site_span.size()
        or values.count < result_required or normalization.count < 1 or gauge.count < 1
        or workspace.active_rank_count < cuts or workspace.record_count < cuts;
    if (insufficient_buffers)
    {
        return set_err(QNPEPS_ERR_BAD_CONFIG);
    }
    auto transient_arena = ctx->transient_arena();
    auto& arena = transient_arena.cursor();
    if (err_state() != QNPEPS_OK) return err_state();
    const auto internal_workspace = densitymatrix::take_workspace(
        ctx->linalg(), arena, density_geometry(site_span, input->upper_bond)
    );
    if (err_state() != QNPEPS_OK) return err_state();
    densitymatrix::apply(
        ctx->linalg(),
        {.settings = input->settings,
         .upper_bond = input->upper_bond,
         .normalize = input->normalize != 0_u32,
         .input_gauge = input->input_gauge,
         .sites = site_span,
         .workspace = internal_workspace,
         .state_values = state.values,
         .state_value_count = state.count,
         .operator_values = operator_values.values,
         .operator_value_count = operator_values.count,
         .result_dimensions = dimensions.values,
         .result_dimension_count = dimensions.count,
         .result_values = values.values,
         .result_value_count = values.count,
         .normalization_log = normalization.values,
         .output_gauge = gauge.values,
         .active_ranks_out = workspace.active_ranks,
         .records_out = workspace.records}
    );
    return err_state();
}

extern "C" qnpeps_status qnpeps_density_filter_apply(
    qnpeps_ctx* ctx, const QnpepsDensityFilterArgs* input, void* stream
)
{
    reset_err();
    if (not input or not bound_stream(ctx, stream)) return err_state();
    const auto invalid_input =
        input->struct_size != sizeof(QnpepsDensityFilterArgs) or input->reserved != 0_u32
        or input->settings.struct_size != sizeof(QnpepsDensitySettings)
        or input->settings.precision != 1_u32 or input->settings.mindim != 1_u32
        or input->settings.reserved != 0_u32 or input->order < 1 or input->natural_cap < 1
        or input->applied_cap < 1;
    if (invalid_input) return set_err(QNPEPS_ERR_BAD_CONFIG);
    const auto solver = buffer<const f64>(input->solver_values);
    const auto sorted = buffer<f64>(input->sorted_values);
    const auto retained = buffer<f64>(input->retained_values);
    const auto indices = buffer<i32>(input->sorted_indices);
    const auto rank = buffer<i32>(input->active_rank);
    const auto record = buffer<QnpepsDensityRankRecord>(input->record);
    const auto order = static_cast<usize>(input->order);
    const auto insufficient_buffers =
        err_state() != QNPEPS_OK or solver.count < order or sorted.count < order
        or retained.count < order or indices.count < order or rank.count < 1 or record.count < 1;
    if (insufficient_buffers)
        return err_state() == QNPEPS_OK ? set_err(QNPEPS_ERR_BAD_CONFIG) : err_state();
    const auto filter_args = densitymatrix::FilterArgs{
        .solver_values = solver.values,
        .order = input->order,
        .natural_cap = input->natural_cap,
        .applied_cap = input->applied_cap,
        .mindim = input->settings.mindim,
        .cutoff = input->settings.relative_cutoff,
        .sorted_values = sorted.values,
        .retained_values = retained.values,
        .sorted_indices = indices.values,
        .active_rank = rank.values,
        .record = record.values,
    };
    densitymatrix::cu_filter<<<1, 1, 0, ctx->stream()>>>(filter_args);
    CUDA_CHECK(cudaGetLastError());
    return err_state();
}
