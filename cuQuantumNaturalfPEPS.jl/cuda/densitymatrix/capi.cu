#include "capi/internal/format.cuh"
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
    require(
        not invalid_buffer,
        Status::bad_version,
        "device buffer struct_size {}, reserved {}, {} bytes not a multiple of {}",
        input.struct_size,
        input.reserved,
        input.bytes,
        sizeof(T)
    );
    require(
        input.values != 0_u64 or input.bytes == 0_u64,
        Status::null_argument,
        "device buffer of {} bytes has a null address",
        input.bytes
    );
    const auto is_empty = input.values == 0_u64 or input.bytes == 0_u64;
    require(optional or not is_empty, Status::null_argument, "required device buffer is empty");
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
    const auto valid_header =
        input.struct_size == sizeof(QnpepsDensityWorkspace) and input.reserved == 0_u32;
    require(
        valid_header,
        Status::bad_version,
        "density workspace struct_size {}, expected {}",
        input.struct_size,
        sizeof(QnpepsDensityWorkspace)
    );
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

auto require_trace_disabled(const QnpepsDensityTrace& input) -> void
{
    const auto valid_header =
        input.struct_size == sizeof(QnpepsDensityTrace) and input.enabled <= 1_u32;
    require(
        valid_header,
        Status::bad_version,
        "density trace struct_size {}, enabled {}",
        input.struct_size,
        input.enabled
    );
    require(input.enabled == 0_u32, Status::bad_config, "density trace capture is not available");
}

auto sites(const QnpepsDensityApplyArgs& input) -> std::vector<densitymatrix::Site>
{
    std::vector<densitymatrix::Site> output{};
    const auto invalid_sites =
        input.num_sites < 2_u64 or input.sites == 0_u64
        or input.sites_bytes != input.num_sites * sizeof(QnpepsDensitySitePlan);
    require(
        not invalid_sites,
        Status::bad_config,
        "density apply rejects {} sites described by {} plan bytes",
        input.num_sites,
        input.sites_bytes
    );
    const auto* plans =
        reinterpret_cast<const QnpepsDensitySitePlan*>(static_cast<uintptr_t>(input.sites));
    output.reserve(static_cast<usize>(input.num_sites));
    for (auto index = 0_u64; index < input.num_sites; ++index)
    {
        const auto& plan = plans[index];
        const auto invalid_plan = plan.struct_size != sizeof(QnpepsDensitySitePlan)
                                  or plan.reserved != 0_u32
                                  or plan.operator_input != plan.physical_input;
        require(
            not invalid_plan,
            Status::bad_version,
            "density site plan {} has struct_size {}, operator input {}, physical input {}",
            index,
            plan.struct_size,
            plan.operator_input,
            plan.physical_input
        );
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

auto require_bound_stream(qnpeps_ctx* ctx, void* stream) -> void
{
    require(ctx != nullptr, Status::null_argument, "density call without a context");
    const auto foreign_stream = stream and static_cast<cudaStream_t>(stream) != ctx->stream();
    require(
        not foreign_stream,
        Status::bad_config,
        "density call stream differs from the context stream"
    );
}

auto density_workspace_sizes(qnpeps_ctx* ctx, const QnpepsDensityWorkspaceQuery* query) -> void
{
    require(
        ctx != nullptr and query != nullptr,
        Status::null_argument,
        "density workspace query without a context or query"
    );
    const auto valid_query_header =
        query->struct_size == sizeof(QnpepsDensityWorkspaceQuery) and query->reserved == 0_u32;
    require(
        valid_query_header,
        Status::bad_version,
        "density workspace query struct_size {}, expected {}",
        query->struct_size,
        sizeof(QnpepsDensityWorkspaceQuery)
    );
    require(
        query->sizes_out != 0_u64, Status::null_argument, "density workspace query has no output"
    );
    auto* output =
        reinterpret_cast<QnpepsDensityWorkspaceSizes*>(static_cast<uintptr_t>(query->sizes_out));
    const auto valid_output_header =
        output->struct_size == sizeof(QnpepsDensityWorkspaceSizes) and output->reserved == 0_u32;
    require(
        valid_output_header,
        Status::bad_version,
        "density workspace sizes struct_size {}, expected {}",
        output->struct_size,
        sizeof(QnpepsDensityWorkspaceSizes)
    );
    const auto invalid_query = query->num_sites < 2 or query->input_bond < 1
                               or query->operator_bond < 1 or query->output_dimension < 1
                               or query->upper_bond < 1 or query->lanes < 1;
    require(not invalid_query, Status::bad_config, "density workspace query rejects {}", *query);
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
}

auto densitymatrix_apply(qnpeps_ctx* ctx, const QnpepsDensityApplyArgs* input, void* stream) -> void
{
    require(input != nullptr, Status::null_argument, "density apply without arguments");
    require_bound_stream(ctx, stream);
    const auto valid_header =
        input->struct_size == sizeof(QnpepsDensityApplyArgs) and input->normalize <= 1_u32;
    require(
        valid_header,
        Status::bad_version,
        "density apply struct_size {}, normalize {}",
        input->struct_size,
        input->normalize
    );
    auto planned_sites = sites(*input);
    const auto state = buffer<const cuDoubleComplex>(input->state_values);
    const auto operator_values = buffer<const cuDoubleComplex>(input->operator_values);
    const auto dimensions = buffer<i32>(input->result_dimensions);
    const auto values = buffer<cuDoubleComplex>(input->result_values);
    const auto normalization = buffer<f64>(input->normalization_log);
    const auto gauge = buffer<f64>(input->output_gauge);
    const auto workspace = workspace_outputs(input->workspace);
    require_trace_disabled(input->trace);
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
    require(
        not insufficient_buffers,
        Status::bad_config,
        "density apply buffers too small, {} result values for {} required, {} dimensions for {} "
        "sites",
        values.count,
        result_required,
        dimensions.count,
        site_span.size()
    );
    auto transient_arena = ctx->transient_arena();
    auto& arena = transient_arena.cursor();
    const auto internal_workspace = densitymatrix::take_workspace(
        ctx->linalg(), arena, density_geometry(site_span, input->upper_bond)
    );
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
}

auto density_filter_apply(qnpeps_ctx* ctx, const QnpepsDensityFilterArgs* input, void* stream)
    -> void
{
    require(input != nullptr, Status::null_argument, "density filter without arguments");
    require_bound_stream(ctx, stream);
    const auto invalid_input =
        input->struct_size != sizeof(QnpepsDensityFilterArgs) or input->reserved != 0_u32
        or input->settings.struct_size != sizeof(QnpepsDensitySettings)
        or input->settings.precision != 1_u32 or input->settings.mindim != 1_u32
        or input->settings.reserved != 0_u32 or input->order < 1 or input->natural_cap < 1
        or input->applied_cap < 1;
    require(
        not invalid_input,
        Status::bad_config,
        "density filter rejects struct_size {}, order {}, natural cap {}, applied cap {}",
        input->struct_size,
        input->order,
        input->natural_cap,
        input->applied_cap
    );
    const auto solver = buffer<const f64>(input->solver_values);
    const auto sorted = buffer<f64>(input->sorted_values);
    const auto retained = buffer<f64>(input->retained_values);
    const auto indices = buffer<i32>(input->sorted_indices);
    const auto rank = buffer<i32>(input->active_rank);
    const auto record = buffer<QnpepsDensityRankRecord>(input->record);
    const auto order = static_cast<usize>(input->order);
    const auto insufficient_buffers = solver.count < order or sorted.count < order
                                      or retained.count < order or indices.count < order
                                      or rank.count < 1 or record.count < 1;
    require(
        not insufficient_buffers,
        Status::bad_config,
        "density filter buffers too small for order {}",
        order
    );
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
    check(cudaGetLastError());
}
}

extern "C" qnpeps_status qnpeps_density_workspace_sizes(
    qnpeps_ctx* ctx, const QnpepsDensityWorkspaceQuery* query
)
{
    return guarded([&] { density_workspace_sizes(ctx, query); });
}

extern "C" qnpeps_status qnpeps_densitymatrix_apply(
    qnpeps_ctx* ctx, const QnpepsDensityApplyArgs* input, void* stream
)
{
    return guarded([&] { densitymatrix_apply(ctx, input, stream); });
}

extern "C" qnpeps_status qnpeps_density_filter_apply(
    qnpeps_ctx* ctx, const QnpepsDensityFilterArgs* input, void* stream
)
{
    return guarded([&] { density_filter_apply(ctx, input, stream); });
}
