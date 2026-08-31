function _densitymatrix_apply_args(
    plan::DensityMatrixPlan,
    state_values::CUDA.CuVector{ComplexF64},
    operator_values::CUDA.CuVector{ComplexF64};
    normalize::Bool,
    input_gauge::Float64,
)::QnpepsDensityApplyArgs
    return QnpepsDensityApplyArgs(
        normalize=UInt32(normalize),
        settings=_density_protocol_settings(plan.config.settings),
        upper_bond=Int64(plan.config.dims.chi),
        num_sites=UInt64(length(plan.sites)),
        sites=UInt64(UInt(pointer(plan.sites))),
        sites_bytes=UInt64(sizeof(QnpepsDensitySitePlan) * length(plan.sites)),
        input_gauge=input_gauge,
        workspace=plan.workspace_wire,
        trace=plan.trace_wire,
        state_values=protocol_buffer(state_values),
        operator_values=protocol_buffer(operator_values),
        result_dimensions=protocol_buffer(plan.result_dimensions),
        result_values=protocol_buffer(plan.result_values),
        normalization_log=protocol_buffer(plan.normalization_log),
        output_gauge=protocol_buffer(plan.output_gauge),
    )
end

function densitymatrix_apply!(
    plan::DensityMatrixPlan,
    operator::DensityMatrixProductOperator,
    state::DensityMatrixProductState;
    normalize::Bool=false,
    input_gauge::Float64=0.0,
)::Nothing
    isopen(plan) || throw(ArgumentError("density plan is destroyed"))
    CUDA.device() == plan.device || throw(ArgumentError("density plan device mismatch"))
    CUDA.stream().handle == plan.stream.handle ||
        throw(ArgumentError("density plan stream mismatch"))
    isfinite(input_gauge) || throw(ArgumentError("density input gauge must be finite"))
    args =
        Ref(_densitymatrix_apply_args(plan, state.values, operator.values; normalize, input_gauge))
    status = GC.@preserve plan operator state args FFI.densitymatrix_apply(
        plan.handle,
        args,
        Ptr{Cvoid}(plan.stream.handle),
    )
    _check(; status, what="qnpeps_densitymatrix_apply")
    return nothing
end

function densitymatrix_apply(
    operator::DensityMatrixProductOperator,
    state::DensityMatrixProductState;
    dims::ZipupDims,
    settings::ZipupSettings,
    normalize::Bool=false,
    input_gauge::Float64=0.0,
    trace::Bool=false,
)::NamedTuple
    config = ZipupConfig(; dims, settings)
    plan = DensityMatrixPlan(config, operator, state; trace)
    try
        densitymatrix_apply!(plan, operator, state; normalize, input_gauge)
        CUDA.synchronize(plan.stream)
        ranks = densitymatrix_rank_records(plan)
        return (
            mps=DensityMatrixProductState(plan.result_dimensions, plan.result_values),
            normalization_log=plan.normalization_log,
            output_gauge=plan.output_gauge,
            ranks,
            trace=plan.trace,
            solver_workspace_bytes=plan.sizes.solver_workspace_bytes,
        )
    finally
        destroy!(plan)
    end
end

function qnpeps_density_filter_apply(
    context::Ptr{Cvoid},
    args::Base.RefValue{QnpepsDensityFilterArgs},
    stream::Ptr{Cvoid},
)::Cint
    return GC.@preserve args FFI.density_filter_apply(context, args, stream)
end
