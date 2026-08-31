mutable struct DensityMatrixPlan{D,S}
    config::ZipupConfig
    handle::Ptr{Cvoid}
    sizes::QnpepsDensityWorkspaceSizes
    workspace::DensityMatrixWorkspaceBuffers
    workspace_wire::QnpepsDensityWorkspace
    sites::Vector{QnpepsDensitySitePlan}
    trace::DensityMatrixTraceBuffers
    trace_wire::QnpepsDensityTrace
    result_dimensions::CUDA.CuVector{Int32}
    result_values::CUDA.CuVector{ComplexF64}
    normalization_log::CUDA.CuVector{Float64}
    output_gauge::CUDA.CuVector{Float64}
    device::D
    stream::S
end

Base.isopen(plan::DensityMatrixPlan)::Bool = plan.handle != C_NULL

function Base.close(plan::DensityMatrixPlan)::Nothing
    isopen(plan) || return nothing
    handle = plan.handle
    plan.handle = C_NULL
    _ffi_ctx_destroy(handle)
    return nothing
end

Base.copy(::DensityMatrixPlan) = throw(ArgumentError("DensityMatrixPlan cannot be copied"))

is_destroyed(plan::DensityMatrixPlan)::Bool = !isopen(plan)

function destroy!(plan::DensityMatrixPlan)::Nothing
    close(plan)
    return nothing
end

function _density_workspace_sizes(
    context::Ptr{Cvoid},
    geometry::DensityMatrixWorkspaceGeometry,
)::QnpepsDensityWorkspaceSizes
    output = Ref(QnpepsDensityWorkspaceSizes())
    query = Ref(
        QnpepsDensityWorkspaceQuery(
            num_sites=Int64(geometry.num_sites),
            input_bond=Int64(geometry.input_bond),
            operator_bond=Int64(geometry.operator_bond),
            output_dimension=Int64(geometry.output_dimension),
            upper_bond=Int64(geometry.upper_bond),
            lanes=Int64(geometry.lanes),
            sizes_out=UInt64(UInt(Base.unsafe_convert(Ptr{QnpepsDensityWorkspaceSizes}, output))),
        ),
    )
    status = GC.@preserve output query FFI.density_workspace_sizes(context, query)
    _check(; status, what="qnpeps_density_workspace_sizes")
    return output[]
end

function densitymatrix_workspace_sizes(
    geometry::DensityMatrixWorkspaceGeometry;
    stream=CUDA.stream(),
)::QnpepsDensityWorkspaceSizes
    min(
        geometry.num_sites,
        geometry.input_bond,
        geometry.operator_bond,
        geometry.output_dimension,
        geometry.upper_bond,
        geometry.lanes,
    ) >= 1 || throw(ArgumentError("density workspace geometry must be positive"))
    context =
        _ffi_ctx_create(config=_density_context_config(geometry), stream=Ptr{Cvoid}(stream.handle))
    try
        return _density_workspace_sizes(context, geometry)
    finally
        _ffi_ctx_destroy(context)
    end
end

function DensityMatrixPlan(
    config::ZipupConfig,
    operator::DensityMatrixProductOperator,
    state::DensityMatrixProductState;
    trace::Bool=false,
)::DensityMatrixPlan
    _densitymatrix_validate(config)
    state_dimensions = Array(state.dims)
    operator_dimensions = Array(operator.dims)
    resolved = _densitymatrix_site_plans(
        state_dimensions,
        operator_dimensions,
        config.dims.num_sites,
        config.dims.chi,
    )
    UInt64(length(state.values)) == resolved.state_count ||
        throw(DimensionMismatch("density state values differ from dimensions"))
    UInt64(length(operator.values)) == resolved.operator_count ||
        throw(DimensionMismatch("density operator values differ from dimensions"))
    geometry = DensityMatrixWorkspaceGeometry(
        num_sites=config.dims.num_sites,
        input_bond=resolved.maximum_state_bond,
        operator_bond=resolved.maximum_operator_bond,
        output_dimension=resolved.maximum_output,
        upper_bond=config.dims.chi,
    )
    device = CUDA.device()
    stream = CUDA.stream()
    handle =
        _ffi_ctx_create(config=_density_context_config(geometry), stream=Ptr{Cvoid}(stream.handle))
    try
        sizes = _density_workspace_sizes(handle, geometry)
        workspace = _density_workspace_buffers(sizes)
        trace_buffers =
            trace ? _densitymatrix_trace(sizes, config.dims.num_sites, config.dims.chi) :
            _empty_densitymatrix_trace()
        plan = DensityMatrixPlan(
            config,
            handle,
            sizes,
            workspace,
            _density_workspace_wire(workspace),
            resolved.sites,
            trace_buffers,
            _densitymatrix_trace_wire(trace_buffers),
            CUDA.zeros(Int32, 3 * config.dims.num_sites),
            CUDA.zeros(ComplexF64, Int(resolved.output_count)),
            CUDA.zeros(Float64, 1),
            CUDA.zeros(Float64, 1),
            device,
            stream,
        )
        finalizer(close, plan)
        return plan
    catch
        _ffi_ctx_destroy(handle)
        rethrow()
    end
end

function densitymatrix_rank_records(plan::DensityMatrixPlan)::NamedTuple
    isopen(plan) || throw(ArgumentError("density plan is destroyed"))
    return (
        active_ranks=Array(plan.workspace.active_ranks),
        records=Array(plan.workspace.truncation_records),
    )
end
