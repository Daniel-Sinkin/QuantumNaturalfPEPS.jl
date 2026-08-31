struct DensityMatrixWorkspaceBuffers
    left_environments::CUDA.CuVector{ComplexF64}
    right_blocks::CUDA.CuVector{ComplexF64}
    temporaries::CUDA.CuVector{ComplexF64}
    density::CUDA.CuVector{ComplexF64}
    eigenvalues::CUDA.CuVector{Float64}
    sorted_eigenvalues::CUDA.CuVector{Float64}
    sort_indices::CUDA.CuVector{Int32}
    active_ranks::CUDA.CuVector{Int32}
    truncation_records::CUDA.CuVector{QnpepsDensityRankRecord}
    solver_workspace::CUDA.CuVector{UInt8}
    solver_information::CUDA.CuVector{Int32}
end

struct DensityMatrixTraceBuffers
    left_environments::Union{CUDA.CuVector{ComplexF64},Nothing}
    right_blocks::Union{CUDA.CuVector{ComplexF64},Nothing}
    conjugate_right_blocks::Union{CUDA.CuVector{ComplexF64},Nothing}
    density_matrices::Union{CUDA.CuVector{ComplexF64},Nothing}
    solver_spectra::Union{CUDA.CuVector{Float64},Nothing}
    sorted_spectra::Union{CUDA.CuVector{Float64},Nothing}
    retained_spectra::Union{CUDA.CuVector{Float64},Nothing}
    right_bases::Union{CUDA.CuVector{ComplexF64},Nothing}
    left_bases::Union{CUDA.CuVector{ComplexF64},Nothing}
    projectors::Union{CUDA.CuVector{ComplexF64},Nothing}
    accumulated_gauges::Union{CUDA.CuVector{Float64},Nothing}
    matrix_orders::Union{CUDA.CuVector{Int32},Nothing}
    combined_inputs::Union{CUDA.CuVector{Int32},Nothing}
end

function _density_count(bytes::UInt64, ::Type{T})::Int where {T}
    bytes % UInt64(sizeof(T)) == 0 || error("density workspace byte count is misaligned")
    return Int(bytes ÷ UInt64(sizeof(T)))
end

function _density_workspace_buffers(
    sizes::QnpepsDensityWorkspaceSizes,
)::DensityMatrixWorkspaceBuffers
    return DensityMatrixWorkspaceBuffers(
        CUDA.zeros(ComplexF64, _density_count(sizes.left_environment_bytes, ComplexF64)),
        CUDA.zeros(ComplexF64, _density_count(sizes.right_block_bytes, ComplexF64)),
        CUDA.zeros(ComplexF64, _density_count(sizes.temporary_bytes, ComplexF64)),
        CUDA.zeros(ComplexF64, _density_count(sizes.density_bytes, ComplexF64)),
        CUDA.zeros(Float64, _density_count(sizes.eigenvalues_bytes, Float64)),
        CUDA.zeros(Float64, _density_count(sizes.sort_keys_bytes, Float64)),
        CUDA.zeros(Int32, _density_count(sizes.sort_indices_bytes, Int32)),
        CUDA.zeros(Int32, _density_count(sizes.active_ranks_bytes, Int32)),
        CuArray{QnpepsDensityRankRecord}(
            undef,
            _density_count(sizes.truncation_records_bytes, QnpepsDensityRankRecord),
        ),
        CUDA.zeros(UInt8, Int(sizes.solver_workspace_bytes)),
        CUDA.zeros(Int32, _density_count(sizes.solver_information_bytes, Int32)),
    )
end

function _density_workspace_wire(buffers::DensityMatrixWorkspaceBuffers)::QnpepsDensityWorkspace
    return QnpepsDensityWorkspace(
        left_environments=protocol_buffer(buffers.left_environments),
        right_blocks=protocol_buffer(buffers.right_blocks),
        temporaries=protocol_buffer(buffers.temporaries),
        density=protocol_buffer(buffers.density),
        eigenvalues=protocol_buffer(buffers.eigenvalues),
        sorted_eigenvalues=protocol_buffer(buffers.sorted_eigenvalues),
        sort_indices=protocol_buffer(buffers.sort_indices),
        active_ranks=protocol_buffer(buffers.active_ranks),
        truncation_records=protocol_buffer(buffers.truncation_records),
        solver_workspace=protocol_buffer(buffers.solver_workspace),
        solver_information=protocol_buffer(buffers.solver_information),
    )
end

function _empty_densitymatrix_trace()::DensityMatrixTraceBuffers
    return DensityMatrixTraceBuffers(ntuple(_ -> nothing, 13)...)
end

function _densitymatrix_trace(
    sizes::QnpepsDensityWorkspaceSizes,
    num_sites::Int,
    upper_bond::Int,
)::DensityMatrixTraceBuffers
    cuts = num_sites - 1
    combined = Int(sizes.combined_input)
    output = Int(sizes.capped_output)
    return DensityMatrixTraceBuffers(
        CUDA.zeros(ComplexF64, cuts * combined * combined),
        CUDA.zeros(ComplexF64, cuts * combined * output),
        CUDA.zeros(ComplexF64, cuts * combined * output),
        CUDA.zeros(ComplexF64, cuts * output * output),
        CUDA.zeros(Float64, cuts * output),
        CUDA.zeros(Float64, cuts * output),
        CUDA.zeros(Float64, cuts * upper_bond),
        CUDA.zeros(ComplexF64, cuts * output * upper_bond),
        CUDA.zeros(ComplexF64, cuts * output * upper_bond),
        CUDA.zeros(ComplexF64, cuts * output * output),
        CUDA.zeros(Float64, cuts),
        CUDA.zeros(Int32, cuts),
        CUDA.zeros(Int32, 2 * cuts),
    )
end

function _densitymatrix_trace_wire(trace::DensityMatrixTraceBuffers)::QnpepsDensityTrace
    return QnpepsDensityTrace(
        enabled=UInt32(trace.left_environments !== nothing),
        left_environments=protocol_buffer(trace.left_environments),
        right_blocks=protocol_buffer(trace.right_blocks),
        conjugate_right_blocks=protocol_buffer(trace.conjugate_right_blocks),
        density_matrices=protocol_buffer(trace.density_matrices),
        solver_spectra=protocol_buffer(trace.solver_spectra),
        sorted_spectra=protocol_buffer(trace.sorted_spectra),
        retained_spectra=protocol_buffer(trace.retained_spectra),
        right_bases=protocol_buffer(trace.right_bases),
        left_bases=protocol_buffer(trace.left_bases),
        projectors=protocol_buffer(trace.projectors),
        accumulated_gauges=protocol_buffer(trace.accumulated_gauges),
        matrix_orders=protocol_buffer(trace.matrix_orders),
        combined_inputs=protocol_buffer(trace.combined_inputs),
    )
end
