
function double_layer_step(
    tensors::AbstractMatrix,
    env_row::Integer,
    env_below::Union{Nothing,MPS};
    maxdim::Integer,
    route=:default,
    density_cutoff::Real=1.0e-13,
)::Tuple{MPS,Float64}
    lx, ly = size(tensors)
    1 <= env_row <= lx - 1 || throw(ArgumentError("env_row must be in 1:$(lx-1) (got $env_row)"))
    row_index = env_row + 1
    config = QnpepsConfig(
        lx=lx,
        ly=ly,
        dim_bond=_itensor_bond_dim(tensors),
        chi_s=maxdim,
        dim_phys=dim(_phys_index(tensors, 1, 1)),
        dlenv_truncation_route=route,
        dlenv_density_cutoff=density_cutoff,
    )
    row_buffer = _pack_peps_row([_site_array(tensors, row_index, col) for col in 1:ly])

    mps_dims, mps_values = if env_below !== nothing
        vbonds = [_vbond(tensors, row_index, col) for col in 1:ly]
        _pack_grouped_mps(_mps_to_env_arrays(env_below, vbonds))
    else
        nothing, nothing
    end
    workspace = _ZipupWorkspace(config, maxdim)
    output_dims = zeros(Int32, 3 * ly)
    output_values = CUDA.zeros(UInt8, workspace.row_value_bytes)
    scales = Vector{Float64}(undef, ly)
    try
        _ffi_zipup_ctx_begin(workspace.handle)
        mps = _ZipupMpsInput(; dims=mps_dims, values=mps_values)
        output = _ZipupRowOutput(; dims=output_dims, values=output_values)
        arguments = _ZipupPepsRowArguments(;
            workspace,
            row=row_index,
            peps_values=row_buffer,
            peps_offset=0,
            peps_elements=length(row_buffer),
            mps,
            output,
        )
        _enqueue_peps_row!(arguments)
        _ffi_zipup_ctx_finish(workspace.handle, scales)
    finally
        close(workspace)
    end

    output_elements = _grouped_mps_elements(output_dims)
    output_bytes = Array(view(output_values, 1:(output_elements*sizeof(ComplexF32))))
    env = _ungroup_mps(output_dims, copy(reinterpret(ComplexF32, output_bytes)))
    row_log = sum(scale > 0.0 ? log(scale) : 0.0 for scale in scales)
    return _env_arrays_to_mps(env, [_vbond(tensors, env_row, col) for col in 1:ly]), row_log
end

function double_layer(
    tensors::AbstractMatrix;
    maxdim::Integer,
    route=:default,
    density_cutoff::Real=1.0e-13,
)::Tuple{Vector{MPS},Vector{Float64}}
    lx, ly = size(tensors)
    lx >= 2 || throw(ArgumentError("double-layer envs need at least 2 rows (got $lx)"))
    ly >= 2 || throw(ArgumentError("double-layer envs need at least 2 columns (got $ly)"))
    maxdim >= 1 || throw(ArgumentError("maxdim must be positive (got $maxdim)"))

    num_env_rows = lx - 1
    rows = Vector{MPS}(undef, num_env_rows)
    cumulative_row_logs = Vector{Float64}(undef, num_env_rows)
    env_below = nothing
    cumulative_log = 0.0
    for env_row in num_env_rows:-1:1
        row, row_log = double_layer_step(tensors, env_row, env_below; maxdim, route, density_cutoff)
        rows[env_row] = row
        cumulative_log += row_log
        cumulative_row_logs[env_row] = cumulative_log
        env_below = row
    end
    return rows, cumulative_row_logs
end
