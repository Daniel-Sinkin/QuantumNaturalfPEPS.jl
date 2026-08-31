
function double_layer_rowwise(
    device_peps::CuPeps;
    chi_s::Integer=device_peps.dim_bond,
    chi_dl::Integer=device_peps.dim_bond,
    route=:default,
    density_cutoff::Real=1.0e-13,
    workspace::Union{Nothing,ZipupWorkspace}=nothing,
)::CuDlenv
    config = _cfg_of(
        device_peps;
        chi_s,
        chi_dl,
        dlenv_truncation_route=route,
        dlenv_density_cutoff=density_cutoff,
    )
    owned_workspace = workspace === nothing
    active_workspace =
        workspace === nothing ? ZipupWorkspace(device_peps; chi_s, chi_dl, route, density_cutoff) :
        workspace
    _validate_zipup_workspace(active_workspace, device_peps, config)

    try
        num_env_rows = device_peps.lx - 1
        peps_row_offsets = _peps_row_element_offsets(device_peps)
        row_dims = [zeros(Int32, 3 * device_peps.ly) for _ in 1:num_env_rows]
        row_values = [CUDA.zeros(UInt8, active_workspace.row_value_bytes) for _ in 1:num_env_rows]
        scales = Vector{Float64}(undef, num_env_rows * device_peps.ly)

        _ffi_zipup_ctx_begin(active_workspace.handle)
        for env_row in num_env_rows:-1:1
            peps_row = env_row + 1
            peps_elements = _peps_row_elements(device_peps, peps_row)
            mps_dims = env_row == num_env_rows ? nothing : row_dims[env_row+1]
            mps_values = env_row == num_env_rows ? nothing : row_values[env_row+1]
            mps = _ZipupMpsInput(; dims=mps_dims, values=mps_values)
            output = _ZipupRowOutput(; dims=row_dims[env_row], values=row_values[env_row])
            arguments = _ZipupPepsRowArguments(;
                workspace=active_workspace,
                row=peps_row,
                peps_values=device_peps.data,
                peps_offset=peps_row_offsets[peps_row],
                peps_elements,
                mps,
                output,
            )
            _enqueue_peps_row!(arguments)
        end
        _ffi_zipup_ctx_finish(active_workspace.handle, scales)

        cumulative_row_logs = Vector{Float64}(undef, num_env_rows)
        cumulative_log = 0.0
        for step in 1:num_env_rows
            env_row = num_env_rows - step + 1
            scale_range = ((step-1)*device_peps.ly+1):(step*device_peps.ly)
            cumulative_log += sum(scale > 0.0 ? log(scale) : 0.0 for scale in scales[scale_range])
            cumulative_row_logs[env_row] = cumulative_log
        end
        data = _pack_dlenv_rows(row_dims, row_values, config)
        return CuDlenv(
            data,
            cumulative_row_logs,
            device_peps.lx,
            device_peps.ly,
            device_peps.dim_phys,
            device_peps.dim_bond,
            chi_s,
            chi_dl,
        )
    finally
        owned_workspace && close(active_workspace)
    end
end
