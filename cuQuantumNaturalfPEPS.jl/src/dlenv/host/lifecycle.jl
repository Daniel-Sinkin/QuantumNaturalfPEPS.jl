
function DlenvHost(
    device_peps::CuPeps;
    chi_s::Integer=device_peps.dim_bond,
    chi_dl::Integer=device_peps.dim_bond,
    route=:default,
    density_cutoff::Real=1.0e-13,
    capture::Symbol=:auto,
)
    config = _cfg_of(
        device_peps;
        chi_s,
        chi_dl,
        dlenv_truncation_route=route,
        dlenv_density_cutoff=density_cutoff,
    )
    plan = plan_dlenv_host(device_peps; chi_s, chi_dl, route, density_cutoff)
    workspace = _ZipupWorkspace(config, chi_dl)
    arena = CUDA.zeros(UInt8, plan.total_bytes)
    row_values, lane_buffers = _carve_dlenv_arena(arena, plan)
    row_value_pointers = CUDA.CuPtr{UInt8}[pointer(values) for values in row_values]
    lane_buffer_pointers =
        ntuple(lane -> CUDA.CuPtr{UInt8}(pointer(lane_buffers[lane])), _DLENV_HOST_LANES)
    num_rows = device_peps.lx - 1
    row_dims = [zeros(Int32, 3 * device_peps.ly) for _ in 1:num_rows]
    logs = ntuple(_DLENV_HOST_LANES) do _
        Vector{Float64}(undef, num_rows)
    end
    lanes = ntuple(_DLENV_HOST_LANES) do lane
        CuDlenv(
            lane_buffers[lane],
            logs[lane],
            device_peps.lx,
            device_peps.ly,
            device_peps.dim_phys,
            device_peps.dim_bond,
            Int(chi_s),
            Int(chi_dl),
        )
    end
    peps_offsets = _peps_row_element_offsets(device_peps)
    peps_elements = [_peps_row_elements(device_peps, row) for row in 1:device_peps.lx]
    capture_enabled =
        config.dlenv_truncation_route == TRUNCATION_DEFAULT && _dlenv_capture_enabled(capture)
    capture_state = capture === :eager || !capture_enabled ? :eager : :warming
    capture_reason = if capture === :eager
        :policy_eager
    elseif capture_enabled
        :awaiting_lane_warmup
    else
        :route_not_capturable
    end
    host = DlenvHost(;
        config,
        plan,
        workspace,
        arena,
        peps_data=device_peps.data,
        row_values,
        row_value_pointers,
        lane_buffers,
        lane_buffer_pointers,
        lanes,
        row_dims,
        row_bytes=zeros(Int, num_rows),
        value_offsets=zeros(Int, num_rows),
        header=UInt8[],
        scales=Vector{Float64}(undef, num_rows * device_peps.ly),
        peps_offsets,
        peps_elements,
        row_args=Vector{QnpepsZipupPepsRowArgs}(undef, num_rows),
        lane_warmed=fill(false, _DLENV_HOST_LANES),
        graphs=Union{Nothing,CUDA.CuGraphExec}[nothing for _ in 1:_DLENV_HOST_LANES],
        device=CUDA.device(),
        stream=CUDA.stream(),
        generation=-1,
        active_lane=0,
        build_count=0,
        layout_ready=false,
        args_ready=false,
        capture_enabled,
        capture_state,
        capture_reason,
        open=true,
    )
    finalizer(close, host)
    return host
end

Base.isopen(host::DlenvHost)::Bool = host.open

function Base.close(host::DlenvHost)::Nothing
    host.open || return nothing
    host.graphs .= nothing
    close(host.workspace)
    CUDA.unsafe_free!(host.arena)
    host.open = false
    return nothing
end

Base.copy(::DlenvHost) = throw(ArgumentError("DlenvHost cannot be copied"))

function _validate_dlenv_host(host::DlenvHost, device_peps::CuPeps)::Nothing
    host.open || throw(ArgumentError("DlenvHost is closed"))
    CUDA.device() == host.device || throw(ArgumentError("DlenvHost device mismatch"))
    CUDA.stream().handle == host.stream.handle || throw(ArgumentError("DlenvHost stream mismatch"))
    config = host.config
    device_peps.lx == config.lx &&
    device_peps.ly == config.ly &&
    device_peps.dim_phys == config.dim_phys &&
    device_peps.dim_bond == config.dim_bond &&
    device_peps.data === host.peps_data ||
        throw(DimensionMismatch("CuPeps and DlenvHost geometry or storage mismatch"))
    return nothing
end
