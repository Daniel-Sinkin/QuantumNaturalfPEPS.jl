
function _warm_dlenv_rows!(host::DlenvHost, device_peps::CuPeps)::Nothing
    num_rows = device_peps.lx - 1
    _ffi_zipup_ctx_begin(host.workspace.handle)
    for env_row in num_rows:-1:1
        peps_row = env_row + 1
        mps_dims = env_row == num_rows ? nothing : host.row_dims[env_row+1]
        mps_values = env_row == num_rows ? nothing : host.row_values[env_row+1]
        mps = _ZipupMpsInput(; dims=mps_dims, values=mps_values)
        output = _ZipupRowOutput(; dims=host.row_dims[env_row], values=host.row_values[env_row])
        arguments = _ZipupPepsRowArguments(;
            workspace=host.workspace,
            row=peps_row,
            peps_values=device_peps.data,
            peps_offset=host.peps_offsets[peps_row],
            peps_elements=host.peps_elements[peps_row],
            mps,
            output,
        )
        _enqueue_peps_row!(arguments)
    end
    _ffi_zipup_ctx_finish(host.workspace.handle, host.scales)
    return nothing
end

function _prepare_dlenv_layout!(host::DlenvHost, device_peps::CuPeps)::Nothing
    num_rows = device_peps.lx - 1
    num_cols = device_peps.ly
    header_i32 = Vector{Int32}(undef, num_rows * num_cols * 4)
    header_cursor = 0
    value_cursor = sizeof(Int32) * length(header_i32)
    for env_row in 1:num_rows
        dims = host.row_dims[env_row]
        host.row_bytes[env_row] = _grouped_mps_elements(dims) * sizeof(ComplexF32)
        host.value_offsets[env_row] = value_cursor
        for col in 1:num_cols
            base = 3 * (col - 1)
            left = dims[base+1]
            physical = dims[base+2]
            right = dims[base+3]
            vertical = isqrt(physical)
            vertical * vertical == physical ||
                throw(DimensionMismatch("non-square dl-env physical dimension"))
            header_i32[header_cursor+1] = left
            header_i32[header_cursor+2] = Int32(vertical)
            header_i32[header_cursor+3] = Int32(vertical)
            header_i32[header_cursor+4] = right
            header_cursor += 4
        end
        value_cursor += host.row_bytes[env_row]
    end
    value_cursor <= host.plan.packed_bytes ||
        throw(DimensionMismatch("packed dl-env rows exceed the planned buffer"))
    host.header = collect(reinterpret(UInt8, header_i32))
    header = host.header
    for lane in 1:_DLENV_HOST_LANES
        GC.@preserve header unsafe_copyto!(
            pointer(host.lane_buffers[lane]),
            pointer(header),
            length(header);
            stream=host.stream,
            async=true,
        )
    end
    CUDA.synchronize(host.stream)
    _prepare_dlenv_args!(host, device_peps)
    host.layout_ready = true
    return nothing
end

function _prepare_dlenv_args!(host::DlenvHost, device_peps::CuPeps)::Nothing
    host.args_ready && return nothing
    isbitstype(QnpepsZipupPepsRowArgs) || throw(ArgumentError("row arguments must be isbits"))
    num_rows = device_peps.lx - 1
    peps_data = host.peps_data
    row_dims = host.row_dims
    row_values = host.row_values
    row_args = host.row_args
    GC.@preserve peps_data row_dims row_values row_args begin
        for env_row in 1:num_rows
            peps_row = env_row + 1
            has_environment = env_row != num_rows
            row_args[env_row] = QnpepsZipupPepsRowArgs(;
                struct_size=UInt32(sizeof(QnpepsZipupPepsRowArgs)),
                row=Int32(peps_row),
                peps_row=UInt(pointer(peps_data, host.peps_offsets[peps_row] + 1)),
                peps_row_bytes=UInt64(host.peps_elements[peps_row] * sizeof(ComplexF32)),
                mps_dims=has_environment ? UInt(pointer(row_dims[env_row+1])) : UInt(0),
                mps_values=has_environment ? UInt(pointer(row_values[env_row+1])) : UInt(0),
                mps_bytes=has_environment ? UInt64(host.row_bytes[env_row+1]) : UInt64(0),
                output_dims=UInt(pointer(row_dims[env_row])),
                output_values=UInt(pointer(row_values[env_row])),
                output_bytes=UInt64(host.plan.row_value_bytes),
            )
        end
    end
    host.args_ready = true
    return nothing
end

@inline function _run_prepared_rows_begin!(host::DlenvHost)::Nothing
    begin_status = FFI.zipup_ctx_begin(host.workspace.handle)
    begin_status == 0 || _check(; status=begin_status, what="qnpeps_zipup_ctx_begin")
    return nothing
end

@inline function _run_prepared_rows_enqueue!(
    host::DlenvHost,
    device_peps::CuPeps,
    peps_data,
    row_dims,
    row_values,
    row_args,
    scales,
)::Nothing
    GC.@preserve peps_data row_dims row_values row_args scales begin
        for env_row in (device_peps.lx-1):-1:1
            enqueue_status =
                FFI.zipup_ctx_enqueue_peps_row(host.workspace.handle, pointer(row_args, env_row))
            enqueue_status == 0 ||
                _check(; status=enqueue_status, what="qnpeps_zipup_ctx_enqueue_peps_row")
        end
    end
    return nothing
end

@inline function _run_prepared_rows_finish!(host::DlenvHost, scales)::Nothing
    finish_status = GC.@preserve scales FFI.zipup_ctx_finish(
        host.workspace.handle,
        pointer(scales),
        length(scales),
    )
    finish_status == 0 || _check(; status=finish_status, what="qnpeps_zipup_ctx_finish")
    return nothing
end

function _run_prepared_rows!(host::DlenvHost, device_peps::CuPeps)::Nothing
    peps_data = host.peps_data
    row_dims = host.row_dims
    row_values = host.row_values
    row_args = host.row_args
    scales = host.scales
    _run_prepared_rows_begin!(host)
    _run_prepared_rows_enqueue!(
        host,
        device_peps,
        peps_data,
        row_dims,
        row_values,
        row_args,
        scales,
    )
    _run_prepared_rows_finish!(host, scales)
    return nothing
end

function _enqueue_prepared_rows!(host::DlenvHost, device_peps::CuPeps)::Nothing
    _prepare_dlenv_args!(host, device_peps)
    _run_prepared_rows!(host, device_peps)
    return nothing
end
