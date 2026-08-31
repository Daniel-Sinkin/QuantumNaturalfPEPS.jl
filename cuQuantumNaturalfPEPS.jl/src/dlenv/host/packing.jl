
function _extract_dlenv_logs!(host::DlenvHost, lane::Int)::Nothing
    logs = host.lanes[lane].cumulative_row_logs
    num_rows = Int(host.config.lx) - 1
    num_cols = Int(host.config.ly)
    cumulative = 0.0
    for step in 1:num_rows
        for col in 1:num_cols
            scale = host.scales[(step-1)*num_cols+col]
            isfinite(scale) || throw(ErrorException("non-finite dl-env scale"))
            scale > 0.0 && (cumulative += log(scale))
        end
        logs[num_rows-step+1] = cumulative
    end
    return nothing
end

function _pack_dlenv_lane!(host::DlenvHost, lane::Int)::Nothing
    destination = host.lane_buffer_pointers[lane]
    stream = host.stream
    for env_row in eachindex(host.row_value_pointers)
        status = FFI.cuda_memcpy_dto_d_async(
            destination + host.value_offsets[env_row],
            host.row_value_pointers[env_row],
            host.row_bytes[env_row],
            stream,
        )::CUDA.CUresult
        status == CUDA.CUDA_SUCCESS || CUDA.throw_api_error(status)
    end
    return nothing
end

@inline function _launch_dlenv_graph!(executable::CUDA.CUgraphExec, stream::CUDA.CUstream)::Nothing
    status = FFI.cuda_graph_launch(executable, stream)
    status == CUDA.CUDA_SUCCESS || CUDA.throw_api_error(status)
    return nothing
end

function _pack_dlenv_reordered!(host::DlenvHost, lane::Int)::Nothing
    destination = host.lane_buffer_pointers[lane]
    num_rows = length(host.row_value_pointers)
    for target_row in 1:num_rows
        source_row = num_rows - target_row + 1
        CUDA.cuMemcpyDtoDAsync_v2(
            destination + host.value_offsets[target_row],
            host.row_value_pointers[source_row],
            min(host.row_bytes[target_row], host.row_bytes[source_row]),
            host.stream,
        )
    end
    return nothing
end

function _pack_dlenv_columns_reordered!(host::DlenvHost, lane::Int)::Nothing
    destination = host.lane_buffer_pointers[lane]
    num_cols = Int(host.config.ly)
    for env_row in eachindex(host.row_values)
        dims = host.row_dims[env_row]
        source_offsets = Vector{Int}(undef, num_cols)
        cursor = 0
        for col in 1:num_cols
            source_offsets[col] = cursor
            base = 3 * (col - 1)
            cursor += Int(dims[base+1]) * Int(dims[base+2]) * Int(dims[base+3]) * sizeof(ComplexF32)
        end
        target_offset = host.value_offsets[env_row]
        for target_col in 1:num_cols
            source_col = num_cols - target_col + 1
            source_base = 3 * (source_col - 1)
            bytes =
                Int(dims[source_base+1]) *
                Int(dims[source_base+2]) *
                Int(dims[source_base+3]) *
                sizeof(ComplexF32)
            CUDA.cuMemcpyDtoDAsync_v2(
                destination + target_offset,
                host.row_value_pointers[env_row] + source_offsets[source_col],
                bytes,
                host.stream,
            )
            target_offset += bytes
        end
    end
    return nothing
end

function _pack_dlenv_with_policy!(host::DlenvHost, lane::Int)::Nothing
    if !host.capture_enabled || host.capture_state === :fallback_eager
        _pack_dlenv_lane!(host, lane)
        host.lane_warmed[lane] = true
        return nothing
    end
    executable = host.graphs[lane]
    if executable !== nothing
        _launch_dlenv_graph!(executable.handle, host.stream.handle)
        host.capture_state = :captured_pack
        host.capture_reason = :capture_succeeded
        return nothing
    end
    if !host.lane_warmed[lane]
        _pack_dlenv_lane!(host, lane)
        host.lane_warmed[lane] = true
        return nothing
    end
    graph = try
        CUDA.capture() do
            _pack_dlenv_lane!(host, lane)
        end
    catch
        host.capture_enabled = false
        host.capture_state = :fallback_eager
        host.capture_reason = :capture_api_error
        _pack_dlenv_lane!(host, lane)
        return nothing
    end
    executable = try
        CUDA.instantiate(graph)
    catch
        host.capture_enabled = false
        host.capture_state = :fallback_eager
        host.capture_reason = :graph_instantiate_error
        _pack_dlenv_lane!(host, lane)
        return nothing
    end
    host.graphs[lane] = executable
    host.capture_state = :captured_pack
    host.capture_reason = :capture_succeeded
    try
        _launch_dlenv_graph!(executable.handle, host.stream.handle)
    catch
        host.capture_enabled = false
        host.capture_state = :fallback_eager
        host.capture_reason = :graph_launch_error
        _pack_dlenv_lane!(host, lane)
    end
    return nothing
end
