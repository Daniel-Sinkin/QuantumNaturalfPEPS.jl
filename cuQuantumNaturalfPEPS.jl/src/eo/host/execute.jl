
@inline function _eo_host_peer_copy!(
    destination::CUDA.CuPtr{Cvoid},
    destination_context::CUDA.CuContext,
    source::CUDA.CuPtr{Cvoid},
    source_context::CUDA.CuContext,
    bytes::UInt64,
    stream::CUDA.CUstream,
)
    status = if destination_context == source_context
        FFI.cuda_memcpy_dto_d_async(destination, source, bytes, stream)
    else
        FFI.cuda_memcpy_peer_async(
            destination,
            destination_context.handle,
            source,
            source_context.handle,
            bytes,
            stream,
        )
    end
    status == CUDA.CUDA_SUCCESS || CUDA.throw_api_error(status)
    return nothing
end

@inline function _eo_host_stream_synchronize!(stream::CUDA.CUstream)
    status = FFI.cuda_stream_synchronize(stream)
    status == CUDA.CUDA_SUCCESS || CUDA.throw_api_error(status)
    return nothing
end

function eo_host_execute!(host::EoHost, lane_index::Int)::Int32
    isopen(host) || throw(ArgumentError("EoHost is closed"))
    1 <= lane_index <= EO_HOST_LANES || throw(BoundsError(host.lanes, lane_index))
    lane = host.lanes[lane_index]
    binding = lane.binding
    lane.status = _EO_HOST_OK
    lane.failure_location = ""
    lane.failure_message = ""
    _eo_host_peer_copy!(
        binding.peps,
        lane.context,
        host.source_peps_pointer,
        host.source_context,
        host.source_peps_bytes,
        lane.stream,
    )
    _eo_host_peer_copy!(
        binding.samples,
        lane.context,
        lane.source_sample_pointer,
        host.source_context,
        UInt64(lane.row_count * host.sites),
        lane.stream,
    )
    handle = lane.handle
    arguments = binding.args
    status = GC.@preserve host lane binding arguments FFI.eloc_ctx_run(
        handle,
        Base.unsafe_convert(Ptr{Cvoid}, arguments),
    )
    if status == 0
        scalar_bytes = UInt64(2 * lane.row_count * sizeof(Float64))
        _eo_host_peer_copy!(
            lane.destination_logpsi_pointer,
            host.source_context,
            binding.logpsi,
            lane.context,
            scalar_bytes,
            lane.stream,
        )
        _eo_host_peer_copy!(
            lane.destination_e_loc_pointer,
            host.source_context,
            binding.e_loc,
            lane.context,
            scalar_bytes,
            lane.stream,
        )
        _eo_host_stream_synchronize!(lane.stream)
    end
    lane.status = Int32(status)
    if status != 0
        lane.failure_location = _last_error_location()
        lane.failure_message = _last_error_message()
    end
    return lane.status
end

function eo_host_warm!(host::EoHost, iterations::Integer=3)::EoHost
    iterations >= 1 || throw(ArgumentError("iterations must be positive"))
    caller_context = CUDA.context()
    try
        for _ in 1:iterations, lane_index in 1:EO_HOST_LANES
            CUDA.context!(host.lanes[lane_index].context)
            status = eo_host_execute!(host, lane_index)
            if status != _EO_HOST_OK
                lane = host.lanes[lane_index]
                location = isempty(lane.failure_location) ? "" : " at $(lane.failure_location)"
                message = isempty(lane.failure_message) ? "" : "; $(lane.failure_message)"
                error("qnpeps_eloc_ctx_run failed with status $status$location$message")
            end
        end
    finally
        CUDA.context!(caller_context)
    end
    return host
end

function bind_eo_host!(host::EoHost, peps::CuVector{ComplexF32})::EoHost
    isopen(host) || throw(ArgumentError("EoHost is closed"))
    host.source_peps === peps || throw(ArgumentError("EoHost requires stable PEPS storage"))
    CUDA.CuPtr{Cvoid}(pointer(peps)) == host.source_peps_pointer ||
        throw(ArgumentError("EoHost PEPS pointer changed"))
    host.source_generation += 1
    return host
end

@inline function _eo_host_run_args_with_selector(
    arguments::_EoHostRunArgs,
    selector::EoSelector,
)::_EoHostRunArgs
    return _EoHostRunArgs(;
        struct_size=arguments.struct_size,
        padding=arguments.padding,
        device_peps=arguments.device_peps,
        device_samples=arguments.device_samples,
        logpsi_out=arguments.logpsi_out,
        e_loc_out=arguments.e_loc_out,
        o_rows_dev=arguments.o_rows_dev,
        o_rows_host=arguments.o_rows_host,
        gram=arguments.gram,
        lambda=arguments.lambda,
        j2_mode=selector.mode,
        j2_draw=selector.draw,
        j2_seed=selector.seed,
        j2_epoch=selector.epoch,
    )
end

function set_eo_selector!(host::EoHost, selector::EoSelector)::EoHost
    isopen(host) || throw(ArgumentError("EoHost is closed"))
    for lane in host.lanes
        lane.binding.args[] = _eo_host_run_args_with_selector(lane.binding.args[], selector)
    end
    host.selector = selector
    return host
end

function advance_eo_selector!(host::EoHost, epoch::UInt64)::EoHost
    selector = host.selector
    return set_eo_selector!(host, EoSelector(selector.mode, selector.draw, selector.seed, epoch))
end

function eo_host_selector(host::EoHost)::EoSelector
    isopen(host) || throw(ArgumentError("EoHost is closed"))
    return host.selector
end

function eo_host_term_table(host::EoHost)::EoTermTable
    isopen(host) || throw(ArgumentError("EoHost is closed"))
    table = host.term_table
    table === nothing && throw(ArgumentError("EoHost term state is cleared"))
    return table
end

function seal_eo_host!(host::EoHost)::EoHost
    isopen(host) || throw(ArgumentError("EoHost is closed"))
    host.sealed = true
    return host
end

function eo_host_replace_buffers!(host::EoHost, lane_index::Int)::EoHost
    isopen(host) || throw(ArgumentError("EoHost is closed"))
    host.sealed && throw(ArgumentError("sealed EoHost bindings cannot be replaced"))
    1 <= lane_index <= EO_HOST_LANES || throw(BoundsError(host.lanes, lane_index))
    table = host.term_table
    table === nothing && throw(ArgumentError("EoHost term state is cleared"))
    lane = host.lanes[lane_index]
    CUDA.context!(lane.context)
    next_epoch = lane.binding.epoch + UInt64(1)
    next_binding = _eo_host_create_binding(
        Int(host.source_peps_bytes ÷ UInt64(sizeof(ComplexF32))),
        host.sites,
        host.compact,
        lane.row_base,
        lane.row_count,
        host.destination_rows_pointer,
        next_epoch,
        host.selector,
    )
    next_handle = Ptr{Cvoid}(C_NULL)
    try
        next_handle = _eo_host_create_handle(host.config_ref, table, lane.row_count, lane.stream)
    catch
        CUDA.free(next_binding.arena)
        rethrow()
    end
    _eo_host_stream_synchronize!(lane.stream)
    old_handle = lane.handle
    old_binding = lane.binding
    FFI.eloc_ctx_destroy(old_handle)
    CUDA.free(old_binding.arena)
    lane.handle = next_handle
    lane.binding = next_binding
    lane.status = _EO_HOST_OK
    lane.failure_location = ""
    lane.failure_message = ""
    host.contexts_created += 1
    host.contexts_destroyed += 1
    host.arenas_created += 1
    host.arenas_freed += 1
    host.replacements += 1
    return host
end

function eo_host_stats(host::EoHost, lane_index::Int)::EoHostStats
    isopen(host) || throw(ArgumentError("EoHost is closed"))
    1 <= lane_index <= EO_HOST_LANES || throw(BoundsError(host.lanes, lane_index))
    output = Ref(
        EoHostStats(;
            struct_size=UInt32(sizeof(EoHostStats)),
            graph_enabled=UInt32(0),
            runs=UInt64(0),
            bindings=UInt64(0),
            graph_captures=UInt64(0),
            graph_replays=UInt64(0),
            graph_capture_failures=UInt64(0),
            graph_nodes=UInt64(0),
            graph_edges=UInt64(0),
            graph_introspection_failures=UInt64(0),
            j2_mode_last=UInt32(0),
            j2_draw_last=UInt32(0),
            j2_seed_last=UInt64(0),
            j2_epoch_last=UInt64(0),
            j2_waves=UInt64(0),
            j2_group0_waves=UInt64(0),
            j2_group1_waves=UInt64(0),
            j2_row_groups_total=UInt64(0),
            j2_row_groups_retained=UInt64(0),
            j2_column_groups_total=UInt64(0),
            j2_column_groups_retained=UInt64(0),
            j2_diag_terms_total=UInt64(0),
            j2_diag_terms_retained=UInt64(0),
            j2_flip_terms_total=UInt64(0),
            j2_flip_terms_retained=UInt64(0),
        ),
    )
    status = GC.@preserve output FFI.eloc_ctx_stats(
        host.lanes[lane_index].handle,
        Base.unsafe_convert(Ptr{Cvoid}, output),
    )
    status == 0 || error("qnpeps_eloc_ctx_stats failed with status $status")
    return output[]
end

function eo_host_binding_state(host::EoHost, lane_index::Int)::EoHostBindingState
    isopen(host) || throw(ArgumentError("EoHost is closed"))
    1 <= lane_index <= EO_HOST_LANES || throw(BoundsError(host.lanes, lane_index))
    lane = host.lanes[lane_index]
    binding = lane.binding
    return EoHostBindingState(;
        lane=Int32(lane_index - 1),
        handle=UInt(lane.handle),
        stream=UInt(lane.stream),
        arena=UInt(pointer(binding.arena)),
        arena_bytes=binding.arena_bytes,
        peps=UInt(binding.peps),
        samples=UInt(binding.samples),
        logpsi=UInt(binding.logpsi),
        e_loc=UInt(binding.e_loc),
        rows=UInt(binding.rows),
        epoch=binding.epoch,
    )
end

function eo_host_row_pointers(host::EoHost)
    isopen(host) || throw(ArgumentError("EoHost is closed"))
    return ntuple(lane -> UInt(host.lanes[lane].binding.rows), EO_HOST_LANES)
end

function eo_host_lifecycle_state(host::EoHost)::EoHostLifecycleState
    order = (
        host.teardown_order[1],
        host.teardown_order[2],
        host.teardown_order[3],
        host.teardown_order[4],
        host.teardown_order[5],
        host.teardown_order[6],
    )
    return EoHostLifecycleState(;
        contexts_created=host.contexts_created,
        contexts_destroyed=host.contexts_destroyed,
        arenas_created=host.arenas_created,
        arenas_freed=host.arenas_freed,
        streams_created=host.streams_created,
        streams_destroyed=host.streams_destroyed,
        replacements=host.replacements,
        gc_runs=host.gc_runs,
        teardown_order=order,
        refs_cleared=host.refs_cleared,
        open=(!host.closed),
        sealed=host.sealed,
    )
end
