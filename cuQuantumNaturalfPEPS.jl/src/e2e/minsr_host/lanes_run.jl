
function _minsr_host_create_lane(
    lane_index::Int,
    device::CUDA.CuDevice,
    context::CUDA.CuContext,
    gram_descriptor::_MinsrHostGramDesc,
    minsr_descriptor::_MinsrHostMinsrDesc,
    sites::Int64,
    compact::Int64,
    dense::Int64,
)
    CUDA.context!(context)
    stream_out = Ref{CUDA.CUstream}(CUDA.CUstream(C_NULL))
    CUDA.cuStreamCreate(stream_out, CUDA.CU_STREAM_DEFAULT)
    stream = stream_out[]
    gram = Ptr{Cvoid}(C_NULL)
    minsr = Ptr{Cvoid}(C_NULL)
    arena = nothing
    try
        gram_out = Ref{Ptr{Cvoid}}(C_NULL)
        status = FFI.gram_host_ctx_create(gram_descriptor, Ptr{Cvoid}(stream), gram_out)
        gram = gram_out[]
        status == 0 || throw(
            MinsrHostError(
                Int32(lane_index - 1),
                Int32(status),
                _last_error_location(),
                _last_error_message(),
            ),
        )
        footprint_ref = Ref(
            _MinsrHostGramFootprint(;
                struct_size=UInt32(sizeof(_MinsrHostGramFootprint)),
                reserved=UInt32(0),
                context_device_bytes=UInt64(0),
                geometry_device_bytes=UInt64(0),
                dense_a_device_bytes=UInt64(0),
                dense_b_device_bytes=UInt64(0),
                caller_samples_bytes=UInt64(0),
                caller_rows_bytes=UInt64(0),
                caller_gram_bytes=UInt64(0),
            ),
        )
        status = FFI.gram_host_ctx_footprint(gram, footprint_ref)
        status == 0 || throw(
            MinsrHostError(
                Int32(lane_index - 1),
                Int32(status),
                _last_error_location(),
                _last_error_message(),
            ),
        )
        minsr_out = Ref{Ptr{Cvoid}}(C_NULL)
        status = FFI.minsr_host_ctx_create(minsr_descriptor, Ptr{Cvoid}(stream), minsr_out)
        minsr = minsr_out[]
        status == 0 || throw(
            MinsrHostError(
                Int32(lane_index - 1),
                Int32(status),
                _last_error_location(),
                _last_error_message(),
            ),
        )
        footprint = footprint_ref[]
        sizes = (
            footprint.caller_samples_bytes,
            UInt64(2 * minsr_descriptor.n_samples) * UInt64(sizeof(Float64)),
            UInt64(2 * minsr_descriptor.n_samples) * UInt64(sizeof(Float64)),
            UInt64(minsr_descriptor.n_samples) * UInt64(sizeof(Float64)),
            footprint.caller_rows_bytes,
            footprint.caller_gram_bytes,
            UInt64(dense) * UInt64(sizeof(ComplexF32)),
        )
        arena_bytes = UInt64(0)
        for bytes in sizes
            arena_bytes = _minsr_host_arena_total(arena_bytes, bytes)
        end
        arena = CUDA.alloc(CUDA.DeviceMemory, Int(arena_bytes))
        base = UInt(pointer(arena))
        offset = UInt64(0)
        samples, offset = _minsr_host_arena_take(base, offset, sizes[1])
        logpsi, offset = _minsr_host_arena_take(base, offset, sizes[2])
        e_loc, offset = _minsr_host_arena_take(base, offset, sizes[3])
        logq, offset = _minsr_host_arena_take(base, offset, sizes[4])
        rows, offset = _minsr_host_arena_take(base, offset, sizes[5])
        raw_gram, offset = _minsr_host_arena_take(base, offset, sizes[6])
        theta, offset = _minsr_host_arena_take(base, offset, sizes[7])
        samples_pointer = CUDA.CuPtr{Cvoid}(samples)
        logpsi_pointer = CUDA.CuPtr{Cvoid}(logpsi)
        e_loc_pointer = CUDA.CuPtr{Cvoid}(e_loc)
        logq_pointer = CUDA.CuPtr{Cvoid}(logq)
        rows_pointer = CUDA.CuPtr{Cvoid}(rows)
        gram_pointer = CUDA.CuPtr{Cvoid}(raw_gram)
        theta_pointer = CUDA.CuPtr{Cvoid}(theta)
        statistics = zeros(Float64, 4)
        statistics_pointer = UInt(pointer(statistics))
        row_bytes = UInt64(compact) * UInt64(sizeof(ComplexF32))
        gram_bytes =
            UInt64(minsr_descriptor.n_samples) *
            UInt64(minsr_descriptor.n_samples) *
            UInt64(sizeof(ComplexF32))
        gram_args = Ref(
            _MinsrHostGramArgs(;
                struct_size=UInt32(sizeof(_MinsrHostGramArgs)),
                reserved=UInt32(0),
                samples=UInt(samples_pointer),
                samples_bytes=UInt64(minsr_descriptor.n_samples * sites),
                o_rows=UInt(rows_pointer),
                o_rows_bytes=UInt64(minsr_descriptor.n_samples) * row_bytes,
                gram_out=UInt(gram_pointer),
                gram_out_bytes=gram_bytes,
                stream=UInt(stream),
            ),
        )
        minsr_args = Ref(
            _MinsrHostMinsrArgs(;
                struct_size=UInt32(sizeof(_MinsrHostMinsrArgs)),
                reserved=UInt32(0),
                samples=UInt(samples_pointer),
                samples_bytes=UInt64(minsr_descriptor.n_samples * sites),
                logpsi=UInt(logpsi_pointer),
                logpsi_bytes=UInt64(2 * minsr_descriptor.n_samples) * UInt64(sizeof(Float64)),
                e_loc=UInt(e_loc_pointer),
                e_loc_bytes=UInt64(2 * minsr_descriptor.n_samples) * UInt64(sizeof(Float64)),
                logq=UInt(logq_pointer),
                logq_bytes=UInt64(minsr_descriptor.n_samples) * UInt64(sizeof(Float64)),
                gram=UInt(gram_pointer),
                gram_bytes,
                o_rows_device=UInt(rows_pointer),
                o_rows_host=UInt(0),
                o_rows_bytes=UInt64(minsr_descriptor.n_samples) * row_bytes,
                theta_dot_out=UInt(theta_pointer),
                theta_dot_out_bytes=UInt64(dense) * UInt64(sizeof(ComplexF32)),
                relative_cut=0.0,
                absolute_cut=0.0,
                e_mean_out=statistics_pointer,
                e_var_out=statistics_pointer + UInt(2 * sizeof(Float64)),
                ess_out=statistics_pointer + UInt(3 * sizeof(Float64)),
                stream=UInt(stream),
            ),
        )
        return _MinsrHostLane(;
            device,
            context,
            stream,
            gram,
            minsr,
            arena,
            arena_bytes,
            samples=samples_pointer,
            logpsi=logpsi_pointer,
            e_loc=e_loc_pointer,
            logq=logq_pointer,
            rows=rows_pointer,
            raw_gram=gram_pointer,
            theta=theta_pointer,
            statistics,
            statistics_pointer,
            gram_args,
            minsr_args,
            original_affinity=_MinsrHostCpuSet(ntuple(_ -> UInt64(0), Val(16))),
            affinity_saved=Int32(0),
            affinity_applied=Int32(0),
            status=Int32(0),
            failure_location="",
            failure_message="",
            input_copy_s=0.0,
            gram_s=0.0,
            minsr_s=0.0,
            complete_s=0.0,
        )
    catch
        _minsr_host_release_lane_resources!(context, stream, gram, minsr, arena)
        rethrow()
    end
end

function _minsr_host_destroy_lane!(host::MinsrHost, lane_index::Int)
    lane = host.lanes[lane_index]
    _minsr_host_release_lane_resources!(
        lane.context,
        lane.stream,
        lane.gram,
        lane.minsr,
        lane.arena,
    )
    lane.minsr = C_NULL
    lane.gram = C_NULL
    lane.stream = CUDA.CUstream(C_NULL)
    return nothing
end

@inline function _minsr_host_copy_inputs!(host::MinsrHost, lane_index::Int)
    lane = host.lanes[lane_index]
    inputs = host.peer_inputs
    sample_bytes = UInt64(host.n_samples * host.sites)
    complex_scalar_bytes = UInt64(2 * host.n_samples) * UInt64(sizeof(Float64))
    scalar_bytes = UInt64(host.n_samples) * UInt64(sizeof(Float64))
    source_context = host.lanes[1].context
    _minsr_host_peer_copy!(
        lane.samples,
        lane.context,
        inputs.samples,
        source_context,
        sample_bytes,
        lane.stream,
    )
    _minsr_host_peer_copy!(
        lane.logpsi,
        lane.context,
        inputs.logpsi,
        source_context,
        complex_scalar_bytes,
        lane.stream,
    )
    _minsr_host_peer_copy!(
        lane.e_loc,
        lane.context,
        inputs.e_loc,
        source_context,
        complex_scalar_bytes,
        lane.stream,
    )
    _minsr_host_peer_copy!(
        lane.logq,
        lane.context,
        inputs.logq,
        source_context,
        scalar_bytes,
        lane.stream,
    )
    row_bytes = UInt64(host.compact) * UInt64(sizeof(ComplexF32))
    for source_lane in 1:MINSR_HOST_LANES
        bytes = UInt64(host.row_counts[source_lane]) * row_bytes
        destination = lane.rows + UInt(host.row_bases[source_lane]) * UInt(row_bytes)
        source = inputs.row_shards[source_lane]
        offset = UInt64(0)
        while offset < bytes
            count = min(MINSR_HOST_PEER_TILE_BYTES, bytes - offset)
            _minsr_host_peer_copy!(
                destination + offset,
                lane.context,
                source + offset,
                host.lanes[source_lane].context,
                count,
                lane.stream,
            )
            offset += count
        end
    end
    _minsr_host_stream_synchronize!(lane.stream)
    return nothing
end

function _minsr_host_run_lane!(host::MinsrHost, lane_index::Int, command::Int32)
    lane = host.lanes[lane_index]
    lane.status = _MINSR_HOST_OK
    lane.failure_location = ""
    lane.failure_message = ""
    lane.input_copy_s = 0.0
    lane.gram_s = 0.0
    lane.minsr_s = 0.0
    lane.complete_s = 0.0
    command == _MINSR_HOST_COMMAND_NOOP && return nothing
    started = _minsr_host_seconds()
    copy_started = started
    _minsr_host_copy_inputs!(host, lane_index)
    lane.input_copy_s = _minsr_host_seconds() - copy_started
    if command == _MINSR_HOST_COMMAND_COPY
        lane.complete_s = _minsr_host_seconds() - started
        return nothing
    end
    gram_started = _minsr_host_seconds()
    status = FFI.gram_host_ctx_run(lane.gram, lane.gram_args)
    lane.gram_s = _minsr_host_seconds() - gram_started
    if status != 0
        lane.status = Int32(status)
        lane.failure_location = _last_error_location()
        lane.failure_message = _last_error_message()
        lane.complete_s = _minsr_host_seconds() - started
        return nothing
    end
    minsr_args = lane.minsr_args[]
    lane.minsr_args[] = _MinsrHostMinsrArgs(;
        struct_size=minsr_args.struct_size,
        reserved=minsr_args.reserved,
        samples=minsr_args.samples,
        samples_bytes=minsr_args.samples_bytes,
        logpsi=minsr_args.logpsi,
        logpsi_bytes=minsr_args.logpsi_bytes,
        e_loc=minsr_args.e_loc,
        e_loc_bytes=minsr_args.e_loc_bytes,
        logq=minsr_args.logq,
        logq_bytes=minsr_args.logq_bytes,
        gram=minsr_args.gram,
        gram_bytes=minsr_args.gram_bytes,
        o_rows_device=minsr_args.o_rows_device,
        o_rows_host=minsr_args.o_rows_host,
        o_rows_bytes=minsr_args.o_rows_bytes,
        theta_dot_out=minsr_args.theta_dot_out,
        theta_dot_out_bytes=minsr_args.theta_dot_out_bytes,
        relative_cut=host.inputs.relative_cut,
        absolute_cut=host.inputs.absolute_cut,
        e_mean_out=minsr_args.e_mean_out,
        e_var_out=minsr_args.e_var_out,
        ess_out=minsr_args.ess_out,
        stream=minsr_args.stream,
    )
    minsr_started = _minsr_host_seconds()
    status = FFI.minsr_host_ctx_run(lane.minsr, lane.minsr_args)
    lane.minsr_s = _minsr_host_seconds() - minsr_started
    lane.status = Int32(status)
    if status != 0
        lane.failure_location = _last_error_location()
        lane.failure_message = _last_error_message()
    end
    if status == 0 && lane_index == 1
        _minsr_host_peer_copy!(
            host.peer_inputs.theta_dot,
            host.lanes[1].context,
            lane.theta,
            lane.context,
            UInt64(host.dense) * UInt64(sizeof(ComplexF32)),
            lane.stream,
        )
        _minsr_host_stream_synchronize!(lane.stream)
    end
    lane.complete_s = _minsr_host_seconds() - started
    return nothing
end

function _minsr_host_worker!(host::MinsrHost, lane_index::Int)
    lane = host.lanes[lane_index]
    saved, applied, original = _minsr_host_pin_worker!(host.topology, lane_index)
    lane.original_affinity = original
    lane.affinity_saved = saved ? Int32(1) : Int32(0)
    lane.affinity_applied = applied ? Int32(1) : Int32(0)
    if lane.affinity_applied == 0
        lane.status = _MINSR_HOST_ERR_INTERNAL
    end
    CUDA.context!(lane.context)
    observed = host.epoch[]
    Threads.atomic_add!(host.ready, 1)
    while true
        while host.epoch[] == observed
            GC.safepoint()
            ccall(:jl_cpu_pause, Cvoid, ())
        end
        observed = host.epoch[]
        command = host.command[]
        if command == _MINSR_HOST_COMMAND_STOP
            _minsr_host_destroy_lane!(host, lane_index)
            if lane.affinity_saved == 1 && !_minsr_host_restore_worker!(lane.original_affinity)
                lane.status = _MINSR_HOST_ERR_INTERNAL
            end
            Threads.atomic_add!(host.done, 1)
            return nothing
        end
        try
            _minsr_host_run_lane!(host, lane_index, command)
        catch failure
            lane.status = _MINSR_HOST_ERR_INTERNAL
            lane.failure_location = "Julia lane worker"
            lane.failure_message = sprint(showerror, failure)
        end
        Threads.atomic_add!(host.done, 1)
    end
end

function _minsr_host_start_workers!(host::MinsrHost)
    tids = Threads.threadpooltids(:default)
    for lane_index in 1:MINSR_HOST_LANES
        task = Task(() -> _minsr_host_worker!(host, lane_index))
        task.sticky = true
        status = ccall(:jl_set_task_tid, Cint, (Any, Cint), task, Cint(tids[lane_index] - 1))
        status == 1 || error("failed to assign a Julia lane worker")
        push!(host.workers, task)
    end
    for task in host.workers
        schedule(task)
    end
    while host.ready[] != MINSR_HOST_LANES
        GC.safepoint()
        yield()
    end
    for lane in host.lanes
        lane.affinity_applied == 1 || throw(MinsrHostError(Int32(lane.device.handle), lane.status))
    end
    return nothing
end
