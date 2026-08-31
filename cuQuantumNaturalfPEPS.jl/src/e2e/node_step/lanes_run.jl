
@inline _node_step_eo_lane(host::NodeStepHost, ::Nothing, lane_index::Int) =
    host.eo_lanes[lane_index]

@inline _node_step_eo_lane(host::NodeStepHost, eo::EoHost, lane_index::Int) = eo.lanes[lane_index]

@inline _node_step_eo_lanes(host::NodeStepHost, ::Nothing) = host.eo_lanes
@inline _node_step_eo_lanes(host::NodeStepHost, eo::EoHost) = eo.lanes

function _node_step_precompile_eo_host!(dlenv, sampler)::Nothing
    host_type = NodeStepHost{typeof(dlenv),typeof(sampler),MinsrHost,MinsrHostInputs,EoHost}
    Base.precompile(eo_host_execute!, (EoHost, Int)) || error("failed to precompile E/O execute")
    Base.precompile(_node_step_eo_lane, (host_type, EoHost, Int)) ||
        error("failed to precompile E/O lane selection")
    Base.precompile(_node_step_run_eo_lane!, (host_type, EoHost, Int)) ||
        error("failed to precompile E/O lane run")
    Base.precompile(_node_step_run_eo_lane!, (host_type, Int)) ||
        error("failed to precompile E/O lane dispatch")
    Base.precompile(_node_step_eo_worker!, (host_type, Int)) ||
        error("failed to precompile E/O worker")
    return nothing
end

@inline function _node_step_run_eo_lane!(host::NodeStepHost, lane_index::Int)::Nothing
    return _node_step_run_eo_lane!(host, host.eo_host, lane_index)
end

function _node_step_run_eo_lane!(host::NodeStepHost, ::Nothing, lane_index::Int)::Nothing
    lane = host.eo_lanes[lane_index]
    source_context = host.minsr.lanes[1].context
    lane.status = _NODE_STEP_OK
    lane.failure_location = ""
    lane.failure_message = ""
    _minsr_host_peer_copy!(
        lane.peps_pointer,
        lane.context,
        host.source_peps_pointer,
        source_context,
        host.source_peps_bytes,
        lane.stream,
    )
    _minsr_host_peer_copy!(
        lane.samples_pointer,
        lane.context,
        lane.source_sample_pointer,
        source_context,
        UInt64(lane.row_count * host.sites),
        lane.stream,
    )
    status = GC.@preserve lane FFI.eloc_ctx_run(lane.handle, Ptr{Cvoid}(pointer(lane.args)))
    if status == 0
        scalar_bytes = UInt64(2 * lane.row_count * sizeof(Float64))
        _minsr_host_peer_copy!(
            lane.destination_logpsi_pointer,
            source_context,
            lane.logpsi_pointer,
            lane.context,
            scalar_bytes,
            lane.stream,
        )
        _minsr_host_peer_copy!(
            lane.destination_e_loc_pointer,
            source_context,
            lane.e_loc_pointer,
            lane.context,
            scalar_bytes,
            lane.stream,
        )
        _minsr_host_stream_synchronize!(lane.stream)
    end
    lane.status = Int32(status)
    if status != 0
        lane.failure_location = _last_error_location()
        lane.failure_message = _last_error_message()
    end
    return nothing
end

function _node_step_run_eo_lane!(host::NodeStepHost, eo::EoHost, lane_index::Int)::Nothing
    eo_host_execute!(eo, lane_index)
    return nothing
end

function _node_step_eo_worker!(host::NodeStepHost, lane_index::Int)::Nothing
    lane = _node_step_eo_lane(host, host.eo_host, lane_index)
    saved, applied, original = _minsr_host_pin_worker!(host.minsr.topology, lane_index)
    lane.original_affinity = original.words
    lane.affinity_saved = saved ? Int32(1) : Int32(0)
    lane.affinity_applied = applied ? Int32(1) : Int32(0)
    applied || (lane.status = Int32(6))
    CUDA.context!(lane.context)
    observed = host.command_epoch[]
    Threads.atomic_add!(host.ready, 1)
    while true
        while host.command_epoch[] == observed
            GC.safepoint()
            ccall(:jl_cpu_pause, Cvoid, ())
        end
        observed = host.command_epoch[]
        if host.command[] == _NODE_STEP_EO_STOP
            if lane.affinity_saved == 1 &&
               !_minsr_host_restore_worker!(_MinsrHostCpuSet(lane.original_affinity))
                lane.status = Int32(6)
            end
            Threads.atomic_add!(host.done, 1)
            return nothing
        end
        try
            _node_step_run_eo_lane!(host, lane_index)
        catch failure
            lane.status = Int32(6)
            lane.failure_location = "Julia lane worker"
            lane.failure_message = sprint(showerror, failure)
        end
        Threads.atomic_add!(host.done, 1)
    end
end

function _node_step_start_workers!(host::NodeStepHost)::Nothing
    tids = Threads.threadpooltids(:default)
    for lane_index in 1:NODE_STEP_LANES
        task = Task(() -> _node_step_eo_worker!(host, lane_index))
        task.sticky = true
        status = ccall(
            :jl_set_task_tid,
            Cint,
            (Any, Cint),
            task,
            Cint(tids[NODE_STEP_LANES+lane_index] - 1),
        )
        status == 1 || error("failed to assign an E/O lane worker")
        push!(host.workers, task)
    end
    for task in host.workers
        schedule(task)
    end
    while host.ready[] != NODE_STEP_LANES
        GC.safepoint()
        yield()
    end
    for lane in _node_step_eo_lanes(host, host.eo_host)
        lane.affinity_applied == 1 || error("failed to pin an E/O lane worker")
    end
    return nothing
end

function _node_step_dispatch_eo!(host::NodeStepHost)::Nothing
    host.stage = :eo
    host.done[] = 0
    host.command[] = _NODE_STEP_EO_RUN
    Threads.atomic_add!(host.command_epoch, 1)
    while host.done[] != NODE_STEP_LANES
        GC.safepoint()
        ccall(:jl_cpu_pause, Cvoid, ())
    end
    for lane in _node_step_eo_lanes(host, host.eo_host)
        if lane.status != 0
            location = isempty(lane.failure_location) ? "" : " at $(lane.failure_location)"
            message = isempty(lane.failure_message) ? "" : "; $(lane.failure_message)"
            error("qnpeps_eloc_ctx_run failed with status $(lane.status)$location$message")
        end
    end
    return nothing
end
