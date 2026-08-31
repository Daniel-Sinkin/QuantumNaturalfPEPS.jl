
function MinsrHost(;
    lx::Integer,
    ly::Integer,
    dim_bond::Integer,
    n_samples::Integer,
    dim_phys::Integer=2,
    topology::MinsrHostTopology=MINSR_HOST_TOPOLOGY_JURECA,
)
    _minsr_host_validate_descriptor(lx, ly, dim_phys, dim_bond, n_samples, topology)
    gram_descriptor = _MinsrHostGramDesc(;
        struct_size=UInt32(sizeof(_MinsrHostGramDesc)),
        lx=Int32(lx),
        ly=Int32(ly),
        dim_phys=Int32(dim_phys),
        dim_bond=Int32(dim_bond),
        consumer=Int32(0),
        reserved=Int32(0),
        n_samples=Int64(n_samples),
    )
    minsr_descriptor = _MinsrHostMinsrDesc(;
        struct_size=UInt32(sizeof(_MinsrHostMinsrDesc)),
        lx=Int32(lx),
        ly=Int32(ly),
        dim_phys=Int32(dim_phys),
        dim_bond=Int32(dim_bond),
        diagnostics=Int32(0),
        reserved=Int32(0),
        tail_padding=UInt32(0),
        n_samples=Int64(n_samples),
        host_tile_bytes=Int64(0),
    )
    compact = FFI.minsr_host_compact_count(minsr_descriptor)
    dense = FFI.minsr_host_dense_count(minsr_descriptor)
    compact > 0 && dense > 0 || throw(ArgumentError("published minSR rejected the descriptor"))
    quotient, remainder = divrem(Int64(n_samples), Int64(MINSR_HOST_LANES))
    row_counts =
        ntuple(lane -> quotient + (lane <= remainder ? Int64(1) : Int64(0)), MINSR_HOST_LANES)
    row_bases = ntuple(lane -> lane == 1 ? Int64(0) : sum(row_counts[1:(lane-1)]), MINSR_HOST_LANES)
    caller_context = CUDA.context()
    devices = ntuple(lane -> CUDA.CuDevice(lane - 1), MINSR_HOST_LANES)
    contexts = ntuple(lane -> CUDA.context(devices[lane]), MINSR_HOST_LANES)
    peer_pairs = Int32(0)
    lanes = Vector{_MinsrHostLane}(undef, MINSR_HOST_LANES)
    built = 0
    try
        peer_pairs = _minsr_host_enable_peers!(devices, contexts)
        try
            for lane_index in 1:MINSR_HOST_LANES
                saved, applied, original = _minsr_host_pin_worker!(topology, lane_index)
                saved && applied ||
                    throw(MinsrHostError(Int32(lane_index - 1), _MINSR_HOST_ERR_INTERNAL))
                restored = false
                try
                    lanes[lane_index] = _minsr_host_create_lane(
                        lane_index,
                        devices[lane_index],
                        contexts[lane_index],
                        gram_descriptor,
                        minsr_descriptor,
                        Int64(lx) * Int64(ly),
                        compact,
                        dense,
                    )
                    built += 1
                finally
                    restored = _minsr_host_restore_worker!(original)
                end
                restored || throw(MinsrHostError(Int32(lane_index - 1), _MINSR_HOST_ERR_INTERNAL))
            end
        catch
            for lane_index in 1:built
                lane = lanes[lane_index]
                _minsr_host_release_lane_resources!(
                    lane.context,
                    lane.stream,
                    lane.gram,
                    lane.minsr,
                    lane.arena,
                )
            end
            rethrow()
        end
    finally
        CUDA.context!(caller_context)
    end
    stats = MinsrHostStats(topology)
    stats.peer_pairs_enabled = peer_pairs
    for lane_index in 1:MINSR_HOST_LANES
        stats.row_counts[lane_index] = row_counts[lane_index]
        stats.logical_devices[lane_index] = Int32(lane_index - 1)
        stats.arena_bytes[lane_index] = lanes[lane_index].arena_bytes
    end
    descriptor_state = _MinsrHostDescriptor(;
        lx=Int32(lx),
        ly=Int32(ly),
        dim_phys=Int32(dim_phys),
        dim_bond=Int32(dim_bond),
        n_samples=Int64(n_samples),
        sites=Int64(lx) * Int64(ly),
        compact,
        dense,
        topology,
    )
    lane_state = _MinsrHostLanes(; row_bases, row_counts, lanes, workers=Task[])
    runtime_state = _MinsrHostRuntime(;
        inputs=MinsrHostInputs(),
        peer_inputs=_MinsrHostPeerInputs(),
        stats,
        command=Threads.Atomic{Int32}(_MINSR_HOST_COMMAND_NOOP),
        epoch=Threads.Atomic{Int}(0),
        ready=Threads.Atomic{Int}(0),
        done=Threads.Atomic{Int}(0),
        closed=Threads.Atomic{Bool}(false),
        running=Threads.Atomic{Bool}(false),
    )
    state = _MinsrHostState(; descriptor=descriptor_state, lanes=lane_state, runtime=runtime_state)
    host = MinsrHost(state)
    try
        _minsr_host_start_workers!(host)
    catch
        if host.ready[] == MINSR_HOST_LANES
            close(host)
        else
            host.closed[] = true
            for lane_index in 1:MINSR_HOST_LANES
                _minsr_host_destroy_lane!(host, lane_index)
            end
        end
        rethrow()
    end
    finalizer(close, host)
    return host
end

Base.isopen(host::MinsrHost) = !host.closed[]

function _minsr_host_fail!(host::MinsrHost, lane::Int32, status::Int32)
    host.stats.failed_lane = lane
    lane >= 0 && (host.stats.lane_status[Int(lane)+1] = status)
    return status
end

function _minsr_host_prepare!(host::MinsrHost, inputs::MinsrHostInputs)
    isopen(host) || return _minsr_host_fail!(host, Int32(-1), _MINSR_HOST_ERR_CONFIG)
    host.stats.failed_lane = Int32(-1)
    for lane in 1:MINSR_HOST_LANES
        host.stats.lane_status[lane] = _MINSR_HOST_OK
    end
    inputs.samples == 0 && return _minsr_host_fail!(host, Int32(-1), _MINSR_HOST_ERR_NULL)
    inputs.logpsi == 0 && return _minsr_host_fail!(host, Int32(-1), _MINSR_HOST_ERR_NULL)
    inputs.e_loc == 0 && return _minsr_host_fail!(host, Int32(-1), _MINSR_HOST_ERR_NULL)
    inputs.logq == 0 && return _minsr_host_fail!(host, Int32(-1), _MINSR_HOST_ERR_NULL)
    inputs.theta_dot == 0 && return _minsr_host_fail!(host, Int32(-1), _MINSR_HOST_ERR_NULL)
    sample_bytes = UInt64(host.n_samples * host.sites)
    complex_scalar_bytes = UInt64(2 * host.n_samples) * UInt64(sizeof(Float64))
    scalar_bytes = UInt64(host.n_samples) * UInt64(sizeof(Float64))
    theta_bytes = UInt64(host.dense) * UInt64(sizeof(ComplexF32))
    if inputs.samples_bytes < sample_bytes ||
       inputs.logpsi_bytes < complex_scalar_bytes ||
       inputs.e_loc_bytes < complex_scalar_bytes ||
       inputs.logq_bytes < scalar_bytes ||
       inputs.theta_dot_bytes < theta_bytes
        return _minsr_host_fail!(host, Int32(-1), _MINSR_HOST_ERR_CONFIG)
    end
    row_bytes = UInt64(host.compact) * UInt64(sizeof(ComplexF32))
    for lane in 1:MINSR_HOST_LANES
        inputs.row_shards[lane] == 0 &&
            return _minsr_host_fail!(host, Int32(lane - 1), _MINSR_HOST_ERR_NULL)
        inputs.row_shard_bytes[lane] < UInt64(host.row_counts[lane]) * row_bytes &&
            return _minsr_host_fail!(host, Int32(lane - 1), _MINSR_HOST_ERR_CONFIG)
    end
    host.inputs.samples = inputs.samples
    host.inputs.samples_bytes = inputs.samples_bytes
    host.inputs.logpsi = inputs.logpsi
    host.inputs.logpsi_bytes = inputs.logpsi_bytes
    host.inputs.e_loc = inputs.e_loc
    host.inputs.e_loc_bytes = inputs.e_loc_bytes
    host.inputs.logq = inputs.logq
    host.inputs.logq_bytes = inputs.logq_bytes
    host.inputs.row_shards = inputs.row_shards
    host.inputs.row_shard_bytes = inputs.row_shard_bytes
    host.inputs.theta_dot = inputs.theta_dot
    host.inputs.theta_dot_bytes = inputs.theta_dot_bytes
    host.inputs.relative_cut = inputs.relative_cut
    host.inputs.absolute_cut = inputs.absolute_cut
    host.peer_inputs.samples = CUDA.CuPtr{Cvoid}(inputs.samples)
    host.peer_inputs.logpsi = CUDA.CuPtr{Cvoid}(inputs.logpsi)
    host.peer_inputs.e_loc = CUDA.CuPtr{Cvoid}(inputs.e_loc)
    host.peer_inputs.logq = CUDA.CuPtr{Cvoid}(inputs.logq)
    host.peer_inputs.row_shards = (
        CUDA.CuPtr{Cvoid}(inputs.row_shards[1]),
        CUDA.CuPtr{Cvoid}(inputs.row_shards[2]),
        CUDA.CuPtr{Cvoid}(inputs.row_shards[3]),
        CUDA.CuPtr{Cvoid}(inputs.row_shards[4]),
    )
    host.peer_inputs.theta_dot = CUDA.CuPtr{Cvoid}(inputs.theta_dot)
    return _MINSR_HOST_OK
end

function _minsr_host_dispatch!(host::MinsrHost, command::Int32)
    host.done[] = 0
    host.command[] = command
    Threads.atomic_add!(host.epoch, 1)
    while host.done[] != MINSR_HOST_LANES
        GC.safepoint()
        ccall(:jl_cpu_pause, Cvoid, ())
    end
    return nothing
end

function _minsr_host_assemble_stats!(host::MinsrHost)
    failed = Int32(-1)
    status = _MINSR_HOST_OK
    for lane_index in 1:MINSR_HOST_LANES
        lane = host.lanes[lane_index]
        host.stats.affinity_applied[lane_index] = lane.affinity_applied
        host.stats.lane_status[lane_index] = lane.status
        host.stats.input_copy_s[lane_index] = lane.input_copy_s
        host.stats.gram_s[lane_index] = lane.gram_s
        host.stats.minsr_s[lane_index] = lane.minsr_s
        host.stats.complete_s[lane_index] = lane.complete_s
        if status == _MINSR_HOST_OK && lane.status != _MINSR_HOST_OK
            failed = Int32(lane_index - 1)
            status = lane.status
        end
    end
    host.stats.failed_lane = failed
    if status == _MINSR_HOST_OK
        lane = host.lanes[1]
        host.stats.e_mean[1] = lane.statistics[1]
        host.stats.e_mean[2] = lane.statistics[2]
        host.stats.e_var = lane.statistics[3]
        host.stats.ess = lane.statistics[4]
    end
    return status
end
