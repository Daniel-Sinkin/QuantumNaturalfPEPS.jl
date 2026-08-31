
function Base.close(host::EoHost)::Nothing
    host.closed && return nothing
    host.closed = true
    caller_context = CUDA.context()
    try
        for lane in host.lanes
            CUDA.context!(lane.context)
            _eo_host_stream_synchronize!(lane.stream)
        end
        host.teardown_order[1] = Int32(1)
        for lane in host.lanes
            CUDA.context!(lane.context)
            lane.handle == C_NULL || FFI.eloc_ctx_destroy(lane.handle)
            lane.handle == C_NULL || (host.contexts_destroyed += 1)
            lane.handle = C_NULL
        end
        host.teardown_order[2] = Int32(2)
        for lane in host.lanes
            CUDA.context!(lane.context)
            CUDA.free(lane.binding.arena)
            host.arenas_freed += 1
        end
        host.teardown_order[3] = Int32(3)
        for lane in host.lanes
            CUDA.context!(lane.context)
            lane.stream == C_NULL || CUDA.cuStreamDestroy_v2(lane.stream)
            lane.stream == C_NULL || (host.streams_destroyed += 1)
            lane.stream = CUDA.CUstream(C_NULL)
        end
        host.teardown_order[4] = Int32(4)
    finally
        CUDA.context!(caller_context)
    end
    empty!(host.lanes)
    host.term_table = nothing
    host.source_peps = nothing
    host.source_samples = nothing
    host.destination_logpsi = nothing
    host.destination_e_loc = nothing
    host.destination_rows = nothing
    host.source_peps_pointer = CUDA.CuPtr{Cvoid}(0)
    host.source_samples_pointer = CUDA.CuPtr{Cvoid}(0)
    host.destination_logpsi_pointer = CUDA.CuPtr{Cvoid}(0)
    host.destination_e_loc_pointer = CUDA.CuPtr{Cvoid}(0)
    host.destination_rows_pointer = Ptr{Cvoid}(0)
    host.refs_cleared = true
    host.teardown_order[5] = Int32(5)
    GC.gc()
    host.gc_runs += 1
    host.teardown_order[6] = Int32(6)
    return nothing
end
