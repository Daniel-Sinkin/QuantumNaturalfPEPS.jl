
function _node_step_destroy_eo_lane!(lane::_NodeStepEoLane)::Nothing
    CUDA.context!(lane.context)
    lane.stream == C_NULL || CUDA.cuStreamSynchronize(lane.stream)
    lane.handle == C_NULL || FFI.eloc_ctx_destroy(lane.handle)
    lane.handle = C_NULL
    CUDA.unsafe_free!(lane.rows)
    CUDA.unsafe_free!(lane.e_loc)
    CUDA.unsafe_free!(lane.logpsi)
    CUDA.unsafe_free!(lane.samples)
    CUDA.unsafe_free!(lane.peps)
    lane.stream == C_NULL || CUDA.cuStreamDestroy_v2(lane.stream)
    lane.stream = CUDA.CUstream(C_NULL)
    return nothing
end

function Base.close(host::NodeStepHost)::Nothing
    Threads.atomic_cas!(host.closed, false, true) == false || return nothing
    while host.running[]
        GC.safepoint()
        yield()
    end
    host.done[] = 0
    host.command[] = _NODE_STEP_EO_STOP
    Threads.atomic_add!(host.command_epoch, 1)
    while host.done[] != NODE_STEP_LANES
        GC.safepoint()
        yield()
    end
    for worker in host.workers
        wait(worker)
    end
    empty!(host.workers)
    caller_context = CUDA.context()
    try
        if host.eo_host === nothing
            for lane in host.eo_lanes
                _node_step_destroy_eo_lane!(lane)
            end
        else
            close(host.eo_host)
        end
    finally
        CUDA.context!(caller_context)
    end
    CUDA.device!(0) do
        CUDA.stream!(host.sampler.stream) do
            close(host.sampler)
        end
        CUDA.stream!(host.dlenv.stream) do
            close(host.dlenv)
        end
        CUDA.unsafe_free!(host.theta_dot)
        CUDA.unsafe_free!(host.e_loc)
        CUDA.unsafe_free!(host.logpsi)
        CUDA.unsafe_free!(host.log_gauge)
        CUDA.unsafe_free!(host.logq)
        CUDA.unsafe_free!(host.samples)
        CUDA.unsafe_free!(host.gram_scratch)
        CUDA.unregister(host.o_rows_registration)
    end
    close(host.minsr)
    return nothing
end
