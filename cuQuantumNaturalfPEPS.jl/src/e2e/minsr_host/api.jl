
function minsr_host_try_run!(host::MinsrHost, inputs::MinsrHostInputs)
    Threads.atomic_cas!(host.running, false, true) == false ||
        return _minsr_host_fail!(host, Int32(-1), _MINSR_HOST_ERR_CONFIG)
    status = _minsr_host_prepare!(host, inputs)
    if status == _MINSR_HOST_OK
        _minsr_host_dispatch!(host, _MINSR_HOST_COMMAND_FULL)
        status = _minsr_host_assemble_stats!(host)
    end
    host.running[] = false
    return status
end

function minsr_host_run!(host::MinsrHost, inputs::MinsrHostInputs)
    status = minsr_host_try_run!(host, inputs)
    if status != _MINSR_HOST_OK
        failed_lane_index = Int(host.stats.failed_lane) + 1
        if checkbounds(Bool, host.lanes, failed_lane_index)
            failed_lane = host.lanes[failed_lane_index]
            throw(
                MinsrHostError(
                    host.stats.failed_lane,
                    status,
                    failed_lane.failure_location,
                    failed_lane.failure_message,
                ),
            )
        end
        throw(MinsrHostError(host.stats.failed_lane, status))
    end
    return host.stats
end

function Base.close(host::MinsrHost)
    Threads.atomic_cas!(host.closed, false, true) == false || return nothing
    while host.running[]
        GC.safepoint()
        yield()
    end
    _minsr_host_dispatch!(host, _MINSR_HOST_COMMAND_STOP)
    for worker in host.workers
        wait(worker)
    end
    empty!(host.workers)
    GC.gc()
    return nothing
end
