
function node_step!(
    host::NodeStepHost;
    relative_cut::Real=1.0e-3,
    absolute_cut::Real=1.0e-8,
)::NodeStepTelemetry
    isopen(host) || throw(ArgumentError("NodeStepHost is closed"))
    Threads.atomic_cas!(host.running, false, true) == false ||
        throw(ArgumentError("NodeStepHost is already running"))
    try
        started = _node_step_seconds()
        _node_step_dlenv!(host)
        dlenv_done = _node_step_seconds()
        _node_step_draw!(host)
        sampled = _node_step_seconds()
        _node_step_stage_samples!(host)
        staged = _node_step_seconds()
        _node_step_dispatch_eo!(host)
        eo_done = _node_step_seconds()
        _node_step_minsr!(host, Float64(relative_cut), Float64(absolute_cut))
        minsr_done = _node_step_seconds()
        outputs_done = _node_step_seconds()
        _node_step_assemble_telemetry!(
            host,
            started,
            dlenv_done,
            sampled,
            staged,
            eo_done,
            minsr_done,
            outputs_done,
        )
        host.stage = :complete
        host.epoch += 1
        return host.telemetry
    finally
        host.running[] = false
    end
end
