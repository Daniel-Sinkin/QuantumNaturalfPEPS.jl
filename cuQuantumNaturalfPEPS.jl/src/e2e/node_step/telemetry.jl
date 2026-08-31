
@inline _node_step_seconds() = Float64(time_ns()) * 1.0e-9

function _node_step_assemble_telemetry!(
    host::NodeStepHost,
    started::Float64,
    dlenv_done::Float64,
    sampled::Float64,
    staged::Float64,
    eo_done::Float64,
    minsr_done::Float64,
    outputs_done::Float64,
)::Nothing
    telemetry = host.telemetry
    telemetry.epoch = host.epoch
    telemetry.ns = host.n_samples
    telemetry.predraw_join_s = 0.0
    telemetry.env_wait_s = dlenv_done - started
    telemetry.sampling_s = sampled - dlenv_done
    telemetry.ring_s = staged - sampled
    telemetry.eo_s = eo_done - staged
    telemetry.predraw_issue_s = 0.0
    telemetry.minsr_s = minsr_done - eo_done
    telemetry.output_s = outputs_done - minsr_done
    telemetry.complete_s = outputs_done - started
    telemetry.status = _NODE_STEP_OK
    return nothing
end
