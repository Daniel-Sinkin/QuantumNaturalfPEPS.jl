
function _node_step_minsr!(
    host::NodeStepHost,
    relative_cut::Float64,
    absolute_cut::Float64,
)::Nothing
    host.stage = :minsr
    host.minsr_inputs.relative_cut = relative_cut
    host.minsr_inputs.absolute_cut = absolute_cut
    minsr = host.minsr
    prepare_status = _minsr_host_prepare!(minsr, host.minsr_inputs)
    prepare_status == _MINSR_HOST_OK ||
        throw(MinsrHostError(minsr.stats.failed_lane, prepare_status))
    lane = minsr.lanes[1]
    started = _node_step_seconds()
    copy_started = started
    _minsr_host_copy_inputs!(minsr, 1)
    lane.input_copy_s = _node_step_seconds() - copy_started
    gram_started = _node_step_seconds()
    _node_step_gram!(host, lane)
    lane.gram_s = _node_step_seconds() - gram_started
    samples = CUDA.CuPtr{UInt8}(lane.samples)
    logpsi = CUDA.CuPtr{Float64}(lane.logpsi)
    e_loc = CUDA.CuPtr{Float64}(lane.e_loc)
    logq = CUDA.CuPtr{Float64}(lane.logq)
    gram = CUDA.CuPtr{Cvoid}(lane.raw_gram)
    rows_device = CUDA.CuPtr{Cvoid}(UInt(0))
    theta = host.theta_dot_pointer
    statistics = host.minsr_statistics_pointer
    e_mean = statistics
    e_var = Ptr{Float64}(UInt(statistics) + UInt(2 * sizeof(Float64)))
    ess = Ptr{Float64}(UInt(statistics) + UInt(3 * sizeof(Float64)))
    config = Base.unsafe_convert(Ptr{QnpepsE2eConfig}, host.config_ref)
    stream = Ptr{Cvoid}(lane.stream)
    minsr_started = _node_step_seconds()
    inputs = FFI._E2eMinsrInputs(;
        device_samples=samples,
        logpsi,
        e_loc,
        logq,
        gram,
        o_rows_device=rows_device,
        o_rows_host=host.o_rows_host_pointer,
    )
    cuts = FFI._E2eCuts(; relative_cut, absolute_cut)
    outputs = FFI._E2eMinsrOutputs(; theta_dot=theta, e_mean, e_var, ess)
    arguments = FFI._E2eMinsrArguments(;
        config,
        n_samples=host.n_samples,
        inputs,
        host_tile_bytes=host.host_tile_bytes,
        cuts,
        outputs,
        stream,
    )
    status = GC.@preserve host FFI.e2e_minsr_ptr(arguments)
    status == 0 || error("qnpeps_e2e_minsr failed with status $status")
    _minsr_host_stream_synchronize!(lane.stream)
    lane.minsr_s = _node_step_seconds() - minsr_started
    lane.complete_s = _node_step_seconds() - started
    stats = minsr.stats
    values = host.minsr_statistics
    stats.e_mean[1] = values[1]
    stats.e_mean[2] = values[2]
    stats.e_var = values[3]
    stats.ess = values[4]
    return nothing
end
