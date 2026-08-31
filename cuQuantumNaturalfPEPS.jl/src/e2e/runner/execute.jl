
function _iteration_payload(result::IterationResult, wall_seconds::Float64)
    payload = Dict{String,Any}(
        "samples_requested" => result.samples_requested,
        "samples_used" => result.samples_used,
        "ess_estimated" => result.ess_estimated,
        "ess_measured" => result.ess_measured,
        "ess_fraction" => (
            result.ess_measured === nothing || result.samples_used == 0 ? nothing :
            result.ess_measured / result.samples_used
        ),
        "energy_before" => result.energy_before,
        "energy_after" => result.energy_after,
        "energy_variance" => result.energy_variance,
        "direction_norm" => result.direction_norm,
        "step_norm" => result.step_norm,
        "wall_seconds" => wall_seconds,
        "phase_seconds" => result.phase_seconds,
        "fallback_count" => length(result.fallbacks),
        "fallbacks" => result.fallbacks,
    )
    for (key, value) in result.metrics
        haskey(payload, key) &&
            throw(ArgumentError("adapter metric $(repr(key)) shadows a runner field"))
        payload[key] = value
    end
    return payload
end

function _error_payload(err, backtrace)
    payload = Dict{String,Any}(
        "exception_type" => string(typeof(err)),
        "message" => sprint(showerror, err),
    )
    frames = stacktrace(backtrace)
    if !isempty(frames)
        frame = first(frames)
        payload["file"] = string(frame.file)
        payload["line"] = frame.line
        payload["function"] = string(frame.func)
    end
    return payload
end

function _validate_runner_config(config::RunnerConfig)
    config.iterations >= 0 || throw(ArgumentError("iterations must be nonnegative"))
    config.checkpoint_every >= 0 || throw(ArgumentError("checkpoint_every must be nonnegative"))
    isempty(config.run_id) && throw(ArgumentError("run_id must not be empty"))
    return nothing
end

function run_iterations!(
    adapter::AbstractRunAdapter,
    config::RunnerConfig;
    sink::AbstractEventSink=StreamEventSink(stdout),
    control::RunnerControl=RunnerControl(),
)
    _validate_runner_config(config)
    state = _RunnerState(config.run_id, 0)
    status = :completed
    initialized = false
    completed = 0
    first_iteration = 0
    last_iteration = 0
    saved_error = nothing

    _emit!(
        sink,
        state,
        :lifecycle,
        :started;
        payload=Dict(
            "iterations_requested" => config.iterations,
            "checkpoint_every" => config.checkpoint_every,
            "expensive_debug" => config.expensive_debug,
        ),
    )

    try
        initialization = _payload_dict(initialize_adapter!(adapter))
        initialized = true
        last_iteration = Int(adapter_iteration(adapter))
        first_iteration = last_iteration + 1
        _emit!(
            sink,
            state,
            :lifecycle,
            :adapter_initialized;
            iteration=last_iteration,
            payload=initialization,
        )

        command_status = _process_commands!(adapter, sink, state, control, last_iteration)
        command_status === nothing || (status = command_status)

        while status == :completed && completed < config.iterations
            iteration = first_iteration + completed
            started_ns = time_ns()
            result = step_adapter!(adapter, iteration; expensive_debug=config.expensive_debug)
            result isa IterationResult ||
                throw(ArgumentError("step_adapter! must return IterationResult"))
            wall_seconds = Float64(time_ns() - started_ns) / 1.0e9
            completed += 1
            last_iteration = iteration

            _emit!(
                sink,
                state,
                :iteration,
                :completed;
                iteration=iteration,
                payload=_iteration_payload(result, wall_seconds),
            )
            if config.expensive_debug && !isempty(result.debug)
                _emit!(
                    sink,
                    state,
                    :debug,
                    :iteration_diagnostics;
                    iteration=iteration,
                    payload=result.debug,
                )
            end
            for notice in result.warnings
                payload = copy(notice.payload)
                payload["code"] = notice.code
                payload["message"] = notice.message
                _emit!(
                    sink,
                    state,
                    :warning,
                    :adapter_warning;
                    iteration=iteration,
                    payload=payload,
                )
            end

            if config.checkpoint_every > 0 && iteration % config.checkpoint_every == 0
                _checkpoint!(adapter, sink, state, iteration, :periodic)
            end

            command_status = _process_commands!(adapter, sink, state, control, iteration)
            command_status === nothing || (status = command_status)
        end
    catch err
        bt = catch_backtrace()
        status = :error
        saved_error = (err, bt)
        _emit!(
            sink,
            state,
            :error,
            :run_failed;
            iteration=last_iteration,
            payload=_error_payload(err, bt),
        )
    finally
        if initialized
            try
                close_metadata = _payload_dict(close_adapter!(adapter))
                _emit!(
                    sink,
                    state,
                    :lifecycle,
                    :adapter_closed;
                    iteration=last_iteration,
                    payload=close_metadata,
                )
            catch close_err
                close_bt = catch_backtrace()
                status = :error
                saved_error === nothing && (saved_error = (close_err, close_bt))
                _emit!(
                    sink,
                    state,
                    :error,
                    :adapter_close_failed;
                    iteration=last_iteration,
                    payload=_error_payload(close_err, close_bt),
                )
            end
        end
        _emit!(
            sink,
            state,
            :lifecycle,
            :finished;
            iteration=last_iteration,
            payload=Dict("status" => string(status), "iterations_completed" => completed),
        )
        flush_event_sink!(sink)
    end

    if config.rethrow_errors && saved_error !== nothing
        throw(saved_error[1])
    end
    return RunnerResult(status, config.run_id, first_iteration, last_iteration, completed)
end
