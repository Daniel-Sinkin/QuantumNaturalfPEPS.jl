
function parse_runner_command(text::AbstractString)
    word = lowercase(strip(text))
    isempty(word) && return nothing
    word == "help" && return :help
    command = Symbol(word)
    command in (:status, :checkpoint, :pause, :stop) ||
        throw(ArgumentError("unknown runner command $(repr(word))"))
    return command
end

function request_command!(control::RunnerControl, command::Union{Symbol,AbstractString})
    parsed = command isa Symbol ? command : parse_runner_command(command)
    parsed === nothing && return nothing
    parsed == :help && throw(ArgumentError("help is not a runner command"))
    parsed in (:status, :checkpoint, :pause, :stop) ||
        throw(ArgumentError("unknown runner command $(repr(parsed))"))
    put!(control.commands, parsed)
    return parsed
end

function start_stdin_control!(control::RunnerControl; input::IO=stdin, output::IO=stderr)
    println(output, "[runner] commands status checkpoint pause stop help")
    return @async begin
        while !eof(input)
            line = readline(input)
            try
                command = parse_runner_command(line)
                command === nothing && continue
                if command == :help
                    println(output, "[runner] commands status checkpoint pause stop")
                else
                    request_command!(control, command)
                    println(output, "[runner] queued ", command)
                end
            catch err
                println(output, "[runner] ", sprint(showerror, err))
            end
        end
    end
end

mutable struct _RunnerState
    run_id::String
    sequence::Int
end

function _emit!(
    sink::AbstractEventSink,
    state::_RunnerState,
    kind::Symbol,
    name::Symbol;
    iteration::Union{Nothing,Integer}=nothing,
    payload=Dict{String,Any}(),
)
    state.sequence += 1
    event = RunEvent(
        1,
        state.run_id,
        state.sequence,
        round(Int64, time() * 1.0e9),
        kind,
        name,
        iteration === nothing ? nothing : Int(iteration),
        _payload_dict(payload),
    )
    emit_event!(sink, event)
    return event
end

function _checkpoint!(
    adapter::AbstractRunAdapter,
    sink::AbstractEventSink,
    state::_RunnerState,
    iteration::Int,
    reason::Symbol,
)
    metadata = _payload_dict(checkpoint_adapter!(adapter, iteration))
    metadata["reason"] = string(reason)
    _emit!(sink, state, :checkpoint, :written; iteration=iteration, payload=metadata)
    return metadata
end

function _process_commands!(
    adapter::AbstractRunAdapter,
    sink::AbstractEventSink,
    state::_RunnerState,
    control::RunnerControl,
    iteration::Int,
)
    while isready(control.commands)
        command = take!(control.commands)
        _emit!(
            sink,
            state,
            :command,
            :accepted;
            iteration=iteration,
            payload=Dict("command" => string(command)),
        )
        if command == :status
            _emit!(
                sink,
                state,
                :status,
                :snapshot;
                iteration=iteration,
                payload=adapter_status(adapter),
            )
        elseif command == :checkpoint
            _checkpoint!(adapter, sink, state, iteration, :command)
        elseif command == :pause
            _checkpoint!(adapter, sink, state, iteration, :pause)
            return :paused
        elseif command == :stop
            _checkpoint!(adapter, sink, state, iteration, :stop)
            return :stopped
        end
    end
    return nothing
end
