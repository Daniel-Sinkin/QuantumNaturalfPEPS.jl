
mutable struct StreamEventSink <: AbstractEventSink
    io::IO
    flush_each::Bool
    close_on_finish::Bool
    max_value_chars::Int
end

function StreamEventSink(
    io::IO;
    flush_each::Bool=true,
    close_on_finish::Bool=false,
    max_value_chars::Integer=240,
)
    max_value_chars > 0 || throw(ArgumentError("max_value_chars must be positive"))
    return StreamEventSink(io, flush_each, close_on_finish, Int(max_value_chars))
end

function StreamEventSink(
    path::AbstractString;
    append::Bool=false,
    flush_each::Bool=true,
    max_value_chars::Integer=240,
)
    mkpath(dirname(abspath(path)))
    io = open(path, append ? "a" : "w")
    return StreamEventSink(
        io;
        flush_each=flush_each,
        close_on_finish=true,
        max_value_chars=max_value_chars,
    )
end

mutable struct JsonlEventSink <: AbstractEventSink
    io::IO
    flush_each::Bool
    close_on_finish::Bool
end

function _jsonl_header()
    return (format="QNfPEPS-events", schema_version=1, encoding="jsonl")
end

function _write_jsonl_header!(io::IO)
    JSON3.write(io, _jsonl_header())
    write(io, '\n')
    return nothing
end

function _validate_jsonl_header(path::AbstractString)
    first_line = open(readline, path)
    header = try
        JSON3.read(first_line)
    catch err
        throw(ArgumentError("invalid JSONL header at $(path)"))
    end
    valid =
        hasproperty(header, :format) &&
        hasproperty(header, :schema_version) &&
        hasproperty(header, :encoding) &&
        header.format == "QNfPEPS-events" &&
        header.schema_version == 1 &&
        header.encoding == "jsonl"
    valid || throw(ArgumentError("incompatible JSONL header at $(path)"))
    return nothing
end

function JsonlEventSink(
    io::IO;
    flush_each::Bool=true,
    close_on_finish::Bool=false,
    write_header::Bool=true,
)
    write_header && _write_jsonl_header!(io)
    flush_each && flush(io)
    return JsonlEventSink(io, flush_each, close_on_finish)
end

function JsonlEventSink(path::AbstractString; append::Bool=false, flush_each::Bool=true)
    mkpath(dirname(abspath(path)))
    has_header = append && isfile(path) && filesize(path) > 0
    has_header && _validate_jsonl_header(path)
    io = open(path, append ? "a" : "w")
    return JsonlEventSink(
        io;
        flush_each=flush_each,
        close_on_finish=true,
        write_header=(!has_header),
    )
end

struct TeeEventSink <: AbstractEventSink
    sinks::Vector{AbstractEventSink}
end

TeeEventSink(sinks::AbstractEventSink...) = TeeEventSink(AbstractEventSink[sinks...])

struct CallbackEventSink{F} <: AbstractEventSink
    callback::F
end

function _payload_dict(payload)
    out = Dict{String,Any}()
    for (key, value) in pairs(payload)
        out[string(key)] = value
    end
    return out
end

function _json_value(value)
    if value === nothing || value isa Bool || value isa Integer || value isa AbstractString
        return value
    elseif value isa AbstractFloat
        return isfinite(value) ? value : string(value)
    elseif value isa Complex
        return Dict("re" => _json_value(real(value)), "im" => _json_value(imag(value)))
    elseif value isa Symbol
        return string(value)
    elseif value isa NamedTuple || value isa AbstractDict
        out = Dict{String,Any}()
        for key in sort!(collect(keys(value)); by=string)
            out[string(key)] = _json_value(value[key])
        end
        return out
    elseif value isa Tuple || value isa AbstractVector
        return [_json_value(item) for item in value]
    else
        return string(value)
    end
end

function event_record(event::RunEvent)
    return (
        schema_version=event.schema_version,
        run_id=event.run_id,
        sequence=event.sequence,
        timestamp_unix_ns=event.timestamp_unix_ns,
        kind=string(event.kind),
        name=string(event.name),
        iteration=event.iteration,
        payload=_json_value(event.payload),
    )
end

function _display_value(value, max_chars::Int)
    text = replace(repr(value), '\n' => ' ')
    return length(text) <= max_chars ? text : string(first(text, max_chars), "…")
end

function emit_event!(sink::StreamEventSink, event::RunEvent)
    print(sink.io, '[', event.kind, "]")
    print(sink.io, " seq=", event.sequence)
    print(sink.io, " run=", _display_value(event.run_id, sink.max_value_chars))
    print(sink.io, " name=", event.name)
    event.iteration === nothing || print(sink.io, " iteration=", event.iteration)
    for key in sort!(collect(keys(event.payload)))
        print(sink.io, ' ', key, '=', _display_value(event.payload[key], sink.max_value_chars))
    end
    println(sink.io)
    sink.flush_each && flush(sink.io)
    return nothing
end

function emit_event!(sink::JsonlEventSink, event::RunEvent)
    JSON3.write(sink.io, event_record(event))
    write(sink.io, '\n')
    sink.flush_each && flush(sink.io)
    return nothing
end

function emit_event!(sink::TeeEventSink, event::RunEvent)
    first_error = nothing
    for child in sink.sinks
        try
            emit_event!(child, event)
        catch err
            first_error === nothing && (first_error = err)
        end
    end
    first_error === nothing || throw(first_error)
    return nothing
end

function emit_event!(sink::CallbackEventSink, event::RunEvent)
    sink.callback(event)
    return nothing
end

flush_event_sink!(::AbstractEventSink) = nothing
close_event_sink!(::AbstractEventSink) = nothing

function flush_event_sink!(sink::StreamEventSink)
    flush(sink.io)
    return nothing
end

function flush_event_sink!(sink::JsonlEventSink)
    flush(sink.io)
    return nothing
end

function flush_event_sink!(sink::TeeEventSink)
    first_error = nothing
    for child in sink.sinks
        try
            flush_event_sink!(child)
        catch err
            first_error === nothing && (first_error = err)
        end
    end
    first_error === nothing || throw(first_error)
    return nothing
end

function close_event_sink!(sink::StreamEventSink)
    sink.close_on_finish ? close(sink.io) : flush(sink.io)
    return nothing
end

function close_event_sink!(sink::JsonlEventSink)
    sink.close_on_finish ? close(sink.io) : flush(sink.io)
    return nothing
end

function close_event_sink!(sink::TeeEventSink)
    first_error = nothing
    for child in sink.sinks
        try
            close_event_sink!(child)
        catch err
            first_error === nothing && (first_error = err)
        end
    end
    first_error === nothing || throw(first_error)
    return nothing
end
