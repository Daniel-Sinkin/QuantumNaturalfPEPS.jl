abstract type AbstractRunAdapter end
abstract type AbstractEventSink end

struct RunNotice
    code::String
    message::String
    payload::Dict{String,Any}
end

RunNotice(code::AbstractString, message::AbstractString) =
    RunNotice(String(code), String(message), Dict{String,Any}())

Base.@kwdef struct IterationResult
    samples_requested::Int = 0
    samples_used::Int = 0
    ess_estimated::Union{Nothing,Float64} = nothing
    ess_measured::Union{Nothing,Float64} = nothing
    energy_before::Union{Nothing,Float64} = nothing
    energy_after::Union{Nothing,Float64} = nothing
    energy_variance::Union{Nothing,Float64} = nothing
    direction_norm::Union{Nothing,Float64} = nothing
    step_norm::Union{Nothing,Float64} = nothing
    phase_seconds::Dict{String,Float64} = Dict{String,Float64}()
    fallbacks::Vector{String} = String[]
    metrics::Dict{String,Any} = Dict{String,Any}()
    debug::Dict{String,Any} = Dict{String,Any}()
    warnings::Vector{RunNotice} = RunNotice[]
end

struct RunEvent
    schema_version::Int
    run_id::String
    sequence::Int
    timestamp_unix_ns::Int64
    kind::Symbol
    name::Symbol
    iteration::Union{Nothing,Int}
    payload::Dict{String,Any}
end

Base.@kwdef struct RunnerConfig
    iterations::Int
    run_id::String = "run-$(round(Int64, time() * 1.0e9))-$(getpid())"
    checkpoint_every::Int = 0
    expensive_debug::Bool = false
    rethrow_errors::Bool = false
end

struct RunnerResult
    status::Symbol
    run_id::String
    first_iteration::Int
    last_iteration::Int
    iterations_completed::Int
end

struct RunnerControl
    commands::Channel{Symbol}
end

function RunnerControl(capacity::Integer=32)
    capacity > 0 || throw(ArgumentError("runner control capacity must be positive"))
    return RunnerControl(Channel{Symbol}(Int(capacity)))
end
