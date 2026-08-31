struct SyntheticAdapterFailure <: Exception
    iteration::Int
end

function Base.showerror(io::IO, err::SyntheticAdapterFailure)
    print(io, "synthetic failure at iteration ", err.iteration)
end

Base.@kwdef mutable struct SyntheticAdapter <: AbstractRunAdapter
    current_iteration::Int = 0
    energy::Float64 = -20.0
    target_energy::Float64 = -36.7568282608
    relaxation::Float64 = 0.08
    ripple::Float64 = 0.03
    samples::Int = 1024
    ess_fraction::Float64 = 0.20
    ess_prediction_bias::Float64 = 0.08
    checkpoint_path::Union{Nothing,String} = nothing
    resume::Bool = false
    warning_iterations::Set{Int} = Set{Int}()
    failure_iteration::Union{Nothing,Int} = nothing
    sleep_seconds::Float64 = 0.0
    initialized::Bool = false
    checkpoint_count::Int = 0
end

function _validate_synthetic(adapter::SyntheticAdapter)
    adapter.samples > 0 || throw(ArgumentError("samples must be positive"))
    0.0 < adapter.ess_fraction <= 1.0 ||
        throw(ArgumentError("ess_fraction must be in the interval (0, 1]"))
    0.0 < adapter.relaxation <= 1.0 ||
        throw(ArgumentError("relaxation must be in the interval (0, 1]"))
    adapter.sleep_seconds >= 0.0 || throw(ArgumentError("sleep_seconds must be nonnegative"))
    return nothing
end

function _restore_synthetic!(adapter::SyntheticAdapter)
    path = adapter.checkpoint_path
    path === nothing && throw(ArgumentError("resume requires checkpoint_path"))
    isfile(path) || throw(ArgumentError("checkpoint does not exist at $(path)"))
    record = JSON3.read(read(path, String))
    hasproperty(record, :format) && record.format == "QNfPEPS-checkpoint" ||
        throw(ArgumentError("invalid synthetic checkpoint format"))
    Int(record.schema_version) == 1 ||
        throw(ArgumentError("unsupported synthetic checkpoint schema"))
    hasproperty(record, :adapter) && record.adapter == "synthetic" ||
        throw(ArgumentError("synthetic checkpoint adapter mismatch"))
    adapter.current_iteration = Int(record.iteration)
    adapter.energy = Float64(record.energy)
    adapter.checkpoint_count = Int(record.checkpoint_count)
    return nothing
end

function initialize_adapter!(adapter::SyntheticAdapter)
    _validate_synthetic(adapter)
    adapter.initialized && throw(ArgumentError("synthetic adapter is already initialized"))
    adapter.resume && _restore_synthetic!(adapter)
    adapter.initialized = true
    return Dict{String,Any}(
        "adapter" => "synthetic",
        "resumed" => adapter.resume,
        "restored_iteration" => adapter.current_iteration,
        "target_energy" => adapter.target_energy,
        "samples" => adapter.samples,
    )
end

function step_adapter!(adapter::SyntheticAdapter, iteration::Integer; expensive_debug::Bool=false)
    adapter.initialized || throw(ArgumentError("synthetic adapter is not initialized"))
    expected = adapter.current_iteration + 1
    Int(iteration) == expected ||
        throw(ArgumentError("expected iteration $(expected), got $(iteration)"))
    adapter.failure_iteration == iteration && throw(SyntheticAdapterFailure(Int(iteration)))
    adapter.sleep_seconds == 0.0 || sleep(adapter.sleep_seconds)

    before = adapter.energy
    damped_ripple = adapter.ripple * exp(-0.04 * iteration) * sin(0.71 * iteration)
    after = before + adapter.relaxation * (adapter.target_energy - before) + damped_ripple
    adapter.energy = after
    adapter.current_iteration = Int(iteration)

    measured_fraction = clamp(adapter.ess_fraction + 0.02 * sin(0.53 * iteration), 0.01, 1.0)
    estimated_fraction = clamp(
        measured_fraction * (1.0 + adapter.ess_prediction_bias * cos(0.31 * iteration)),
        0.01,
        1.0,
    )
    measured_ess = adapter.samples * measured_fraction
    estimated_ess = adapter.samples * estimated_fraction
    delta = after - before
    fallbacks =
        iteration in adapter.warning_iterations ? ["synthetic_regularization_fallback"] : String[]
    warnings =
        isempty(fallbacks) ? RunNotice[] :
        [
            RunNotice(
                "synthetic_fallback",
                "synthetic regularization fallback was exercised",
                Dict{String,Any}("fallback" => first(fallbacks)),
            ),
        ]
    debug =
        expensive_debug ?
        Dict{String,Any}(
            "condition_estimate" => 1.0 + iteration^2,
            "orthogonality_residual" => eps(Float64) * (1.0 + iteration),
            "energy_distance_to_target" => abs(after - adapter.target_energy),
        ) : Dict{String,Any}()

    return IterationResult(
        samples_requested=adapter.samples,
        samples_used=adapter.samples,
        ess_estimated=estimated_ess,
        ess_measured=measured_ess,
        energy_before=before,
        energy_after=after,
        energy_variance=max(0.0, 0.6 * exp(-0.05 * iteration)),
        direction_norm=abs(delta) / max(adapter.relaxation, eps(Float64)),
        step_norm=abs(delta),
        phase_seconds=Dict(
            "double_layer" => 0.0015,
            "sampling" => 0.0025,
            "e_o" => 0.012,
            "minsr" => 0.003,
            "update" => 0.0005,
        ),
        fallbacks=fallbacks,
        metrics=Dict{String,Any}("execution_mode" => "synthetic", "sample_policy" => "fixed"),
        debug=debug,
        warnings=warnings,
    )
end

function checkpoint_adapter!(adapter::SyntheticAdapter, iteration::Integer)
    adapter.initialized || throw(ArgumentError("synthetic adapter is not initialized"))
    Int(iteration) == adapter.current_iteration ||
        throw(ArgumentError("checkpoint iteration does not match adapter state"))
    adapter.checkpoint_count += 1
    path = adapter.checkpoint_path
    if path === nothing
        return Dict{String,Any}(
            "adapter" => "synthetic",
            "durable" => false,
            "checkpoint_count" => adapter.checkpoint_count,
        )
    end

    absolute_path = abspath(path)
    mkpath(dirname(absolute_path))
    temporary_path = string(absolute_path, ".tmp.", getpid())
    record = (
        format="QNfPEPS-checkpoint",
        schema_version=1,
        adapter="synthetic",
        iteration=adapter.current_iteration,
        energy=adapter.energy,
        checkpoint_count=adapter.checkpoint_count,
    )
    open(temporary_path, "w") do io
        JSON3.write(io, record)
        write(io, '\n')
    end
    mv(temporary_path, absolute_path; force=true)
    return Dict{String,Any}(
        "adapter" => "synthetic",
        "durable" => true,
        "path" => absolute_path,
        "checkpoint_count" => adapter.checkpoint_count,
    )
end

function close_adapter!(adapter::SyntheticAdapter)
    was_initialized = adapter.initialized
    adapter.initialized = false
    return Dict{String,Any}("adapter" => "synthetic", "was_initialized" => was_initialized)
end

adapter_iteration(adapter::SyntheticAdapter) = adapter.current_iteration

function adapter_status(adapter::SyntheticAdapter)
    return Dict{String,Any}(
        "adapter" => "synthetic",
        "initialized" => adapter.initialized,
        "iteration" => adapter.current_iteration,
        "energy" => adapter.energy,
        "checkpoint_count" => adapter.checkpoint_count,
    )
end
