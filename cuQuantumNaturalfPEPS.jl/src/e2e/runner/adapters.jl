
initialize_adapter!(::AbstractRunAdapter) = Dict{String,Any}()

function step_adapter!(adapter::AbstractRunAdapter, iteration::Integer; expensive_debug::Bool=false)
    throw(MethodError(step_adapter!, (adapter, iteration, expensive_debug)))
end

function checkpoint_adapter!(adapter::AbstractRunAdapter, iteration::Integer)
    throw(ArgumentError("adapter $(typeof(adapter)) does not implement checkpoints"))
end

close_adapter!(::AbstractRunAdapter) = Dict{String,Any}()
adapter_iteration(::AbstractRunAdapter) = 0
adapter_status(adapter::AbstractRunAdapter) =
    Dict{String,Any}("adapter_type" => string(typeof(adapter)))
