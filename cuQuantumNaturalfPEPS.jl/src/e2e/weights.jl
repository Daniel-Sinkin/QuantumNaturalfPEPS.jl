function importance_weights(logpsi::AbstractVector{<:Complex}, logq::AbstractVector{<:Real})
    length(logpsi) == length(logq) ||
        throw(DimensionMismatch("logpsi has $(length(logpsi)) samples, logq has $(length(logq))"))
    return _importance_weights(Float64.(real.(Array(logpsi))), Float64.(Array(logq)))
end

function importance_weights(logpsi::AbstractVector{<:Real}, logq::AbstractVector{<:Real})
    sample_count = length(logq)
    length(logpsi) == 2 * sample_count || throw(
        DimensionMismatch(
            "interleaved logpsi has $(length(logpsi)) values, expected $(2 * sample_count) " *
            "values for $(sample_count) samples",
        ),
    )
    paired = Float64.(Array(logpsi))
    return _importance_weights(paired[1:2:end], Float64.(Array(logq)))
end

function _importance_weights(log_amplitude::Vector{Float64}, log_proposal::Vector{Float64})
    sample_count = length(log_proposal)
    sample_count >= 2 ||
        throw(ArgumentError("importance weights need at least 2 samples (got $sample_count)"))
    log_ratios = 2 .* log_amplitude .- log_proposal
    any(isnan, log_ratios) && throw(ArgumentError("log importance ratios contain NaN"))
    largest_log_ratio = maximum(log_ratios)
    isfinite(largest_log_ratio) ||
        throw(ArgumentError("log importance ratios need a finite maximum (got $largest_log_ratio)"))
    shifted_sum = sum(exp.(log_ratios .- largest_log_ratio))
    log_mean_weight = largest_log_ratio + log(shifted_sum) - log(sample_count)
    unnormalized = exp.(log_ratios .- log_mean_weight)
    weights = unnormalized ./ (sum(unnormalized) / sample_count)
    weight_sum = sum(weights)
    largest = argmax(weights)
    return (
        weights=weights,
        ess=weight_sum^2 / sum(weights .^ 2),
        largest_share=weights[largest] / weight_sum,
        ess_without_largest=_effective_sample_size(deleteat!(copy(log_ratios), largest)),
    )
end

function _effective_sample_size(log_ratios::Vector{Float64})
    largest_log_ratio = maximum(log_ratios)
    isfinite(largest_log_ratio) || return 0.0
    shifted = exp.(log_ratios .- largest_log_ratio)
    return sum(shifted)^2 / sum(shifted .^ 2)
end
