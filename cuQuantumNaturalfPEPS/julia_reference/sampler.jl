struct SamplerConfig
    lx::Int
    ly::Int
    dim_phys::Int
    dim_bond::Int
    chi_dl::Int
    chi_s::Int
    sampling_mode::Symbol
    chi_c::Int
    seed::UInt64
end

function SamplerConfig(
    peps::AbstractMatrix,
    double_layer::DoubleLayerStack;
    chi_s::Integer=double_layer.chi_s,
    sampling_mode::Symbol=:fast,
    chi_c::Integer=3 * double_layer.dim_bond,
    seed::Integer=0,
)
    dimensions = _validate_peps_grid(peps)
    dimensions.lx == double_layer.lx && dimensions.ly == double_layer.ly ||
        throw(DimensionMismatch("PEPS and double-layer grids differ"))
    dimensions.dim_phys == double_layer.dim_phys &&
        dimensions.dim_bond == double_layer.dim_bond ||
        throw(DimensionMismatch("PEPS and double-layer local dimensions differ"))
    chi_s >= 1 || throw(ArgumentError("chi_s must be positive"))
    chi_c >= 1 || throw(ArgumentError("chi_c must be positive"))
    sampling_mode in (:fast, :full) ||
        throw(ArgumentError("sampling_mode must be :fast or :full"))
    seed >= 0 || throw(ArgumentError("seed must be nonnegative"))
    return SamplerConfig(
        dimensions.lx,
        dimensions.ly,
        dimensions.dim_phys,
        dimensions.dim_bond,
        double_layer.chi_dl,
        Int(chi_s),
        sampling_mode,
        Int(chi_c),
        UInt64(seed),
    )
end

mutable struct SamplerContext{T}
    peps::AbstractMatrix
    double_layer::DoubleLayerStack{T}
    config::SamplerConfig
    factorizer::RangefinderFactorizer
end

function SamplerContext(
    peps::AbstractMatrix,
    double_layer::DoubleLayerStack{T};
    factorizer::RangefinderFactorizer=RangefinderFactorizer(),
    kwargs...,
) where {T}
    return SamplerContext{T}(peps, double_layer, SamplerConfig(peps, double_layer; kwargs...), factorizer)
end

function ctx_sample_refresh!(context::SamplerContext, peps::AbstractMatrix)
    dimensions = _validate_peps_grid(peps)
    config = context.config
    (dimensions.lx, dimensions.ly, dimensions.dim_phys, dimensions.dim_bond) ==
    (config.lx, config.ly, config.dim_phys, config.dim_bond) ||
        throw(DimensionMismatch("refreshed PEPS dimensions differ from the sampler context"))
    eltype(peps[1, 1]) == eltype(context.double_layer.environments[1][1]) ||
        throw(ArgumentError("refreshed PEPS element type differs from the double layer"))
    context.peps = peps
    return context
end

function _top_ket_row(peps::AbstractMatrix)
    return [Array(@view _site_internal(peps[1, col])[:, 1, :, :, :]) for col in axes(peps, 2)]
end

function _build_ket_row(
    context::SamplerContext,
    row::Int,
    environment_above::AbstractVector,
)
    config = context.config
    element_type = eltype(context.peps[1, 1])
    carried = ones(element_type, 1, 1, 1)
    ket_row = Vector{Array{element_type,4}}(undef, config.ly)

    for col in 1:config.ly
        site = _site_internal(context.peps[row, col])
        above = environment_above[col]
        left_above = contract_arrays(carried, (2,), above, (1,))
        contracted = contract_arrays(left_above, (2, 3), site, (1, 2))
        grouped = permutedims(contracted, (1, 3, 4, 2, 5))
        ket_left, physical, below, above_right, site_right = size(grouped)
        matrix = reshape(grouped, ket_left * physical * below, above_right * site_right)
        rank = min(config.chi_s, size(matrix, 1), size(matrix, 2))
        basis, factor = factorize_matrix(context.factorizer, matrix, rank)
        normalize_log!(factor)
        ket_row[col] = reshape(basis, ket_left, physical, below, rank)
        carried = reshape(factor, rank, above_right, site_right)
    end
    size(carried) == (1, 1, 1) ||
        throw(DimensionMismatch("ket-row reduction ended with a nontrivial right boundary"))
    return ket_row
end

_dlenv_environment(site::AbstractArray) = permutedims(site, (2, 4, 3, 1))
_dlenv_sigma(site::AbstractArray) = permutedims(site, (2, 1, 3, 4))

function _build_env_unsampled(ket_row::AbstractVector, dlenv_row)
    num_cols = length(ket_row)
    element_type = eltype(ket_row[1])
    right_environments = Vector{Array{element_type,3}}(undef, num_cols + 1)
    right_environments[end] = ones(element_type, 1, 1, 1)
    unit_double_layer = ones(element_type, 1, 1, 1, 1)

    for col in num_cols:-1:2
        ket = ket_row[col]
        environment = dlenv_row === nothing ? unit_double_layer : _dlenv_environment(dlenv_row[col])
        ket_right = contract_arrays(ket, (4,), right_environments[col+1], (1,))
        with_below = contract_arrays(ket_right, (3, 4), environment, (1, 2))
        right = contract_arrays(with_below, (2, 4, 3), ket, (2, 3, 4); conj_b=true)
        normalize_log!(right)
        right_environments[col] = Array(right)
    end
    return right_environments
end

function _probabilities_from_rho(rho::AbstractMatrix)
    weights = Float64[abs(rho[spin, spin]) for spin in axes(rho, 1)]
    total = sum(weights)
    if !(isfinite(total) && total > 0.0) || any(weight -> !isfinite(weight), weights)
        fill!(weights, inv(length(weights)))
    else
        weights ./= total
    end
    return weights
end

function _draw_categorical(rng::AbstractRNG, probabilities::AbstractVector{<:Real})
    target = rand(rng)
    cumulative = 0.0
    for spin in eachindex(probabilities)
        cumulative += probabilities[spin]
        target <= cumulative && return spin
    end
    return lastindex(probabilities)
end

function _draw_sigma(
    ket_row::AbstractVector,
    dlenv_row,
    right_environments::AbstractVector,
    rng::AbstractRNG;
    forced_row=nothing,
)
    num_cols = length(ket_row)
    dim_phys = size(ket_row[1], 2)
    element_type = eltype(ket_row[1])
    sigma = ones(element_type, 1, 1, 1)
    unit_double_layer = ones(element_type, 1, 1, 1, 1)
    spins = Vector{UInt8}(undef, num_cols)
    log_probability = 0.0

    for col in 1:num_cols
        ket = ket_row[col]
        environment = dlenv_row === nothing ? unit_double_layer : _dlenv_sigma(dlenv_row[col])
        sigma_ket = contract_arrays(sigma, (1,), ket, (1,))
        sigma_below = contract_arrays(sigma_ket, (4, 1), environment, (1, 2))
        sigma_full = contract_arrays(sigma_below, (1, 4), ket, (1, 3); conj_b=true)
        rho_raw = contract_arrays(
            sigma_full,
            (2, 3, 5),
            right_environments[col+1],
            (1, 2, 3),
        )
        rho = reshape(rho_raw, dim_phys, dim_phys)
        probabilities = _probabilities_from_rho(rho)
        chosen = if forced_row === nothing
            _draw_categorical(rng, probabilities)
        else
            spin = Int(forced_row[col]) + 1
            1 <= spin <= dim_phys || throw(BoundsError(probabilities, spin))
            spin
        end
        spins[col] = UInt8(chosen - 1)
        log_probability += log(probabilities[chosen])
        divisor = abs(rho[chosen, chosen])
        !(isfinite(divisor) && divisor > 0.0) && (divisor = 1.0)
        sigma = Array(@view sigma_full[chosen, :, :, chosen, :]) ./ divisor
    end
    return spins, log_probability
end

function _slice_sampled_ket(ket_row::AbstractVector, spins::AbstractVector)
    element_type = eltype(ket_row[1])
    environment = Vector{Array{element_type,3}}(undef, length(ket_row))
    log_gauge = 0.0
    for col in eachindex(ket_row, spins)
        site = Array(@view ket_row[col][:, Int(spins[col])+1, :, :])
        log_gauge += normalize_log!(site)
        environment[col] = site
    end
    return environment, log_gauge
end

function _build_full_env_above(
    context::SamplerContext,
    row::Int,
    old_environment::AbstractVector,
    spins::AbstractVector,
)
    config = context.config
    element_type = eltype(context.peps[1, 1])
    carried = ones(element_type, 1, 1, 1)
    environment = Vector{Array{element_type,3}}(undef, config.ly)
    log_gauge = 0.0

    for col in 1:config.ly
        projected = _site_projected_internal(context.peps[row, col], spins[col])
        left_above = contract_arrays(carried, (2,), old_environment[col], (1,))
        contracted = contract_arrays(left_above, (2, 3), projected, (1, 2))
        grouped = permutedims(contracted, (1, 3, 2, 4))
        full_left, below, above_right, site_right = size(grouped)
        matrix = reshape(grouped, full_left * below, above_right * site_right)
        rank = min(config.chi_c, size(matrix, 1), size(matrix, 2))
        basis, factor = factorize_matrix(context.factorizer, matrix, rank)
        log_gauge += normalize_log!(factor)
        environment[col] = reshape(basis, full_left, below, rank)
        carried = reshape(factor, rank, above_right, site_right)
    end
    size(carried) == (1, 1, 1) ||
        throw(DimensionMismatch("full sampled boundary ended with a nontrivial right boundary"))
    return environment, log_gauge
end

function _sample_one(
    context::SamplerContext,
    rng::AbstractRNG;
    forced_config=nothing,
)
    config = context.config
    if forced_config !== nothing
        size(forced_config) == (config.lx, config.ly) ||
            throw(DimensionMismatch("forced configuration has the wrong grid shape"))
    end
    sampled = Matrix{UInt8}(undef, config.lx, config.ly)
    log_probability = 0.0
    log_gauge = 0.0
    environment_above = [ones(eltype(context.peps[1, 1]), 1, 1, 1) for _ in 1:config.ly]

    for row in 1:config.lx
        ket_row = row == 1 ? _top_ket_row(context.peps) :
                  _build_ket_row(context, row, environment_above)
        dlenv_row = row < config.lx ? context.double_layer.environments[row] : nothing
        right_environments = _build_env_unsampled(ket_row, dlenv_row)
        forced_row = forced_config === nothing ? nothing : @view forced_config[row, :]
        row_spins, row_log_probability = _draw_sigma(
            ket_row,
            dlenv_row,
            right_environments,
            rng;
            forced_row,
        )
        sampled[row, :] .= row_spins
        log_probability += row_log_probability

        if row < config.lx
            if config.sampling_mode === :fast || row == 1
                environment_above, row_log_gauge = _slice_sampled_ket(ket_row, row_spins)
            else
                environment_above, row_log_gauge = _build_full_env_above(
                    context,
                    row,
                    environment_above,
                    row_spins,
                )
            end
            log_gauge += row_log_gauge
        end
    end
    return sampled, log_probability, log_gauge
end

function ctx_sample_run(context::SamplerContext, num_samples::Integer; batch_base::Integer=0)
    num_samples >= 1 || throw(ArgumentError("num_samples must be positive"))
    batch_base >= 0 || throw(ArgumentError("batch_base must be nonnegative"))
    configs = Vector{Matrix{UInt8}}(undef, num_samples)
    log_prob_config = Vector{Float64}(undef, num_samples)
    log_gauge = Vector{Float64}(undef, num_samples)
    seed_multiplier = UInt64(1_000_003)

    for sample in 1:num_samples
        sample_seed = context.config.seed * seed_multiplier + UInt64(batch_base + sample - 1)
        rng = MersenneTwister(sample_seed)
        configs[sample], log_prob_config[sample], log_gauge[sample] = _sample_one(context, rng)
    end
    return (; configs, log_prob_config, log_gauge)
end

function proposal_log_probability(context::SamplerContext, configuration::AbstractMatrix)
    rng = MersenneTwister(context.config.seed)
    _, log_probability, log_gauge = _sample_one(context, rng; forced_config=configuration)
    return (; log_probability, log_gauge)
end

function sample_peps(
    peps::AbstractMatrix,
    double_layer::DoubleLayerStack,
    num_samples::Integer;
    factorizer::RangefinderFactorizer=RangefinderFactorizer(),
    batch_base::Integer=0,
    kwargs...,
)
    context = SamplerContext(peps, double_layer; factorizer, kwargs...)
    return ctx_sample_run(context, num_samples; batch_base)
end
