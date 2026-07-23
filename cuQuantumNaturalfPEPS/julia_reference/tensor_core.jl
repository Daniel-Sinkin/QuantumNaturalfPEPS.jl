const PEPS_PHYSICAL = 1
const PEPS_UP = 2
const PEPS_RIGHT = 3
const PEPS_DOWN = 4
const PEPS_LEFT = 5

function _checked_axes(array::AbstractArray, axes, operand::Symbol)
    ordered = Tuple(Int(axis) for axis in axes)
    length(Set(ordered)) == length(ordered) ||
        throw(ArgumentError("contracted axes for $operand contain a duplicate: $ordered"))
    all(axis -> 1 <= axis <= ndims(array), ordered) || throw(
        ArgumentError("contracted axes for $operand do not fit rank $(ndims(array)): $ordered"),
    )
    return ordered
end

_axis_extents(array::AbstractArray, axes::Tuple) = Tuple(size(array, axis) for axis in axes)
_axis_product(array::AbstractArray, axes::Tuple) =
    isempty(axes) ? 1 : prod(size(array, axis) for axis in axes)

function contract_plan(
    tensor_a::AbstractArray,
    axes_a,
    tensor_b::AbstractArray,
    axes_b,
)
    contracted_a = _checked_axes(tensor_a, axes_a, :tensor_a)
    contracted_b = _checked_axes(tensor_b, axes_b, :tensor_b)
    length(contracted_a) == length(contracted_b) || throw(
        DimensionMismatch(
            "contractions need the same number of axes, got " *
            "$(length(contracted_a)) and $(length(contracted_b))",
        ),
    )
    extents_a = _axis_extents(tensor_a, contracted_a)
    extents_b = _axis_extents(tensor_b, contracted_b)
    extents_a == extents_b ||
        throw(DimensionMismatch("ordered contracted extents differ: $extents_a != $extents_b"))

    free_a = Tuple(axis for axis in 1:ndims(tensor_a) if !(axis in contracted_a))
    free_b = Tuple(axis for axis in 1:ndims(tensor_b) if !(axis in contracted_b))
    permutation_a = (free_a..., contracted_a...)
    permutation_b = (contracted_b..., free_b...)
    result_shape = (_axis_extents(tensor_a, free_a)..., _axis_extents(tensor_b, free_b)...)
    isempty(result_shape) && (result_shape = (1,))
    return (;
        contracted_a,
        contracted_b,
        free_a,
        free_b,
        permutation_a,
        permutation_b,
        rows=_axis_product(tensor_a, free_a),
        inner=_axis_product(tensor_a, contracted_a),
        columns=_axis_product(tensor_b, free_b),
        result_shape,
    )
end

function contract_arrays(
    tensor_a::AbstractArray,
    axes_a,
    tensor_b::AbstractArray,
    axes_b;
    conj_a::Bool=false,
    conj_b::Bool=false,
)
    plan = contract_plan(tensor_a, axes_a, tensor_b, axes_b)
    input_a = conj_a ? conj.(tensor_a) : tensor_a
    input_b = conj_b ? conj.(tensor_b) : tensor_b
    matrix_a = reshape(permutedims(input_a, plan.permutation_a), plan.rows, plan.inner)
    matrix_b = reshape(permutedims(input_b, plan.permutation_b), plan.inner, plan.columns)
    return reshape(matrix_a * matrix_b, plan.result_shape)
end

function _component_l1_scale(values::AbstractArray)::Float64
    scale = 0.0
    for value in values
        scale = max(scale, abs(Float64(real(value))) + abs(Float64(imag(value))))
    end
    return scale
end

function normalize_log!(values::AbstractArray)::Float64
    scale = _component_l1_scale(values)
    isfinite(scale) || throw(ErrorException("non-finite normalization scale"))
    if scale > 0.0
        real_type = typeof(real(zero(eltype(values))))
        values .*= convert(real_type, inv(scale))
        return log(scale)
    end
    return 0.0
end

Base.@kwdef struct RangefinderConfig
    seed::UInt64 = 777
    power_steps::Int = 1
    chol_shift::Float64 = 1.0e-5
    recovery::Symbol = :householder
    force_fallback::Bool = false
    effective_rank_rtol::Float64 = sqrt(eps(Float64))
end

struct RangefinderDiagnostic
    requested_rank::Int
    capped_rank::Int
    effective_rank::Int
    fallback_used::Bool
    status::Symbol
    message::String
end

struct RangefinderResult{T}
    q::Union{Nothing,Matrix{T}}
    r::Union{Nothing,Matrix{T}}
    diagnostic::RangefinderDiagnostic
end

struct FixedRankRangefinderError <: Exception
    diagnostic::RangefinderDiagnostic
end

function Base.showerror(io::IO, error::FixedRankRangefinderError)
    diagnostic = error.diagnostic
    print(
        io,
        diagnostic.message,
        " requested_rank=",
        diagnostic.requested_rank,
        " capped_rank=",
        diagnostic.capped_rank,
        " effective_rank=",
        diagnostic.effective_rank,
    )
end

mutable struct RangefinderFactorizer
    config::RangefinderConfig
    omegas::Dict{Tuple{DataType,Int,Int},Any}
    diagnostics::Vector{RangefinderDiagnostic}
end

function RangefinderFactorizer(; kwargs...)
    config = RangefinderConfig(; kwargs...)
    config.power_steps >= 0 || throw(ArgumentError("power_steps must be nonnegative"))
    config.chol_shift >= 0 || throw(ArgumentError("chol_shift must be nonnegative"))
    config.recovery in (:none, :householder) ||
        throw(ArgumentError("recovery must be :none or :householder"))
    config.effective_rank_rtol >= 0 ||
        throw(ArgumentError("effective_rank_rtol must be nonnegative"))
    return RangefinderFactorizer(config, Dict{Tuple{DataType,Int,Int},Any}(), RangefinderDiagnostic[])
end

function _complex_normal(rng::AbstractRNG, ::Type{T}, rows::Int, cols::Int) where {T<:Number}
    if T <: Complex
        real_type = typeof(real(zero(T)))
        return complex.(
            randn(rng, real_type, rows, cols),
            randn(rng, real_type, rows, cols),
        )
    end
    return randn(rng, T, rows, cols)
end

function _omega!(factorizer::RangefinderFactorizer, ::Type{T}, columns::Int, rank::Int) where {T}
    key = (T, columns, rank)
    return get!(factorizer.omegas, key) do
        width_seed = factorizer.config.seed ⊻ (UInt64(columns) << 20)
        _complex_normal(MersenneTwister(width_seed), T, columns, rank)
    end
end

function _effective_rank(matrix::AbstractMatrix, rtol::Real)::Int
    singular_values = svdvals(Matrix{ComplexF64}(matrix))
    isempty(singular_values) && return 0
    threshold = Float64(rtol) * singular_values[1]
    return count(value -> value > threshold, singular_values)
end

function _shifted_cholesky_qr2(sketch::AbstractMatrix, config::RangefinderConfig)
    q = Matrix(sketch)
    rank = size(q, 2)
    real_type = typeof(real(zero(eltype(q))))
    for _ in 1:2
        gram = q' * q
        trace_scale = real(sum(diag(gram))) / max(rank, 1)
        shift = convert(
            real_type,
            config.chol_shift * (trace_scale > 0 ? trace_scale : 1.0) + 1.0e-30,
        )
        shifted = Hermitian(gram + shift * I)
        factor = cholesky(shifted; check=true)
        q = q / factor.U
    end
    all(isfinite, q) || throw(DomainError(q, "non-finite CholeskyQR basis"))
    return q
end

function batched_rangefinder(
    matrix::AbstractMatrix{T},
    requested_rank::Integer;
    factorizer::RangefinderFactorizer=RangefinderFactorizer(),
    force_fallback::Bool=factorizer.config.force_fallback,
) where {T<:Number}
    rows, columns = size(matrix)
    requested_rank >= 1 || throw(ArgumentError("requested_rank must be positive"))
    capped_rank = min(Int(requested_rank), rows, columns)
    capped_rank >= 1 || throw(ArgumentError("rangefinder matrix must be nonempty"))

    # Proposal sketch to mix rows
    omega = _omega!(factorizer, T, columns, capped_rank)
    sketch = matrix * omega
    for _ in 1:factorizer.config.power_steps # 1 or 2
        sketch = matrix * (matrix' * sketch)
    end
    effective_rank = _effective_rank(sketch, factorizer.config.effective_rank_rtol)

    q = nothing
    fallback_used = force_fallback
    chol_failed = force_fallback
    if !force_fallback
        try
            q = _shifted_cholesky_qr2(sketch, factorizer.config)
        catch error
            if error isa PosDefException ||
               error isa LinearAlgebra.LAPACKException ||
               error isa DomainError
                chol_failed = true
            else
                rethrow()
            end
        end
    end

    if chol_failed
        if factorizer.config.recovery !== :householder
            diagnostic = RangefinderDiagnostic(
                Int(requested_rank),
                capped_rank,
                effective_rank,
                false,
                :cholesky_failed,
                "shifted CholeskyQR failed and recovery is disabled",
            )
            push!(factorizer.diagnostics, diagnostic)
            return RangefinderResult{T}(nothing, nothing, diagnostic)
        end
        fallback_used = true
        if effective_rank >= capped_rank
            q = Matrix(qr(sketch).Q)[:, 1:capped_rank]
        end
    end

    if effective_rank < capped_rank
        diagnostic = RangefinderDiagnostic(
            Int(requested_rank),
            capped_rank,
            effective_rank,
            fallback_used,
            :unsupported_fixed_rank,
            "the rangefinder sketch does not support the requested fixed rank",
        )
        push!(factorizer.diagnostics, diagnostic)
        return RangefinderResult{T}(nothing, nothing, diagnostic)
    end

    q_matrix = Matrix{T}(q)
    r_matrix = q_matrix' * matrix
    status = fallback_used ? :householder_recovered : :ok
    diagnostic = RangefinderDiagnostic(
        Int(requested_rank),
        capped_rank,
        effective_rank,
        fallback_used,
        status,
        fallback_used ?
        "reference-only Householder replay preserved the capped fixed rank" :
        "shifted two-pass CholeskyQR completed",
    )
    push!(factorizer.diagnostics, diagnostic)
    return RangefinderResult(q_matrix, Matrix{T}(r_matrix), diagnostic)
end

function factorize_matrix(
    factorizer::RangefinderFactorizer,
    matrix::AbstractMatrix,
    requested_rank::Integer,
)
    result = batched_rangefinder(matrix, requested_rank; factorizer)
    result.q === nothing && throw(FixedRankRangefinderError(result.diagnostic))
    return result.q, result.r
end

function qr_factorize(matrix::AbstractMatrix, maxdim::Integer)
    rank = min(Int(maxdim), size(matrix, 1), size(matrix, 2))
    rank >= 1 || throw(ArgumentError("factorization rank must be positive"))
    q = Matrix(qr(matrix).Q)[:, 1:rank]
    return q, q' * matrix
end

function zipup_mpo_mps(
    mpo::AbstractVector,
    mps::AbstractVector;
    maxdim::Integer,
    factorizer::RangefinderFactorizer=RangefinderFactorizer(),
)
    length(mpo) == length(mps) ||
        throw(DimensionMismatch("MPO and MPS must contain the same number of sites"))
    isempty(mpo) && throw(ArgumentError("zip-up needs at least one site"))
    maxdim >= 1 || throw(ArgumentError("maxdim must be positive"))
    element_type = promote_type(eltype(mpo[1]), eltype(mps[1]))
    output = Vector{Array{element_type,3}}(undef, length(mpo))
    carried = ones(element_type, 1, 1, 1)
    log_gauge = 0.0

    for site in eachindex(mpo, mps)
        operator = mpo[site]
        state = mps[site]
        ndims(operator) == 4 ||
            throw(DimensionMismatch("MPO sites use [left,physical_in,physical_out,right]"))
        ndims(state) == 3 ||
            throw(DimensionMismatch("MPS sites use [left,physical,right]"))
        size(operator, 2) == size(state, 2) ||
            throw(DimensionMismatch("MPO physical input does not match MPS site $site"))

        left_state = contract_arrays(carried, (3,), state, (1,))
        contracted = contract_arrays(left_state, (2, 3), operator, (1, 2))
        grouped = permutedims(contracted, (1, 3, 2, 4))
        output_left, physical_out, state_right, mpo_right = size(grouped)
        matrix = reshape(grouped, output_left * physical_out, state_right * mpo_right)
        rank = min(Int(maxdim), size(matrix, 1), size(matrix, 2))
        q, r = factorize_matrix(factorizer, matrix, rank)
        log_gauge += normalize_log!(r)
        output[site] = reshape(q, output_left, physical_out, rank)
        carried = permutedims(reshape(r, rank, state_right, mpo_right), (1, 3, 2))
    end

    size(carried) == (1, 1, 1) ||
        throw(DimensionMismatch("zip-up ended with a nontrivial right boundary"))
    output[end] .*= carried[1]
    return output, log_gauge
end

function _validate_peps_grid(peps::AbstractMatrix)
    lx, ly = size(peps)
    lx >= 2 && ly >= 2 ||
        throw(DimensionMismatch("PEPS grid must be at least 2x2, got $(size(peps))"))
    first_site = peps[1, 1]
    first_site isa AbstractArray || throw(ArgumentError("PEPS sites must be arrays"))
    ndims(first_site) == 5 ||
        throw(DimensionMismatch("PEPS sites use five axes [p,u,r,d,l]"))
    element_type = eltype(first_site)
    physical_dim = size(first_site, PEPS_PHYSICAL)
    bond_dim = size(first_site, PEPS_RIGHT)

    for row in 1:lx, col in 1:ly
        site = peps[row, col]
        site isa AbstractArray || throw(ArgumentError("PEPS sites must be arrays"))
        ndims(site) == 5 ||
            throw(DimensionMismatch("site ($row,$col) does not use axes [p,u,r,d,l]"))
        eltype(site) == element_type ||
            throw(ArgumentError("all PEPS sites must have the same element type"))
        size(site, PEPS_PHYSICAL) == physical_dim ||
            throw(DimensionMismatch("physical dimension differs at site ($row,$col)"))
    end
    for col in 1:ly
        size(peps[1, col], PEPS_UP) == 1 ||
            throw(DimensionMismatch("top boundary up leg at (1,$col) must have extent 1"))
        size(peps[lx, col], PEPS_DOWN) == 1 ||
            throw(DimensionMismatch("bottom boundary down leg at ($lx,$col) must have extent 1"))
    end
    for row in 1:lx
        size(peps[row, 1], PEPS_LEFT) == 1 ||
            throw(DimensionMismatch("left boundary leg at ($row,1) must have extent 1"))
        size(peps[row, ly], PEPS_RIGHT) == 1 ||
            throw(DimensionMismatch("right boundary leg at ($row,$ly) must have extent 1"))
    end
    for row in 1:lx, col in 1:(ly-1)
        size(peps[row, col], PEPS_RIGHT) == bond_dim ||
            throw(DimensionMismatch("horizontal bond at ($row,$col) is not uniform"))
        size(peps[row, col+1], PEPS_LEFT) == bond_dim ||
            throw(DimensionMismatch("horizontal bond at ($row,$col) does not match its neighbor"))
    end
    for row in 1:(lx-1), col in 1:ly
        size(peps[row, col], PEPS_DOWN) == bond_dim ||
            throw(DimensionMismatch("vertical bond at ($row,$col) is not uniform"))
        size(peps[row+1, col], PEPS_UP) == bond_dim ||
            throw(DimensionMismatch("vertical bond at ($row,$col) does not match its neighbor"))
    end
    return (lx=lx, ly=ly, dim_phys=physical_dim, dim_bond=bond_dim, element_type=element_type)
end

function _site_internal(site::AbstractArray)
    return permutedims(site, (PEPS_LEFT, PEPS_UP, PEPS_PHYSICAL, PEPS_DOWN, PEPS_RIGHT))
end

function _site_projected_internal(site::AbstractArray, spin::Integer)
    0 <= spin < size(site, PEPS_PHYSICAL) || throw(BoundsError(site, spin + 1))
    internal = _site_internal(site)
    return Array(@view internal[:, :, spin+1, :, :])
end
