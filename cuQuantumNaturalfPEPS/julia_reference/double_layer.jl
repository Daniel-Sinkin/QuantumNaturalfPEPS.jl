struct DoubleLayerStack{T}
    environments::Vector{Vector{Array{T,4}}}
    cumulative_row_logs::Vector{Float64}
    lx::Int
    ly::Int
    dim_phys::Int
    dim_bond::Int
    chi_s::Int
    chi_dl::Int
    diagnostics::Vector{RangefinderDiagnostic}
end

function _validate_double_layer_row(row::AbstractVector, env_below, maxdim::Integer)
    maxdim >= 1 || throw(ArgumentError("maxdim must be positive"))
    isempty(row) && throw(ArgumentError("a PEPS row cannot be empty"))
    for (col, site) in pairs(row)
        ndims(site) == 5 ||
            throw(DimensionMismatch("row site $col does not use axes [p,u,r,d,l]"))
    end
    size(row[1], PEPS_LEFT) == 1 ||
        throw(DimensionMismatch("row left boundary must have extent 1"))
    size(row[end], PEPS_RIGHT) == 1 ||
        throw(DimensionMismatch("row right boundary must have extent 1"))
    for col in 1:(length(row)-1)
        size(row[col], PEPS_RIGHT) == size(row[col+1], PEPS_LEFT) ||
            throw(DimensionMismatch("horizontal bond $col does not match"))
    end
    if env_below === nothing
        all(site -> size(site, PEPS_DOWN) == 1, row) || throw(
            DimensionMismatch("a unit lower environment is valid only at the bottom boundary"),
        )
        return nothing
    end
    length(env_below) == length(row) ||
        throw(DimensionMismatch("the lower environment and PEPS row need equal lengths"))
    for col in eachindex(row, env_below)
        environment = env_below[col]
        ndims(environment) == 4 || throw(
            DimensionMismatch(
                "environment site $col does not use [bond_left,ket,bra,bond_right]",
            ),
        )
        size(environment, 2) == size(row[col], PEPS_DOWN) ||
            throw(DimensionMismatch("lower ket leg does not match row site $col"))
        size(environment, 3) == size(row[col], PEPS_DOWN) ||
            throw(DimensionMismatch("lower bra leg does not match row site $col"))
    end
    return nothing
end

# Corresponds to generate_double_layer_env_row
function apply_fused_double_layer_row(
    row::AbstractVector,
    env_below::Union{Nothing,AbstractVector};
    maxdim::Integer,
    factorizer::RangefinderFactorizer=RangefinderFactorizer(),
) # TODO: Add return type as a Named tuple
    _validate_double_layer_row(row, env_below, maxdim)

    element_type = eltype(row[1])
    num_cols = length(row)
    # TODO: Consider if we should remove this check b.c. that PEPS size is not allowed anyways
    bond_dim = num_cols == 1 ? size(row[1], PEPS_UP) : size(row[1], PEPS_RIGHT)
    effective_maxdim = min(Int(maxdim), bond_dim * bond_dim)

    environment_row = Vector{Array{element_type,4}}(undef, num_cols)
    carried_factor = ones(element_type, 1, 1, 1, 1)
    unit_environment = ones(element_type, 1, 1, 1, 1)
    row_log = 0.0

    # Correponds to MPO-MPS overload of ITensors (zipup instead of the default densitymatrix)
    # sampling.jl:6 and sampling.jl:16
    # For more details see 2.4 RF-ZipUp Algorithm in the document
    for col in 1:num_cols
        ket = row[col]
        environment_site = env_below === nothing ? unit_environment : env_below[col]

        # TODO: Add named variables to give some names to the indices, like below on the permutedims
        left_environment = contract_arrays(carried_factor, (4,), environment_site, (1,))
        left_environment_ket = contract_arrays(left_environment, (2, 4), ket, (5, 4))
        column_tensor = contract_arrays(
            left_environment_ket,
            (2, 3, 5),
            ket,
            (5, 4, 1);
            conj_b=true,
        )

        bond_left,
        bond_below_right,
        ket_vertical,
        ket_horizontal,
        bra_vertical,
        bra_horizontal = size(column_tensor)
        grouped = permutedims(column_tensor, (1, 3, 5, 2, 4, 6))
        matrix_rows, matrix_columns, matrix = let
            rows = bond_left * ket_vertical * bra_vertical
            columns = bond_below_right * ket_horizontal * bra_horizontal
            rows, columns, reshape(grouped, rows, columns)
        end

        rank = min(effective_maxdim, matrix_rows, matrix_columns)
        # Apply rangefinder
        basis, factor = factorize_matrix(factorizer, matrix, rank)
        row_log += normalize_log!(factor)

        # Carry the R factor to the right
        environment_row[col] = reshape(basis, bond_left, ket_vertical, bra_vertical, rank)
        carried_factor = permutedims(
            reshape(factor, rank, bond_below_right, ket_horizontal, bra_horizontal),
            (1, 3, 4, 2),
        )
    end

    folded = contract_arrays(environment_row[end], (4,), carried_factor, (1,))
    prior_shape = size(environment_row[end])
    environment_row[end] = reshape(
        folded,
        prior_shape[1],
        prior_shape[2],
        prior_shape[3],
        1,
    )
    return environment_row, row_log
end

# Corresponds to generate_double_layer_envs()
function build_double_layer_stack(
    peps::AbstractMatrix;
    chi_s::Integer=_validate_peps_grid(peps).dim_bond,
    chi_dl::Integer=_validate_peps_grid(peps).dim_bond,
    maxdim::Integer=chi_dl,
    factorizer::RangefinderFactorizer=RangefinderFactorizer(),
)
    dimensions = _validate_peps_grid(peps)
    chi_s >= 1 || throw(ArgumentError("chi_s must be positive"))
    chi_dl >= 1 || throw(ArgumentError("chi_dl must be positive"))
    if maxdim != chi_dl
        throw(ArgumentError("maxdim is a compatibility alias and must equal chi_dl"))
    end
    lx, ly = dimensions.lx, dimensions.ly
    element_type = dimensions.element_type

    # TODO: Add a type alias here so this is Vector{Environment}

    # Corresponds to double_layer_envs, in QNPEPS the row logs are fused into
    # the environment and accumulated in the constructor, corresponds to Environment type
    environments = Vector{Vector{Array{element_type,4}}}(undef, lx - 1)
    cumulative_row_logs = Vector{Float64}(undef, lx - 1)

    # In QNPEPS the base case is handled as a seperate overload so there are two functions
    # Corresponds to generate_double_layer_env_row()
    env_below = nothing
    cumulative_log = 0.0
    for peps_row in lx:-1:2
        env_row = peps_row - 1
        row = [peps[peps_row, col] for col in 1:ly]
        environment, row_log = apply_fused_double_layer_row(
            row,
            env_below;
            maxdim=chi_dl,
            factorizer,
        )
        environments[env_row] = environment
        cumulative_log += row_log
        cumulative_row_logs[env_row] = cumulative_log
        env_below = environment
    end

    return DoubleLayerStack(
        environments,
        cumulative_row_logs,
        lx,
        ly,
        dimensions.dim_phys,
        dimensions.dim_bond,
        Int(chi_s),
        Int(chi_dl),
        copy(factorizer.diagnostics),
    )
end

double_layer(peps::AbstractMatrix; kwargs...) = build_double_layer_stack(peps; kwargs...)
