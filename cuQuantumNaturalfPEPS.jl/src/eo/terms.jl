using Libdl

struct QnpepsElocConfig
    struct_size::UInt32
    lx::Int32
    ly::Int32
    dim_phys::Int32
    dim_bond::Int32
    chi_eo::Int32
    meo::Int32
    truncation_route::Int32
    density_cutoff::Float64
    svd_cutoff::Float64
end

function QnpepsElocConfig(;
    lx,
    ly,
    dim_bond,
    chi_eo,
    meo,
    dim_phys=2,
    route=:default,
    density_cutoff::Real=1.0e-13,
    svd_cutoff::Real=1.0e-13,
)
    isfinite(density_cutoff) || throw(ArgumentError("density cutoff must be finite"))
    density_cutoff >= 0 || throw(ArgumentError("density cutoff must be nonnegative"))
    0 <= svd_cutoff <= 1 || throw(ArgumentError("svd cutoff must lie in [0, 1]"))
    return QnpepsElocConfig(
        UInt32(sizeof(QnpepsElocConfig)),
        Int32(lx),
        Int32(ly),
        Int32(dim_phys),
        Int32(dim_bond),
        Int32(chi_eo),
        Int32(meo),
        _truncation_route_value(route),
        Float64(density_cutoff),
        Float64(svd_cutoff),
    )
end

struct QnpepsElocDiagonalTerm
    value_offset::Int64
    site_count::Int32
    state_count::Int32
    site::NTuple{4,Int32}
end

struct QnpepsElocOffDiagonalPiece
    state_offset::Int64
    weight_offset::Int64
    site_count::Int32
    state_count::Int32
    site::NTuple{4,Int32}
end

struct QnpepsElocTermTable
    struct_size::UInt32
    dim_phys::Int32
    diagonal_count::Int32
    piece_count::Int32
    diagonal::Ptr{QnpepsElocDiagonalTerm}
    piece::Ptr{QnpepsElocOffDiagonalPiece}
    diagonal_value_count::Int64
    diagonal_values::Ptr{ComplexF64}
    piece_state_count::Int64
    piece_active::Ptr{UInt8}
    piece_coefficients::Ptr{ComplexF64}
    piece_weight_count::Int64
    piece_weights::Ptr{ComplexF64}
end

const ELOC_CAP_LOCAL_OPERATORS = UInt64(2)
const ELOC_LOCAL_DIMENSION = 2
const ELOC_MAX_OPERATOR_SITES = 2

struct OperatorTerms
    lx::Int
    ly::Int
    supports::Vector{Vector{Int32}}
    matrices::Vector{Matrix{ComplexF64}}
    support_index::Dict{Vector{Int32},Int}
end

function OperatorTerms(lx::Integer, ly::Integer)
    lx >= 2 || throw(ArgumentError("lx must be at least two"))
    ly >= 2 || throw(ArgumentError("ly must be at least two"))
    return OperatorTerms(
        Int(lx),
        Int(ly),
        Vector{Int32}[],
        Matrix{ComplexF64}[],
        Dict{Vector{Int32},Int}(),
    )
end

function _eloc_capabilities()::UInt64
    handle = Libdl.dlopen(_ffi_library())
    try
        symbol = Libdl.dlsym_e(handle, :qnpeps_eloc_capabilities)
        (isnothing(symbol) || symbol == C_NULL) && return UInt64(0)
        return ccall(symbol, UInt64, ())
    finally
        Libdl.dlclose(handle)
    end
end

_operator_sites(site::Tuple{Integer,Integer}) = (site,)
_operator_sites(sites::Tuple{Vararg{Tuple{Integer,Integer}}}) = sites

function _linear_site(terms::OperatorTerms, site::Tuple{Integer,Integer})::Int32
    row, col = site
    1 <= row <= terms.lx || throw(ArgumentError("operator row $row is outside 1:$(terms.lx)"))
    1 <= col <= terms.ly || throw(ArgumentError("operator column $col is outside 1:$(terms.ly)"))
    return Int32((row - 1) * terms.ly + (col - 1))
end

const _SWAPPED_PAIR_ORDER = [1, 3, 2, 4]

function add_operator!(terms::OperatorTerms, sites, matrix::AbstractMatrix)
    site_list = _operator_sites(sites)
    count = length(site_list)
    1 <= count <= ELOC_MAX_OPERATOR_SITES ||
        throw(ArgumentError("an operator acts on one or two sites"))
    states = ELOC_LOCAL_DIMENSION^count
    size(matrix) == (states, states) ||
        throw(DimensionMismatch("an operator on $count site(s) is a $states x $states matrix"))
    all(isfinite, matrix) || throw(ArgumentError("operator entries must be finite"))
    linear = [_linear_site(terms, site) for site in site_list]
    allunique(linear) || throw(ArgumentError("operator sites must be distinct"))
    canonical = ComplexF64.(matrix)
    if count == 2 && linear[1] > linear[2]
        reverse!(linear)
        canonical = canonical[_SWAPPED_PAIR_ORDER, _SWAPPED_PAIR_ORDER]
    end
    index = get(terms.support_index, linear, 0)
    if index == 0
        push!(terms.supports, linear)
        push!(terms.matrices, canonical)
        terms.support_index[linear] = length(terms.supports)
    else
        terms.matrices[index] .+= canonical
    end
    return terms
end

const _PAULI_X = ComplexF64[0 1; 1 0]
const _PAULI_Z = ComplexF64[1 0; 0 -1]
const _IDENTITY = ComplexF64[1 0; 0 1]

function _exchange_matrix(coupling::Float64)::Matrix{ComplexF64}
    exchange = 2.0 * coupling
    return ComplexF64[
        coupling 0 0 0
        0 -coupling exchange 0
        0 exchange -coupling 0
        0 0 0 coupling
    ]
end

function heisenberg_terms(lx::Integer, ly::Integer; J1::Real=1.0, J2::Real=0.0)
    terms = OperatorTerms(lx, ly)
    bond!(a, b, coupling) = add_operator!(terms, (a, b), _exchange_matrix(Float64(coupling)))
    for i in 1:lx, j in 1:(ly-1)
        bond!((i, j), (i, j + 1), J1)
    end
    for i in 1:(lx-1), j in 1:ly
        bond!((i, j), (i + 1, j), J1)
    end
    if J2 != 0
        for i in 1:(lx-1), j in 1:(ly-1)
            bond!((i, j), (i + 1, j + 1), J2)
            bond!((i + 1, j), (i, j + 1), J2)
        end
    end
    return terms
end

function truncated_rydberg_terms(
    lx::Integer,
    ly::Integer;
    Omega::Real=1.0,
    delta::Real=0.0,
    V_nn::Real=2.0,
    V_diag::Real=V_nn / 8,
)
    lx >= 2 || throw(ArgumentError("lx must be at least two"))
    ly >= 2 || throw(ArgumentError("ly must be at least two"))
    all(isfinite, (Omega, delta, V_nn, V_diag)) ||
        throw(ArgumentError("Rydberg coefficients must be finite"))
    terms = OperatorTerms(lx, ly)
    for i in 1:lx, j in 1:ly
        add_operator!(terms, (i, j), (0.5 * Float64(Omega)) .* _PAULI_X)
    end
    interactions = Tuple{NTuple{2,Int},NTuple{2,Int},Float64}[]
    for i in 1:lx, j in 1:(ly-1)
        push!(interactions, ((i, j), (i, j + 1), Float64(V_nn)))
    end
    for i in 1:(lx-1), j in 1:ly
        push!(interactions, ((i, j), (i + 1, j), Float64(V_nn)))
    end
    for i in 1:(lx-1), j in 1:(ly-1)
        push!(interactions, ((i, j), (i + 1, j + 1), Float64(V_diag)))
        push!(interactions, ((i + 1, j), (i, j + 1), Float64(V_diag)))
    end
    fields = fill(0.5 * Float64(delta), lx, ly)
    constant = -0.5 * lx * ly * Float64(delta)
    for (site_a, site_b, interaction) in interactions
        quarter = 0.25 * interaction
        constant += quarter
        fields[site_a...] -= quarter
        fields[site_b...] -= quarter
    end
    constant == 0.0 || add_operator!(terms, (1, 1), constant .* _IDENTITY)
    for i in 1:lx, j in 1:ly
        fields[i, j] == 0.0 || add_operator!(terms, (i, j), fields[i, j] .* _PAULI_Z)
    end
    for (site_a, site_b, interaction) in interactions
        interaction == 0.0 ||
            add_operator!(terms, (site_a, site_b), (0.25 * interaction) .* kron(_PAULI_Z, _PAULI_Z))
    end
    return terms
end
