struct ElocTermBuffers
    ly::Int
    diagonal::Vector{QnpepsElocDiagonalTerm}
    diagonal_values::Vector{ComplexF64}
    piece::Vector{QnpepsElocOffDiagonalPiece}
    piece_active::Vector{UInt8}
    piece_coefficients::Vector{ComplexF64}
    piece_weights::Vector{ComplexF64}
end

ElocTermBuffers(ly::Integer) = ElocTermBuffers(
    Int(ly),
    QnpepsElocDiagonalTerm[],
    ComplexF64[],
    QnpepsElocOffDiagonalPiece[],
    UInt8[],
    ComplexF64[],
    ComplexF64[],
)

struct RowComponent
    coefficient::ComplexF64
    weights::Vector{Vector{ComplexF64}}
end

function _unit_vector(value::Int)::Vector{ComplexF64}
    vector = zeros(ComplexF64, ELOC_LOCAL_DIMENSION)
    vector[value+1] = one(ComplexF64)
    return vector
end

function _normalized_weights(vector::AbstractVector{ComplexF64})
    nonzero = findall(!iszero, vector)
    length(nonzero) == 1 && return vector[nonzero[1]], _unit_vector(nonzero[1] - 1)
    pivot = argmax(abs.(vector))
    scale = vector[pivot]
    weights = vector ./ scale
    weights[pivot] = one(ComplexF64)
    return scale, weights
end

function _row_component(vectors::Vector{ComplexF64}...)::RowComponent
    coefficient = one(ComplexF64)
    weights = Vector{ComplexF64}[]
    for vector in vectors
        scale, normalized = _normalized_weights(vector)
        coefficient *= scale
        push!(weights, normalized)
    end
    return RowComponent(coefficient, weights)
end

function _one_site_components(row::Vector{ComplexF64})::Vector{RowComponent}
    any(!iszero, row) || return RowComponent[]
    return [_row_component(row)]
end

function _two_site_components(row::Vector{ComplexF64})::Vector{RowComponent}
    targets = [row[1+ELOC_LOCAL_DIMENSION*first+second] for first in 0:1, second in 0:1]
    first_rows = [any(!iszero, targets[value, :]) for value in 1:ELOC_LOCAL_DIMENSION]
    second_columns = [any(!iszero, targets[:, value]) for value in 1:ELOC_LOCAL_DIMENSION]
    any(first_rows) || return RowComponent[]
    if count(first_rows) == 1
        value = findfirst(first_rows)
        return [_row_component(_unit_vector(value - 1), targets[value, :])]
    end
    if count(second_columns) == 1
        value = findfirst(second_columns)
        return [_row_component(targets[:, value], _unit_vector(value - 1))]
    end
    return [
        _row_component(_unit_vector(value - 1), targets[value, :]) for
        value in 1:ELOC_LOCAL_DIMENSION
    ]
end

const _PRODUCT_TOLERANCE = 64 * eps(Float64)

function _site_targets(row::Vector{ComplexF64}, site_count::Int)::Array{ComplexF64}
    shape = ntuple(_ -> ELOC_LOCAL_DIMENSION, site_count)
    return permutedims(reshape(row, shape), Tuple(site_count:-1:1))
end

function _product_factors(targets::Array{ComplexF64})
    site_count = ndims(targets)
    pivot = argmax(abs.(targets))
    scale = targets[pivot]
    factors = [
        targets[ntuple(axis -> axis == position ? Colon() : pivot[axis], site_count)...] for
        position in 1:site_count
    ]
    for position in 1:(site_count-1)
        factors[position] = factors[position] ./ scale
    end
    product = ones(ComplexF64, size(targets))
    for position in 1:site_count
        shape = ntuple(axis -> axis == position ? ELOC_LOCAL_DIMENSION : 1, site_count)
        product = product .* reshape(factors[position], shape)
    end
    return isapprox(product, targets; rtol=_PRODUCT_TOLERANCE) ? factors : nothing
end

function _prefix_components(targets::Array{ComplexF64})::Vector{RowComponent}
    prefix_count = ndims(targets) - 1
    components = RowComponent[]
    for prefix in 0:(ELOC_LOCAL_DIMENSION^prefix_count-1)
        values = _state_values(prefix, prefix_count)
        slice = targets[(values .+ 1)..., :]
        any(!iszero, slice) || continue
        push!(components, _row_component(_unit_vector.(values)..., slice))
    end
    return components
end

function _multi_site_components(row::Vector{ComplexF64}, site_count::Int)::Vector{RowComponent}
    any(!iszero, row) || return RowComponent[]
    targets = _site_targets(row, site_count)
    factors = _product_factors(targets)
    isnothing(factors) || return [_row_component(factors...)]
    return _prefix_components(targets)
end

function _row_components(matrix::Matrix{ComplexF64}, state::Int, site_count::Int)
    row = matrix[state, :]
    row[state] = zero(ComplexF64)
    site_count == 1 && return _one_site_components(row)
    site_count == 2 && return _two_site_components(row)
    return _multi_site_components(row, site_count)
end

_padded_sites(support::Vector{Int32}) =
    ntuple(index -> index <= length(support) ? support[index] : Int32(0), Val(4))

function _push_diagonal!(
    buffers::ElocTermBuffers,
    support::Vector{Int32},
    matrix::Matrix{ComplexF64},
)
    values = [matrix[state, state] for state in axes(matrix, 1)]
    any(!iszero, values) || return nothing
    push!(
        buffers.diagonal,
        QnpepsElocDiagonalTerm(
            Int64(length(buffers.diagonal_values)),
            Int32(length(support)),
            Int32(length(values)),
            _padded_sites(support),
        ),
    )
    append!(buffers.diagonal_values, values)
    return nothing
end

function _moving_pattern(component::RowComponent, values::Vector{Int})::BitVector
    return BitVector([
        findall(!iszero, weights) != [value + 1] for
        (weights, value) in zip(component.weights, values)
    ])
end

const PieceRows = Vector{Union{Nothing,RowComponent}}

function _pattern_pieces(components::Vector{Vector{RowComponent}}, site_count::Int)
    piece_keys = Tuple{BitVector,Int}[]
    pieces = PieceRows[]
    for (state, row) in enumerate(components)
        values = _state_values(state - 1, site_count)
        row_patterns = BitVector[]
        for component in row
            pattern = _moving_pattern(component, values)
            key = (pattern, count(==(pattern), row_patterns))
            push!(row_patterns, pattern)
            index = findfirst(==(key), piece_keys)
            if isnothing(index)
                push!(piece_keys, key)
                push!(pieces, PieceRows(nothing, length(components)))
                index = length(pieces)
            end
            pieces[index][state] = component
        end
    end
    return pieces
end

function _push_pieces!(buffers::ElocTermBuffers, support::Vector{Int32}, matrix::Matrix{ComplexF64})
    site_count = length(support)
    state_count = size(matrix, 1)
    components = [_row_components(matrix, state, site_count) for state in 1:state_count]
    for rows in _pattern_pieces(components, site_count)
        push!(
            buffers.piece,
            QnpepsElocOffDiagonalPiece(
                Int64(length(buffers.piece_active)),
                Int64(length(buffers.piece_weights)),
                Int32(site_count),
                Int32(state_count),
                _padded_sites(support),
            ),
        )
        for component in rows
            is_active = !isnothing(component)
            push!(buffers.piece_active, is_active ? UInt8(1) : UInt8(0))
            push!(buffers.piece_coefficients, is_active ? component.coefficient : zero(ComplexF64))
            for position in 1:site_count
                weights =
                    is_active ? component.weights[position] :
                    zeros(ComplexF64, ELOC_LOCAL_DIMENSION)
                append!(buffers.piece_weights, weights)
            end
        end
    end
    return nothing
end

function _moving_sites(buffers::ElocTermBuffers, piece::QnpepsElocOffDiagonalPiece)::Vector{Int32}
    site_count = Int(piece.site_count)
    is_moving = falses(site_count)
    for state in 0:(Int(piece.state_count)-1)
        buffers.piece_active[piece.state_offset+state+1] == 0 && continue
        values = _state_values(state, site_count)
        for position in 1:site_count
            slot = state * site_count + position - 1
            offset = piece.weight_offset + slot * ELOC_LOCAL_DIMENSION
            weights = @view buffers.piece_weights[(offset+1):(offset+ELOC_LOCAL_DIMENSION)]
            findall(!iszero, weights) == [values[position] + 1] || (is_moving[position] = true)
        end
    end
    return [piece.site[position] for position in 1:site_count if is_moving[position]]
end

function _is_line_or_plaquette(moving::Vector{Int32}, ly::Int)::Bool
    isempty(moving) && return true
    rows = div.(moving, ly)
    columns = moving .% ly
    row_span = maximum(rows) - minimum(rows)
    column_span = maximum(columns) - minimum(columns)
    return row_span == 0 || column_span == 0 || (row_span == 1 && column_span == 1)
end

function _has_piece_beyond_plaquette(buffers::ElocTermBuffers)::Bool
    return any(
        piece -> !_is_line_or_plaquette(_moving_sites(buffers, piece), buffers.ly),
        buffers.piece,
    )
end

const _PAIR_OPERATOR_SITES = 2

function _require_eloc_term_capabilities(buffers::ElocTermBuffers)::Nothing
    capabilities = _eloc_capabilities()
    (capabilities & ELOC_CAP_LOCAL_OPERATORS) != 0 ||
        throw(ArgumentError("the loaded CUDA library does not support local operator term tables"))
    has_wide_piece = any(piece -> piece.site_count > _PAIR_OPERATOR_SITES, buffers.piece)
    has_plaquette_operators = (capabilities & ELOC_CAP_PLAQUETTE_OPERATORS) != 0
    (has_wide_piece && !has_plaquette_operators) && throw(
        ArgumentError("the loaded CUDA library does not support operators on three or four sites"),
    )
    has_slab_operators = (capabilities & ELOC_CAP_SLAB_OPERATORS) != 0
    (!has_slab_operators && _has_piece_beyond_plaquette(buffers)) && throw(
        ArgumentError(
            "the loaded CUDA library supports only operators whose changed sites lie in one row, " *
            "in one column, or in one 2 x 2 plaquette",
        ),
    )
    return nothing
end

function _eloc_term_buffers(terms::OperatorTerms, config)::ElocTermBuffers
    Int(config.lx) == terms.lx ||
        throw(DimensionMismatch("operator terms lx does not match config"))
    Int(config.ly) == terms.ly ||
        throw(DimensionMismatch("operator terms ly does not match config"))
    Int(config.dim_phys) == ELOC_LOCAL_DIMENSION ||
        throw(ArgumentError("operator terms require dim_phys equal to two"))
    buffers = ElocTermBuffers(terms.ly)
    for (support, matrix) in zip(terms.supports, terms.matrices)
        _push_diagonal!(buffers, support, matrix)
        _push_pieces!(buffers, support, matrix)
    end
    length(buffers.diagonal) <= typemax(Int32) || throw(ArgumentError("too many diagonal terms"))
    length(buffers.piece) <= typemax(Int32) || throw(ArgumentError("too many off-diagonal pieces"))
    return buffers
end

_buffer_pointer(vector::Vector{T}) where {T} = isempty(vector) ? Ptr{T}(0) : pointer(vector)

function _eloc_term_table(buffers::ElocTermBuffers)::QnpepsElocTermTable
    _require_eloc_term_capabilities(buffers)
    return _eloc_abi_table(buffers)
end

function _eloc_abi_table(buffers::ElocTermBuffers)::QnpepsElocTermTable
    return QnpepsElocTermTable(
        UInt32(sizeof(QnpepsElocTermTable)),
        Int32(ELOC_LOCAL_DIMENSION),
        Int32(length(buffers.diagonal)),
        Int32(length(buffers.piece)),
        _buffer_pointer(buffers.diagonal),
        _buffer_pointer(buffers.piece),
        Int64(length(buffers.diagonal_values)),
        _buffer_pointer(buffers.diagonal_values),
        Int64(length(buffers.piece_active)),
        _buffer_pointer(buffers.piece_active),
        _buffer_pointer(buffers.piece_coefficients),
        Int64(length(buffers.piece_weights)),
        _buffer_pointer(buffers.piece_weights),
    )
end
