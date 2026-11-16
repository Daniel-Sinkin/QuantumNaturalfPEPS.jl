struct ElocTermBuffers
    diagonal::Vector{QnpepsElocDiagonalTerm}
    diagonal_values::Vector{ComplexF64}
    piece::Vector{QnpepsElocOffDiagonalPiece}
    piece_active::Vector{UInt8}
    piece_coefficients::Vector{ComplexF64}
    piece_weights::Vector{ComplexF64}
end

ElocTermBuffers() = ElocTermBuffers(
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

function _row_components(matrix::Matrix{ComplexF64}, state::Int, site_count::Int)
    row = matrix[state, :]
    row[state] = zero(ComplexF64)
    return site_count == 1 ? _one_site_components(row) : _two_site_components(row)
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

function _push_pieces!(buffers::ElocTermBuffers, support::Vector{Int32}, matrix::Matrix{ComplexF64})
    site_count = length(support)
    state_count = size(matrix, 1)
    components = [_row_components(matrix, state, site_count) for state in 1:state_count]
    piece_count = maximum(length, components; init=0)
    for piece_index in 1:piece_count
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
        for row in components
            is_active = length(row) >= piece_index
            push!(buffers.piece_active, is_active ? UInt8(1) : UInt8(0))
            push!(
                buffers.piece_coefficients,
                is_active ? row[piece_index].coefficient : zero(ComplexF64),
            )
            for position in 1:site_count
                weights =
                    is_active ? row[piece_index].weights[position] :
                    zeros(ComplexF64, ELOC_LOCAL_DIMENSION)
                append!(buffers.piece_weights, weights)
            end
        end
    end
    return nothing
end

function _require_eloc_term_capabilities()::Nothing
    (_eloc_capabilities() & ELOC_CAP_LOCAL_OPERATORS) != 0 ||
        throw(ArgumentError("the loaded CUDA library does not support local operator term tables"))
    return nothing
end

function _eloc_term_buffers(terms::OperatorTerms, config)::ElocTermBuffers
    Int(config.lx) == terms.lx ||
        throw(DimensionMismatch("operator terms lx does not match config"))
    Int(config.ly) == terms.ly ||
        throw(DimensionMismatch("operator terms ly does not match config"))
    Int(config.dim_phys) == ELOC_LOCAL_DIMENSION ||
        throw(ArgumentError("operator terms require dim_phys equal to two"))
    buffers = ElocTermBuffers()
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
    _require_eloc_term_capabilities()
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
