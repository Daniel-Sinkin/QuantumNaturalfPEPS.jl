const CONVERGENCE_REFERENCE_ROOT = joinpath(
    normpath(joinpath(@__DIR__, "..", "..", "..")),
    "experiments",
    "E0261_reference_l4_cpu_jureca",
)
const CONVERGENCE_REFERENCE_WINDOW = joinpath(
    CONVERGENCE_REFERENCE_ROOT,
    "out",
    "jureca-reference_20260823T160926Z",
)

read_u32(io) = Int(ltoh(read(io, UInt32)))

function read_u64(io)
    value = ltoh(read(io, UInt64))
    value <= UInt64(typemax(Int)) || error("fixture integer exceeds host range")
    return Int(value)
end

function read_convergence_qnf(path::AbstractString)
    return open(path, "r") do io
        String(read(io, 7)) == "QNfPEPS" || error("fixture signature is invalid")
        read(io, UInt8) == 1 || error("fixture version is unsupported")
        read_u32(io) == 44 || error("fixture header size is unsupported")
        lx = read_u32(io)
        ly = read_u32(io)
        bond = read_u32(io)
        seed = read_u32(io)
        read_u64(io) == lx * ly || error("fixture site count is invalid")
        count = read_u64(io)
        dimensions = NTuple{5,Int}[]
        for row in 1:lx, column in 1:ly
            push!(
                dimensions,
                (
                    column == 1 ? 1 : bond,
                    row == lx ? 1 : bond,
                    column == ly ? 1 : bond,
                    row == 1 ? 1 : bond,
                    2,
                ),
            )
        end
        count == sum(prod, dimensions) || error("fixture value count is invalid")
        words = Vector{UInt32}(undef, 2 * count)
        read!(io, words)
        eof(io) || error("fixture has trailing bytes")
        values = Vector{ComplexF32}(undef, count)
        for index in eachindex(values)
            values[index] = ComplexF32(
                reinterpret(Float32, ltoh(words[2index-1])),
                reinterpret(Float32, ltoh(words[2index])),
            )
        end
        device_tensors = Matrix{Array{ComplexF32,5}}(undef, lx, ly)
        tensors = Matrix{Array{ComplexF32,5}}(undef, lx, ly)
        offset = 1
        for row in 1:lx, column in 1:ly
            dims = dimensions[(row - 1) * ly + column]
            site_count = prod(dims)
            device = reshape(values[offset:(offset+site_count-1)], dims)
            device_tensors[row, column] = device
            tensors[row, column] = permutedims(device, (5, 4, 3, 2, 1))
            offset += site_count
        end
        return (; lx, ly, bond, seed, values, device_tensors, tensors)
    end
end

function csv_dict_rows(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("CSV is empty")
    header = split(first(lines), ',')
    rows = Dict{String,String}[]
    for line in Iterators.drop(lines, 1)
        isempty(line) && continue
        fields = split(line, ','; keepempty=true)
        length(fields) == length(header) || error("CSV row width differs")
        push!(rows, Dict(zip(header, fields)))
    end
    return rows
end

function reference_trajectory()
    path = joinpath(CONVERGENCE_REFERENCE_WINDOW, "trajectory_reference_J.csv")
    rows = csv_dict_rows(path)
    length(rows) == 2000 || error("E0261 J trajectory length differs")
    for (index, row) in enumerate(rows)
        parse(Int, row["iteration"]) == index || error("E0261 iteration order differs")
    end
    return rows
end

function pack_convergence_peps(tensors::AbstractMatrix)
    packed = ComplexF32[]
    for row in axes(tensors, 1), column in axes(tensors, 2)
        append!(packed, vec(ComplexF32.(permutedims(tensors[row, column], (5, 4, 3, 2, 1)))))
    end
    return packed
end

function unpack_convergence_peps(packed::AbstractVector{T}, template::AbstractMatrix) where {T<:Complex}
    output = Matrix{Array{T,5}}(undef, size(template))
    offset = 1
    for row in axes(template, 1), column in axes(template, 2)
        dims = size(template[row, column])
        count = prod(dims)
        reversed = reshape(packed[offset:(offset+count-1)], reverse(dims))
        output[row, column] = permutedims(reversed, (5, 4, 3, 2, 1))
        offset += count
    end
    offset == length(packed) + 1 || error("packed PEPS length differs")
    return output
end

function write_csv_row(io, values)
    println(io, join(values, ','))
    flush(io)
    return nothing
end
