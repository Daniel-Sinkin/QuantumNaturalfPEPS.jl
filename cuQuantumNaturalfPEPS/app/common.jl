using ITensors
using Random

const LX, LY, DIM_BOND, DIM_PHYS = 4, 4, 4, 2
const CHI = DIM_BOND * DIM_BOND
const SEED = 1
const SEED_SAMPLE = 7
const NS = 4096

function array_peps(lx, ly, dim_bond, dim_phys; seed=SEED)::Matrix{Array{ComplexF32,5}}
    Random.seed!(seed)
    arrays = Matrix{Array{ComplexF32,5}}(undef, lx, ly)
    for row in 1:lx, col in 1:ly
        u = row == 1 ? 1 : dim_bond
        r = col == ly ? 1 : dim_bond
        d = row == lx ? 1 : dim_bond
        l = col == 1 ? 1 : dim_bond
        arrays[row, col] = rand(ComplexF32, dim_phys, u, r, d, l)
    end
    return arrays
end

function random_unitary(::Type{S}, ingoing, outgoing)::ITensor where {S<:Number}
    t = ITensors.NDTensors.random_unitary(S, dim(ingoing), dim(outgoing))
    return ITensor(t, ingoing..., outgoing...)
end

function grid_peps(lx, ly, dim_bond; seed=SEED)::Matrix{ITensor}
    Random.seed!(seed)
    hilbert = [siteind("S=1/2"; addtags="nx=$row,ny=$col") for row in 1:lx, col in 1:ly]
    h_links = Matrix{Index{Int64}}(undef, lx, ly - 1)
    v_links = Matrix{Index{Int64}}(undef, lx - 1, ly)
    for row in 1:lx, col in 1:(ly-1)
        h_links[row, col] = Index(dim_bond, "h_link, $(row);$(col) -> $(row);$(col+1)")
    end
    for row in 1:(lx-1), col in 1:ly
        v_links[row, col] = Index(dim_bond, "v_link, $(row);$(col) -> $(row+1);$(col)")
    end
    tensors = Matrix{ITensor}(undef, lx, ly)
    for row in 1:lx, col in 1:ly
        ingoing = Index{Int64}[hilbert[row, col]]
        outgoing = Index{Int64}[]
        col != ly && push!(outgoing, h_links[row, col])
        row != lx && push!(outgoing, v_links[row, col])
        col != 1 && push!(ingoing, h_links[row, col-1])
        row != 1 && push!(ingoing, v_links[row-1, col])
        tensors[row, col] = random_unitary(ComplexF64, ingoing, outgoing)
    end
    return tensors
end
