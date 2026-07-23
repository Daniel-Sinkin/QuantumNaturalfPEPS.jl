using ITensors
using Random

# Experiment settings shared by all app/ files
const LX = 4
const LY = 4
const DIM_BOND = 4
const DIM_PHYS = 2

# Allows for reproducible results
const SEED = 1 
const SEED_SAMPLE = 2

const NS = 256 # Number of samples

# The two PEPS representations that we support for direct device upload, all
# others must first be transformed to one of these representations
const ArrayPeps = Matrix{Array{ComplexF32,5}}
const ITensorPeps = Matrix{ITensor}

# This is closer to the CUDA representation, although the legs are column major
# here while the CUDA code uses row major legs
function peps_as_arrays(lx, ly, dim_bond, dim_phys; seed=SEED)::ArrayPeps
    Random.seed!(seed) # Fix the seed to make sure we get reproducible results
    out = ArrayPeps(undef, lx, ly)
    trivial_bond = 1
    for row in 1:lx, col in 1:ly
        #       u
        #       |
        # l - [  ] - r
        #      |  \
        #     d    p
        u = (row == 1 ) ? trivial_bond : dim_bond
        r = (col == ly) ? trivial_bond : dim_bond
        d = (row == lx) ? trivial_bond : dim_bond
        l = (col == 1 ) ? trivial_bond : dim_bond
        out[row, col] = rand(ComplexF32, dim_phys, u, r, d, l)
    end
    return out
end

# This matches the representation of the existing library code
function peps_as_itensors(lx, ly, dim_bond; seed=SEED)::ITensorPeps
    Random.seed!(seed) # Fix the seed to make sure we get reproducible results
    hilbert = [siteind("S=1/2"; addtags="nx=$row,ny=$col") for row in 1:lx, col in 1:ly]
    h_links = Matrix{Index{Int64}}(undef, lx, ly - 1)
    v_links = Matrix{Index{Int64}}(undef, lx - 1, ly)
    for row in 1:lx, col in 1:(ly-1)
        h_links[row, col] = Index(dim_bond, "h_link, $(row);$(col) -> $(row);$(col+1)")
    end
    for row in 1:(lx-1), col in 1:ly
        v_links[row, col] = Index(dim_bond, "v_link, $(row);$(col) -> $(row+1);$(col)")
    end
    out = ITensorPeps(undef, lx, ly)
    for row in 1:lx, col in 1:ly
        ingoing = Index{Int64}[hilbert[row, col]]
        outgoing = Index{Int64}[]
        col != ly && push!(outgoing, h_links[row, col])
        row != lx && push!(outgoing, v_links[row, col])
        col != 1 && push!(ingoing, h_links[row, col-1])
        row != 1 && push!(ingoing, v_links[row-1, col])

        t = ITensors.NDTensors.random_unitary(ComplexF64, dim(ingoing), dim(outgoing))
        out[row, col] = ITensor(t, ingoing..., outgoing...)
    end
    return out
end
