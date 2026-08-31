using ITensors
using ITensorMPS
using CUDA

function _env_arrays_to_mps(env::AbstractVector, site_indices::AbstractVector)::MPS
    ly = length(env)
    links = [Index(size(env[col], 4); tags="Link,l=$col") for col in 1:(ly-1)]
    sites = Vector{ITensor}(undef, ly)
    for col in 1:ly
        site_index = site_indices[col]
        site_index_prime = prime(site_index, 1)
        env_site = env[col]
        bond_left = size(env_site, 1)
        ket_dim = size(env_site, 2)
        bra_dim = size(env_site, 3)
        bond_right = size(env_site, 4)
        if ly == 1
            reshaped = reshape(env_site, ket_dim, bra_dim)
            sites[col] = itensor(reshaped, site_index, site_index_prime)
        elseif col == 1
            reshaped = reshape(env_site, ket_dim, bra_dim, bond_right)
            sites[col] = itensor(reshaped, site_index, site_index_prime, links[1])
        elseif col == ly
            reshaped = reshape(env_site, bond_left, ket_dim, bra_dim)
            sites[col] = itensor(reshaped, links[ly-1], site_index, site_index_prime)
        else
            sites[col] = itensor(env_site, links[col-1], site_index, site_index_prime, links[col])
        end
    end
    return MPS(sites)
end

function _mps_to_env_arrays(env::MPS, site_indices::AbstractVector)::Vector{<:Array{<:Number,4}}
    ly = length(env)
    T = eltype(env[1])
    out = Vector{Array{T,4}}(undef, ly)
    for col in 1:ly
        site_tensor = env[col]
        link_left = col > 1 ? commonind(site_tensor, env[col-1]) : nothing
        link_right = col < ly ? commonind(env[col], env[col+1]) : nothing
        site_index = site_indices[col]
        legs = (link_left, site_index, prime(site_index, 1), link_right)
        present = Tuple(leg for leg in legs if leg !== nothing)
        dims = Tuple(leg === nothing ? 1 : dim(leg) for leg in legs)
        raw = ITensors.array(site_tensor, present...)
        out[col] = reshape(Array{T}(raw), dims)
    end
    return out
end

_vbond(tensors, row, col)::Index = commonind(tensors[row, col], tensors[row+1, col])
_hbond(tensors, row, col)::Index = commonind(tensors[row, col], tensors[row, col+1])

function _itensor_bond_dim(tensors::AbstractMatrix)::Int
    lx, ly = size(tensors)
    dim_bond = 1
    for row in 1:(lx-1), col in 1:ly
        dim_bond = max(dim_bond, dim(_vbond(tensors, row, col)))
    end
    for row in 1:lx, col in 1:(ly-1)
        dim_bond = max(dim_bond, dim(_hbond(tensors, row, col)))
    end
    return dim_bond
end

function _pack_peps_row(row::AbstractVector)::CuVector{ComplexF32}
    buffer = ComplexF32[]
    for site in row
        append!(buffer, vec(ComplexF32.(_to_device_order(site))))
    end
    return CuArray(buffer)
end

function _pack_grouped_mps(env::AbstractVector)::Tuple{Vector{Int32},CuVector{ComplexF32}}
    dims = Vector{Int32}(undef, 3 * length(env))
    values = ComplexF32[]
    for (col, site) in enumerate(env)
        left, ket, bra, right = size(site)
        dims[(3*col-2):(3*col)] .= Int32[left, ket*bra, right]
        append!(values, vec(ComplexF32.(site)))
    end
    return dims, CuArray(values)
end

function _grouped_mps_elements(dims::AbstractVector{Int32})::Int
    length(dims) % 3 == 0 || throw(DimensionMismatch("grouped MPS dimensions are incomplete"))
    elements = 0
    for site in 1:(length(dims)÷3)
        site_dims = view(dims, (3*site-2):(3*site))
        all(>(0), site_dims) || throw(DimensionMismatch("grouped MPS dimensions must be positive"))
        elements += prod(Int, site_dims)
    end
    return elements
end

function _ungroup_mps(dims::Vector{Int32}, values::Vector{ComplexF32})::Vector{Array{ComplexF32,4}}
    sites = Vector{Array{ComplexF32,4}}(undef, length(dims) ÷ 3)
    offset = 0
    for site in eachindex(sites)
        left, physical, right = Int.(view(dims, (3*site-2):(3*site)))
        vertical = isqrt(physical)
        vertical * vertical == physical ||
            throw(DimensionMismatch("double-layer physical dimension $physical is not square"))
        elements = left * physical * right
        sites[site] = reshape(
            copy(view(values, (offset+1):(offset+elements))),
            left,
            vertical,
            vertical,
            right,
        )
        offset += elements
    end
    offset == length(values) || error("grouped MPS value count does not match its dimensions")
    return sites
end
