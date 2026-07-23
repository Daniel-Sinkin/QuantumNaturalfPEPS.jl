using CUDA
using ITensors
using ITensorMPS

struct _ZipupITensorSite
    mpo_left::Union{Nothing,Index}
    physical_input::Index
    physical_output::Index
    mpo_right::Union{Nothing,Index}
    mps_left::Union{Nothing,Index}
    mps_right::Union{Nothing,Index}
end

function _zipup_site_dims(sites::AbstractVector, rank::Integer, label::AbstractString)::Vector{Int32}
    dimensions = Int32[]
    sizehint!(dimensions, rank * length(sites))
    for (site_index, site) in enumerate(sites)
        ndims(site) == rank || throw(
            DimensionMismatch(
                "$label site $site_index has rank $(ndims(site)); expected rank $rank",
            ),
        )
        site isa CuArray || throw(ArgumentError("$label site $site_index must be a CuArray"))
        eltype(site) == ComplexF32 || throw(
            ArgumentError("$label site $site_index must have element type ComplexF32"),
        )
        for extent in size(site)
            1 <= extent <= typemax(Int32) || throw(
                DimensionMismatch("$label site $site_index has invalid extent $extent"),
            )
            push!(dimensions, Int32(extent))
        end
    end
    return dimensions
end

function _zipup_validate_topology(
    mpo::AbstractVector,
    mps::AbstractVector,
    mpo_dims::Vector{Int32},
    mps_dims::Vector{Int32},
)::Nothing
    length(mpo) == length(mps) || throw(
        DimensionMismatch("MPO and MPS site counts differ: $(length(mpo)) != $(length(mps))"),
    )
    isempty(mpo) && throw(ArgumentError("MPO and MPS must contain at least one site"))
    for site in 1:length(mpo)
        mpo_base = 4 * (site - 1)
        mps_base = 3 * (site - 1)
        mpo_left = mpo_dims[mpo_base+1]
        mpo_physical_in = mpo_dims[mpo_base+2]
        mpo_right = mpo_dims[mpo_base+4]
        mps_left = mps_dims[mps_base+1]
        mps_physical = mps_dims[mps_base+2]
        mps_right = mps_dims[mps_base+3]
        expected_mpo_left = site == 1 ? 1 : mpo_dims[mpo_base]
        expected_mps_left = site == 1 ? 1 : mps_dims[mps_base]
        mpo_left == expected_mpo_left || throw(
            DimensionMismatch("MPO bond before site $site is not open/adjacent"),
        )
        mps_left == expected_mps_left || throw(
            DimensionMismatch("MPS bond before site $site is not open/adjacent"),
        )
        mpo_physical_in == mps_physical || throw(
            DimensionMismatch(
                "MPO physical input and MPS physical dimension differ at site $site",
            ),
        )
        if site == length(mpo)
            mpo_right == 1 || throw(DimensionMismatch("MPO right boundary must be one"))
            mps_right == 1 || throw(DimensionMismatch("MPS right boundary must be one"))
        end
    end
    return nothing
end

function _zipup_output_dims(
    mpo_dims::Vector{Int32},
    mps_dims::Vector{Int32},
    num_sites::Integer,
    maxdim::Integer,
)::Vector{NTuple{3,Int}}
    output = Vector{NTuple{3,Int}}(undef, num_sites)
    output_left = 1
    for site in 1:num_sites
        physical_out = Int(mpo_dims[4*(site-1)+3])
        mps_right = Int(mps_dims[3*(site-1)+3])
        mpo_right = Int(mpo_dims[4*(site-1)+4])
        output_right = min(maxdim, output_left * physical_out, mps_right * mpo_right)
        output[site] = (output_left, physical_out, output_right)
        output_left = output_right
    end
    return output
end

function _zipup_pack(sites::AbstractVector)::CuVector{ComplexF32}
    packed = CUDA.zeros(ComplexF32, sum(length, sites; init=0))
    offset = 1
    for site in sites
        elements = length(site)
        copyto!(packed, offset, vec(site), 1, elements)
        offset += elements
    end
    return packed
end

function _zipup_chain_bonds(chain, label::AbstractString)::Vector{Index}
    isempty(chain) && throw(ArgumentError("$label must contain at least one site"))
    bonds = Vector{Index}(undef, length(chain) - 1)
    for site in 1:(length(chain)-1)
        shared = commoninds(chain[site], chain[site+1])
        if length(shared) != 1
            throw(
                DimensionMismatch(
                    "$label sites $site and $(site+1) must share exactly one bond index",
                ),
            )
        end
        bonds[site] = only(shared)
    end
    for first_site in 1:(length(chain)-2)
        for second_site in (first_site+2):length(chain)
            if !isempty(commoninds(chain[first_site], chain[second_site]))
                throw(
                    DimensionMismatch(
                        "$label sites $first_site and $second_site share a nonadjacent index",
                    ),
                )
            end
        end
    end
    return bonds
end

function _zipup_site_indices(
    tensor::ITensor,
    left::Union{Nothing,Index},
    right::Union{Nothing,Index},
    label::AbstractString,
    site::Integer,
)::Vector{Index}
    physical = Index[]
    for index in inds(tensor)
        if index != left && index != right
            push!(physical, index)
        end
    end
    isempty(physical) && throw(
        DimensionMismatch("$label site $site has no physical index"),
    )
    return physical
end

function _zipup_itensor_sites(mpo::MPO, mps::MPS)::Vector{_ZipupITensorSite}
    length(mpo) == length(mps) || throw(
        DimensionMismatch("MPO and MPS site counts differ: $(length(mpo)) != $(length(mps))"),
    )
    isempty(mpo) && throw(ArgumentError("MPO and MPS must contain at least one site"))
    mpo_bonds = _zipup_chain_bonds(mpo, "MPO")
    mps_bonds = _zipup_chain_bonds(mps, "MPS")
    sites = Vector{_ZipupITensorSite}(undef, length(mpo))
    for site in eachindex(mpo)
        mpo_left = site == 1 ? nothing : mpo_bonds[site-1]
        mpo_right = site == length(mpo) ? nothing : mpo_bonds[site]
        mps_left = site == 1 ? nothing : mps_bonds[site-1]
        mps_right = site == length(mps) ? nothing : mps_bonds[site]
        mpo_physical = _zipup_site_indices(mpo[site], mpo_left, mpo_right, "MPO", site)
        mps_physical = _zipup_site_indices(mps[site], mps_left, mps_right, "MPS", site)
        if length(mpo_physical) != 2
            throw(
                DimensionMismatch(
                    "MPO site $site must have one physical input and one physical output index",
                ),
            )
        end
        if length(mps_physical) != 1
            throw(DimensionMismatch("MPS site $site must have one physical index"))
        end
        physical_input = only(mps_physical)
        output_position = findfirst(!=(physical_input), mpo_physical)
        if physical_input ∉ mpo_physical || output_position === nothing
            throw(
                DimensionMismatch(
                    "MPO and MPS site $site must share exactly one physical input index",
                ),
            )
        end
        sites[site] = _ZipupITensorSite(
            mpo_left,
            physical_input,
            mpo_physical[output_position],
            mpo_right,
            mps_left,
            mps_right,
        )
    end
    return sites
end

function _zipup_itensor_host_arrays(mpo::MPO, mps::MPS)::NamedTuple
    if hasqns(mpo) || hasqns(mps)
        throw(ArgumentError("zipup_mpo_mps does not support quantum-number tensors"))
    end
    layout = _zipup_itensor_sites(mpo, mps)
    mpo_arrays = Vector{Array{ComplexF32,4}}(undef, length(layout))
    mps_arrays = Vector{Array{ComplexF32,3}}(undef, length(layout))
    physical_outputs = Vector{Index}(undef, length(layout))
    for site in eachindex(layout)
        axes = layout[site]
        mpo_indices = Tuple(
            index for
            index in
            (axes.mpo_left, axes.physical_input, axes.physical_output, axes.mpo_right) if
            index !== nothing
        )
        mps_indices = Tuple(
            index for
            index in (axes.mps_left, axes.physical_input, axes.mps_right) if index !== nothing
        )
        mpo_dims = (
            axes.mpo_left === nothing ? 1 : dim(axes.mpo_left),
            dim(axes.physical_input),
            dim(axes.physical_output),
            axes.mpo_right === nothing ? 1 : dim(axes.mpo_right),
        )
        mps_dims = (
            axes.mps_left === nothing ? 1 : dim(axes.mps_left),
            dim(axes.physical_input),
            axes.mps_right === nothing ? 1 : dim(axes.mps_right),
        )
        mpo_values = Array{ComplexF32}(ITensors.array(mpo[site], mpo_indices...))
        mps_values = Array{ComplexF32}(ITensors.array(mps[site], mps_indices...))
        mpo_arrays[site] = reshape(mpo_values, mpo_dims)
        mps_arrays[site] = reshape(mps_values, mps_dims)
        physical_outputs[site] = axes.physical_output
    end
    return (mpo=mpo_arrays, mps=mps_arrays, physical_outputs=physical_outputs)
end

function _zipup_arrays_to_mps(
    sites::AbstractVector,
    physical_outputs::AbstractVector{<:Index},
)::MPS
    length(sites) == length(physical_outputs) || throw(
        DimensionMismatch(
            "MPS site and physical-index counts differ: $(length(sites)) != $(length(physical_outputs))",
        ),
    )
    isempty(sites) && throw(ArgumentError("MPS must contain at least one site"))
    links = Vector{Index}(undef, length(sites) - 1)
    for site in 1:(length(sites)-1)
        size(sites[site], 3) == size(sites[site+1], 1) || throw(
            DimensionMismatch("output MPS bond after site $site is not adjacent"),
        )
        links[site] = Index(size(sites[site], 3); tags="Link,l=$site")
    end
    tensors = Vector{ITensor}(undef, length(sites))
    for site in eachindex(sites)
        ndims(sites[site]) == 3 || throw(
            DimensionMismatch("output MPS site $site must have rank three"),
        )
        left, physical, right = size(sites[site])
        expected_left = site == 1 ? 1 : dim(links[site-1])
        expected_right = site == length(sites) ? 1 : dim(links[site])
        left == expected_left || throw(
            DimensionMismatch("output MPS left dimension is invalid at site $site"),
        )
        right == expected_right || throw(
            DimensionMismatch("output MPS right dimension is invalid at site $site"),
        )
        physical == dim(physical_outputs[site]) || throw(
            DimensionMismatch("output MPS physical dimension is invalid at site $site"),
        )
        values = ComplexF32.(Array(sites[site]))
        if length(sites) == 1
            tensors[site] = itensor(
                reshape(values, physical),
                physical_outputs[site],
            )
        elseif site == 1
            tensors[site] = itensor(
                reshape(values, physical, right),
                physical_outputs[site],
                links[site],
            )
        elseif site == length(sites)
            tensors[site] = itensor(
                reshape(values, left, physical),
                dag(links[site-1]),
                physical_outputs[site],
            )
        else
            tensors[site] = itensor(
                values,
                dag(links[site-1]),
                physical_outputs[site],
                links[site],
            )
        end
    end
    return MPS(tensors)
end

function zipup_mpo_mps(
    mpo::AbstractVector,
    mps::AbstractVector;
    maxdim::Integer,
)::NamedTuple
    1 <= maxdim <= typemax(Int32) || throw(
        ArgumentError("maxdim must be in 1:$(typemax(Int32)) (got $maxdim)"),
    )
    length(mpo) == length(mps) || throw(
        DimensionMismatch("MPO and MPS site counts differ: $(length(mpo)) != $(length(mps))"),
    )
    isempty(mpo) && throw(ArgumentError("MPO and MPS must contain at least one site"))
    length(mpo) <= typemax(Int32) || throw(ArgumentError("too many MPO sites"))
    mpo_dims = _zipup_site_dims(mpo, 4, "MPO")
    mps_dims = _zipup_site_dims(mps, 3, "MPS")
    _zipup_validate_topology(mpo, mps, mpo_dims, mps_dims)
    output_dims = _zipup_output_dims(mpo_dims, mps_dims, length(mpo), maxdim)

    packed_mpo = _zipup_pack(mpo)
    packed_mps = _zipup_pack(mps)
    descriptor = GC.@preserve mpo_dims mps_dims QnpepsZipupMpoMpsDesc(
        UInt32(sizeof(QnpepsZipupMpoMpsDesc)),
        Int32(length(mpo)),
        Int32(maxdim),
        Int32(0),
        UInt(pointer(mpo_dims)),
        UInt(pointer(mps_dims)),
    )
    output_bytes = GC.@preserve mpo_dims mps_dims _zipup_mpo_mps_bytes(; descriptor)
    output_bytes >= 0 || error("validated zip-up descriptor was rejected by the CUDA library")
    output_bytes % sizeof(ComplexF32) == 0 || error("invalid zip-up output byte count")
    packed_output = CUDA.zeros(ComplexF32, output_bytes ÷ sizeof(ComplexF32))
    log_gauge = Ref{Float64}(0.0)

    GC.@preserve mpo_dims mps_dims packed_mpo packed_mps packed_output log_gauge begin
        args = QnpepsZipupMpoMpsArgs(
            UInt32(sizeof(QnpepsZipupMpoMpsArgs)),
            UInt32(0),
            UInt(pointer(packed_mpo)),
            UInt(sizeof(ComplexF32) * length(packed_mpo)),
            UInt(pointer(packed_mps)),
            UInt(sizeof(ComplexF32) * length(packed_mps)),
            UInt(pointer(packed_output)),
            UInt(output_bytes),
            UInt(Base.unsafe_convert(Ptr{Float64}, log_gauge)),
            UInt(CUDA.stream().handle),
        )
        _ffi_zipup_mpo_mps(; descriptor, args)
    end

    output_sites = Vector{CuArray{ComplexF32,3}}(undef, length(output_dims))
    offset = 1
    for site in eachindex(output_dims)
        dims = output_dims[site]
        elements = prod(dims)
        output_sites[site] = CUDA.zeros(ComplexF32, dims)
        copyto!(vec(output_sites[site]), 1, packed_output, offset, elements)
        offset += elements
    end
    return (mps=output_sites, log_gauge=log_gauge[])
end

function zipup_mpo_mps(
    mpo::MPO,
    mps::MPS;
    maxdim::Integer,
)::NamedTuple
    host = _zipup_itensor_host_arrays(mpo, mps)
    result = zipup_mpo_mps(CuArray.(host.mpo), CuArray.(host.mps); maxdim)
    return (
        mps=_zipup_arrays_to_mps(result.mps, host.physical_outputs),
        log_gauge=result.log_gauge,
    )
end
