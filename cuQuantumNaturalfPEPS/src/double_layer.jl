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
        bond_left, ket_dim, bra_dim, bond_right = size(env_site)
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
        site_index = site_indices[col]
        site_index_prime = prime(site_index, 1)
        link_left = col > 1 ? commonind(site_tensor, env[col-1]) : nothing
        link_right = col < ly ? commonind(site_tensor, env[col+1]) : nothing
        legs = (link_left, site_index, site_index_prime, link_right)
        present = Tuple(leg for leg in legs if leg !== nothing)
        dims = Tuple(leg !== nothing ? dim(leg) : 1 for leg in legs)
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

function _pack_env_row(env::AbstractVector)::CuVector{UInt8}
    ly = length(env)
    header = Vector{Int32}(undef, ly * 4)
    for (col, site) in enumerate(env)
        header[(4*col-3):(4*col)] .= size(site)
    end
    values = ComplexF32[]
    for site in env
        append!(values, vec(ComplexF32.(site)))
    end
    return CuArray(vcat(collect(reinterpret(UInt8, header)), collect(reinterpret(UInt8, values))))
end

function _parse_env_row(bytes::Vector{UInt8}, ly::Integer)::Vector{Array{ComplexF32,4}}
    header = reinterpret(Int32, view(bytes, 1:(ly*4*sizeof(Int32))))
    values_offset = ly * 4 * sizeof(Int32)
    values = reinterpret(ComplexF32, view(bytes, (values_offset+1):length(bytes)))
    out = Vector{Array{ComplexF32,4}}(undef, ly)
    header_pos = 0
    values_pos = 0
    for col in 1:ly
        bond_left = Int(header[header_pos+1])
        ket_dim = Int(header[header_pos+2])
        bra_dim = Int(header[header_pos+3])
        bond_right = Int(header[header_pos+4])
        header_pos += 4
        site_size = bond_left * ket_dim * bra_dim * bond_right
        chunk = Array{ComplexF32}(values[(values_pos+1):(values_pos+site_size)])
        out[col] = reshape(chunk, bond_left, ket_dim, bra_dim, bond_right)
        values_pos += site_size
    end
    return out
end

function double_layer_step(
    tensors::AbstractMatrix,
    env_row::Integer,
    env_below::Union{Nothing,MPS};
    maxdim::Integer,
)::Tuple{MPS,Float64}
    lx, ly = size(tensors)
    1 <= env_row <= lx - 1 ||
        throw(ArgumentError("env_row must be in 1:$(lx-1) (got $env_row)"))
    row_index = env_row + 1
    config = QnpepsConfig(
        lx=lx,
        ly=ly,
        dim_bond=_itensor_bond_dim(tensors),
        chi_s=maxdim,
        dim_phys=dim(_phys_index(tensors, 1, 1)),
    )
    row_buffer = _pack_peps_row([_site_array(tensors, row_index, col) for col in 1:ly])
    env_below_buffer = if env_below === nothing
        nothing
    else
        vbonds = [_vbond(tensors, row_index, col) for col in 1:ly]
        _pack_env_row(_mps_to_env_arrays(env_below, vbonds))
    end
    env_below_ptr =
        env_below_buffer !== nothing ? pointer(env_below_buffer) : CuPtr{Cvoid}(0)
    out_bytes = CUDA.zeros(UInt8, _dlenv_row_bytes(; config, maxdim))
    row_log = Ref{Float64}(0.0)
    GC.@preserve row_buffer env_below_buffer out_bytes begin
        _ffi_double_layer_row(;
            config,
            row=row_index,
            maxdim,
            peps_row=pointer(row_buffer),
            env_below=env_below_ptr,
            out=pointer(out_bytes),
            row_log,
        )
    end
    CUDA.synchronize()
    env = _parse_env_row(Array(out_bytes), ly)
    return _env_arrays_to_mps(env, [_vbond(tensors, env_row, col) for col in 1:ly]), row_log[]
end

struct CuDlenv
    data::CuArray{UInt8,1}
    cumulative_row_logs::Vector{Float64}
    lx::Int
    ly::Int
    dim_phys::Int
    dim_bond::Int
    chi_s::Int
    chi_dl::Int
end

function Base.show(io::IO, dlenv::CuDlenv)::Nothing
    return print(
        io,
        "CuDlenv(",
        dlenv.lx,
        "×",
        dlenv.ly,
        ", dim_phys=",
        dlenv.dim_phys,
        ", dim_bond=",
        dlenv.dim_bond,
        ", chi_s=",
        dlenv.chi_s,
        ", chi_dl=",
        dlenv.chi_dl,
        ")",
    )
end

function _cfg_of(
    dlenv::CuDlenv;
    seed::Integer=0,
    sampling_mode=:fast,
    chi_c::Integer=3 * dlenv.dim_bond,
)::QnpepsConfig
    return QnpepsConfig(
        lx=dlenv.lx,
        ly=dlenv.ly,
        dim_phys=dlenv.dim_phys,
        dim_bond=dlenv.dim_bond,
        chi_s=dlenv.chi_s,
        chi_dl=dlenv.chi_dl,
        seed=seed,
        sampling_mode=sampling_mode,
        chi_c=chi_c,
    )
end

function double_layer(
    device_peps::CuPeps;
    chi_s::Integer=device_peps.dim_bond,
    chi_dl::Integer=device_peps.dim_bond,
)::CuDlenv
    config = _cfg_of(device_peps; chi_s=chi_s, chi_dl=chi_dl)
    n_bytes = _dlenv_bytes(; config)
    n_bytes >= 0 || throw(
        ArgumentError(
            "invalid double-layer configuration: $device_peps, chi_s=$chi_s, chi_dl=$chi_dl",
        ),
    )
    data = CUDA.zeros(UInt8, n_bytes)
    logs = CUDA.zeros(Float64, device_peps.lx - 1)
    GC.@preserve device_peps data logs begin
        _ffi_build_dlenv(;
            config,
            peps=pointer(device_peps.data),
            dlenv=pointer(data),
            cumulative_row_logs=pointer(logs),
        )
    end
    return CuDlenv(
        data,
        Array(logs),
        device_peps.lx,
        device_peps.ly,
        device_peps.dim_phys,
        device_peps.dim_bond,
        chi_s,
        chi_dl,
    )
end
