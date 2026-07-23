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
        dims[(3*col-2):(3*col)] .= Int32[left, ket * bra, right]
        append!(values, vec(ComplexF32.(site)))
    end
    return dims, CuArray(values)
end

function _grouped_mps_elements(dims::AbstractVector{Int32})::Int
    length(dims) % 3 == 0 || throw(DimensionMismatch("grouped MPS dimensions are incomplete"))
    elements = 0
    for site in 1:(length(dims) ÷ 3)
        site_dims = view(dims, (3*site-2):(3*site))
        all(>(0), site_dims) || throw(DimensionMismatch("grouped MPS dimensions must be positive"))
        elements += prod(Int, site_dims)
    end
    return elements
end

function _ungroup_mps(
    dims::Vector{Int32}, values::Vector{ComplexF32}
)::Vector{Array{ComplexF32,4}}
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

    mps_dims, mps_values = if env_below !== nothing
        vbonds = [_vbond(tensors, row_index, col) for col in 1:ly]
        _pack_grouped_mps(_mps_to_env_arrays(env_below, vbonds))
    else
        nothing, nothing
    end
    workspace = _ZipupWorkspace(config, maxdim)
    output_dims = zeros(Int32, 3 * ly)
    output_values = CUDA.zeros(UInt8, workspace.row_value_bytes)
    scales = Vector{Float64}(undef, ly)
    try
        _ffi_zipup_ctx_begin(workspace.handle)
        _enqueue_peps_row!(
            workspace,
            row_index,
            row_buffer,
            mps_dims,
            mps_values,
            output_dims,
            output_values,
        )
        _ffi_zipup_ctx_finish(workspace.handle, scales)
    finally
        close(workspace)
    end

    output_elements = _grouped_mps_elements(output_dims)
    output_bytes = Array(view(output_values, 1:(output_elements*sizeof(ComplexF32))))
    env = _ungroup_mps(output_dims, copy(reinterpret(ComplexF32, output_bytes)))
    row_log = sum(scale > 0.0 ? log(scale) : 0.0 for scale in scales)
    return _env_arrays_to_mps(env, [_vbond(tensors, env_row, col) for col in 1:ly]), row_log
end

function double_layer(
    tensors::AbstractMatrix;
    maxdim::Integer,
)::Tuple{Vector{MPS},Vector{Float64}}
    lx, ly = size(tensors)
    lx >= 2 || throw(ArgumentError("double-layer envs need at least 2 rows (got $lx)"))
    ly >= 2 || throw(ArgumentError("double-layer envs need at least 2 columns (got $ly)"))
    maxdim >= 1 || throw(ArgumentError("maxdim must be positive (got $maxdim)"))

    num_env_rows = lx - 1
    rows = Vector{MPS}(undef, num_env_rows)
    cumulative_row_logs = Vector{Float64}(undef, num_env_rows)
    env_below = nothing
    cumulative_log = 0.0
    for env_row in num_env_rows:-1:1
        row, row_log = double_layer_step(tensors, env_row, env_below; maxdim)
        rows[env_row] = row
        cumulative_log += row_log
        cumulative_row_logs[env_row] = cumulative_log
        env_below = row
    end
    return rows, cumulative_row_logs
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

mutable struct ZipupWorkspace{D,S}
    handle::Ptr{Cvoid}
    config::QnpepsConfig
    maxdim::Int
    row_value_bytes::Int
    device::D
    stream::S
end

function _ZipupWorkspace(config::QnpepsConfig, maxdim::Integer)
    row_bytes = _zipup_peps_row_bytes(; config, maxdim)
    row_bytes >= 0 && row_bytes % sizeof(ComplexF32) == 0 || throw(
        ArgumentError(
            "invalid PEPS-row zip-up configuration: config=$config, maxdim=$maxdim",
        ),
    )

    device = CUDA.device()
    stream = CUDA.stream()
    handle = _ffi_zipup_ctx_create(;
        config,
        maxdim,
        stream=Ptr{Cvoid}(stream.handle),
    )
    workspace = ZipupWorkspace(
        handle,
        config,
        Int(maxdim),
        row_bytes,
        device,
        stream,
    )
    finalizer(close, workspace)
    return workspace
end

function ZipupWorkspace(
    device_peps::CuPeps;
    chi_s::Integer=device_peps.dim_bond,
    chi_dl::Integer=device_peps.dim_bond,
)
    return _ZipupWorkspace(_cfg_of(device_peps; chi_s, chi_dl), chi_dl)
end

Base.isopen(workspace::ZipupWorkspace)::Bool = workspace.handle != C_NULL

function Base.close(workspace::ZipupWorkspace)::Nothing
    isopen(workspace) || return nothing
    handle = workspace.handle
    workspace.handle = C_NULL
    _ffi_zipup_ctx_destroy(handle)
    return nothing
end

Base.copy(::ZipupWorkspace) = throw(ArgumentError("ZipupWorkspace cannot be copied"))

function _validate_zipup_workspace(
    workspace::ZipupWorkspace,
    device_peps::CuPeps,
    config::QnpepsConfig,
)::Nothing
    isopen(workspace) || throw(ArgumentError("ZipupWorkspace is closed"))
    workspace.config == config || throw(ArgumentError("ZipupWorkspace configuration mismatch"))
    workspace.maxdim == config.chi_dl || throw(ArgumentError("ZipupWorkspace rank mismatch"))
    CUDA.device() == workspace.device || throw(ArgumentError("ZipupWorkspace device mismatch"))
    CUDA.stream().handle == workspace.stream.handle ||
        throw(ArgumentError("ZipupWorkspace stream mismatch"))
    _peps_bytes(; config) == sizeof(ComplexF32) * length(device_peps.data) ||
        throw(ArgumentError("ZipupWorkspace PEPS layout mismatch"))
    return nothing
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

function _peps_row_elements(device_peps::CuPeps, row::Integer)::Int
    bond_up = row > 1 ? device_peps.dim_bond : 1
    bond_down = row < device_peps.lx ? device_peps.dim_bond : 1
    elements = 0
    for col in 1:device_peps.ly
        bond_left = col > 1 ? device_peps.dim_bond : 1
        bond_right = col < device_peps.ly ? device_peps.dim_bond : 1
        site_elements = device_peps.dim_phys * bond_up * bond_right * bond_down * bond_left
        elements += site_elements
    end
    return elements
end

function _peps_row_element_offsets(device_peps::CuPeps)::Vector{Int}
    offsets = Vector{Int}(undef, device_peps.lx)
    offset = 0
    for row in 1:device_peps.lx
        offsets[row] = offset
        offset += _peps_row_elements(device_peps, row)
    end
    offset == length(device_peps.data) || error("invalid packed PEPS row layout")
    return offsets
end

function _enqueue_peps_row!(
    workspace::ZipupWorkspace,
    row::Integer,
    peps_values::CuVector{ComplexF32},
    peps_offset::Integer,
    peps_elements::Integer,
    mps_dims::Union{Nothing,Vector{Int32}},
    mps_values::Union{Nothing,CuVector},
    output_dims::Vector{Int32},
    output_values::CuVector{UInt8},
)::Nothing
    (mps_dims === nothing) == (mps_values === nothing) ||
        throw(ArgumentError("MPS dimensions and values must either both be present or both absent"))
    length(output_dims) == 3 * workspace.config.ly ||
        throw(DimensionMismatch("PEPS-row zip-up output dimension buffer has the wrong size"))
    length(output_values) >= workspace.row_value_bytes ||
        throw(DimensionMismatch("PEPS-row zip-up output value buffer is too small"))
    0 <= peps_offset && peps_elements >= 1 && peps_offset + peps_elements <= length(peps_values) ||
        throw(BoundsError(peps_values, (peps_offset+1):(peps_offset+peps_elements)))

    mps_bytes = 0
    if mps_dims !== nothing
        length(mps_dims) == 3 * workspace.config.ly ||
            throw(DimensionMismatch("PEPS-row zip-up input dimension buffer has the wrong size"))
        mps_bytes = _grouped_mps_elements(mps_dims) * sizeof(ComplexF32)
        mps_bytes <= sizeof(eltype(mps_values)) * length(mps_values) ||
            throw(DimensionMismatch("PEPS-row zip-up input value buffer is too small"))
    end

    GC.@preserve peps_values mps_dims mps_values output_dims output_values begin
        args = QnpepsZipupPepsRowArgs(
            UInt32(sizeof(QnpepsZipupPepsRowArgs)),
            Int32(row),
            UInt(pointer(peps_values, peps_offset + 1)),
            UInt64(peps_elements * sizeof(ComplexF32)),
            mps_dims === nothing ? UInt(0) : UInt(pointer(mps_dims)),
            mps_values === nothing ? UInt(0) : UInt(pointer(mps_values)),
            UInt64(mps_bytes),
            UInt(pointer(output_dims)),
            UInt(pointer(output_values)),
            UInt64(length(output_values)),
        )
        _ffi_zipup_ctx_enqueue_peps_row(workspace.handle, args)
    end
    return nothing
end

function _enqueue_peps_row!(
    workspace::ZipupWorkspace,
    row::Integer,
    peps_values::CuVector{ComplexF32},
    mps_dims::Union{Nothing,Vector{Int32}},
    mps_values::Union{Nothing,CuVector},
    output_dims::Vector{Int32},
    output_values::CuVector{UInt8},
)::Nothing
    return _enqueue_peps_row!(
        workspace,
        row,
        peps_values,
        0,
        length(peps_values),
        mps_dims,
        mps_values,
        output_dims,
        output_values,
    )
end

function _pack_dlenv_rows(
    row_dims::Vector{Vector{Int32}},
    row_values::AbstractVector{<:CuVector{UInt8}},
    config::QnpepsConfig,
)::CuVector{UInt8}
    num_env_rows = config.lx - 1
    header = Vector{Int32}(undef, num_env_rows * config.ly * 4)
    header_offset = 0
    value_bytes = Vector{Int}(undef, num_env_rows)
    for env_row in 1:num_env_rows
        dims = row_dims[env_row]
        length(dims) == 3 * config.ly ||
            throw(DimensionMismatch("double-layer row $env_row has the wrong site count"))
        value_bytes[env_row] = _grouped_mps_elements(dims) * sizeof(ComplexF32)
        for col in 1:config.ly
            left, physical, right = Int.(view(dims, (3*col-2):(3*col)))
            vertical = isqrt(physical)
            vertical * vertical == physical || throw(
                DimensionMismatch(
                    "double-layer row $env_row site $col has non-square physical dimension $physical",
                ),
            )
            header[(header_offset+1):(header_offset+4)] .=
                Int32[left, vertical, vertical, right]
            header_offset += 4
        end
    end

    header_bytes = sizeof(Int32) * length(header)
    required_bytes = header_bytes + sum(value_bytes)
    capacity = _dlenv_bytes(; config)
    capacity >= required_bytes || error("packed double-layer rows exceed output capacity")
    data = CUDA.zeros(UInt8, capacity)
    host_header = collect(reinterpret(UInt8, header))
    copyto!(data, 1, host_header, 1, header_bytes)
    value_offset = header_bytes
    for env_row in 1:num_env_rows
        copyto!(data, value_offset + 1, row_values[env_row], 1, value_bytes[env_row])
        value_offset += value_bytes[env_row]
    end
    return data
end

function double_layer_rowwise(
    device_peps::CuPeps;
    chi_s::Integer=device_peps.dim_bond,
    chi_dl::Integer=device_peps.dim_bond,
    workspace::Union{Nothing,ZipupWorkspace}=nothing,
)::CuDlenv
    config = _cfg_of(device_peps; chi_s, chi_dl)
    owned_workspace = workspace === nothing
    active_workspace = workspace === nothing ? ZipupWorkspace(device_peps; chi_s, chi_dl) : workspace
    _validate_zipup_workspace(active_workspace, device_peps, config)

    try
        num_env_rows = device_peps.lx - 1
        peps_row_offsets = _peps_row_element_offsets(device_peps)
        row_dims = [zeros(Int32, 3 * device_peps.ly) for _ in 1:num_env_rows]
        row_values = [
            CUDA.zeros(UInt8, active_workspace.row_value_bytes) for _ in 1:num_env_rows
        ]
        scales = Vector{Float64}(undef, num_env_rows * device_peps.ly)

        _ffi_zipup_ctx_begin(active_workspace.handle)
        for env_row in num_env_rows:-1:1
            peps_row = env_row + 1
            peps_elements = _peps_row_elements(device_peps, peps_row)
            mps_dims = env_row == num_env_rows ? nothing : row_dims[env_row+1]
            mps_values = env_row == num_env_rows ? nothing : row_values[env_row+1]
            _enqueue_peps_row!(
                active_workspace,
                peps_row,
                device_peps.data,
                peps_row_offsets[peps_row],
                peps_elements,
                mps_dims,
                mps_values,
                row_dims[env_row],
                row_values[env_row],
            )
        end
        _ffi_zipup_ctx_finish(active_workspace.handle, scales)

        cumulative_row_logs = Vector{Float64}(undef, num_env_rows)
        cumulative_log = 0.0
        for step in 1:num_env_rows
            env_row = num_env_rows - step + 1
            scale_range = ((step-1)*device_peps.ly+1):(step*device_peps.ly)
            cumulative_log += sum(scale > 0.0 ? log(scale) : 0.0 for scale in scales[scale_range])
            cumulative_row_logs[env_row] = cumulative_log
        end
        data = _pack_dlenv_rows(row_dims, row_values, config)
        return CuDlenv(
            data,
            cumulative_row_logs,
            device_peps.lx,
            device_peps.ly,
            device_peps.dim_phys,
            device_peps.dim_bond,
            chi_s,
            chi_dl,
        )
    finally
        owned_workspace && close(active_workspace)
    end
end

function _dlenv_arrays(
    bytes::AbstractVector{UInt8},
    lx::Integer,
    ly::Integer,
)::Vector{Vector{Array{ComplexF32,4}}}
    num_env_rows = lx - 1
    num_sites = num_env_rows * ly
    header_elements = num_sites * 4
    header_bytes = header_elements * sizeof(Int32)
    length(bytes) >= header_bytes ||
        throw(DimensionMismatch("packed double-layer environment header is incomplete"))
    header = reinterpret(Int32, view(bytes, 1:header_bytes))
    values = reinterpret(ComplexF32, view(bytes, (header_bytes+1):length(bytes)))
    rows = Vector{Vector{Array{ComplexF32,4}}}(undef, num_env_rows)
    header_offset = 0
    values_offset = 0
    for env_row in 1:num_env_rows
        row = Vector{Array{ComplexF32,4}}(undef, ly)
        for col in 1:ly
            dims = Tuple(Int(header[header_offset+axis]) for axis in 1:4)
            header_offset += 4
            all(>(0), dims) ||
                throw(DimensionMismatch("packed double-layer dimensions must be positive"))
            site_elements = prod(dims)
            values_offset + site_elements <= length(values) ||
                throw(DimensionMismatch("packed double-layer environment values are incomplete"))
            values_range = (values_offset+1):(values_offset+site_elements)
            row[col] = reshape(Array{ComplexF32}(values[values_range]), dims)
            values_offset += site_elements
        end
        rows[env_row] = row
    end
    return rows
end

function _dlenv_arrays(dlenv::CuDlenv)::Vector{Vector{Array{ComplexF32,4}}}
    return _dlenv_arrays(Array(dlenv.data), dlenv.lx, dlenv.ly)
end

function _check_dlenv_grid(lx::Integer, ly::Integer, tensors::AbstractMatrix)::Nothing
    size(tensors) == (lx, ly) || throw(
        DimensionMismatch(
            "tensor grid $(size(tensors)) does not match double-layer grid " *
            "$(lx)x$(ly)",
        ),
    )
    return nothing
end

function _check_dlenv_grid(dlenv::CuDlenv, tensors::AbstractMatrix)::Nothing
    return _check_dlenv_grid(dlenv.lx, dlenv.ly, tensors)
end

function _materialize_dlenv_arrays(
    arrays::Vector{Vector{Array{ComplexF32,4}}},
    tensors::AbstractMatrix,
)::Vector{MPS}
    lx, ly = size(tensors)
    length(arrays) == lx - 1 ||
        throw(DimensionMismatch("double-layer row count does not match the tensor grid"))
    rows = Vector{MPS}(undef, lx - 1)
    for env_row in eachindex(rows)
        length(arrays[env_row]) == ly ||
            throw(DimensionMismatch("double-layer column count does not match the tensor grid"))
        vertical_bonds = [_vbond(tensors, env_row, col) for col in 1:ly]
        rows[env_row] = _env_arrays_to_mps(arrays[env_row], vertical_bonds)
    end
    return rows
end

function materialize_dlenv(
    dlenv::CuDlenv,
    tensors::AbstractMatrix,
    env_row::Integer,
)::MPS
    _check_dlenv_grid(dlenv, tensors)
    1 <= env_row <= dlenv.lx - 1 || throw(
        ArgumentError("env_row must be in 1:$(dlenv.lx-1) (got $env_row)"),
    )
    arrays = _dlenv_arrays(dlenv)[env_row]
    vertical_bonds = [_vbond(tensors, env_row, col) for col in 1:dlenv.ly]
    return _env_arrays_to_mps(arrays, vertical_bonds)
end

function materialize_dlenv(dlenv::CuDlenv, tensors::AbstractMatrix)::Vector{MPS}
    _check_dlenv_grid(dlenv, tensors)
    return _materialize_dlenv_arrays(_dlenv_arrays(dlenv), tensors)
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

function _double_layer_native(
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

function double_layer(
    device_peps::CuPeps;
    chi_s::Integer=device_peps.dim_bond,
    chi_dl::Integer=device_peps.dim_bond,
    workspace::Union{Nothing,ZipupWorkspace}=nothing,
)::CuDlenv
    return double_layer_rowwise(device_peps; chi_s, chi_dl, workspace)
end
