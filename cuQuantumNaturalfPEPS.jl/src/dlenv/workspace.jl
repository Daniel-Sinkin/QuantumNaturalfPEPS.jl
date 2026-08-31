
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

Base.@kwdef struct _ZipupMpsInput
    dims::Union{Nothing,Vector{Int32}}
    values::Union{Nothing,CuVector}
end

Base.@kwdef struct _ZipupRowOutput
    dims::Vector{Int32}
    values::CuVector{UInt8}
end

Base.@kwdef struct _ZipupPepsRowArguments
    workspace::ZipupWorkspace
    row::Integer
    peps_values::CuVector{ComplexF32}
    peps_offset::Integer
    peps_elements::Integer
    mps::_ZipupMpsInput
    output::_ZipupRowOutput
end

function _ZipupWorkspace(config::QnpepsConfig, maxdim::Integer)
    row_bytes = _zipup_peps_row_bytes(; config, maxdim)
    row_bytes >= 0 && row_bytes % sizeof(ComplexF32) == 0 || throw(
        ArgumentError("invalid PEPS-row zip-up configuration: config=$config, maxdim=$maxdim"),
    )

    device = CUDA.device()
    stream = CUDA.stream()
    handle = _ffi_zipup_ctx_create(; config, maxdim, stream=Ptr{Cvoid}(stream.handle))
    workspace = ZipupWorkspace(handle, config, Int(maxdim), row_bytes, device, stream)
    finalizer(close, workspace)
    return workspace
end

function ZipupWorkspace(
    device_peps::CuPeps;
    chi_s::Integer=device_peps.dim_bond,
    chi_dl::Integer=device_peps.dim_bond,
    route=:default,
    density_cutoff::Real=1.0e-13,
)
    return _ZipupWorkspace(
        _cfg_of(
            device_peps;
            chi_s,
            chi_dl,
            dlenv_truncation_route=route,
            dlenv_density_cutoff=density_cutoff,
        ),
        chi_dl,
    )
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

function _enqueue_peps_row!(arguments::_ZipupPepsRowArguments)::Nothing
    workspace = arguments.workspace
    row = arguments.row
    peps_values = arguments.peps_values
    peps_offset = arguments.peps_offset
    peps_elements = arguments.peps_elements
    mps_dims = arguments.mps.dims
    mps_values = arguments.mps.values
    output_dims = arguments.output.dims
    output_values = arguments.output.values
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
        args = QnpepsZipupPepsRowArgs(;
            struct_size=UInt32(sizeof(QnpepsZipupPepsRowArgs)),
            row=Int32(row),
            peps_row=UInt(pointer(peps_values, peps_offset + 1)),
            peps_row_bytes=UInt64(peps_elements * sizeof(ComplexF32)),
            mps_dims=mps_dims === nothing ? UInt(0) : UInt(pointer(mps_dims)),
            mps_values=mps_values === nothing ? UInt(0) : UInt(pointer(mps_values)),
            mps_bytes=UInt64(mps_bytes),
            output_dims=UInt(pointer(output_dims)),
            output_values=UInt(pointer(output_values)),
            output_bytes=UInt64(length(output_values)),
        )
        _ffi_zipup_ctx_enqueue_peps_row(workspace.handle, args)
    end
    return nothing
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
            header[(header_offset+1):(header_offset+4)] .= Int32[left, vertical, vertical, right]
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
