
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
            "tensor grid $(size(tensors)) does not match double-layer grid " * "$(lx)x$(ly)",
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

function materialize_dlenv(dlenv::CuDlenv, tensors::AbstractMatrix, env_row::Integer)::MPS
    _check_dlenv_grid(dlenv, tensors)
    1 <= env_row <= dlenv.lx - 1 ||
        throw(ArgumentError("env_row must be in 1:$(dlenv.lx-1) (got $env_row)"))
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
    sampler_truncation_route=:default,
    sampler_density_cutoff::Real=1.0e-3,
    projected_density_cutoff::Real=1.0e-4,
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
        sampler_truncation_route=sampler_truncation_route,
        sampler_density_cutoff=sampler_density_cutoff,
        projected_density_cutoff=projected_density_cutoff,
    )
end
