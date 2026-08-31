using CUDA
using LinearAlgebra
using Random

function _dl_permreshape(A::AbstractArray, perm::NTuple{N,Int}, dims) where {N}
    return issorted(perm) ? reshape(A, dims) : reshape(permutedims(A, perm), dims)
end


function _dl_tcon(
    A::AbstractArray{T},
    axes_a::NTuple{NC,Int},
    B::AbstractArray{T},
    axes_b::NTuple{NC,Int};
    conj_b::Bool=false,
) where {T,NC}
    free_a = Tuple(i for i in 1:ndims(A) if !(i in axes_a))
    free_b = Tuple(i for i in 1:ndims(B) if !(i in axes_b))

    left_dim = prod(i -> size(A, i), free_a; init=1)
    contract_dim = prod(i -> size(A, i), axes_a; init=1)
    right_dim = prod(i -> size(B, i), free_b; init=1)

    a_mat = _dl_permreshape(A, (free_a..., axes_a...), (left_dim, contract_dim))
    b_mat = _dl_permreshape(B, (axes_b..., free_b...), (contract_dim, right_dim))

    product = a_mat * (conj_b ? conj(b_mat) : b_mat)
    out_dims = ((size(A, i) for i in free_a)..., (size(B, i) for i in free_b)...)
    return reshape(product, isempty(out_dims) ? (1,) : out_dims)
end

_dl_sketch(rng::AbstractRNG, ::Type{Complex{R}}, n::Integer, k::Integer) where {R} =
    randn(rng, Complex{R}, n, k)

function _dl_sketch(rng::CUDA.RNG, ::Type{Complex{R}}, n::Integer, k::Integer) where {R}
    Random.seed!(rng, UInt64(hash((0x777, n, k)) % typemax(UInt64)))
    real_part = randn(rng, R, n, k)
    imag_part = randn(rng, R, n, k)
    return complex.(real_part, imag_part)
end

function _dl_qr_q(Y::Matrix{T}) where {T}
    rank = size(Y, 2)
    factorization = qr(Y)
    return Matrix(factorization.Q)[:, 1:rank]
end

function _dl_qr_q(Y::CuMatrix{T}) where {T}
    reflectors, tau = CUDA.CUSOLVER.geqrf!(Y)
    CUDA.CUSOLVER.orgqr!(reflectors, tau)
    return reflectors
end

function _dl_rangefinder(sketch, matrix::AbstractMatrix{T}, maxdim::Integer) where {T}
    rows, cols = size(matrix)
    rank = clamp(min(maxdim, rows), 1, cols)
    omega = _dl_sketch(sketch, T, cols, rank)
    y = matrix * omega
    for _ in 1:2
        z = matrix' * y
        y = matrix * z
    end
    q = _dl_qr_q(y)
    r = q' * matrix
    return q, r, rank
end

_dl_factorize_rangefinder(sketch) = (matrix, maxdim) -> _dl_rangefinder(sketch, matrix, maxdim)

function _dl_qr_thin(matrix::Matrix{T}) where {T}
    rank = min(size(matrix)...)
    factorization = qr(matrix)
    return Matrix(factorization.Q)[:, 1:rank]
end

function _dl_exact_qr(matrix::AbstractMatrix, maxdim::Integer)
    q = _dl_qr_thin(matrix)
    return q, q' * matrix, size(q, 2)
end

_dl_absmax_l1(r::AbstractArray)::Float64 =
    maximum(z -> abs(Float64(real(z))) + abs(Float64(imag(z))), r)

function _dl_scale_column!(r::AbstractMatrix{Complex{R}})::Float64 where {R}
    scale = _dl_absmax_l1(r)
    if scale > 0.0 && isfinite(scale)
        r .*= R(1.0 / scale)
    end
    return scale
end

function _dl_ones_scalar(template::AbstractArray, ::Type{T}) where {T}
    return fill!(similar(template, T, (1, 1, 1, 1)), one(T))
end

function _dl_row(
    row_kets::AbstractVector,
    env_below::Union{Nothing,AbstractVector},
    maxdim::Integer,
    factorize,
)
    T = eltype(row_kets[1])
    ly = length(row_kets)

    envs = Vector{AbstractArray{T,4}}(undef, ly)
    scales = Vector{Float64}(undef, ly)

    carried_r = _dl_ones_scalar(row_kets[1], T)
    trivial_env = _dl_ones_scalar(row_kets[1], T)

    for col in 1:ly
        ket = row_kets[col]
        env = env_below === nothing ? trivial_env : env_below[col]
        left_environment = _dl_tcon(carried_r, (4,), env, (1,))
        left_environment_ket = _dl_tcon(left_environment, (2, 4), ket, (5, 4))
        column_tensor = _dl_tcon(left_environment_ket, (2, 3, 5), ket, (5, 4, 1); conj_b=true)
        bond_left, bond_right, ket_vertical, ket_horizontal, bra_vertical, bra_horizontal =
            size(column_tensor)
        matrixized = permutedims(column_tensor, (1, 3, 5, 2, 4, 6))
        rows = bond_left * ket_vertical * bra_vertical
        cols = bond_right * ket_horizontal * bra_horizontal
        matrix = reshape(matrixized, rows, cols)
        q, r, rank = factorize(matrix, maxdim)
        scales[col] = _dl_scale_column!(r)
        envs[col] = reshape(q, bond_left, ket_vertical, bra_vertical, rank)
        r_folded = reshape(r, rank, bond_right, ket_horizontal, bra_horizontal)
        carried_r = permutedims(r_folded, (1, 3, 4, 2))
    end

    folded = _dl_tcon(envs[ly], (4,), carried_r, (1,))
    last_size = size(envs[ly])
    envs[ly] = reshape(folded, last_size[1], last_size[2], last_size[3], 1)
    return envs, scales
end

function _dl_stack(sites::AbstractMatrix, maxdim::Integer, factorize)
    lx, ly = size(sites)
    lx >= 2 || throw(ArgumentError("double-layer envs need at least 2 rows (got $lx)"))
    num_env_rows = lx - 1
    envs = Vector{Vector{AbstractArray{eltype(sites[1, 1]),4}}}(undef, num_env_rows)
    scales_all = Matrix{Float64}(undef, num_env_rows, ly)

    bottom, bottom_scales = _dl_row([sites[lx, col] for col in 1:ly], nothing, maxdim, factorize)
    envs[num_env_rows] = bottom
    scales_all[1, :] = bottom_scales

    build_step = 1
    for peps_row in (lx-1):-1:2
        row_env, row_scales =
            _dl_row([sites[peps_row, col] for col in 1:ly], envs[peps_row], maxdim, factorize)
        envs[peps_row-1] = row_env
        scales_all[build_step+1, :] = row_scales
        build_step += 1
    end

    cumulative_row_logs = zeros(Float64, num_env_rows)
    total_log = 0.0
    for step in 1:num_env_rows
        for col in 1:ly
            scale = scales_all[step, col]
            isfinite(scale) ||
                throw(ErrorException("non-finite dl-env scale at build step $step col $col"))
            scale > 0.0 && (total_log += log(scale))
        end
        cumulative_row_logs[num_env_rows-step+1] = total_log
    end
    return envs, cumulative_row_logs
end

struct CuDlenvBuilt
    data::CuArray{UInt8,1}
    cumulative_row_logs::Vector{Float64}
    envs::Vector{Vector{AbstractArray{ComplexF32,4}}}
    lx::Int
    ly::Int
    dim_phys::Int
    dim_bond::Int
    chi_s::Int
    chi_dl::Int
end

function _dl_pack_buffer(envs::AbstractVector, lx::Integer, ly::Integer)::CuArray{UInt8,1}
    num_env_rows = lx - 1
    nsites = num_env_rows * ly

    header = Vector{Int32}(undef, nsites * 4)

    site_order = AbstractArray[]
    total_values = 0
    site_index = 0
    for env_row in 1:num_env_rows
        for col in 1:ly
            site = envs[env_row][col]
            header[(4*site_index+1):(4*site_index+4)] .= Int32.(size(site))
            total_values += length(site)
            push!(site_order, site)
            site_index += 1
        end
    end

    header_bytes = nsites * 4 * sizeof(Int32)
    buffer = CUDA.zeros(UInt8, header_bytes + total_values * sizeof(ComplexF32))
    copyto!(buffer, 1, collect(reinterpret(UInt8, header)), 1, header_bytes)

    values_view = reinterpret(ComplexF32, buffer)
    offset = header_bytes ÷ sizeof(ComplexF32)

    for site in site_order
        flat = vec(site)
        copyto!(view(values_view, (offset+1):(offset+length(flat))), flat)
        offset += length(flat)
    end

    return buffer
end

function build_dlenv_device(
    peps::Peps;
    chi_s::Integer=_dim_bond(peps),
    chi_dl::Integer=_dim_bond(peps),
)::CuDlenvBuilt
    lx, ly = size(peps)

    dim_bond = _dim_bond(peps)
    dim_phys = _dim_phys(peps)

    maxdim = min(chi_dl, dim_bond * dim_bond)
    sites = [CuArray(ComplexF32.(peps.tensors[row, col])) for row in 1:lx, col in 1:ly]

    factorize = _dl_factorize_rangefinder(CUDA.default_rng())
    envs, cumulative_row_logs = _dl_stack(sites, maxdim, factorize)

    data = _dl_pack_buffer(envs, lx, ly)
    typed_envs = Vector{Vector{AbstractArray{ComplexF32,4}}}(undef, lx - 1)

    for env_row in 1:(lx-1)
        typed_envs[env_row] = envs[env_row]
    end

    return CuDlenvBuilt(
        data,
        cumulative_row_logs,
        typed_envs,
        lx,
        ly,
        dim_phys,
        dim_bond,
        chi_s,
        chi_dl,
    )
end

function build_dlenv_host(
    peps::Peps;
    chi_s::Integer=_dim_bond(peps),
    chi_dl::Integer=_dim_bond(peps),
)
    lx, ly = size(peps)
    dim_bond = _dim_bond(peps)
    maxdim = min(chi_dl, dim_bond * dim_bond)

    sites = [peps.tensors[row, col] for row in 1:lx, col in 1:ly]

    return _dl_stack(sites, maxdim, _dl_factorize_rangefinder(MersenneTwister(0)))
end
