using ITensors
using CUDA

const AX_P = 1
const AX_U = 2
const AX_R = 3
const AX_D = 4
const AX_L = 5

struct PepsError <: Exception
    msg::String
end
function Base.showerror(io::IO, e::PepsError)::Nothing
    return print(io, "cuQuantumNaturalfPEPS PEPS error: ", e.msg)
end

struct Peps{T}
    tensors::Matrix{Array{T,5}}
end

Base.size(p::Peps)::Tuple{Int,Int} = size(p.tensors)
Base.size(p::Peps, d::Integer)::Int = size(p.tensors, d)

_lx(p::Peps)::Int = size(p.tensors, 1)
_ly(p::Peps)::Int = size(p.tensors, 2)
_dim_phys(p::Peps)::Int = size(p.tensors[1, 1], AX_P)
_dim_bond(p::Peps)::Int = size(p.tensors[1, 1], AX_R)

function Base.show(io::IO, p::Peps)::Nothing
    return print(io, "Peps(", _lx(p), "×", _ly(p), ", dim_bond=", _dim_bond(p), ")")
end

struct CuPeps
    data::CuArray{ComplexF32,1}
    lx::Int
    ly::Int
    dim_phys::Int
    dim_bond::Int
end

function _cfg_of(
    device_peps::CuPeps;
    chi_s::Integer,
    chi_dl::Integer=device_peps.dim_bond,
    seed::Integer=0,
)::QnpepsConfig
    return QnpepsConfig(
        lx=device_peps.lx,
        ly=device_peps.ly,
        dim_phys=device_peps.dim_phys,
        dim_bond=device_peps.dim_bond,
        chi_s=chi_s,
        chi_dl=chi_dl,
        seed=seed,
    )
end

function _phys_index(tensors::AbstractMatrix, row::Integer, col::Integer)::Index
    lx, ly = size(tensors)
    neighbors = ITensor[]
    row > 1 && push!(neighbors, tensors[row-1, col])
    col < ly && push!(neighbors, tensors[row, col+1])
    row < lx && push!(neighbors, tensors[row+1, col])
    col > 1 && push!(neighbors, tensors[row, col-1])
    return uniqueind(tensors[row, col], neighbors...)
end

function _site_array(tensors::AbstractMatrix, row::Integer, col::Integer)::Array{<:Number,5}
    lx, ly = size(tensors)
    site = tensors[row, col]
    T = eltype(site)
    phys = _phys_index(tensors, row, col)
    up = row > 1 ? commonind(site, tensors[row-1, col]) : nothing
    right = col < ly ? commonind(site, tensors[row, col+1]) : nothing
    down = row < lx ? commonind(site, tensors[row+1, col]) : nothing
    left = col > 1 ? commonind(site, tensors[row, col-1]) : nothing
    legs = (phys, up, right, down, left)
    present = Tuple(leg for leg in legs if leg !== nothing)
    dims = Tuple(leg === nothing ? 1 : dim(leg) for leg in legs)
    raw = ITensors.array(site, present...)
    return reshape(Array{T}(raw), dims)
end

function _itensor_grid_to_arrays(tensors::AbstractMatrix)::Matrix{<:Array{<:Number,5}}
    lx, ly = size(tensors)
    return [_site_array(tensors, row, col) for row in 1:lx, col in 1:ly]
end

function _validate(arrays::AbstractMatrix)::Nothing
    lx, ly = size(arrays)
    (lx >= 2 && ly >= 2) || throw(PepsError("PEPS must be at least 2x2 (got $(size(arrays)))"))
    dim_phys = size(arrays[1, 1], AX_P)
    for row in 1:lx, col in 1:ly
        A = arrays[row, col]
        if ndims(A) != 5
            throw(PepsError("site ($row,$col) is rank $(ndims(A)), expected 5 [p,u,r,d,l]"))
        end
        if size(A, AX_P) != dim_phys
            throw(PepsError("site ($row,$col) physical dim must be uniform ($dim_phys)"))
        end
    end
    for col in 1:ly
        size(arrays[1, col], AX_U) == 1 || throw(PepsError("up leg of top site (1,$col) must be 1"))
        if size(arrays[lx, col], AX_D) != 1
            throw(PepsError("down leg of bottom site ($lx,$col) must be 1"))
        end
    end
    for row in 1:lx
        if size(arrays[row, 1], AX_L) != 1
            throw(PepsError("left leg of left site ($row,1) must be 1"))
        end
        if size(arrays[row, ly], AX_R) != 1
            throw(PepsError("right leg of right site ($row,$ly) must be 1"))
        end
    end
    for row in 1:lx, col in 1:(ly-1)
        if size(arrays[row, col], AX_R) != size(arrays[row, col+1], AX_L)
            throw(PepsError("horizontal bond ($row,$col) right != left"))
        end
    end
    for row in 1:(lx-1), col in 1:ly
        if size(arrays[row, col], AX_D) != size(arrays[row+1, col], AX_U)
            throw(PepsError("vertical bond ($row,$col) down != up"))
        end
    end
    D = size(arrays[1, 1], AX_R)
    for row in 1:lx, col in 1:(ly-1)
        actual = size(arrays[row, col], AX_R)
        if actual != D
            throw(PepsError("horizontal bond ($row,$col) dim $actual != uniform dim_bond $D"))
        end
    end
    for row in 1:(lx-1), col in 1:ly
        actual = size(arrays[row, col], AX_D)
        if actual != D
            throw(PepsError("vertical bond ($row,$col) dim $actual != uniform dim_bond $D"))
        end
    end
    return nothing
end

function load_peps(tensors::AbstractMatrix)::Peps
    arrays = if tensors[1, 1] isa ITensor
        _itensor_grid_to_arrays(tensors)
    else
        Matrix{Array{eltype(tensors[1, 1]),5}}(tensors)
    end
    _validate(arrays)
    return Peps(arrays)
end

_to_device_order(A)::Array{<:Number,5} = permutedims(A, (5, 4, 3, 2, 1))

function upload_peps(peps::Peps)::CuPeps
    lx, ly = size(peps)
    buf = ComplexF32[]
    for row in 1:lx, col in 1:ly
        append!(buf, vec(ComplexF32.(_to_device_order(peps.tensors[row, col]))))
    end
    config = QnpepsConfig(
        lx=lx,
        ly=ly,
        dim_bond=_dim_bond(peps),
        chi_s=_dim_bond(peps),
        chi_dl=_dim_bond(peps),
        dim_phys=_dim_phys(peps),
    )
    packed_bytes = sizeof(ComplexF32) * length(buf)
    expected_bytes = _peps_bytes(config)
    if packed_bytes != expected_bytes
        throw(PepsError("packed PEPS is $packed_bytes bytes but C layout expects $expected_bytes"))
    end
    return CuPeps(CuArray(buf), lx, ly, _dim_phys(peps), _dim_bond(peps))
end
