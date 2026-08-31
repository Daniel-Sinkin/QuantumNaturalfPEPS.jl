using CUDA

const _DLENV_HOST_LANES = 2
const _DLENV_ARENA_ALIGNMENT = 256

struct DlenvArenaPlan
    row_value_bytes::Int
    packed_bytes::Int
    row_value_offsets::Vector{Int}
    lane_offsets::NTuple{_DLENV_HOST_LANES,Int}
    total_bytes::Int
end

Base.@kwdef mutable struct DlenvHost{B,W,P,D,S}
    config::QnpepsConfig
    plan::DlenvArenaPlan
    workspace::W
    arena::B
    peps_data::P
    row_values::Vector{B}
    row_value_pointers::Vector{CUDA.CuPtr{UInt8}}
    lane_buffers::NTuple{_DLENV_HOST_LANES,B}
    lane_buffer_pointers::NTuple{_DLENV_HOST_LANES,CUDA.CuPtr{UInt8}}
    lanes::NTuple{_DLENV_HOST_LANES,CuDlenv}
    row_dims::Vector{Vector{Int32}}
    row_bytes::Vector{Int}
    value_offsets::Vector{Int}
    header::Vector{UInt8}
    scales::Vector{Float64}
    peps_offsets::Vector{Int}
    peps_elements::Vector{Int}
    row_args::Vector{QnpepsZipupPepsRowArgs}
    lane_warmed::Vector{Bool}
    graphs::Vector{Union{Nothing,CUDA.CuGraphExec}}
    device::D
    stream::S
    generation::Int
    active_lane::Int
    build_count::Int
    layout_ready::Bool
    args_ready::Bool
    capture_enabled::Bool
    capture_state::Symbol
    capture_reason::Symbol
    open::Bool
end

@inline _align_dlenv_arena(bytes::Int)::Int =
    (bytes + _DLENV_ARENA_ALIGNMENT - 1) & -_DLENV_ARENA_ALIGNMENT

function plan_dlenv_host(
    device_peps::CuPeps;
    chi_s::Integer=device_peps.dim_bond,
    chi_dl::Integer=device_peps.dim_bond,
    route=:default,
    density_cutoff::Real=1.0e-13,
)::DlenvArenaPlan
    config = _cfg_of(
        device_peps;
        chi_s,
        chi_dl,
        dlenv_truncation_route=route,
        dlenv_density_cutoff=density_cutoff,
    )
    row_value_bytes = Int(_zipup_peps_row_bytes(; config, maxdim=chi_dl))
    packed_bytes = Int(_dlenv_bytes(; config))
    row_value_bytes >= 0 || throw(ArgumentError("invalid dl-env row size"))
    packed_bytes >= 0 || throw(ArgumentError("invalid packed dl-env size"))
    num_rows = device_peps.lx - 1
    row_value_offsets = Vector{Int}(undef, num_rows)
    cursor = 0
    for row in 1:num_rows
        cursor = _align_dlenv_arena(cursor)
        row_value_offsets[row] = cursor
        cursor += row_value_bytes
    end
    lane_offsets = ntuple(_DLENV_HOST_LANES) do _
        cursor = _align_dlenv_arena(cursor)
        offset = cursor
        cursor += packed_bytes
        offset
    end
    return DlenvArenaPlan(
        row_value_bytes,
        packed_bytes,
        row_value_offsets,
        lane_offsets,
        _align_dlenv_arena(cursor),
    )
end

function _arena_view(arena::B, offset::Int, bytes::Int)::B where {B<:CuVector{UInt8}}
    ptr = CuPtr{UInt8}(pointer(arena) + offset)
    return Base.unsafe_wrap(B, ptr, bytes; own=false)
end

function _carve_dlenv_arena(arena::B, plan::DlenvArenaPlan) where {B<:CuVector{UInt8}}
    rows = B[_arena_view(arena, offset, plan.row_value_bytes) for offset in plan.row_value_offsets]
    lanes = ntuple(_DLENV_HOST_LANES) do lane
        _arena_view(arena, plan.lane_offsets[lane], plan.packed_bytes)
    end
    return rows, lanes
end
