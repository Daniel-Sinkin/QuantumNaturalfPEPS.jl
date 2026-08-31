
struct _MinsrHostCpuSet
    words::NTuple{16,UInt64}
end

Base.@kwdef mutable struct MinsrHostInputs
    samples::UInt = UInt(0)
    samples_bytes::UInt64 = UInt64(0)
    logpsi::UInt = UInt(0)
    logpsi_bytes::UInt64 = UInt64(0)
    e_loc::UInt = UInt(0)
    e_loc_bytes::UInt64 = UInt64(0)
    logq::UInt = UInt(0)
    logq_bytes::UInt64 = UInt64(0)
    row_shards::NTuple{MINSR_HOST_LANES,UInt} = ntuple(_ -> UInt(0), MINSR_HOST_LANES)
    row_shard_bytes::NTuple{MINSR_HOST_LANES,UInt64} = ntuple(_ -> UInt64(0), MINSR_HOST_LANES)
    theta_dot::UInt = UInt(0)
    theta_dot_bytes::UInt64 = UInt64(0)
    relative_cut::Float64 = 0.0
    absolute_cut::Float64 = 0.0
end

mutable struct _MinsrHostPeerInputs
    samples::CUDA.CuPtr{Cvoid}
    logpsi::CUDA.CuPtr{Cvoid}
    e_loc::CUDA.CuPtr{Cvoid}
    logq::CUDA.CuPtr{Cvoid}
    row_shards::NTuple{MINSR_HOST_LANES,CUDA.CuPtr{Cvoid}}
    theta_dot::CUDA.CuPtr{Cvoid}
end

function _MinsrHostPeerInputs()
    null = CUDA.CuPtr{Cvoid}(UInt(0))
    return _MinsrHostPeerInputs(null, null, null, null, (null, null, null, null), null)
end

Base.@kwdef mutable struct MinsrHostStats
    lanes::Int32 = Int32(MINSR_HOST_LANES)
    peer_pairs_enabled::Int32 = Int32(0)
    failed_lane::Int32 = Int32(-1)
    peer_tile_bytes::UInt64 = MINSR_HOST_PEER_TILE_BYTES
    row_counts::Vector{Int64} = zeros(Int64, MINSR_HOST_LANES)
    logical_devices::Vector{Int32} = zeros(Int32, MINSR_HOST_LANES)
    numa_nodes::Vector{Int32} = collect(MINSR_HOST_TOPOLOGY_JURECA.numa_nodes)
    affinity_applied::Vector{Int32} = zeros(Int32, MINSR_HOST_LANES)
    lane_status::Vector{Int32} = zeros(Int32, MINSR_HOST_LANES)
    arena_bytes::Vector{UInt64} = zeros(UInt64, MINSR_HOST_LANES)
    input_copy_s::Vector{Float64} = zeros(Float64, MINSR_HOST_LANES)
    gram_s::Vector{Float64} = zeros(Float64, MINSR_HOST_LANES)
    minsr_s::Vector{Float64} = zeros(Float64, MINSR_HOST_LANES)
    complete_s::Vector{Float64} = zeros(Float64, MINSR_HOST_LANES)
    e_mean::Vector{Float64} = zeros(Float64, 2)
    e_var::Float64 = 0.0
    ess::Float64 = 0.0
end

function MinsrHostStats(topology::MinsrHostTopology)
    return MinsrHostStats(; numa_nodes=collect(topology.numa_nodes))
end

Base.@kwdef mutable struct _MinsrHostLane
    device::CUDA.CuDevice
    context::CUDA.CuContext
    stream::CUDA.CUstream
    gram::Ptr{Cvoid}
    minsr::Ptr{Cvoid}
    arena::CUDA.DeviceMemory
    arena_bytes::UInt64
    samples::CUDA.CuPtr{Cvoid}
    logpsi::CUDA.CuPtr{Cvoid}
    e_loc::CUDA.CuPtr{Cvoid}
    logq::CUDA.CuPtr{Cvoid}
    rows::CUDA.CuPtr{Cvoid}
    raw_gram::CUDA.CuPtr{Cvoid}
    theta::CUDA.CuPtr{Cvoid}
    statistics::Vector{Float64}
    statistics_pointer::UInt
    gram_args::Base.RefValue{_MinsrHostGramArgs}
    minsr_args::Base.RefValue{_MinsrHostMinsrArgs}
    original_affinity::_MinsrHostCpuSet
    affinity_saved::Int32
    affinity_applied::Int32
    status::Int32
    failure_location::String
    failure_message::String
    input_copy_s::Float64
    gram_s::Float64
    minsr_s::Float64
    complete_s::Float64
end

mutable struct MinsrHost
    lx::Int32
    ly::Int32
    dim_phys::Int32
    dim_bond::Int32
    n_samples::Int64
    sites::Int64
    compact::Int64
    dense::Int64
    topology::MinsrHostTopology
    row_bases::NTuple{MINSR_HOST_LANES,Int64}
    row_counts::NTuple{MINSR_HOST_LANES,Int64}
    lanes::Vector{_MinsrHostLane}
    workers::Vector{Task}
    inputs::MinsrHostInputs
    peer_inputs::_MinsrHostPeerInputs
    stats::MinsrHostStats
    command::Threads.Atomic{Int32}
    epoch::Threads.Atomic{Int}
    ready::Threads.Atomic{Int}
    done::Threads.Atomic{Int}
    closed::Threads.Atomic{Bool}
    running::Threads.Atomic{Bool}
end

Base.@kwdef struct _MinsrHostDescriptor
    lx::Int32
    ly::Int32
    dim_phys::Int32
    dim_bond::Int32
    n_samples::Int64
    sites::Int64
    compact::Int64
    dense::Int64
    topology::MinsrHostTopology
end

Base.@kwdef struct _MinsrHostLanes
    row_bases::NTuple{MINSR_HOST_LANES,Int64}
    row_counts::NTuple{MINSR_HOST_LANES,Int64}
    lanes::Vector{_MinsrHostLane}
    workers::Vector{Task}
end

Base.@kwdef struct _MinsrHostRuntime
    inputs::MinsrHostInputs
    peer_inputs::_MinsrHostPeerInputs
    stats::MinsrHostStats
    command::Threads.Atomic{Int32}
    epoch::Threads.Atomic{Int}
    ready::Threads.Atomic{Int}
    done::Threads.Atomic{Int}
    closed::Threads.Atomic{Bool}
    running::Threads.Atomic{Bool}
end

Base.@kwdef struct _MinsrHostState
    descriptor::_MinsrHostDescriptor
    lanes::_MinsrHostLanes
    runtime::_MinsrHostRuntime
end

function MinsrHost(state::_MinsrHostState)
    descriptor =
        ntuple(index -> getfield(state.descriptor, index), fieldcount(_MinsrHostDescriptor))
    lanes = ntuple(index -> getfield(state.lanes, index), fieldcount(_MinsrHostLanes))
    runtime = ntuple(index -> getfield(state.runtime, index), fieldcount(_MinsrHostRuntime))
    values = (descriptor..., lanes..., runtime...)
    return MinsrHost(values...)
end

struct MinsrHostError <: Exception
    lane::Int32
    status::Int32
    location::String
    message::String
end

MinsrHostError(lane::Int32, status::Int32) = MinsrHostError(lane, status, "", "")

function Base.showerror(output_stream::IO, error::MinsrHostError)
    print(output_stream, "minSR Julia host failed lane=", error.lane, " status=", error.status)
    isempty(error.location) || print(output_stream, " at ", error.location)
    isempty(error.message) || print(output_stream, "; ", error.message)
end

@inline _minsr_host_seconds() = Float64(time_ns()) * 1.0e-9
