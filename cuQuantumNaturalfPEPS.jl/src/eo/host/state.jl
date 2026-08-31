
Base.@kwdef struct EoHostBindingState
    lane::Int32
    handle::UInt
    stream::UInt
    arena::UInt
    arena_bytes::UInt64
    peps::UInt
    samples::UInt
    logpsi::UInt
    e_loc::UInt
    rows::UInt
    epoch::UInt64
end

Base.@kwdef struct EoHostLifecycleState
    contexts_created::Int64
    contexts_destroyed::Int64
    arenas_created::Int64
    arenas_freed::Int64
    streams_created::Int64
    streams_destroyed::Int64
    replacements::Int64
    gc_runs::Int64
    teardown_order::NTuple{6,Int32}
    refs_cleared::Bool
    open::Bool
    sealed::Bool
end

Base.@kwdef mutable struct _EoHostBinding
    arena::CUDA.DeviceMemory
    arena_bytes::UInt64
    peps::CUDA.CuPtr{Cvoid}
    samples::CUDA.CuPtr{Cvoid}
    logpsi::CUDA.CuPtr{Cvoid}
    e_loc::CUDA.CuPtr{Cvoid}
    rows::CUDA.CuPtr{Cvoid}
    args::Base.RefValue{_EoHostRunArgs}
    epoch::UInt64
end

Base.@kwdef mutable struct _EoHostLane
    context::CUDA.CuContext
    stream::CUDA.CUstream
    handle::Ptr{Cvoid}
    binding::_EoHostBinding
    source_sample_pointer::CUDA.CuPtr{Cvoid}
    destination_logpsi_pointer::CUDA.CuPtr{Cvoid}
    destination_e_loc_pointer::CUDA.CuPtr{Cvoid}
    row_base::Int64
    row_count::Int64
    original_affinity::NTuple{16,UInt64}
    affinity_saved::Int32
    affinity_applied::Int32
    status::Int32
    failure_location::String
    failure_message::String
end

Base.@kwdef struct _EoHostLaneGeometry
    peps_elements::Int
    sites::Int64
    compact::Int64
    row_base::Int64
    row_count::Int64
end

Base.@kwdef struct _EoHostLaneDestinations
    samples::CUDA.CuPtr{Cvoid}
    logpsi::CUDA.CuPtr{Cvoid}
    e_loc::CUDA.CuPtr{Cvoid}
    rows::Ptr{Cvoid}
end

Base.@kwdef struct _EoHostLaneArguments
    context::CUDA.CuContext
    config_ref::Base.RefValue{QnpepsElocConfig}
    table::EoTermTable
    geometry::_EoHostLaneGeometry
    destinations::_EoHostLaneDestinations
    selector::EoSelector
end

Base.@kwdef struct EoHostArguments
    peps::CuVector{ComplexF32}
    config::QnpepsElocConfig
    table::EoTermTable
    n_samples::Integer
    compact::Integer
    samples::CuVector{UInt8}
    logpsi::CuVector{Float64}
    e_loc::CuVector{Float64}
    rows::Vector{ComplexF32}
end

Base.@kwdef mutable struct EoHost
    config::QnpepsElocConfig
    config_ref::Base.RefValue{QnpepsElocConfig}
    term_table::Union{Nothing,EoTermTable}
    selector::EoSelector
    source_peps::Union{Nothing,CuVector{ComplexF32}}
    source_samples::Union{Nothing,CuVector{UInt8}}
    destination_logpsi::Union{Nothing,CuVector{Float64}}
    destination_e_loc::Union{Nothing,CuVector{Float64}}
    destination_rows::Union{Nothing,Vector{ComplexF32}}
    source_context::CUDA.CuContext
    source_peps_pointer::CUDA.CuPtr{Cvoid}
    source_samples_pointer::CUDA.CuPtr{Cvoid}
    destination_logpsi_pointer::CUDA.CuPtr{Cvoid}
    destination_e_loc_pointer::CUDA.CuPtr{Cvoid}
    destination_rows_pointer::Ptr{Cvoid}
    source_peps_bytes::UInt64
    n_samples::Int64
    sites::Int64
    compact::Int64
    row_bases::NTuple{EO_HOST_LANES,Int64}
    row_counts::NTuple{EO_HOST_LANES,Int64}
    lanes::Vector{_EoHostLane}
    peer_pairs_enabled::Int32
    source_generation::Int64
    contexts_created::Int64
    contexts_destroyed::Int64
    arenas_created::Int64
    arenas_freed::Int64
    streams_created::Int64
    streams_destroyed::Int64
    replacements::Int64
    gc_runs::Int64
    teardown_order::Vector{Int32}
    refs_cleared::Bool
    sealed::Bool
    closed::Bool
end
