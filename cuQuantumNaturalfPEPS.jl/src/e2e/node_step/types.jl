const QNP = cuQuantumNaturalfPEPS
const NODE_STEP_LANES = 4
const _NODE_STEP_EO_RUN = Int32(1)
const _NODE_STEP_EO_STOP = Int32(2)
const _NODE_STEP_OK = Int32(0)
const _NODE_STEP_EO_ROWS = UInt32(1)

Base.@kwdef struct _NodeStepElocArgs
    struct_size::UInt32
    device_peps::CUDA.CuPtr{Cvoid}
    device_samples::CUDA.CuPtr{UInt8}
    logpsi_out::CUDA.CuPtr{Float64}
    e_loc_out::CUDA.CuPtr{Float64}
    o_rows_dev::CUDA.CuPtr{Cvoid}
    o_rows_host::Ptr{Cvoid}
    gram::CUDA.CuPtr{Cvoid}
    lambda::Float64
    j2_mode::UInt32
    j2_draw::UInt32
    j2_seed::UInt64
    j2_epoch::UInt64
end

Base.@kwdef struct _NodeStepEoGeometry
    peps_elements::Int
    sites::Int64
    compact::Int64
    row_base::Int64
    row_count::Int64
end

Base.@kwdef struct _NodeStepEoOutputs
    samples::CUDA.CuPtr{Cvoid}
    logpsi::CUDA.CuPtr{Cvoid}
    e_loc::CUDA.CuPtr{Cvoid}
    rows::Ptr{Cvoid}
end

Base.@kwdef struct _NodeStepEoLaneArguments
    context::CUDA.CuContext
    config::QnpepsElocConfig
    terms::HeisenbergTerms
    selector::EoSelector
    geometry::_NodeStepEoGeometry
    outputs::_NodeStepEoOutputs
end

Base.@kwdef mutable struct NodeStepTelemetry
    epoch::Int64 = 0
    ns::Int64 = 0
    predraw_join_s::Float64 = 0.0
    env_wait_s::Float64 = 0.0
    sampling_s::Float64 = 0.0
    ring_s::Float64 = 0.0
    eo_s::Float64 = 0.0
    predraw_issue_s::Float64 = 0.0
    minsr_s::Float64 = 0.0
    output_s::Float64 = 0.0
    complete_s::Float64 = 0.0
    status::Int32 = 0
end

Base.@kwdef struct _NodeStepMemcpy2D
    src_x_bytes::UInt
    src_y::UInt
    src_memory_type::UInt32
    src_host::Ptr{Cvoid}
    src_device::UInt
    src_array::Ptr{Cvoid}
    src_pitch::UInt
    dst_x_bytes::UInt
    dst_y::UInt
    dst_memory_type::UInt32
    dst_host::Ptr{Cvoid}
    dst_device::UInt
    dst_array::Ptr{Cvoid}
    dst_pitch::UInt
    width_bytes::UInt
    height::UInt
end

Base.@kwdef mutable struct _NodeStepEoLane
    context::CUDA.CuContext
    stream::CUDA.CUstream
    handle::Ptr{Cvoid}
    peps::CuVector{ComplexF32}
    samples::CuVector{UInt8}
    logpsi::CuVector{Float64}
    e_loc::CuVector{Float64}
    rows::CuVector{ComplexF32}
    peps_pointer::CUDA.CuPtr{Cvoid}
    samples_pointer::CUDA.CuPtr{Cvoid}
    logpsi_pointer::CUDA.CuPtr{Cvoid}
    e_loc_pointer::CUDA.CuPtr{Cvoid}
    rows_pointer::CUDA.CuPtr{Cvoid}
    source_sample_pointer::CUDA.CuPtr{Cvoid}
    destination_logpsi_pointer::CUDA.CuPtr{Cvoid}
    destination_e_loc_pointer::CUDA.CuPtr{Cvoid}
    row_base::Int64
    row_count::Int64
    args::Vector{_NodeStepElocArgs}
    original_affinity::NTuple{16,UInt64}
    affinity_saved::Int32
    affinity_applied::Int32
    status::Int32
    failure_location::String
    failure_message::String
end

Base.@kwdef mutable struct NodeStepHost{DH,SH,MH,MI,EH}
    config::QnpepsE2eConfig
    config_ref::Base.RefValue{QnpepsE2eConfig}
    terms::HeisenbergTerms
    peps::QNP.CuPeps
    source_peps_pointer::CUDA.CuPtr{Cvoid}
    source_peps_bytes::UInt64
    peps_generation::Int64
    dlenv::DH
    sampler::SH
    minsr::MH
    eo_lanes::Vector{_NodeStepEoLane}
    eo_host::EH
    eo_backend::Symbol
    workers::Vector{Task}
    eloc_config::QnpepsElocConfig
    eloc_config_ref::Base.RefValue{QnpepsElocConfig}
    gram_scratch::CuVector{UInt8}
    gram_stage_a::CUDA.CuPtr{Cvoid}
    gram_stage_b::CUDA.CuPtr{Cvoid}
    gram_tile::CUDA.CuPtr{Cvoid}
    gram_copy_args::Vector{_NodeStepMemcpy2D}
    samples_host::Vector{UInt8}
    logq_host::Vector{Float64}
    log_gauge_host::Vector{Float64}
    o_rows_host::Vector{ComplexF32}
    o_rows_registration::CUDA.HostMemory
    minsr_statistics::Vector{Float64}
    samples_host_pointer::Ptr{Cvoid}
    logq_host_pointer::Ptr{Cvoid}
    log_gauge_host_pointer::Ptr{Cvoid}
    o_rows_host_pointer::Ptr{Cvoid}
    minsr_statistics_pointer::Ptr{Float64}
    samples::CuVector{UInt8}
    logq::CuVector{Float64}
    log_gauge::CuVector{Float64}
    logpsi::CuVector{Float64}
    e_loc::CuVector{Float64}
    theta_dot::CuVector{ComplexF32}
    samples_pointer::CUDA.CuPtr{Cvoid}
    logq_pointer::CUDA.CuPtr{Cvoid}
    log_gauge_pointer::CUDA.CuPtr{Cvoid}
    logpsi_pointer::CUDA.CuPtr{Cvoid}
    e_loc_pointer::CUDA.CuPtr{Cvoid}
    theta_dot_pointer::CUDA.CuPtr{Cvoid}
    dlenv_value_pointers::NTuple{2,UInt}
    sampler_refresh_args::Vector{QNP.QnpepsSamplerHostRefreshArgs}
    sampler_refresh_lane_args::NTuple{2,QNP.QnpepsSamplerHostRefreshArgs}
    sampler_batch_lane_args::NTuple{2,QNP.QnpepsSamplerHostBatchArgs}
    minsr_inputs::MI
    telemetry::NodeStepTelemetry
    n_samples::Int64
    sites::Int64
    compact::Int64
    dense::Int64
    dim_batch::Int64
    host_tile_bytes::Int64
    next_batch::UInt64
    epoch::Int64
    stage::Symbol
    command::Threads.Atomic{Int32}
    command_epoch::Threads.Atomic{Int}
    ready::Threads.Atomic{Int}
    done::Threads.Atomic{Int}
    running::Threads.Atomic{Bool}
    closed::Threads.Atomic{Bool}
end
