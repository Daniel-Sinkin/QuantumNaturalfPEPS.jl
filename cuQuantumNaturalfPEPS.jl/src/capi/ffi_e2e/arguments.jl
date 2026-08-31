Base.@kwdef struct _E2eCuts
    relative_cut::Float64
    absolute_cut::Float64
end

Base.@kwdef struct _E2eMinsrInputs
    device_samples::CUDA.CuPtr{UInt8}
    logpsi::CUDA.CuPtr{Float64}
    e_loc::CUDA.CuPtr{Float64}
    logq::CUDA.CuPtr{Float64}
    gram::CUDA.CuPtr{Cvoid}
    o_rows_device::CUDA.CuPtr{Cvoid}
    o_rows_host::Ptr{Cvoid}
end

Base.@kwdef struct _E2eMinsrOutputs
    theta_dot::CUDA.CuPtr{Cvoid}
    e_mean::Ptr{Float64}
    e_var::Ptr{Float64}
    ess::Ptr{Float64}
end

Base.@kwdef struct _E2eMinsrArguments{C}
    config::C
    n_samples::Int64
    inputs::_E2eMinsrInputs
    host_tile_bytes::Int64
    cuts::_E2eCuts
    outputs::_E2eMinsrOutputs
    stream::Ptr{Cvoid}
end

Base.@kwdef struct _E2eStepInputs
    device_peps::CUDA.CuPtr{Cvoid}
    terms::Ptr{Cvoid}
end

Base.@kwdef struct _E2eSampleOutputs
    samples::CUDA.CuPtr{UInt8}
    logq::CUDA.CuPtr{Float64}
    log_gauge::CUDA.CuPtr{Float64}
    logpsi::CUDA.CuPtr{Float64}
    e_loc::CUDA.CuPtr{Float64}
    o_rows_host::Ptr{Cvoid}
end

Base.@kwdef struct _E2eStepArguments
    config::QnpepsE2eConfig
    inputs::_E2eStepInputs
    n_samples::Int64
    host_tile_bytes::Int64
    cuts::_E2eCuts
    minsr_outputs::_E2eMinsrOutputs
    sample_outputs::_E2eSampleOutputs
    stream::Ptr{Cvoid}
end

Base.@kwdef struct _E2eNodeStepArguments
    node::Ptr{Cvoid}
    n_samples::Int64
    cuts::_E2eCuts
    minsr_outputs::_E2eMinsrOutputs
    sample_outputs::_E2eSampleOutputs
    epoch::Ptr{Int64}
end
