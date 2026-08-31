Base.@kwdef struct _ElocRunInputs
    peps::CUDA.CuPtr{Cvoid}
    samples::CUDA.CuPtr{UInt8}
    terms::QnpepsElocTermTable
end

Base.@kwdef struct _ElocRunOutputs
    logpsi::CUDA.CuPtr{Float64}
    e_loc::CUDA.CuPtr{Float64}
    rows::CUDA.CuPtr{Cvoid}
end

Base.@kwdef struct _ElocRunWorkspace
    workspace::Ptr{Cvoid}
    scratch::Ptr{Cvoid}
    reference_energy::Float64
    stream::Ptr{Cvoid}
end

Base.@kwdef struct _ElocRunArguments
    config::QnpepsElocConfig
    n_samples::Int64
    inputs::_ElocRunInputs
    outputs::_ElocRunOutputs
    workspace::_ElocRunWorkspace
end

Base.@kwdef struct _ElocGramTileRows
    rows::CUDA.CuPtr{Cvoid}
    samples::CUDA.CuPtr{UInt8}
    count::Int64
end

Base.@kwdef struct _ElocGramTileArguments
    config::Ptr{QnpepsElocConfig}
    source::_ElocGramTileRows
    target::_ElocGramTileRows
    tile::CUDA.CuPtr{Cvoid}
    stream::Ptr{Cvoid}
end
