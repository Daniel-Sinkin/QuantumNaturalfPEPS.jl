
Base.@kwdef struct _EoHostRunArgs
    struct_size::UInt32
    padding::NTuple{4,UInt8}
    device_peps::CUDA.CuPtr{Cvoid}
    device_samples::CUDA.CuPtr{UInt8}
    logpsi_out::CUDA.CuPtr{Float64}
    e_loc_out::CUDA.CuPtr{Float64}
    o_rows_dev::CUDA.CuPtr{Cvoid}
    o_rows_host::Ptr{Cvoid}
    gram::CUDA.CuPtr{Cvoid}
    lambda::Float64
end
