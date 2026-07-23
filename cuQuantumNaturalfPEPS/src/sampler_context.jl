using CUDA
using ITensorMPS

mutable struct SamplerContext{D,S}
    handle::Ptr{Cvoid}
    config::QnpepsConfig
    batch_size::Int
    cumulative_row_logs::Vector{Float64}
    device::D
    stream::S
end

Base.isopen(context::SamplerContext)::Bool = context.handle != C_NULL

function Base.close(context::SamplerContext)::Nothing
    isopen(context) || return nothing
    handle = context.handle
    context.handle = C_NULL
    _ffi_ctx_destroy(handle)
    return nothing
end

Base.copy(::SamplerContext) = throw(ArgumentError("SamplerContext cannot be copied"))

function Base.show(io::IO, context::SamplerContext)::Nothing
    config = context.config
    return print(
        io,
        "SamplerContext(",
        config.lx,
        "×",
        config.ly,
        ", dim_bond=",
        config.dim_bond,
        ", chi_s=",
        config.chi_s,
        ", chi_dl=",
        config.chi_dl,
        ", batch_size=",
        context.batch_size,
        ")",
    )
end

function _validate_sampler_context(context::SamplerContext)::Nothing
    isopen(context) || throw(ArgumentError("SamplerContext is closed"))
    CUDA.device() == context.device || throw(ArgumentError("SamplerContext device mismatch"))
    CUDA.stream().handle == context.stream.handle ||
        throw(ArgumentError("SamplerContext stream mismatch"))
    return nothing
end

function _validate_context_peps(context::SamplerContext, device_peps::CuPeps)::Nothing
    config = context.config
    dimensions_match =
        device_peps.lx == config.lx &&
        device_peps.ly == config.ly &&
        device_peps.dim_phys == config.dim_phys &&
        device_peps.dim_bond == config.dim_bond
    dimensions_match ||
        throw(DimensionMismatch("$device_peps does not match $context"))
    return nothing
end

function build_dlenv!(context::SamplerContext, device_peps::CuPeps)::SamplerContext
    _validate_sampler_context(context)
    _validate_context_peps(context, device_peps)
    logs = CUDA.zeros(Float64, Int(context.config.lx) - 1)
    try
        GC.@preserve device_peps logs begin
            _ffi_ctx_build_dlenv(context.handle, pointer(device_peps.data), pointer(logs))
        end
        context.cumulative_row_logs = Array(logs)
    finally
        CUDA.unsafe_free!(logs)
    end
    return context
end

function build_dlenv!(context::SamplerContext, peps::Peps)::SamplerContext
    device_peps = upload_peps(peps)
    try
        return build_dlenv!(context, device_peps)
    finally
        CUDA.unsafe_free!(device_peps.data)
    end
end

function build_dlenv!(context::SamplerContext, tensors::AbstractMatrix)::SamplerContext
    return build_dlenv!(context, load_peps(tensors))
end

function SamplerContext(
    device_peps::CuPeps;
    chi_s::Integer=device_peps.dim_bond,
    chi_dl::Integer=device_peps.dim_bond,
    seed::Integer=0,
    sampling_mode=:fast,
    chi_c::Integer=3 * device_peps.dim_bond,
    batch_size::Integer=MAX_BATCH_SIZE,
)::SamplerContext
    1 <= batch_size <= MAX_BATCH_SIZE ||
        throw(ArgumentError("batch_size must be between 1 and $MAX_BATCH_SIZE"))
    config = QnpepsConfig(
        lx=device_peps.lx,
        ly=device_peps.ly,
        dim_phys=device_peps.dim_phys,
        dim_bond=device_peps.dim_bond,
        chi_s=chi_s,
        chi_dl=chi_dl,
        seed=seed,
        sampling_mode=sampling_mode,
        chi_c=chi_c,
    )
    device = CUDA.device()
    stream = CUDA.stream()
    handle = _ffi_ctx_create(; config, stream=Ptr{Cvoid}(stream.handle))
    context = SamplerContext(handle, config, Int(batch_size), Float64[], device, stream)
    finalizer(close, context)
    try
        return build_dlenv!(context, device_peps)
    catch
        close(context)
        rethrow()
    end
end

function SamplerContext(
    peps::Peps;
    chi_s::Integer=_dim_bond(peps),
    chi_dl::Integer=_dim_bond(peps),
    seed::Integer=0,
    sampling_mode=:fast,
    chi_c::Integer=3 * _dim_bond(peps),
    batch_size::Integer=MAX_BATCH_SIZE,
)::SamplerContext
    device_peps = upload_peps(peps)
    try
        return SamplerContext(
            device_peps;
            chi_s,
            chi_dl,
            seed,
            sampling_mode,
            chi_c,
            batch_size,
        )
    finally
        CUDA.unsafe_free!(device_peps.data)
    end
end

function SamplerContext(tensors::AbstractMatrix; kwargs...)::SamplerContext
    return SamplerContext(load_peps(tensors); kwargs...)
end

function _context_dlenv_arrays(
    context::SamplerContext,
)::Vector{Vector{Array{ComplexF32,4}}}
    _validate_sampler_context(context)
    n_bytes = _dlenv_bytes(; config=context.config)
    n_bytes >= 0 || throw(ArgumentError("invalid SamplerContext double-layer size"))
    bytes = Vector{UInt8}(undef, n_bytes)
    GC.@preserve bytes begin
        _ffi_ctx_copy_dlenv_host(context.handle, pointer(bytes), length(bytes))
    end
    return _dlenv_arrays(bytes, Int(context.config.lx), Int(context.config.ly))
end

materialize_dlenv(context::SamplerContext) = _context_dlenv_arrays(context)

function materialize_dlenv(
    context::SamplerContext,
    tensors::AbstractMatrix,
)::Vector{MPS}
    _check_dlenv_grid(Int(context.config.lx), Int(context.config.ly), tensors)
    return _materialize_dlenv_arrays(_context_dlenv_arrays(context), tensors)
end
