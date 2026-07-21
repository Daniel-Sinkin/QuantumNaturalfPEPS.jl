using CUDA

mutable struct CuSamplingContext
    handle::Ptr{Cvoid}
    config::QnpepsConfig
    device_peps::CuPeps
    row_logs::CuVector{Float64}
    sample_bytes::CuVector{UInt8}
    log_prob_config::CuVector{Float64}
    log_gauge::CuVector{Float64}
    host_peps::Vector{ComplexF32}
    host_sample_bytes::Vector{UInt8}
    host_log_prob_config::Vector{Float64}
    host_log_gauge::Vector{Float64}
    configs::Vector{Matrix{UInt8}}
    n_samples::Int
    batch_base::UInt64
    batches_per_call::UInt64
    device::CUDA.CuDevice
    stream::CUDA.CuStream
    dlenv_ready::Bool
end

function Base.show(io::IO, context::CuSamplingContext)::Nothing
    state = isopen(context) ? "open" : "closed"
    return print(
        io,
        "CuSamplingContext(",
        Int(context.config.lx),
        "×",
        Int(context.config.ly),
        ", dim_bond=",
        Int(context.config.dim_bond),
        ", n_samples=",
        context.n_samples,
        ", ",
        state,
        ")",
    )
end

Base.isopen(context::CuSamplingContext)::Bool = context.handle != C_NULL

function _assert_open(context::CuSamplingContext)::Nothing
    isopen(context) || throw(ArgumentError("CuSamplingContext is closed"))
    return nothing
end

function _with_context(f::F, context::CuSamplingContext) where {F}
    _assert_open(context)
    return CUDA.device!(context.device) do
        CUDA.stream!(context.stream) do
            f()
        end
    end
end

_raw_pointer(::Type{T}, data::CuArray) where {T} = reinterpret(Ptr{T}, pointer(data))

function _destroy_context!(context::CuSamplingContext)::Nothing
    handle = context.handle
    handle == C_NULL && return nothing
    CUDA.device!(context.device) do
        FFI.ctx_destroy(handle)
    end
    context.handle = C_NULL
    context.dlenv_ready = false
    return nothing
end

function close!(context::CuSamplingContext)::Nothing
    return _destroy_context!(context)
end

Base.close(context::CuSamplingContext)::Nothing = close!(context)

function _finalize_context!(context::CuSamplingContext)::Nothing
    try
        close!(context)
    catch
    end
    return nothing
end

function CuSamplingContext(
    peps::Peps,
    n_samples::Integer;
    seed::Integer=0,
    sampling_mode=:fast,
    chi_s::Integer=_dim_bond(peps),
    chi_dl::Integer=_dim_bond(peps),
    chi_c::Integer=3 * _dim_bond(peps),
    batch_base::Integer=0,
)::CuSamplingContext
    CUDA.functional() || error("CUDA is not functional")
    n_samples > 0 || throw(ArgumentError("n_samples must be positive"))
    seed >= 0 || throw(ArgumentError("seed must be nonnegative"))
    batch_base >= 0 || throw(ArgumentError("batch_base must be nonnegative"))

    n_samples_int = Int(n_samples)
    batch_base_u64 = UInt64(batch_base)
    host_peps = _pack_peps(peps)
    _check_packed_peps(peps, host_peps)
    device_peps = CuPeps(
        CuArray(host_peps),
        _lx(peps),
        _ly(peps),
        _dim_phys(peps),
        _dim_bond(peps),
    )
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

    row_logs = CUDA.zeros(Float64, device_peps.lx - 1)
    sample_bytes = CUDA.zeros(UInt8, _sample_bytes(; config, n_samples=n_samples_int))
    log_prob_config = CUDA.zeros(Float64, n_samples_int)
    log_gauge = CUDA.zeros(Float64, n_samples_int)
    host_sample_bytes = Vector{UInt8}(undef, length(sample_bytes))
    host_log_prob_config = Vector{Float64}(undef, n_samples_int)
    host_log_gauge = Vector{Float64}(undef, n_samples_int)
    configs = [Matrix{UInt8}(undef, device_peps.lx, device_peps.ly) for _ in 1:n_samples_int]
    dim_batch = min(n_samples_int, MAX_BATCH_SIZE)
    batches_per_call = UInt64(cld(n_samples_int, dim_batch))
    device = CUDA.device()
    stream = CUDA.stream()

    handle_ref = Ref{Ptr{Cvoid}}(C_NULL)
    status = GC.@preserve stream FFI.ctx_create(config, handle_ref; stream=stream.handle)
    if status != 0
        handle_ref[] == C_NULL || FFI.ctx_destroy(handle_ref[])
        _check(; status, what="qnpeps_ctx_create")
    end
    handle_ref[] != C_NULL ||
        error("cuQuantumNaturalfPEPS: qnpeps_ctx_create returned a null context")

    context = CuSamplingContext(
        handle_ref[],
        config,
        device_peps,
        row_logs,
        sample_bytes,
        log_prob_config,
        log_gauge,
        host_peps,
        host_sample_bytes,
        host_log_prob_config,
        host_log_gauge,
        configs,
        n_samples_int,
        batch_base_u64,
        batches_per_call,
        device,
        stream,
        false,
    )
    finalizer(_finalize_context!, context)
    return context
end

function update_peps!(context::CuSamplingContext, peps::Peps)::CuSamplingContext
    _with_context(context) do
        _upload_peps!(context.device_peps, peps, context.host_peps)
    end
    context.dlenv_ready = false
    return context
end

function build_dlenv!(context::CuSamplingContext)::CuSamplingContext
    context.dlenv_ready = false
    _with_context(context) do
        buffers = (context.device_peps.data, context.row_logs)
        status = GC.@preserve buffers FFI.ctx_build_dlenv(
            context.handle,
            _raw_pointer(Cvoid, context.device_peps.data);
            cumulative_row_logs=_raw_pointer(Float64, context.row_logs),
        )
        _check(; status, what="qnpeps_ctx_build_dlenv")
    end
    context.dlenv_ready = true
    return context
end

function build_dlenv!(context::CuSamplingContext, peps::Peps)::CuSamplingContext
    update_peps!(context, peps)
    return build_dlenv!(context)
end

function sample_peps!(context::CuSamplingContext)::SampleResult
    _assert_open(context)
    context.dlenv_ready ||
        throw(ArgumentError("build_dlenv! must succeed before sampling"))

    next_batch_base = context.batch_base + context.batches_per_call
    next_batch_base >= context.batch_base || throw(OverflowError("sample batch_base overflow"))
    _with_context(context) do
        buffers = (
            context.sample_bytes,
            context.log_prob_config,
            context.log_gauge,
        )
        status = GC.@preserve buffers FFI.ctx_sample(
            context.handle,
            _raw_pointer(UInt8, context.sample_bytes);
            log_prob_config=_raw_pointer(Float64, context.log_prob_config),
            log_gauge=_raw_pointer(Float64, context.log_gauge),
            n_samples=context.n_samples,
            batch_base=context.batch_base,
        )
        _check(; status, what="qnpeps_ctx_sample")
        copyto!(context.host_sample_bytes, context.sample_bytes)
        copyto!(context.host_log_prob_config, context.log_prob_config)
        copyto!(context.host_log_gauge, context.log_gauge)
        CUDA.synchronize(context.stream)
    end
    _unpack_samples!(
        context.configs,
        context.host_sample_bytes,
        context.config,
        context.n_samples,
    )
    context.batch_base = next_batch_base
    return (
        configs=context.configs,
        log_prob_config=context.host_log_prob_config,
        log_gauge=context.host_log_gauge,
    )
end

function sample_peps(context::CuSamplingContext)::SampleResult
    result = sample_peps!(context)
    return (
        configs=copy.(result.configs),
        log_prob_config=copy(result.log_prob_config),
        log_gauge=copy(result.log_gauge),
    )
end
