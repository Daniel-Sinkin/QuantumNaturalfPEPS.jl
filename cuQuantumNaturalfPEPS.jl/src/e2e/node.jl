mutable struct VMCContext
    handle::Ptr{Cvoid}
    config::QnpepsE2eConfig
    ns_capacity::Int
    dense::Int
    compact::Int
    sites::Int
    submitted::Bool
end

function VMCContext(
    config::QnpepsE2eConfig,
    terms::HeisenbergTerms;
    gpus::Integer=1,
    ns_capacity::Integer,
    ns_ahead::Integer=0,
    dim_batch::Integer,
    host_tile_bytes::Integer=0,
)
    ngpu = _validate_gpus(gpus)
    ns_capacity >= 2 || throw(ArgumentError("ns_capacity must be at least 2"))
    0 <= ns_ahead <= ns_capacity ||
        throw(ArgumentError("ns_ahead must be between 0 and ns_capacity"))
    1 <= dim_batch <= 2048 || throw(ArgumentError("dim_batch must be between 1 and 2048"))
    host_tile_bytes >= 0 || throw(ArgumentError("host_tile_bytes must be nonnegative"))

    diag = terms.diag
    flip = terms.flip
    handle = GC.@preserve diag flip begin
        table = QnpepsElocTermTable(
            Int32(length(diag)),
            isempty(diag) ? Ptr{QnpepsElocDiagBond}(0) : pointer(diag),
            Int32(length(flip)),
            isempty(flip) ? Ptr{QnpepsElocFlipTerm}(0) : pointer(flip),
        )
        table_ref = Ref(table)
        GC.@preserve table_ref _ffi_node_create(
            config,
            ngpu,
            ns_capacity,
            ns_ahead,
            dim_batch,
            host_tile_bytes,
            Base.unsafe_convert(Ptr{Cvoid}, table_ref),
        )
    end
    context = VMCContext(
        handle,
        config,
        Int(ns_capacity),
        dense_count(config),
        compact_count(config),
        Int(config.lx) * Int(config.ly),
        false,
    )
    finalizer(context) do value
        value.handle == C_NULL && return
        try
            close(value)
        catch
        end
    end
    return context
end

Base.isopen(context::VMCContext) = context.handle != C_NULL

function Base.close(context::VMCContext)
    context.handle == C_NULL && return nothing
    handle = context.handle
    try
        _ffi_node_destroy(handle)
    finally
        context.handle = C_NULL
        context.submitted = false
    end
    return nothing
end

function _require_open(context::VMCContext)
    isopen(context) || throw(ArgumentError("VMCContext is closed"))
    return context.handle
end

function _validate_node_samples(context::VMCContext, n_samples::Integer)
    2 <= n_samples <= context.ns_capacity ||
        throw(ArgumentError("n_samples must be between 2 and $(context.ns_capacity)"))
    return Int(n_samples)
end

function submit_peps!(context::VMCContext, peps::CuArray{ComplexF32})
    handle = _require_open(context)
    length(peps) == context.dense ||
        throw(DimensionMismatch("PEPS has $(length(peps)) values, expected $(context.dense)"))
    GC.@preserve peps _ffi_node_submit_theta(handle, pointer(peps))
    context.submitted = true
    return context
end

function _node_output_buffers(
    context::VMCContext,
    n_samples::Int;
    want_theta::Bool,
    want_samples::Bool,
    want_logq::Bool,
    want_log_gauge::Bool,
    want_logpsi::Bool,
    want_e_loc::Bool,
    want_o_rows::Bool,
    want_epoch::Bool,
)
    return (
        theta=want_theta ? CUDA.zeros(ComplexF32, context.dense) : nothing,
        e_mean=Vector{Float64}(undef, 2),
        e_var=Vector{Float64}(undef, 1),
        ess=Vector{Float64}(undef, 1),
        samples=want_samples ? CUDA.zeros(UInt8, n_samples * context.sites) : nothing,
        logq=want_logq ? CUDA.zeros(Float64, n_samples) : nothing,
        log_gauge=want_log_gauge ? CUDA.zeros(Float64, n_samples) : nothing,
        logpsi=want_logpsi ? CUDA.zeros(Float64, 2 * n_samples) : nothing,
        e_loc=want_e_loc ? CUDA.zeros(Float64, 2 * n_samples) : nothing,
        o_rows=want_o_rows ? Vector{ComplexF32}(undef, n_samples * context.compact) : nothing,
        epoch=want_epoch ? Vector{Int64}(undef, n_samples) : nothing,
    )
end

_device_pointer(::Nothing, ::Type{T}) where {T} = CuPtr{T}(0)
_device_pointer(array::CuArray, ::Type{T}) where {T} = CuPtr{T}(pointer(array))
_host_pointer(::Nothing, ::Type{T}) where {T} = Ptr{T}(0)
_host_pointer(array::Vector{T}, ::Type{T}) where {T} = pointer(array)

function _node_result(buffers; include_theta::Bool)
    result = (
        e_mean=complex(buffers.e_mean[1], buffers.e_mean[2]),
        e_var=buffers.e_var[1],
        ess=buffers.ess[1],
    )
    include_theta && (result = merge(result, (theta_dot=buffers.theta,)))
    buffers.samples === nothing || (result = merge(result, (samples=buffers.samples,)))
    buffers.logq === nothing || (result = merge(result, (logq=buffers.logq,)))
    buffers.log_gauge === nothing || (result = merge(result, (log_gauge=buffers.log_gauge,)))
    buffers.logpsi === nothing || (result = merge(result, (logpsi=buffers.logpsi,)))
    buffers.e_loc === nothing || (result = merge(result, (e_loc=buffers.e_loc,)))
    buffers.o_rows === nothing || (result = merge(result, (o_rows=buffers.o_rows,)))
    buffers.epoch === nothing || (result = merge(result, (epoch=buffers.epoch,)))
    return result
end

function _vmc_step_multigpu_composed!(
    data::CuArray{ComplexF32},
    config::QnpepsE2eConfig,
    terms::HeisenbergTerms,
    gpus::Int,
    n_samples::Integer;
    host_tile_bytes::Integer,
    relative_cut::Real,
    absolute_cut::Real,
    want_samples::Bool,
    want_logq::Bool,
    want_log_gauge::Bool,
    want_logpsi::Bool,
    want_e_loc::Bool,
    want_o_rows::Bool,
)
    sample_count = max(Int(n_samples), 1)
    dim_batch = config.sample_batch > 0 ? Int(config.sample_batch) : min(sample_count, 2048)
    capacity = max(2, cld(sample_count, dim_batch) * dim_batch)
    context = VMCContext(
        config,
        terms;
        gpus,
        ns_capacity=capacity,
        ns_ahead=0,
        dim_batch,
        host_tile_bytes,
    )
    result = try
        submit_peps!(context, data)
        vmc_direction!(
            context;
            n_samples,
            relative_cut,
            absolute_cut,
            want_samples,
            want_logq,
            want_log_gauge,
            want_logpsi,
            want_e_loc,
            want_o_rows,
        )
    finally
        close(context)
    end

    output = (
        theta_dot=Array(result.theta_dot),
        e_mean=result.e_mean,
        e_var=result.e_var,
        ess=result.ess,
    )
    want_samples && (output = merge(output, (samples=Array(result.samples),)))
    want_logq && (output = merge(output, (logq=Array(result.logq),)))
    want_log_gauge && (output = merge(output, (log_gauge=Array(result.log_gauge),)))
    want_logpsi && (output = merge(output, (logpsi=_pack_complex(Array(result.logpsi)),)))
    want_e_loc && (output = merge(output, (e_loc=_pack_complex(Array(result.e_loc)),)))
    want_o_rows && (output = merge(output, (o_rows=result.o_rows,)))
    return output
end

function vmc_direction!(
    context::VMCContext;
    n_samples::Integer,
    relative_cut::Real=1.0e-3,
    absolute_cut::Real=1.0e-8,
    want_samples::Bool=false,
    want_logq::Bool=false,
    want_log_gauge::Bool=false,
    want_logpsi::Bool=false,
    want_e_loc::Bool=false,
    want_o_rows::Bool=false,
    want_epoch::Bool=false,
)
    handle = _require_open(context)
    context.submitted || throw(ArgumentError("submit_peps! must be called before vmc_direction!"))
    ns = _validate_node_samples(context, n_samples)
    buffers = _node_output_buffers(
        context,
        ns;
        want_theta=true,
        want_samples,
        want_logq,
        want_log_gauge,
        want_logpsi,
        want_e_loc,
        want_o_rows,
        want_epoch,
    )
    GC.@preserve buffers begin
        cuts =
            FFI._E2eCuts(; relative_cut=Float64(relative_cut), absolute_cut=Float64(absolute_cut))
        minsr_outputs = FFI._E2eMinsrOutputs(;
            theta_dot=CUDA.CuPtr{Cvoid}(pointer(buffers.theta)),
            e_mean=pointer(buffers.e_mean),
            e_var=pointer(buffers.e_var),
            ess=pointer(buffers.ess),
        )
        sample_outputs = FFI._E2eSampleOutputs(;
            samples=CUDA.CuPtr{UInt8}(_device_pointer(buffers.samples, UInt8)),
            logq=CUDA.CuPtr{Float64}(_device_pointer(buffers.logq, Float64)),
            log_gauge=CUDA.CuPtr{Float64}(_device_pointer(buffers.log_gauge, Float64)),
            logpsi=CUDA.CuPtr{Float64}(_device_pointer(buffers.logpsi, Float64)),
            e_loc=CUDA.CuPtr{Float64}(_device_pointer(buffers.e_loc, Float64)),
            o_rows_host=buffers.o_rows === nothing ? Ptr{Cvoid}(0) :
                        Ptr{Cvoid}(pointer(buffers.o_rows)),
        )
        arguments = FFI._E2eNodeStepArguments(;
            node=handle,
            n_samples=Int64(ns),
            cuts,
            minsr_outputs,
            sample_outputs,
            epoch=_host_pointer(buffers.epoch, Int64),
        )
        _ffi_node_step(arguments)
    end
    return _node_result(buffers; include_theta=true)
end

function vmc_euler_step!(
    context::VMCContext,
    peps_f32::CuArray{ComplexF32};
    state_f64::Union{Nothing,CuArray{ComplexF64}}=nothing,
    precision::Symbol=state_f64 === nothing ? :f32 : :f64,
    n_samples::Integer,
    learning_rate::Real,
    relative_cut::Real=1.0e-3,
    absolute_cut::Real=1.0e-8,
    want_theta::Bool=false,
    want_samples::Bool=false,
    want_logq::Bool=false,
    want_log_gauge::Bool=false,
    want_logpsi::Bool=false,
    want_e_loc::Bool=false,
    want_o_rows::Bool=false,
    want_epoch::Bool=false,
)
    handle = _require_open(context)
    context.submitted || throw(ArgumentError("submit_peps! must be called before vmc_euler_step!"))
    length(peps_f32) == context.dense ||
        throw(DimensionMismatch("PEPS has $(length(peps_f32)) values, expected $(context.dense)"))
    isfinite(learning_rate) || throw(ArgumentError("learning_rate must be finite"))
    ns = _validate_node_samples(context, n_samples)

    mode =
        precision === :f32 ? Int32(0) :
        precision === :f64 ? Int32(1) : throw(ArgumentError("precision must be :f32 or :f64"))
    if mode == 0
        state_f64 === nothing || throw(ArgumentError("state_f64 must be nothing in :f32 mode"))
    else
        state_f64 === nothing && throw(ArgumentError("state_f64 is required in :f64 mode"))
        length(state_f64) == context.dense || throw(
            DimensionMismatch(
                "Float64 state has $(length(state_f64)) values, expected $(context.dense)",
            ),
        )
    end

    buffers = _node_output_buffers(
        context,
        ns;
        want_theta,
        want_samples,
        want_logq,
        want_log_gauge,
        want_logpsi,
        want_e_loc,
        want_o_rows,
        want_epoch,
    )
    GC.@preserve peps_f32 state_f64 buffers begin
        state_pointer = state_f64 === nothing ? CuPtr{Cvoid}(0) : CuPtr{Cvoid}(pointer(state_f64))
        theta_pointer =
            buffers.theta === nothing ? CuPtr{Cvoid}(0) : CuPtr{Cvoid}(pointer(buffers.theta))
        arguments = QnpepsE2eEulerStepArgs(;
            struct_size=UInt32(sizeof(QnpepsE2eEulerStepArgs)),
            precision=mode,
            n_samples=Int64(ns),
            relative_cut=Float64(relative_cut),
            absolute_cut=Float64(absolute_cut),
            learning_rate=Float64(learning_rate),
            state_f64_io=state_pointer,
            state_f64_bytes=state_f64 === nothing ? UInt64(0) :
                            UInt64(sizeof(ComplexF64) * length(state_f64)),
            peps_f32_io=CuPtr{Cvoid}(pointer(peps_f32)),
            peps_f32_bytes=UInt64(sizeof(ComplexF32) * length(peps_f32)),
            theta_dot_out=theta_pointer,
            theta_dot_bytes=buffers.theta === nothing ? UInt64(0) :
                            UInt64(sizeof(ComplexF32) * length(buffers.theta)),
            e_mean_out=pointer(buffers.e_mean),
            e_var_out=pointer(buffers.e_var),
            ess_out=pointer(buffers.ess),
            samples_out=_device_pointer(buffers.samples, UInt8),
            logq_out=_device_pointer(buffers.logq, Float64),
            log_gauge_out=_device_pointer(buffers.log_gauge, Float64),
            logpsi_out=_device_pointer(buffers.logpsi, Float64),
            e_loc_out=_device_pointer(buffers.e_loc, Float64),
            o_rows_host=buffers.o_rows === nothing ? Ptr{Cvoid}(0) :
                        Ptr{Cvoid}(pointer(buffers.o_rows)),
            epoch_out=_host_pointer(buffers.epoch, Int64),
        )
        arguments_ref = Ref(arguments)
        GC.@preserve arguments_ref _ffi_node_step_euler(handle, arguments_ref)
    end
    result = _node_result(buffers; include_theta=want_theta)
    return merge(result, (peps=peps_f32, state_f64=state_f64))
end
