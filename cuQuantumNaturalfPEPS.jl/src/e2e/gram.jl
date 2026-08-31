mutable struct E2eGramContext
    handle::Ptr{Cvoid}
    config::QnpepsE2eConfig
    n_samples::Int
    compact::Int
    sites::Int
    stream::CuStream
end

function E2eGramContext(
    config::QnpepsE2eConfig,
    n_samples::Integer;
    stream::CuStream=CUDA.stream(),
)
    n_samples >= 2 || throw(ArgumentError("n_samples must be at least 2"))
    handle = _ffi_gram_create(config, n_samples, stream)
    context = E2eGramContext(
        handle,
        config,
        Int(n_samples),
        compact_count(config),
        Int(config.lx) * Int(config.ly),
        stream,
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

Base.isopen(context::E2eGramContext) = context.handle != C_NULL

function Base.close(context::E2eGramContext)
    context.handle == C_NULL && return nothing
    handle = context.handle
    try
        _ffi_gram_destroy(handle)
    finally
        context.handle = C_NULL
    end
    return nothing
end

function _require_open(context::E2eGramContext)
    isopen(context) || throw(ArgumentError("E2eGramContext is closed"))
    return context.handle
end

function e2e_gram_footprint(context::E2eGramContext)
    footprint = _ffi_gram_footprint(_require_open(context))
    return (
        context_device_bytes=footprint.context_device_bytes,
        geometry_device_bytes=footprint.geometry_device_bytes,
        dense_a_device_bytes=footprint.dense_a_device_bytes,
        dense_b_device_bytes=footprint.dense_b_device_bytes,
        caller_samples_bytes=footprint.caller_samples_bytes,
        caller_rows_bytes=footprint.caller_rows_bytes,
        caller_gram_bytes=footprint.caller_gram_bytes,
    )
end

function e2e_raw_gram!(
    context::E2eGramContext,
    output::CuArray{ComplexF32},
    samples::CuArray{UInt8},
    o_rows::CuArray{ComplexF32};
    timings::Bool=false,
)
    handle = _require_open(context)
    length(samples) == context.n_samples * context.sites || throw(
        DimensionMismatch(
            "samples have $(length(samples)) values, expected $(context.n_samples * context.sites)",
        ),
    )
    length(o_rows) == context.n_samples * context.compact || throw(
        DimensionMismatch(
            "O rows have $(length(o_rows)) values, expected $(context.n_samples * context.compact)",
        ),
    )
    length(output) == context.n_samples * context.n_samples || throw(
        DimensionMismatch(
            "Gram output has $(length(output)) values, expected $(context.n_samples^2)",
        ),
    )

    timing =
        timings ?
        Ref(
            QnpepsE2eGramTimings(
                UInt32(sizeof(QnpepsE2eGramTimings)),
                Int32(0),
                Int32(0),
                Int32(0),
                UInt64(0),
                0.0,
                0.0,
            ),
        ) : nothing
    GC.@preserve output samples o_rows timing begin
        _ffi_gram_run(
            handle,
            pointer(samples),
            pointer(o_rows),
            pointer(output),
            timing === nothing ? Ptr{QnpepsE2eGramTimings}(0) :
            Base.unsafe_convert(Ptr{QnpepsE2eGramTimings}, timing),
        )
    end
    timing === nothing && return output
    value = timing[]
    return (
        gram=output,
        slabs=Int(value.slabs),
        virtual_shards=Int(value.virtual_shards),
        block_calls=Int(value.block_calls),
        slab_width=value.slab_width,
        compute_s=value.compute_s,
        complete_s=value.complete_s,
    )
end

function e2e_raw_gram(
    context::E2eGramContext,
    samples::CuArray{UInt8},
    o_rows::CuArray{ComplexF32};
    timings::Bool=false,
)
    output = CUDA.zeros(ComplexF32, context.n_samples * context.n_samples)
    return e2e_raw_gram!(context, output, samples, o_rows; timings)
end
