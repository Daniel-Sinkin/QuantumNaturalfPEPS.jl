
function _dlenv_bytes(; config::QnpepsConfig)::Int64
    return FFI.dlenv_bytes(config)
end

function _sample_bytes(; config::QnpepsConfig, n_samples::Integer)::Int64
    return FFI.sample_bytes(config, UInt64(n_samples))
end

function _scratch_bytes(; config::QnpepsConfig, dim_batch::Integer)::Int64
    return FFI.sample_scratch_bytes(config, UInt64(dim_batch))
end

function _sample_footprint_bytes(;
    config::QnpepsConfig,
    n_samples::Integer,
    dim_batch::Integer,
)::Int64
    return FFI.sample_footprint_bytes(config, UInt64(n_samples), UInt64(dim_batch))
end

function _peps_bytes(; config::QnpepsConfig)::Int64
    return FFI.peps_bytes(config)
end

function _ffi_random_unitary_peps(;
    config::QnpepsConfig,
    peps::CuPtr,
    peps_bytes::Integer,
    seed::Integer,
    alpha::Real,
)::Nothing
    status = GC.@preserve peps FFI.random_unitary_peps(
        config,
        peps;
        peps_bytes=UInt64(peps_bytes),
        seed=UInt64(seed),
        alpha=Cdouble(alpha),
        stream=Ptr{Cvoid}(CUDA.stream().handle),
    )
    _check(; status, what="qnpeps_random_unitary_peps")
end

function _ffi_build_dlenv(;
    config::QnpepsConfig,
    peps::CuPtr,
    dlenv::CuPtr,
    cumulative_row_logs::CuPtr,
)::Nothing
    status = GC.@preserve peps dlenv FFI.build_dlenv(
        config,
        peps;
        dlenv_out=dlenv,
        cumulative_row_logs,
        stream=Ptr{Cvoid}(CUDA.stream().handle),
    )
    _check(; status, what="qnpeps_build_dlenv")
end

function _ffi_ctx_create(; config::QnpepsConfig, stream::Ptr{Cvoid})::Ptr{Cvoid}
    context = Ref{Ptr{Cvoid}}(C_NULL)
    status = FFI.ctx_create(config, context; stream)
    _check(; status, what="qnpeps_ctx_create")
    context[] == C_NULL && error("qnpeps_ctx_create returned a null context")
    return context[]
end

function _ffi_ctx_destroy(context::Ptr{Cvoid})::Nothing
    FFI.ctx_destroy(context)
    return nothing
end

function _ffi_ctx_build_dlenv(context::Ptr{Cvoid}, peps::CuPtr, cumulative_row_logs::CuPtr)::Nothing
    status = GC.@preserve peps cumulative_row_logs FFI.ctx_build_dlenv(
        context,
        peps;
        cumulative_row_logs,
    )
    _check(; status, what="qnpeps_ctx_build_dlenv")
end

function _ffi_ctx_copy_dlenv_host(
    context::Ptr{Cvoid},
    output::Ptr{UInt8},
    output_bytes::Integer,
)::Nothing
    status = GC.@preserve output FFI.ctx_copy_dlenv_host(
        context,
        output;
        output_bytes=UInt64(output_bytes),
    )
    _check(; status, what="qnpeps_ctx_copy_dlenv_host")
end

function _ffi_ctx_sample(
    context::Ptr{Cvoid};
    samples::CuPtr,
    log_prob_config::CuPtr,
    log_gauge::CuPtr,
    n_samples::Integer,
    batch_base::Integer,
    dim_batch::Integer,
)::Nothing
    status = GC.@preserve samples log_prob_config log_gauge FFI.ctx_sample(
        context,
        samples;
        log_prob_config,
        log_gauge,
        n_samples,
        batch_base,
        dim_batch,
    )
    _check(; status, what="qnpeps_ctx_sample")
end

function _ffi_ctx_sample_host(
    context::Ptr{Cvoid};
    samples::Ptr{UInt8},
    log_prob_config::Ptr{Float64},
    log_gauge::Ptr{Float64},
    n_samples::Integer,
    batch_base::Integer,
    dim_batch::Integer,
)::Nothing
    status = GC.@preserve samples log_prob_config log_gauge FFI.ctx_sample_host(
        context,
        samples;
        log_prob_config,
        log_gauge,
        n_samples,
        batch_base,
        dim_batch,
    )
    _check(; status, what="qnpeps_ctx_sample_host")
end
