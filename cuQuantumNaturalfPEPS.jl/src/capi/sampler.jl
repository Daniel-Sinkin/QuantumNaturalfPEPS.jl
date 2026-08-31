
function _ffi_sample(;
    config::QnpepsConfig,
    peps::CuPtr,
    dlenv::CuPtr,
    gpus::Integer,
    scratch::CuPtr,
    scratch_bytes::Integer,
    samples::CuPtr,
    log_prob_config::CuPtr,
    log_gauge::CuPtr,
    n_samples::Integer,
    batch_base::Integer,
    dim_batch::Integer,
)::Nothing
    status = GC.@preserve peps dlenv scratch samples FFI.sample(
        config,
        peps,
        dlenv;
        gpus,
        scratch,
        scratch_bytes,
        samples_out=samples,
        log_prob_config,
        log_gauge,
        n_samples,
        batch_base,
        dim_batch,
        stream=Ptr{Cvoid}(CUDA.stream().handle),
    )
    _check(; status, what="qnpeps_sample")
end

function _ffi_sample_host(;
    config::QnpepsConfig,
    peps::CuPtr,
    dlenv::CuPtr,
    gpus::Integer,
    samples::Ptr{UInt8},
    log_prob_config::Ptr{Float64},
    log_gauge::Ptr{Float64},
    n_samples::Integer,
    batch_base::Integer,
    dim_batch::Integer,
)::Nothing
    status = GC.@preserve peps dlenv samples log_prob_config log_gauge FFI.sample_host(
        config,
        peps,
        dlenv;
        gpus,
        samples_out=samples,
        log_prob_config,
        log_gauge,
        n_samples,
        batch_base,
        dim_batch,
        stream=Ptr{Cvoid}(CUDA.stream().handle),
    )
    _check(; status, what="qnpeps_sample_host")
end

function _ffi_pool_release()::Nothing
    FFI.sampler_pool_release()
end

function _batched_rangefinder_scratch_bytes(;
    rows::Integer,
    cols::Integer,
    rank::Integer,
    batch::Integer,
)::Int64
    return FFI.batched_rangefinder_scratch_bytes(rows, cols, rank, batch)
end

function _ffi_batched_rangefinder(;
    input::CuPtr,
    rows::Integer,
    cols::Integer,
    rank::Integer,
    batch::Integer,
    input_stride::Integer,
    seed::Integer,
    q_out::CuPtr,
    q_stride::Integer,
    r_out::CuPtr,
    r_stride::Integer,
    scratch::CuPtr,
    scratch_bytes::Integer,
)::Nothing
    status = GC.@preserve input q_out r_out scratch FFI.batched_rangefinder(
        input;
        rows=Cint(rows),
        cols=Cint(cols),
        rank=Cint(rank),
        batch=Cint(batch),
        input_stride=Int64(input_stride),
        seed=UInt64(seed),
        q_out,
        q_stride=Int64(q_stride),
        r_out,
        r_stride=Int64(r_stride),
        scratch,
        scratch_bytes=UInt64(scratch_bytes),
        stream=Ptr{Cvoid}(CUDA.stream().handle),
    )
    _check(; status, what="qnpeps_batched_rangefinder")
end
