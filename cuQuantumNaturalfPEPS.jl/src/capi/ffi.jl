module FFI

using CUDA

import .._ffi_library
import ..QnpepsConfig
import ..QnpepsSampleArgs
import ..QnpepsCtxSampleArgs
import ..QnpepsSampleHostArgs
import ..QnpepsZipupPepsRowArgs
import ..QnpepsZipupMpoMpsArgs
import ..QnpepsZipupMpoMpsDesc
import ..QnpepsGramDesc
import ..QnpepsGramArgs
import ..QnpepsGramFootprint
import ..QnpepsMinsrDesc
import ..QnpepsMinsrArgs
import ..MAX_BATCH_SIZE

include("ffi_validation.jl")

capi_version()::Cstring = ccall((:qnpeps_capi_version, _ffi_library()), Cstring, ())

strerror(status)::Cstring = ccall((:qnpeps_strerror, _ffi_library()), Cstring, (Cint,), status)

last_error_file()::Cstring = ccall((:qnpeps_last_error_file, _ffi_library()), Cstring, ())

last_error_line()::Cint = ccall((:qnpeps_last_error_line, _ffi_library()), Cint, ())

last_error_message()::Cstring = ccall((:qnpeps_last_error_message, _ffi_library()), Cstring, ())

function ctx_create(config, out; stream)::Cint
    return ccall(
        (:qnpeps_ctx_create, _ffi_library()),
        Cint,
        (Ref{QnpepsConfig}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}),
        config,
        stream,
        out,
    )
end

ctx_destroy(ctx)::Nothing = ccall((:qnpeps_ctx_destroy, _ffi_library()), Cvoid, (Ptr{Cvoid},), ctx)

function ctx_build_dlenv(ctx, peps; cumulative_row_logs)::Cint
    return ccall(
        (:qnpeps_ctx_build_dlenv, _ffi_library()),
        Cint,
        (Ptr{Cvoid}, CUDA.CuPtr{Cvoid}, CUDA.CuPtr{Float64}),
        ctx,
        peps,
        cumulative_row_logs,
    )
end

function ctx_copy_dlenv_host(ctx, output; output_bytes)::Cint
    return ccall(
        (:qnpeps_ctx_copy_dlenv_host, _ffi_library()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cvoid}, UInt64),
        ctx,
        output,
        output_bytes,
    )
end

function ctx_sample(
    ctx,
    samples_out;
    log_prob_config,
    log_gauge,
    n_samples,
    batch_base,
    dim_batch=max(1, min(n_samples, MAX_BATCH_SIZE)),
)::Cint
    args = QnpepsCtxSampleArgs(
        UInt32(sizeof(QnpepsCtxSampleArgs)),
        UInt(samples_out),
        UInt(log_prob_config),
        UInt(log_gauge),
        UInt64(n_samples),
        UInt64(batch_base),
        UInt64(dim_batch),
    )
    return ccall(
        (:qnpeps_ctx_sample, _ffi_library()),
        Cint,
        (Ptr{Cvoid}, Ref{QnpepsCtxSampleArgs}),
        ctx,
        args,
    )
end

function ctx_sample_host(
    ctx,
    samples_out;
    log_prob_config,
    log_gauge,
    n_samples,
    batch_base,
    dim_batch=max(1, min(n_samples, MAX_BATCH_SIZE)),
)::Cint
    args = QnpepsCtxSampleArgs(
        UInt32(sizeof(QnpepsCtxSampleArgs)),
        UInt(samples_out),
        UInt(log_prob_config),
        UInt(log_gauge),
        UInt64(n_samples),
        UInt64(batch_base),
        UInt64(dim_batch),
    )
    return ccall(
        (:qnpeps_ctx_sample_host, _ffi_library()),
        Cint,
        (Ptr{Cvoid}, Ref{QnpepsCtxSampleArgs}),
        ctx,
        args,
    )
end

function build_dlenv(config, peps; dlenv_out, cumulative_row_logs, stream)::Cint
    return ccall(
        (:qnpeps_build_dlenv, _ffi_library()),
        Cint,
        (Ref{QnpepsConfig}, CUDA.CuPtr{Cvoid}, CUDA.CuPtr{Cvoid}, CUDA.CuPtr{Float64}, Ptr{Cvoid}),
        config,
        peps,
        dlenv_out,
        cumulative_row_logs,
        stream,
    )
end

function sample(
    config,
    peps,
    dlenv;
    gpus,
    scratch,
    scratch_bytes,
    samples_out,
    log_prob_config,
    log_gauge,
    n_samples,
    batch_base,
    dim_batch,
    stream,
)::Cint
    args = QnpepsSampleArgs(
        UInt32(sizeof(QnpepsSampleArgs)),
        UInt(peps),
        UInt(dlenv),
        Int32(gpus),
        UInt(scratch),
        UInt64(scratch_bytes),
        UInt(samples_out),
        UInt(log_prob_config),
        UInt(log_gauge),
        UInt64(n_samples),
        UInt64(batch_base),
        UInt64(dim_batch),
        UInt(stream),
    )
    return ccall(
        (:qnpeps_sample, _ffi_library()),
        Cint,
        (Ref{QnpepsConfig}, Ref{QnpepsSampleArgs}),
        config,
        args,
    )
end

function sample_host(
    config,
    peps,
    dlenv;
    gpus,
    samples_out,
    log_prob_config,
    log_gauge,
    n_samples,
    batch_base,
    dim_batch,
    stream,
)::Cint
    args = QnpepsSampleHostArgs(;
        struct_size=UInt32(sizeof(QnpepsSampleHostArgs)),
        gpus=Int32(gpus),
        peps=UInt(peps),
        dlenv=UInt(dlenv),
        samples_out=UInt(samples_out),
        log_prob_config=UInt(log_prob_config),
        log_gauge=UInt(log_gauge),
        n_samples=UInt64(n_samples),
        batch_base=UInt64(batch_base),
        dim_batch=UInt64(dim_batch),
        stream=UInt(stream),
    )
    return ccall(
        (:qnpeps_sample_host, _ffi_library()),
        Cint,
        (Ref{QnpepsConfig}, Ref{QnpepsSampleHostArgs}),
        config,
        args,
    )
end

function random_unitary_peps(config, peps_out; peps_bytes, seed, alpha, stream)::Cint
    return ccall(
        (:qnpeps_random_unitary_peps, _ffi_library()),
        Cint,
        (Ref{QnpepsConfig}, CUDA.CuPtr{Cvoid}, UInt64, UInt64, Cdouble, Ptr{Cvoid}),
        config,
        peps_out,
        peps_bytes,
        seed,
        alpha,
        stream,
    )
end

zipup_peps_row_bytes(config, maxdim)::Int64 = ccall(
    (:qnpeps_zipup_peps_row_bytes, _ffi_library()),
    Int64,
    (Ref{QnpepsConfig}, Cint),
    config,
    maxdim,
)

function zipup_ctx_create(config, maxdim, out; stream)::Cint
    return ccall(
        (:qnpeps_zipup_ctx_create, _ffi_library()),
        Cint,
        (Ref{QnpepsConfig}, Cint, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}),
        config,
        maxdim,
        stream,
        out,
    )
end

zipup_ctx_destroy(context)::Nothing =
    ccall((:qnpeps_zipup_ctx_destroy, _ffi_library()), Cvoid, (Ptr{Cvoid},), context)

zipup_ctx_begin(context)::Cint =
    ccall((:qnpeps_zipup_ctx_begin, _ffi_library()), Cint, (Ptr{Cvoid},), context)

function zipup_ctx_enqueue_peps_row(context, args)::Cint
    return ccall(
        (:qnpeps_zipup_ctx_enqueue_peps_row, _ffi_library()),
        Cint,
        (Ptr{Cvoid}, Ref{QnpepsZipupPepsRowArgs}),
        context,
        args,
    )
end

function zipup_ctx_enqueue_peps_row(context, args::Ptr{QnpepsZipupPepsRowArgs})::Cint
    return ccall(
        (:qnpeps_zipup_ctx_enqueue_peps_row, _ffi_library()),
        Cint,
        (Ptr{Cvoid}, Ptr{QnpepsZipupPepsRowArgs}),
        context,
        args,
    )
end

function zipup_ctx_finish(context, scales, count)::Cint
    return ccall(
        (:qnpeps_zipup_ctx_finish, _ffi_library()),
        Cint,
        (Ptr{Cvoid}, Ptr{Float64}, UInt64),
        context,
        scales,
        count,
    )
end

zipup_mpo_mps_bytes(descriptor)::Int64 = ccall(
    (:qnpeps_zipup_mpo_mps_bytes, _ffi_library()),
    Int64,
    (Ref{QnpepsZipupMpoMpsDesc},),
    descriptor,
)

function zipup_mpo_mps(descriptor, args)::Cint
    return ccall(
        (:qnpeps_zipup_mpo_mps, _ffi_library()),
        Cint,
        (Ref{QnpepsZipupMpoMpsDesc}, Ref{QnpepsZipupMpoMpsArgs}),
        descriptor,
        args,
    )
end

minsr_dense_count(descriptor)::Int64 =
    ccall((:qnpeps_minsr_dense_count, _ffi_library()), Int64, (Ref{QnpepsMinsrDesc},), descriptor)

minsr_compact_count(descriptor)::Int64 =
    ccall((:qnpeps_minsr_compact_count, _ffi_library()), Int64, (Ref{QnpepsMinsrDesc},), descriptor)

minsr_scratch_bytes(descriptor)::Int64 =
    ccall((:qnpeps_minsr_scratch_bytes, _ffi_library()), Int64, (Ref{QnpepsMinsrDesc},), descriptor)

function minsr(descriptor, args)::Cint
    return ccall(
        (:qnpeps_minsr, _ffi_library()),
        Cint,
        (Ref{QnpepsMinsrDesc}, Ref{QnpepsMinsrArgs}),
        descriptor,
        args,
    )
end

function minsr_ctx_create(descriptor, out; stream)::Cint
    return ccall(
        (:qnpeps_minsr_ctx_create, _ffi_library()),
        Cint,
        (Ref{QnpepsMinsrDesc}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}),
        descriptor,
        stream,
        out,
    )
end

function minsr_ctx_run(context, args)::Cint
    return ccall(
        (:qnpeps_minsr_ctx_run, _ffi_library()),
        Cint,
        (Ptr{Cvoid}, Ref{QnpepsMinsrArgs}),
        context,
        args,
    )
end

minsr_ctx_destroy(context)::Nothing =
    ccall((:qnpeps_minsr_ctx_destroy, _ffi_library()), Cvoid, (Ptr{Cvoid},), context)

function gram_ctx_create(descriptor, out; stream)::Cint
    return ccall(
        (:qnpeps_gram_ctx_create, _ffi_library()),
        Cint,
        (Ref{QnpepsGramDesc}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}),
        descriptor,
        stream,
        out,
    )
end

function gram_ctx_run(context, args)::Cint
    return ccall(
        (:qnpeps_gram_ctx_run, _ffi_library()),
        Cint,
        (Ptr{Cvoid}, Ref{QnpepsGramArgs}),
        context,
        args,
    )
end

function gram_ctx_footprint(context, out)::Cint
    return ccall(
        (:qnpeps_gram_ctx_footprint, _ffi_library()),
        Cint,
        (Ptr{Cvoid}, Ref{QnpepsGramFootprint}),
        context,
        out,
    )
end

gram_ctx_destroy(context)::Nothing =
    ccall((:qnpeps_gram_ctx_destroy, _ffi_library()), Cvoid, (Ptr{Cvoid},), context)

function batched_rangefinder(
    input;
    rows,
    cols,
    rank,
    batch,
    input_stride,
    seed,
    q_out,
    q_stride,
    r_out,
    r_stride,
    scratch,
    scratch_bytes,
    stream,
)::Cint
    return ccall(
        (:qnpeps_batched_rangefinder, _ffi_library()),
        Cint,
        (
            Ptr{Cvoid},
            Cint,
            Cint,
            Cint,
            Cint,
            Int64,
            UInt64,
            Ptr{Cvoid},
            Int64,
            Ptr{Cvoid},
            Int64,
            Ptr{Cvoid},
            UInt64,
            Ptr{Cvoid},
        ),
        input,
        rows,
        cols,
        rank,
        batch,
        input_stride,
        seed,
        q_out,
        q_stride,
        r_out,
        r_stride,
        scratch,
        scratch_bytes,
        stream,
    )
end

function batched_rangefinder_scratch_bytes(rows, cols, rank, batch)::Int64
    return ccall(
        (:qnpeps_batched_rangefinder_scratch_bytes, _ffi_library()),
        Int64,
        (Cint, Cint, Cint, Cint),
        rows,
        cols,
        rank,
        batch,
    )
end

peps_bytes(config)::Int64 =
    ccall((:qnpeps_peps_bytes, _ffi_library()), Int64, (Ref{QnpepsConfig},), config)

dlenv_bytes(config)::Int64 =
    ccall((:qnpeps_dlenv_bytes, _ffi_library()), Int64, (Ref{QnpepsConfig},), config)

function sample_bytes(config, n_samples)::Int64
    return ccall(
        (:qnpeps_sample_bytes, _ffi_library()),
        Int64,
        (Ref{QnpepsConfig}, UInt64),
        config,
        n_samples,
    )
end

function sample_footprint_bytes(config, n_samples, dim_batch)::Int64
    return ccall(
        (:qnpeps_sample_footprint_bytes, _ffi_library()),
        Int64,
        (Ref{QnpepsConfig}, UInt64, UInt64),
        config,
        n_samples,
        dim_batch,
    )
end

function sample_scratch_bytes(config, dim_batch)::Int64
    return ccall(
        (:qnpeps_sample_scratch_bytes, _ffi_library()),
        Int64,
        (Ref{QnpepsConfig}, UInt64),
        config,
        dim_batch,
    )
end

sampler_pool_release()::Nothing = ccall((:qnpeps_sampler_pool_release, _ffi_library()), Cvoid, ())

end
