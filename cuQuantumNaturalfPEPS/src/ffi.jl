module FFI

import .._sym
import ..QnpepsConfig
import ..QnpepsSampleArgs
import ..QnpepsCtxSampleArgs
import ..QnpepsSampleHostArgs
import ..QnpepsZipupPepsRowArgs
import ..QnpepsZipupMpoMpsArgs
import ..QnpepsZipupMpoMpsDesc
import ..MAX_BATCH_SIZE

capi_version()::Cstring = @ccall $(_sym(; name=:qnpeps_capi_version))()::Cstring

strerror(status)::Cstring = @ccall $(_sym(; name=:qnpeps_strerror))(status::Cint)::Cstring

last_error_file()::Cstring = @ccall $(_sym(; name=:qnpeps_last_error_file))()::Cstring

last_error_line()::Cint = @ccall $(_sym(; name=:qnpeps_last_error_line))()::Cint

last_error_message()::Cstring =
    @ccall $(_sym(; name=:qnpeps_last_error_message))()::Cstring

function ctx_create(config, out; stream)::Cint
    return @ccall $(_sym(; name=:qnpeps_ctx_create))(
        config::Ref{QnpepsConfig},
        stream::Ptr{Cvoid},
        out::Ptr{Ptr{Cvoid}},
    )::Cint
end

ctx_destroy(ctx)::Nothing =
    @ccall $(_sym(; name=:qnpeps_ctx_destroy))(ctx::Ptr{Cvoid})::Cvoid

function ctx_build_dlenv(ctx, peps; cumulative_row_logs)::Cint
    return @ccall $(_sym(; name=:qnpeps_ctx_build_dlenv))(
        ctx::Ptr{Cvoid},
        peps::Ptr{Cvoid},
        cumulative_row_logs::Ptr{Float64},
    )::Cint
end

function ctx_copy_dlenv_host(ctx, output; output_bytes)::Cint
    return @ccall $(_sym(; name=:qnpeps_ctx_copy_dlenv_host))(
        ctx::Ptr{Cvoid},
        output::Ptr{Cvoid},
        output_bytes::UInt64,
    )::Cint
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
    return @ccall $(_sym(; name=:qnpeps_ctx_sample))(
        ctx::Ptr{Cvoid}, args::Ref{QnpepsCtxSampleArgs}
    )::Cint
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
    return @ccall $(_sym(; name=:qnpeps_ctx_sample_host))(
        ctx::Ptr{Cvoid}, args::Ref{QnpepsCtxSampleArgs}
    )::Cint
end

function build_dlenv(config, peps; dlenv_out, cumulative_row_logs, stream)::Cint
    return @ccall $(_sym(; name=:qnpeps_build_dlenv))(
        config::Ref{QnpepsConfig},
        peps::Ptr{Cvoid},
        dlenv_out::Ptr{Cvoid},
        cumulative_row_logs::Ptr{Float64},
        stream::Ptr{Cvoid},
    )::Cint
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
    return @ccall $(_sym(; name=:qnpeps_sample))(
        config::Ref{QnpepsConfig},
        args::Ref{QnpepsSampleArgs},
    )::Cint
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
    args = QnpepsSampleHostArgs(
        UInt32(sizeof(QnpepsSampleHostArgs)),
        Int32(gpus),
        UInt(peps),
        UInt(dlenv),
        UInt(samples_out),
        UInt(log_prob_config),
        UInt(log_gauge),
        UInt64(n_samples),
        UInt64(batch_base),
        UInt64(dim_batch),
        UInt(stream),
    )
    return @ccall $(_sym(; name=:qnpeps_sample_host))(
        config::Ref{QnpepsConfig},
        args::Ref{QnpepsSampleHostArgs},
    )::Cint
end

zipup_peps_row_bytes(config, maxdim)::Int64 =
    @ccall $(_sym(; name=:qnpeps_zipup_peps_row_bytes))(
        config::Ref{QnpepsConfig}, maxdim::Cint
    )::Int64

function zipup_ctx_create(config, maxdim, out; stream)::Cint
    return @ccall $(_sym(; name=:qnpeps_zipup_ctx_create))(
        config::Ref{QnpepsConfig},
        maxdim::Cint,
        stream::Ptr{Cvoid},
        out::Ptr{Ptr{Cvoid}},
    )::Cint
end

zipup_ctx_destroy(context)::Nothing =
    @ccall $(_sym(; name=:qnpeps_zipup_ctx_destroy))(context::Ptr{Cvoid})::Cvoid

zipup_ctx_begin(context)::Cint =
    @ccall $(_sym(; name=:qnpeps_zipup_ctx_begin))(context::Ptr{Cvoid})::Cint

function zipup_ctx_enqueue_peps_row(context, args)::Cint
    return @ccall $(_sym(; name=:qnpeps_zipup_ctx_enqueue_peps_row))(
        context::Ptr{Cvoid}, args::Ref{QnpepsZipupPepsRowArgs}
    )::Cint
end

function zipup_ctx_finish(context, scales, count)::Cint
    return @ccall $(_sym(; name=:qnpeps_zipup_ctx_finish))(
        context::Ptr{Cvoid},
        scales::Ptr{Float64},
        count::UInt64,
    )::Cint
end

zipup_mpo_mps_bytes(descriptor)::Int64 =
    @ccall $(_sym(; name=:qnpeps_zipup_mpo_mps_bytes))(
        descriptor::Ref{QnpepsZipupMpoMpsDesc}
    )::Int64

function zipup_mpo_mps(descriptor, args)::Cint
    return @ccall $(_sym(; name=:qnpeps_zipup_mpo_mps))(
        descriptor::Ref{QnpepsZipupMpoMpsDesc},
        args::Ref{QnpepsZipupMpoMpsArgs},
    )::Cint
end

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
    return @ccall $(_sym(; name=:qnpeps_batched_rangefinder))(
        input::Ptr{Cvoid},
        rows::Cint,
        cols::Cint,
        rank::Cint,
        batch::Cint,
        input_stride::Int64,
        seed::UInt64,
        q_out::Ptr{Cvoid},
        q_stride::Int64,
        r_out::Ptr{Cvoid},
        r_stride::Int64,
        scratch::Ptr{Cvoid},
        scratch_bytes::UInt64,
        stream::Ptr{Cvoid},
    )::Cint
end

function batched_rangefinder_scratch_bytes(rows, cols, rank, batch)::Int64
    fn = _sym(; name=:qnpeps_batched_rangefinder_scratch_bytes)
    return @ccall $fn(rows::Cint, cols::Cint, rank::Cint, batch::Cint)::Int64
end

peps_bytes(config)::Int64 =
    @ccall $(_sym(; name=:qnpeps_peps_bytes))(config::Ref{QnpepsConfig})::Int64

dlenv_bytes(config)::Int64 =
    @ccall $(_sym(; name=:qnpeps_dlenv_bytes))(config::Ref{QnpepsConfig})::Int64

function sample_bytes(config, n_samples)::Int64
    fn = _sym(; name=:qnpeps_sample_bytes)
    return @ccall $fn(config::Ref{QnpepsConfig}, n_samples::UInt64)::Int64
end

function sample_footprint_bytes(config, n_samples, dim_batch)::Int64
    fn = _sym(; name=:qnpeps_sample_footprint_bytes)
    return @ccall $fn(
        config::Ref{QnpepsConfig},
        n_samples::UInt64,
        dim_batch::UInt64,
    )::Int64
end

function sample_scratch_bytes(config, dim_batch)::Int64
    fn = _sym(; name=:qnpeps_sample_scratch_bytes)
    return @ccall $fn(config::Ref{QnpepsConfig}, dim_batch::UInt64)::Int64
end

sampler_pool_release()::Nothing =
    @ccall $(_sym(; name=:qnpeps_sampler_pool_release))()::Cvoid

end
