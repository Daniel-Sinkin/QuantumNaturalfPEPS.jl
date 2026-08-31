
function _zipup_peps_row_bytes(; config::QnpepsConfig, maxdim::Integer)::Int64
    return FFI.zipup_peps_row_bytes(config, Cint(maxdim))
end

function _ffi_zipup_ctx_create(;
    config::QnpepsConfig,
    maxdim::Integer,
    stream::Ptr{Cvoid},
)::Ptr{Cvoid}
    context = Ref{Ptr{Cvoid}}(C_NULL)
    status = FFI.zipup_ctx_create(config, Cint(maxdim), context; stream)
    _check(; status, what="qnpeps_zipup_ctx_create")
    context[] == C_NULL && error("qnpeps_zipup_ctx_create returned a null context")
    return context[]
end

function _ffi_zipup_ctx_destroy(context::Ptr{Cvoid})::Nothing
    FFI.zipup_ctx_destroy(context)
    return nothing
end

function _ffi_zipup_ctx_begin(context::Ptr{Cvoid})::Nothing
    status = FFI.zipup_ctx_begin(context)
    _check(; status, what="qnpeps_zipup_ctx_begin")
end

function _ffi_zipup_ctx_enqueue_peps_row(context::Ptr{Cvoid}, args::QnpepsZipupPepsRowArgs)::Nothing
    status = FFI.zipup_ctx_enqueue_peps_row(context, args)
    _check(; status, what="qnpeps_zipup_ctx_enqueue_peps_row")
end

function _ffi_zipup_ctx_enqueue_peps_row(
    context::Ptr{Cvoid},
    args::Ptr{QnpepsZipupPepsRowArgs},
)::Nothing
    status = FFI.zipup_ctx_enqueue_peps_row(context, args)
    _check(; status, what="qnpeps_zipup_ctx_enqueue_peps_row")
end

function _ffi_zipup_ctx_finish(context::Ptr{Cvoid}, scales::Vector{Float64})::Nothing
    status =
        GC.@preserve scales FFI.zipup_ctx_finish(context, pointer(scales), UInt64(length(scales)))
    _check(; status, what="qnpeps_zipup_ctx_finish")
end

function _zipup_mpo_mps_bytes(; descriptor::QnpepsZipupMpoMpsDesc)::Int64
    return FFI.zipup_mpo_mps_bytes(descriptor)
end

function _ffi_zipup_mpo_mps(;
    descriptor::QnpepsZipupMpoMpsDesc,
    args::QnpepsZipupMpoMpsArgs,
)::Nothing
    status = FFI.zipup_mpo_mps(descriptor, args)
    _check(; status, what="qnpeps_zipup_mpo_mps")
end
