
function _minsr_dense_count(; descriptor::QnpepsMinsrDesc)::Int64
    return FFI.minsr_dense_count(descriptor)
end

function _minsr_compact_count(; descriptor::QnpepsMinsrDesc)::Int64
    return FFI.minsr_compact_count(descriptor)
end

function _minsr_scratch_bytes(; descriptor::QnpepsMinsrDesc)::Int64
    return FFI.minsr_scratch_bytes(descriptor)
end

function _ffi_minsr(; descriptor::QnpepsMinsrDesc, args::QnpepsMinsrArgs)::Nothing
    status = FFI.minsr(descriptor, args)
    _check(; status, what="qnpeps_minsr")
end

function _ffi_minsr_ctx_create(; descriptor::QnpepsMinsrDesc, stream::Ptr{Cvoid})::Ptr{Cvoid}
    context = Ref{Ptr{Cvoid}}(C_NULL)
    status = FFI.minsr_ctx_create(descriptor, context; stream)
    _check(; status, what="qnpeps_minsr_ctx_create")
    context[] == C_NULL && error("qnpeps_minsr_ctx_create returned a null context")
    return context[]
end

function _ffi_minsr_ctx_run(context::Ptr{Cvoid}, args::QnpepsMinsrArgs)::Nothing
    status = FFI.minsr_ctx_run(context, args)
    _check(; status, what="qnpeps_minsr_ctx_run")
end

function _ffi_minsr_ctx_destroy(context::Ptr{Cvoid})::Nothing
    FFI.minsr_ctx_destroy(context)
    return nothing
end

function _ffi_gram_ctx_create(; descriptor::QnpepsGramDesc, stream::Ptr{Cvoid})::Ptr{Cvoid}
    context = Ref{Ptr{Cvoid}}(C_NULL)
    status = FFI.gram_ctx_create(descriptor, context; stream)
    _check(; status, what="qnpeps_gram_ctx_create")
    context[] == C_NULL && error("qnpeps_gram_ctx_create returned a null context")
    return context[]
end

function _ffi_gram_ctx_run(context::Ptr{Cvoid}, args::QnpepsGramArgs)::Nothing
    status = FFI.gram_ctx_run(context, args)
    _check(; status, what="qnpeps_gram_ctx_run")
end

function _ffi_gram_ctx_footprint(context::Ptr{Cvoid})::QnpepsGramFootprint
    footprint = Ref{QnpepsGramFootprint}(_empty_gram_footprint())
    status = FFI.gram_ctx_footprint(context, footprint)
    _check(; status, what="qnpeps_gram_ctx_footprint")
    return footprint[]
end

function _ffi_gram_ctx_destroy(context::Ptr{Cvoid})::Nothing
    FFI.gram_ctx_destroy(context)
    return nothing
end
