import .._MinsrHostGramArgs
import .._MinsrHostGramDesc
import .._MinsrHostGramFootprint
import .._MinsrHostMinsrArgs
import .._MinsrHostMinsrDesc

minsr_host_dense_count(descriptor)::Int64 = ccall(
    (:qnpeps_minsr_dense_count, _ffi_library()),
    Int64,
    (Ref{_MinsrHostMinsrDesc},),
    descriptor,
)

minsr_host_compact_count(descriptor)::Int64 = ccall(
    (:qnpeps_minsr_compact_count, _ffi_library()),
    Int64,
    (Ref{_MinsrHostMinsrDesc},),
    descriptor,
)

minsr_host_ctx_create(descriptor, stream, out)::Cint = ccall(
    (:qnpeps_minsr_ctx_create, _ffi_library()),
    Cint,
    (Ref{_MinsrHostMinsrDesc}, Ptr{Cvoid}, Ref{Ptr{Cvoid}}),
    descriptor,
    stream,
    out,
)

minsr_host_ctx_run(context, args)::Cint = ccall(
    (:qnpeps_minsr_ctx_run, _ffi_library()),
    Cint,
    (Ptr{Cvoid}, Ref{_MinsrHostMinsrArgs}),
    context,
    args,
)

minsr_host_ctx_destroy(context)::Nothing =
    ccall((:qnpeps_minsr_ctx_destroy, _ffi_library()), Cvoid, (Ptr{Cvoid},), context)

gram_host_ctx_create(descriptor, stream, out)::Cint = ccall(
    (:qnpeps_gram_ctx_create, _ffi_library()),
    Cint,
    (Ref{_MinsrHostGramDesc}, Ptr{Cvoid}, Ref{Ptr{Cvoid}}),
    descriptor,
    stream,
    out,
)

gram_host_ctx_run(context, args)::Cint = ccall(
    (:qnpeps_gram_ctx_run, _ffi_library()),
    Cint,
    (Ptr{Cvoid}, Ref{_MinsrHostGramArgs}),
    context,
    args,
)

gram_host_ctx_footprint(context, out)::Cint = ccall(
    (:qnpeps_gram_ctx_footprint, _ffi_library()),
    Cint,
    (Ptr{Cvoid}, Ref{_MinsrHostGramFootprint}),
    context,
    out,
)

gram_host_ctx_destroy(context)::Nothing =
    ccall((:qnpeps_gram_ctx_destroy, _ffi_library()), Cvoid, (Ptr{Cvoid},), context)
