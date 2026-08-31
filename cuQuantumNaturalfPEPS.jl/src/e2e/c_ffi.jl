using CUDA

struct QnpepsE2eUpdateArgs
    struct_size::UInt32
    reserved::UInt32
    state_f64_io::CuPtr{Cvoid}
    state_f64_bytes::UInt64
    theta_dot::CuPtr{Cvoid}
    theta_dot_bytes::UInt64
    peps_f32_out::CuPtr{Cvoid}
    peps_f32_bytes::UInt64
    learning_rate::Float64
    stream::Ptr{Cvoid}
end

e2e_version() = unsafe_string(FFI.e2e_version())
e2e_strerror(status::Integer) = unsafe_string(FFI.e2e_strerror(Cint(status)))
eloc_version() = unsafe_string(FFI.eloc_version())
sampler_version() = unsafe_string(FFI.capi_version())

@inline function _check(status::Integer, what::AbstractString)
    status == 0 && return
    error("[cuQuantumNaturalfPEPS] $what failed (status $status, $(e2e_strerror(status)))")
end

@inline function _check_node(status::Integer, what::AbstractString, node::Ptr{Cvoid})
    status == 0 && return
    stage = unsafe_string(FFI.e2e_node_error_stage(node))
    error(
        "[cuQuantumNaturalfPEPS] $what failed (stage $stage, status $status, $(e2e_strerror(status)))",
    )
end

function dense_count(config::QnpepsE2eConfig)
    out = Ref{Int64}(0)
    st = FFI.e2e_dense_count(config, out)
    _check(st, "qnpeps_e2e_dense_count")
    return Int(out[])
end

function compact_count(config::QnpepsE2eConfig)
    out = Ref{Int64}(0)
    st = FFI.e2e_compact_count(config, out)
    _check(st, "qnpeps_e2e_compact_count")
    return Int(out[])
end

function eloc_compact_count(config::QnpepsElocConfig)
    out = Ref{Int64}(0)
    st = FFI.eloc_compact_count(config, out)
    st == 0 || error("[cuQuantumNaturalfPEPS] qnpeps_eloc_compact_count failed (status $st)")
    return Int(out[])
end

function minsr_scratch_bytes(
    config::QnpepsE2eConfig,
    n_samples::Integer,
    host_tile_bytes::Integer=0,
)
    out = Ref{UInt64}(0)
    st = FFI.e2e_minsr_scratch_bytes(config, Int64(n_samples), Int64(host_tile_bytes), out)
    _check(st, "qnpeps_e2e_minsr_scratch_bytes")
    return UInt64(out[])
end

function step_scratch_bytes(
    config::QnpepsE2eConfig,
    n_samples::Integer,
    terms::HeisenbergTerms,
    host_tile_bytes::Integer=0,
)
    out = Ref{UInt64}(0)
    diag = terms.diag
    flip = terms.flip
    st = GC.@preserve diag flip begin
        table = QnpepsElocTermTable(
            Int32(length(diag)),
            length(diag) == 0 ? Ptr{QnpepsElocDiagBond}(0) : pointer(diag),
            Int32(length(flip)),
            length(flip) == 0 ? Ptr{QnpepsElocFlipTerm}(0) : pointer(flip),
        )
        table_ref = Ref(table)
        GC.@preserve table_ref FFI.e2e_step_scratch_bytes(
            config,
            Int64(n_samples),
            Base.unsafe_convert(Ptr{Cvoid}, table_ref),
            Int64(host_tile_bytes),
            out,
        )
    end
    _check(st, "qnpeps_e2e_step_scratch_bytes")
    return UInt64(out[])
end

function step_multigpu_scratch_bytes(
    config::QnpepsE2eConfig,
    n_samples::Integer,
    terms::HeisenbergTerms,
    gpus::Integer,
    host_tile_bytes::Integer=0,
)
    out = Ref{UInt64}(0)
    diag = terms.diag
    flip = terms.flip
    st = GC.@preserve diag flip begin
        table = QnpepsElocTermTable(
            Int32(length(diag)),
            length(diag) == 0 ? Ptr{QnpepsElocDiagBond}(0) : pointer(diag),
            Int32(length(flip)),
            length(flip) == 0 ? Ptr{QnpepsElocFlipTerm}(0) : pointer(flip),
        )
        table_ref = Ref(table)
        GC.@preserve table_ref FFI.e2e_step_multigpu_scratch_bytes(
            config,
            Int64(n_samples),
            Base.unsafe_convert(Ptr{Cvoid}, table_ref),
            Cint(gpus),
            Int64(host_tile_bytes),
            out,
        )
    end
    _check(st, "qnpeps_e2e_step_multigpu_scratch_bytes")
    return UInt64(out[])
end

function _ffi_minsr(arguments)
    status = FFI.e2e_minsr(arguments)
    _check(status, "qnpeps_e2e_minsr")
    return nothing
end

function _ffi_step(arguments)
    status = FFI.e2e_step(arguments)
    _check(status, "qnpeps_e2e_step")
    return nothing
end

function _ffi_node_create(
    config::QnpepsE2eConfig,
    gpus::Integer,
    ns_capacity::Integer,
    ns_ahead::Integer,
    dim_batch::Integer,
    host_tile_bytes::Integer,
    terms::Ptr{Cvoid},
)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = FFI.e2e_node_create(
        config,
        Cint(gpus),
        Int64(ns_capacity),
        Int64(ns_ahead),
        Int64(dim_batch),
        Int64(host_tile_bytes),
        terms,
        out,
    )
    _check(status, "qnpeps_e2e_node_create")
    out[] == C_NULL && error("[cuQuantumNaturalfPEPS] node create returned a null handle")
    return out[]
end

function _ffi_update_ctx_create(config::QnpepsE2eConfig)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = FFI.e2e_update_ctx_create(config, out)
    _check(status, "qnpeps_e2e_update_ctx_create")
    out[] == C_NULL && error("[cuQuantumNaturalfPEPS] update context create returned a null handle")
    return out[]
end

function _ffi_update_ctx_run(ctx::Ptr{Cvoid}, arguments::Ref{QnpepsE2eUpdateArgs})
    status = FFI.e2e_update_ctx_run(ctx, arguments)
    _check(status, "qnpeps_e2e_update_ctx_run")
    return nothing
end

function _ffi_update_ctx_destroy(ctx::Ptr{Cvoid})
    status = FFI.e2e_update_ctx_destroy(ctx)
    _check(status, "qnpeps_e2e_update_ctx_destroy")
    return nothing
end

function _ffi_node_submit_theta(node::Ptr{Cvoid}, device_peps::CuPtr)
    status = FFI.e2e_node_submit_theta(node, CUDA.CuPtr{Cvoid}(device_peps))
    _check(status, "qnpeps_e2e_node_submit_theta")
    return nothing
end

function _ffi_node_step(arguments)
    status = FFI.e2e_node_step(arguments)
    _check_node(status, "qnpeps_e2e_node_step", arguments.node)
    return nothing
end

function _ffi_node_step_euler(node::Ptr{Cvoid}, args::Ref{QnpepsE2eEulerStepArgs})
    status = FFI.e2e_node_step_euler(node, args)
    _check_node(status, "qnpeps_e2e_node_step_euler", node)
    return nothing
end

function _ffi_node_destroy(node::Ptr{Cvoid})
    status = FFI.e2e_node_destroy(node)
    _check(status, "qnpeps_e2e_node_destroy")
    return nothing
end

function node_footprint_bytes(
    config::QnpepsE2eConfig,
    terms::HeisenbergTerms;
    gpus::Integer,
    ns_capacity::Integer,
    ns_ahead::Integer=0,
    dim_batch::Integer,
)
    diag = terms.diag
    flip = terms.flip
    out = Ref{UInt64}(0)
    status = GC.@preserve diag flip begin
        table = QnpepsElocTermTable(
            Int32(length(diag)),
            isempty(diag) ? Ptr{QnpepsElocDiagBond}(0) : pointer(diag),
            Int32(length(flip)),
            isempty(flip) ? Ptr{QnpepsElocFlipTerm}(0) : pointer(flip),
        )
        table_ref = Ref(table)
        GC.@preserve table_ref FFI.e2e_node_footprint_bytes(
            config,
            Cint(gpus),
            Int64(ns_capacity),
            Int64(ns_ahead),
            Int64(dim_batch),
            Base.unsafe_convert(Ptr{Cvoid}, table_ref),
            out,
        )
    end
    _check(status, "qnpeps_e2e_node_footprint_bytes")
    return out[]
end

function _ffi_gram_create(config::QnpepsE2eConfig, n_samples::Integer, stream::CuStream)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = FFI.e2e_gram_ctx_create(
        config,
        Int64(n_samples),
        Ptr{Cvoid}(stream.handle),
        out,
    )
    _check(status, "qnpeps_e2e_gram_ctx_create")
    out[] == C_NULL && error("[cuQuantumNaturalfPEPS] Gram create returned a null handle")
    return out[]
end

function _ffi_gram_run(
    context::Ptr{Cvoid},
    samples::CuPtr,
    rows::CuPtr,
    gram::CuPtr,
    timings::Ptr{QnpepsE2eGramTimings},
)
    status = FFI.e2e_gram_ctx_run(
        context,
        CUDA.CuPtr{UInt8}(samples),
        CUDA.CuPtr{Cvoid}(rows),
        CUDA.CuPtr{Cvoid}(gram),
        timings,
    )
    _check(status, "qnpeps_e2e_gram_ctx_run")
    return nothing
end

function _ffi_gram_footprint(context::Ptr{Cvoid})
    output = Ref(
        QnpepsE2eGramFootprint(;
            struct_size=UInt32(sizeof(QnpepsE2eGramFootprint)),
            reserved=UInt32(0),
            context_device_bytes=UInt64(0),
            geometry_device_bytes=UInt64(0),
            dense_a_device_bytes=UInt64(0),
            dense_b_device_bytes=UInt64(0),
            caller_samples_bytes=UInt64(0),
            caller_rows_bytes=UInt64(0),
            caller_gram_bytes=UInt64(0),
        ),
    )
    status = FFI.e2e_gram_ctx_footprint(context, output)
    _check(status, "qnpeps_e2e_gram_ctx_footprint")
    return output[]
end

function _ffi_gram_destroy(context::Ptr{Cvoid})
    FFI.e2e_gram_ctx_destroy(context)
    return nothing
end
