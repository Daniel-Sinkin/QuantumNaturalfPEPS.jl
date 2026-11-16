using CUDA

e2e_version() = unsafe_string(FFI.e2e_version())
eloc_version() = unsafe_string(FFI.eloc_version())
sampler_version() = unsafe_string(FFI.capi_version())

function dense_count(config::QnpepsE2eConfig)
    out = Ref{Int64}(0)
    st = FFI.e2e_dense_count(config, out)
    _check(; status=st, what="qnpeps_e2e_dense_count")
    return Int(out[])
end

function compact_count(config::QnpepsE2eConfig)
    out = Ref{Int64}(0)
    st = FFI.e2e_compact_count(config, out)
    _check(; status=st, what="qnpeps_e2e_compact_count")
    return Int(out[])
end

function eloc_compact_count(config::QnpepsElocConfig)
    out = Ref{Int64}(0)
    st = FFI.eloc_compact_count(config, out)
    _check(; status=st, what="qnpeps_eloc_compact_count")
    return Int(out[])
end

function minsr_scratch_bytes(
    config::QnpepsE2eConfig,
    n_samples::Integer,
    host_tile_bytes::Integer=0,
)
    out = Ref{UInt64}(0)
    st = FFI.e2e_minsr_scratch_bytes(config, Int64(n_samples), Int64(host_tile_bytes), out)
    _check(; status=st, what="qnpeps_e2e_minsr_scratch_bytes")
    return UInt64(out[])
end

function _ffi_minsr(arguments)
    status = FFI.e2e_minsr(arguments)
    _check(; status, what="qnpeps_e2e_minsr")
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
    _check(; status, what="qnpeps_e2e_node_create")
    out[] == C_NULL && error("[cuQuantumNaturalfPEPS] node create returned a null handle")
    return out[]
end

function _ffi_node_submit_theta(node::Ptr{Cvoid}, device_peps::CuPtr)
    status = FFI.e2e_node_submit_theta(node, CUDA.CuPtr{Cvoid}(device_peps))
    _check(; status, what="qnpeps_e2e_node_submit_theta")
    return nothing
end

function _ffi_node_step(arguments)
    status = FFI.e2e_node_step(arguments)
    _check(; status, what="qnpeps_e2e_node_step")
    return nothing
end

function _ffi_node_step_euler(node::Ptr{Cvoid}, args::Ref{QnpepsE2eEulerStepArgs})
    status = FFI.e2e_node_step_euler(node, args)
    _check(; status, what="qnpeps_e2e_node_step_euler")
    return nothing
end

function _ffi_node_wait_environment(node::Ptr{Cvoid})
    status = FFI.e2e_node_wait_environment(node)
    _check(; status, what="qnpeps_e2e_node_wait_environment")
    return nothing
end

function _ffi_node_destroy(node::Ptr{Cvoid})
    status = FFI.e2e_node_destroy(node)
    _check(; status, what="qnpeps_e2e_node_destroy")
    return nothing
end

function node_footprint_bytes(
    config::QnpepsE2eConfig,
    terms::OperatorTerms;
    gpus::Integer,
    ns_capacity::Integer,
    ns_ahead::Integer=0,
    dim_batch::Integer,
)
    buffers = _eloc_term_buffers(terms, config)
    out = Ref{UInt64}(0)
    status = GC.@preserve buffers begin
        table = _eloc_term_table(buffers)
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
    _check(; status, what="qnpeps_e2e_node_footprint_bytes")
    return out[]
end
