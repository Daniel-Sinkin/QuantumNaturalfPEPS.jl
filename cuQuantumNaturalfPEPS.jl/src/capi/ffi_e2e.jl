import ..QnpepsE2eConfig
import ..QnpepsE2eEulerStepArgs
import ..QnpepsE2eGramFootprint
import ..QnpepsE2eGramTimings
import ..QnpepsE2eUpdateArgs
import ..QnpepsElocConfig

include("ffi_e2e/arguments.jl")

e2e_version()::Cstring = ccall((:qnpeps_e2e_version, _ffi_library()), Cstring, ())

e2e_strerror(status)::Cstring =
    ccall((:qnpeps_e2e_strerror, _ffi_library()), Cstring, (Cint,), status)

eloc_version()::Cstring = ccall((:qnpeps_eloc_version, _ffi_library()), Cstring, ())

e2e_node_error_stage(node)::Cstring =
    ccall((:qnpeps_e2e_node_error_stage, _ffi_library()), Cstring, (Ptr{Cvoid},), node)

e2e_dense_count(config, out)::Cint = ccall(
    (:qnpeps_e2e_dense_count, _ffi_library()),
    Cint,
    (Ref{QnpepsE2eConfig}, Ref{Int64}),
    config,
    out,
)

e2e_compact_count(config, out)::Cint = ccall(
    (:qnpeps_e2e_compact_count, _ffi_library()),
    Cint,
    (Ref{QnpepsE2eConfig}, Ref{Int64}),
    config,
    out,
)

eloc_compact_count(config, out)::Cint = ccall(
    (:qnpeps_eloc_compact_count, _ffi_library()),
    Cint,
    (Ref{QnpepsElocConfig}, Ref{Int64}),
    config,
    out,
)

e2e_minsr_scratch_bytes(config, n_samples, host_tile_bytes, out)::Cint = ccall(
    (:qnpeps_e2e_minsr_scratch_bytes, _ffi_library()),
    Cint,
    (Ref{QnpepsE2eConfig}, Int64, Int64, Ref{UInt64}),
    config,
    n_samples,
    host_tile_bytes,
    out,
)

e2e_step_scratch_bytes(config, n_samples, terms, host_tile_bytes, out)::Cint = ccall(
    (:qnpeps_e2e_step_scratch_bytes, _ffi_library()),
    Cint,
    (Ref{QnpepsE2eConfig}, Int64, Ptr{Cvoid}, Int64, Ref{UInt64}),
    config,
    n_samples,
    terms,
    host_tile_bytes,
    out,
)

e2e_step_multigpu_scratch_bytes(config, n_samples, terms, gpus, host_tile_bytes, out)::Cint = ccall(
    (:qnpeps_e2e_step_multigpu_scratch_bytes, _ffi_library()),
    Cint,
    (Ref{QnpepsE2eConfig}, Int64, Ptr{Cvoid}, Cint, Int64, Ref{UInt64}),
    config,
    n_samples,
    terms,
    gpus,
    host_tile_bytes,
    out,
)

function e2e_minsr(arguments::_E2eMinsrArguments{QnpepsE2eConfig})::Cint
    inputs = arguments.inputs
    cuts = arguments.cuts
    outputs = arguments.outputs
    return ccall(
        (:qnpeps_e2e_minsr, _ffi_library()),
        Cint,
        (
            Ref{QnpepsE2eConfig},
            Int64,
            CUDA.CuPtr{UInt8},
            CUDA.CuPtr{Float64},
            CUDA.CuPtr{Float64},
            CUDA.CuPtr{Float64},
            CUDA.CuPtr{Cvoid},
            CUDA.CuPtr{Cvoid},
            Ptr{Cvoid},
            Int64,
            Float64,
            Float64,
            CUDA.CuPtr{Cvoid},
            Ptr{Float64},
            Ptr{Float64},
            Ptr{Float64},
            Ptr{Cvoid},
        ),
        arguments.config,
        arguments.n_samples,
        inputs.device_samples,
        inputs.logpsi,
        inputs.e_loc,
        inputs.logq,
        inputs.gram,
        inputs.o_rows_device,
        inputs.o_rows_host,
        arguments.host_tile_bytes,
        cuts.relative_cut,
        cuts.absolute_cut,
        outputs.theta_dot,
        outputs.e_mean,
        outputs.e_var,
        outputs.ess,
        arguments.stream,
    )
end

function e2e_minsr_ptr(arguments::_E2eMinsrArguments{Ptr{QnpepsE2eConfig}})::Cint
    inputs = arguments.inputs
    cuts = arguments.cuts
    outputs = arguments.outputs
    return ccall(
        (:qnpeps_e2e_minsr, _ffi_library()),
        Cint,
        (
            Ptr{QnpepsE2eConfig},
            Int64,
            CUDA.CuPtr{UInt8},
            CUDA.CuPtr{Float64},
            CUDA.CuPtr{Float64},
            CUDA.CuPtr{Float64},
            CUDA.CuPtr{Cvoid},
            CUDA.CuPtr{Cvoid},
            Ptr{Cvoid},
            Int64,
            Float64,
            Float64,
            CUDA.CuPtr{Cvoid},
            Ptr{Float64},
            Ptr{Float64},
            Ptr{Float64},
            Ptr{Cvoid},
        ),
        arguments.config,
        arguments.n_samples,
        inputs.device_samples,
        inputs.logpsi,
        inputs.e_loc,
        inputs.logq,
        inputs.gram,
        inputs.o_rows_device,
        inputs.o_rows_host,
        arguments.host_tile_bytes,
        cuts.relative_cut,
        cuts.absolute_cut,
        outputs.theta_dot,
        outputs.e_mean,
        outputs.e_var,
        outputs.ess,
        arguments.stream,
    )
end

function e2e_step(arguments::_E2eStepArguments)::Cint
    inputs = arguments.inputs
    cuts = arguments.cuts
    minsr_outputs = arguments.minsr_outputs
    sample_outputs = arguments.sample_outputs
    return ccall(
        (:qnpeps_e2e_step, _ffi_library()),
        Cint,
        (
            Ref{QnpepsE2eConfig},
            CUDA.CuPtr{Cvoid},
            Int64,
            Ptr{Cvoid},
            Int64,
            Float64,
            Float64,
            CUDA.CuPtr{Cvoid},
            Ptr{Float64},
            Ptr{Float64},
            Ptr{Float64},
            CUDA.CuPtr{UInt8},
            CUDA.CuPtr{Float64},
            CUDA.CuPtr{Float64},
            CUDA.CuPtr{Float64},
            CUDA.CuPtr{Float64},
            Ptr{Cvoid},
            Ptr{Cvoid},
        ),
        arguments.config,
        inputs.device_peps,
        arguments.n_samples,
        inputs.terms,
        arguments.host_tile_bytes,
        cuts.relative_cut,
        cuts.absolute_cut,
        minsr_outputs.theta_dot,
        minsr_outputs.e_mean,
        minsr_outputs.e_var,
        minsr_outputs.ess,
        sample_outputs.samples,
        sample_outputs.logq,
        sample_outputs.log_gauge,
        sample_outputs.logpsi,
        sample_outputs.e_loc,
        sample_outputs.o_rows_host,
        arguments.stream,
    )
end

function e2e_node_create(
    config,
    gpus,
    ns_capacity,
    ns_ahead,
    dim_batch,
    host_tile_bytes,
    terms,
    out,
)::Cint
    return ccall(
        (:qnpeps_e2e_node_create, _ffi_library()),
        Cint,
        (Ref{QnpepsE2eConfig}, Cint, Int64, Int64, Int64, Int64, Ptr{Cvoid}, Ref{Ptr{Cvoid}}),
        config,
        gpus,
        ns_capacity,
        ns_ahead,
        dim_batch,
        host_tile_bytes,
        terms,
        out,
    )
end

e2e_update_ctx_create(config, out)::Cint = ccall(
    (:qnpeps_e2e_update_ctx_create, _ffi_library()),
    Cint,
    (Ref{QnpepsE2eConfig}, Ref{Ptr{Cvoid}}),
    config,
    out,
)

e2e_update_ctx_run(ctx, arguments)::Cint = ccall(
    (:qnpeps_e2e_update_ctx_run, _ffi_library()),
    Cint,
    (Ptr{Cvoid}, Ref{QnpepsE2eUpdateArgs}),
    ctx,
    arguments,
)

e2e_update_ctx_destroy(ctx)::Cint =
    ccall((:qnpeps_e2e_update_ctx_destroy, _ffi_library()), Cint, (Ptr{Cvoid},), ctx)

e2e_node_submit_theta(node, device_peps)::Cint = ccall(
    (:qnpeps_e2e_node_submit_theta, _ffi_library()),
    Cint,
    (Ptr{Cvoid}, CUDA.CuPtr{Cvoid}),
    node,
    device_peps,
)

function e2e_node_step(arguments::_E2eNodeStepArguments)::Cint
    cuts = arguments.cuts
    minsr_outputs = arguments.minsr_outputs
    sample_outputs = arguments.sample_outputs
    return ccall(
        (:qnpeps_e2e_node_step, _ffi_library()),
        Cint,
        (
            Ptr{Cvoid},
            Int64,
            Float64,
            Float64,
            CUDA.CuPtr{Cvoid},
            Ptr{Float64},
            Ptr{Float64},
            Ptr{Float64},
            CUDA.CuPtr{UInt8},
            CUDA.CuPtr{Float64},
            CUDA.CuPtr{Float64},
            CUDA.CuPtr{Float64},
            CUDA.CuPtr{Float64},
            Ptr{Cvoid},
            Ptr{Int64},
        ),
        arguments.node,
        arguments.n_samples,
        cuts.relative_cut,
        cuts.absolute_cut,
        minsr_outputs.theta_dot,
        minsr_outputs.e_mean,
        minsr_outputs.e_var,
        minsr_outputs.ess,
        sample_outputs.samples,
        sample_outputs.logq,
        sample_outputs.log_gauge,
        sample_outputs.logpsi,
        sample_outputs.e_loc,
        sample_outputs.o_rows_host,
        arguments.epoch,
    )
end

e2e_node_step_euler(node, args)::Cint = ccall(
    (:qnpeps_e2e_node_step_euler, _ffi_library()),
    Cint,
    (Ptr{Cvoid}, Ref{QnpepsE2eEulerStepArgs}),
    node,
    args,
)

e2e_node_destroy(node)::Cint =
    ccall((:qnpeps_e2e_node_destroy, _ffi_library()), Cint, (Ptr{Cvoid},), node)

function e2e_node_footprint_bytes(config, gpus, ns_capacity, ns_ahead, dim_batch, terms, out)::Cint
    return ccall(
        (:qnpeps_e2e_node_footprint_bytes, _ffi_library()),
        Cint,
        (Ref{QnpepsE2eConfig}, Cint, Int64, Int64, Int64, Ptr{Cvoid}, Ref{UInt64}),
        config,
        gpus,
        ns_capacity,
        ns_ahead,
        dim_batch,
        terms,
        out,
    )
end

e2e_gram_ctx_create(config, n_samples, stream, out)::Cint = ccall(
    (:qnpeps_e2e_gram_ctx_create, _ffi_library()),
    Cint,
    (Ref{QnpepsE2eConfig}, Int64, Ptr{Cvoid}, Ref{Ptr{Cvoid}}),
    config,
    n_samples,
    stream,
    out,
)

e2e_gram_ctx_run(context, samples, rows, gram, timings)::Cint = ccall(
    (:qnpeps_e2e_gram_ctx_run, _ffi_library()),
    Cint,
    (
        Ptr{Cvoid},
        CUDA.CuPtr{UInt8},
        CUDA.CuPtr{Cvoid},
        CUDA.CuPtr{Cvoid},
        Ptr{QnpepsE2eGramTimings},
    ),
    context,
    samples,
    rows,
    gram,
    timings,
)

e2e_gram_ctx_footprint(context, out)::Cint = ccall(
    (:qnpeps_e2e_gram_ctx_footprint, _ffi_library()),
    Cint,
    (Ptr{Cvoid}, Ref{QnpepsE2eGramFootprint}),
    context,
    out,
)

e2e_gram_ctx_destroy(context)::Nothing =
    ccall((:qnpeps_e2e_gram_ctx_destroy, _ffi_library()), Cvoid, (Ptr{Cvoid},), context)
