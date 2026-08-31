import ..QnpepsDensityApplyArgs
import ..QnpepsDensityFilterArgs
import ..QnpepsDensityWorkspaceQuery
import ..QnpepsElocConfig
import ..QnpepsElocTermTable
import ..QnpepsSamplerHostBatchArgs
import ..QnpepsSamplerHostRefreshArgs
import ..SweepSiteArgs

include("ffi_host/arguments.jl")

density_workspace_sizes(context, query)::Cint = ccall(
    (:qnpeps_density_workspace_sizes, _ffi_library()),
    Cint,
    (Ptr{Cvoid}, Ref{QnpepsDensityWorkspaceQuery}),
    context,
    query,
)

densitymatrix_apply(context, args, stream)::Cint = ccall(
    (:qnpeps_densitymatrix_apply, _ffi_library()),
    Cint,
    (Ptr{Cvoid}, Ref{QnpepsDensityApplyArgs}, Ptr{Cvoid}),
    context,
    args,
    stream,
)

density_filter_apply(context, args, stream)::Cint = ccall(
    (:qnpeps_density_filter_apply, _ffi_library()),
    Cint,
    (Ptr{Cvoid}, Ref{QnpepsDensityFilterArgs}, Ptr{Cvoid}),
    context,
    args,
    stream,
)

sampler_host_batch(context, args)::Cint = ccall(
    (:qnpeps_sampler_host_batch, _ffi_library()),
    Cint,
    (Ptr{Cvoid}, Ptr{QnpepsSamplerHostBatchArgs}),
    context,
    args,
)

sampler_host_refresh(context, args)::Cint = ccall(
    (:qnpeps_sampler_host_refresh, _ffi_library()),
    Cint,
    (Ptr{Cvoid}, Ptr{QnpepsSamplerHostRefreshArgs}),
    context,
    args,
)

sweep_begin(context, seed)::Cint =
    ccall((:qnpeps_sweep_begin, _ffi_library()), Cint, (Ptr{Cvoid}, Ptr{UInt64}), context, seed)

sweep_build_ket_site(context, args)::Cint = ccall(
    (:qnpeps_sweep_build_ket_site, _ffi_library()),
    Cint,
    (Ptr{Cvoid}, Ptr{SweepSiteArgs}),
    context,
    args,
)

sweep_build_env_unsampled_site(context, args)::Cint = ccall(
    (:qnpeps_sweep_build_env_unsampled_site, _ffi_library()),
    Cint,
    (Ptr{Cvoid}, Ptr{SweepSiteArgs}),
    context,
    args,
)

sweep_draw_sigma_site(context, args)::Cint = ccall(
    (:qnpeps_sweep_draw_sigma_site, _ffi_library()),
    Cint,
    (Ptr{Cvoid}, Ptr{SweepSiteArgs}),
    context,
    args,
)

sweep_build_env_above_site(context, args)::Cint = ccall(
    (:qnpeps_sweep_build_env_above_site, _ffi_library()),
    Cint,
    (Ptr{Cvoid}, Ptr{SweepSiteArgs}),
    context,
    args,
)

sweep_finish(context, samples, logpc, lognorm)::Cint = ccall(
    (:qnpeps_sweep_finish, _ffi_library()),
    Cint,
    (Ptr{Cvoid}, Ptr{UInt8}, Ptr{Float64}, Ptr{Float64}),
    context,
    samples,
    logpc,
    lognorm,
)

sweep_graph_policy(context, policy, use_graph, reason)::Cint = ccall(
    (:qnpeps_sweep_graph_policy, _ffi_library()),
    Cint,
    (Ptr{Cvoid}, Int32, Ptr{Int32}, Ptr{Int32}),
    context,
    policy,
    use_graph,
    reason,
)

function eloc_ctx_create(config, rows, terms, selector_mode, stream, out)::Cint
    return ccall(
        (:qnpeps_eloc_ctx_create, _ffi_library()),
        Cint,
        (Ref{QnpepsElocConfig}, Int64, Ptr{Cvoid}, UInt32, Ptr{Cvoid}, Ref{Ptr{Cvoid}}),
        config,
        rows,
        terms,
        selector_mode,
        stream,
        out,
    )
end

function eloc_run(arguments::_ElocRunArguments)::Cint
    inputs = arguments.inputs
    outputs = arguments.outputs
    workspace = arguments.workspace
    return ccall(
        (:qnpeps_eloc_run, _ffi_library()),
        Cint,
        (
            Ref{QnpepsElocConfig},
            CUDA.CuPtr{Cvoid},
            CUDA.CuPtr{UInt8},
            Int64,
            Ref{QnpepsElocTermTable},
            CUDA.CuPtr{Float64},
            CUDA.CuPtr{Float64},
            CUDA.CuPtr{Cvoid},
            Ptr{Cvoid},
            Ptr{Cvoid},
            Float64,
            Ptr{Cvoid},
        ),
        arguments.config,
        inputs.peps,
        inputs.samples,
        arguments.n_samples,
        inputs.terms,
        outputs.logpsi,
        outputs.e_loc,
        outputs.rows,
        workspace.workspace,
        workspace.scratch,
        workspace.reference_energy,
        workspace.stream,
    )
end

eloc_ctx_run(context, args)::Cint =
    ccall((:qnpeps_eloc_ctx_run, _ffi_library()), Cint, (Ptr{Cvoid}, Ptr{Cvoid}), context, args)

eloc_ctx_destroy(context)::Nothing =
    ccall((:qnpeps_eloc_ctx_destroy, _ffi_library()), Cvoid, (Ptr{Cvoid},), context)

eloc_ctx_stats(context, output)::Cint =
    ccall((:qnpeps_eloc_ctx_stats, _ffi_library()), Cint, (Ptr{Cvoid}, Ptr{Cvoid}), context, output)

function eloc_gram_tile(arguments::_ElocGramTileArguments)::Cint
    source = arguments.source
    target = arguments.target
    return ccall(
        (:qnpeps_eloc_gram_tile, _ffi_library()),
        Cint,
        (
            Ptr{QnpepsElocConfig},
            CUDA.CuPtr{Cvoid},
            CUDA.CuPtr{UInt8},
            Int64,
            CUDA.CuPtr{Cvoid},
            CUDA.CuPtr{UInt8},
            Int64,
            CUDA.CuPtr{Cvoid},
            Ptr{Cvoid},
        ),
        arguments.config,
        source.rows,
        source.samples,
        source.count,
        target.rows,
        target.samples,
        target.count,
        arguments.tile,
        arguments.stream,
    )
end

cuda_graph_launch(executable, stream)::CUDA.CUresult = ccall(
    (:cuGraphLaunch, CUDA.libcuda),
    CUDA.CUresult,
    (CUDA.CUgraphExec, CUDA.CUstream),
    executable,
    stream,
)

cuda_memcpy_dto_d_async(destination, source, bytes, stream)::CUDA.CUresult = ccall(
    (:cuMemcpyDtoDAsync_v2, CUDA.libcuda),
    CUDA.CUresult,
    (CUDA.CuPtr{Cvoid}, CUDA.CuPtr{Cvoid}, Csize_t, CUDA.CUstream),
    destination,
    source,
    bytes,
    stream,
)

cuda_memcpy_peer_async(
    destination,
    destination_context,
    source,
    source_context,
    bytes,
    stream,
)::CUDA.CUresult = ccall(
    (:cuMemcpyPeerAsync, CUDA.libcuda),
    CUDA.CUresult,
    (CUDA.CuPtr{Cvoid}, CUDA.CUcontext, CUDA.CuPtr{Cvoid}, CUDA.CUcontext, Csize_t, CUDA.CUstream),
    destination,
    destination_context,
    source,
    source_context,
    bytes,
    stream,
)

cuda_stream_synchronize(stream)::CUDA.CUresult =
    ccall((:cuStreamSynchronize, CUDA.libcuda), CUDA.CUresult, (CUDA.CUstream,), stream)

cuda_memcpy_htod_async(destination, source, bytes, stream)::CUDA.CUresult = ccall(
    (:cuMemcpyHtoDAsync_v2, CUDA.libcuda),
    CUDA.CUresult,
    (CUDA.CuPtr{Cvoid}, Ptr{Cvoid}, Csize_t, CUDA.CUstream),
    destination,
    source,
    bytes,
    stream,
)

cuda_memcpy_2d_async(copy, stream)::CUDA.CUresult = ccall(
    (:cuMemcpy2DAsync_v2, CUDA.libcuda),
    CUDA.CUresult,
    (Ptr{Cvoid}, CUDA.CUstream),
    copy,
    stream,
)

cuda_memset_d8_async(destination, value, bytes, stream)::CUDA.CUresult = ccall(
    (:cuMemsetD8Async, CUDA.libcuda),
    CUDA.CUresult,
    (CUDA.CuPtr{Cvoid}, UInt8, Csize_t, CUDA.CUstream),
    destination,
    value,
    bytes,
    stream,
)
