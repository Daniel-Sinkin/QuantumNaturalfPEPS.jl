
function _eo_host_create_lane(arguments::_EoHostLaneArguments)
    context = arguments.context
    config_ref = arguments.config_ref
    table = arguments.table
    geometry = arguments.geometry
    destinations = arguments.destinations
    selector = arguments.selector
    peps_elements = geometry.peps_elements
    sites = geometry.sites
    compact = geometry.compact
    row_base = geometry.row_base
    row_count = geometry.row_count
    source_samples = destinations.samples
    destination_logpsi = destinations.logpsi
    destination_e_loc = destinations.e_loc
    destination_rows = destinations.rows
    CUDA.context!(context)
    stream_ref = Ref{CUDA.CUstream}(CUDA.CUstream(C_NULL))
    CUDA.cuStreamCreate(stream_ref, CUDA.CU_STREAM_DEFAULT)
    stream = stream_ref[]
    binding = nothing
    handle = Ptr{Cvoid}(C_NULL)
    try
        binding = _eo_host_create_binding(
            peps_elements,
            sites,
            compact,
            row_base,
            row_count,
            destination_rows,
            UInt64(1),
            selector,
        )
        handle = _eo_host_create_handle(config_ref, table, row_count, stream)
        sample_offset = UInt(row_base * sites)
        scalar_offset = UInt(2 * row_base * sizeof(Float64))
        return _EoHostLane(;
            context,
            stream,
            handle,
            binding,
            source_sample_pointer=CUDA.CuPtr{Cvoid}(UInt(source_samples) + sample_offset),
            destination_logpsi_pointer=CUDA.CuPtr{Cvoid}(UInt(destination_logpsi) + scalar_offset),
            destination_e_loc_pointer=CUDA.CuPtr{Cvoid}(UInt(destination_e_loc) + scalar_offset),
            row_base,
            row_count,
            original_affinity=ntuple(_ -> UInt64(0), Val(16)),
            affinity_saved=Int32(0),
            affinity_applied=Int32(0),
            status=Int32(0),
            failure_location="",
            failure_message="",
        )
    catch
        handle == C_NULL || FFI.eloc_ctx_destroy(handle)
        binding === nothing || CUDA.free(binding.arena)
        stream == C_NULL || CUDA.cuStreamDestroy_v2(stream)
        rethrow()
    end
end

function _eo_host_release_partial!(lanes::Vector{_EoHostLane})
    for lane in lanes
        CUDA.context!(lane.context)
        lane.stream == C_NULL || CUDA.cuStreamSynchronize(lane.stream)
    end
    for lane in lanes
        CUDA.context!(lane.context)
        lane.handle == C_NULL || FFI.eloc_ctx_destroy(lane.handle)
        lane.handle = C_NULL
    end
    for lane in lanes
        CUDA.context!(lane.context)
        CUDA.free(lane.binding.arena)
    end
    for lane in lanes
        CUDA.context!(lane.context)
        lane.stream == C_NULL || CUDA.cuStreamDestroy_v2(lane.stream)
        lane.stream = CUDA.CUstream(C_NULL)
    end
    empty!(lanes)
    return nothing
end

function _eo_host_arguments(
    peps::CuVector{ComplexF32},
    config::QnpepsElocConfig,
    table::EoTermTable,
    values::Tuple{
        Integer,
        Integer,
        CuVector{UInt8},
        CuVector{Float64},
        CuVector{Float64},
        Vector{ComplexF32},
    },
)::EoHostArguments
    n_samples, compact, samples, logpsi, e_loc, rows = values
    return EoHostArguments(; peps, config, table, n_samples, compact, samples, logpsi, e_loc, rows)
end

function EoHost(
    peps::CuVector{ComplexF32},
    config::QnpepsElocConfig,
    terms::HeisenbergTerms,
    values::Vararg{Any,6},
)
    table = eo_term_table(config, terms)
    return EoHost(_eo_host_arguments(peps, config, table, values))
end

function EoHost(
    peps::CuVector{ComplexF32},
    config::QnpepsElocConfig,
    table::EoTermTable,
    values::Vararg{Any,6},
)
    return EoHost(_eo_host_arguments(peps, config, table, values))
end

function EoHost(input::EoHostArguments)
    _eo_host_validate(input)
    peps = input.peps
    config = input.config
    table = input.table
    n_samples = input.n_samples
    compact = input.compact
    samples = input.samples
    logpsi = input.logpsi
    e_loc = input.e_loc
    rows = input.rows
    config_ref = Ref(config)
    selector = _EO_SELECTOR_DEFAULT
    n_samples_i64 = Int64(n_samples)
    sites = Int64(config.lx) * Int64(config.ly)
    compact_i64 = Int64(compact)
    row_bases, row_counts = _eo_host_partition(n_samples_i64)
    source_peps_pointer = CUDA.CuPtr{Cvoid}(pointer(peps))
    source_samples_pointer = CUDA.CuPtr{Cvoid}(pointer(samples))
    destination_logpsi_pointer = CUDA.CuPtr{Cvoid}(pointer(logpsi))
    destination_e_loc_pointer = CUDA.CuPtr{Cvoid}(pointer(e_loc))
    destination_rows_pointer = Ptr{Cvoid}(pointer(rows))
    caller_context = CUDA.context()
    devices = ntuple(lane -> CUDA.CuDevice(lane - 1), EO_HOST_LANES)
    contexts = ntuple(lane -> CUDA.context(devices[lane]), EO_HOST_LANES)
    lanes = _EoHostLane[]
    peer_pairs = Int32(0)
    try
        peer_pairs = _eo_host_enable_peers!(contexts)
        sizehint!(lanes, EO_HOST_LANES)
        for lane_index in 1:EO_HOST_LANES
            geometry = _EoHostLaneGeometry(;
                peps_elements=length(peps),
                sites,
                compact=compact_i64,
                row_base=row_bases[lane_index],
                row_count=row_counts[lane_index],
            )
            destinations = _EoHostLaneDestinations(;
                samples=source_samples_pointer,
                logpsi=destination_logpsi_pointer,
                e_loc=destination_e_loc_pointer,
                rows=destination_rows_pointer,
            )
            arguments = _EoHostLaneArguments(;
                context=contexts[lane_index],
                config_ref,
                table,
                geometry,
                destinations,
                selector,
            )
            push!(lanes, _eo_host_create_lane(arguments))
        end
    catch
        _eo_host_release_partial!(lanes)
        rethrow()
    finally
        CUDA.context!(caller_context)
    end
    host = EoHost(;
        config,
        config_ref,
        term_table=table,
        selector,
        source_peps=peps,
        source_samples=samples,
        destination_logpsi=logpsi,
        destination_e_loc=e_loc,
        destination_rows=rows,
        source_context=contexts[1],
        source_peps_pointer,
        source_samples_pointer,
        destination_logpsi_pointer,
        destination_e_loc_pointer,
        destination_rows_pointer,
        source_peps_bytes=UInt64(length(peps) * sizeof(ComplexF32)),
        n_samples=n_samples_i64,
        sites,
        compact=compact_i64,
        row_bases,
        row_counts,
        lanes,
        peer_pairs_enabled=peer_pairs,
        source_generation=Int64(0),
        contexts_created=Int64(EO_HOST_LANES),
        contexts_destroyed=Int64(0),
        arenas_created=Int64(EO_HOST_LANES),
        arenas_freed=Int64(0),
        streams_created=Int64(EO_HOST_LANES),
        streams_destroyed=Int64(0),
        replacements=Int64(0),
        gc_runs=Int64(0),
        teardown_order=Int32[0, 0, 0, 0, 0, 0],
        refs_cleared=false,
        sealed=false,
        closed=false,
    )
    advance_eo_selector!(host, selector.epoch)
    finalizer(close, host)
    return host
end

Base.isopen(host::EoHost) = !host.closed
