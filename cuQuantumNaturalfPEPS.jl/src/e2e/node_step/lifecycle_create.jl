
function _node_step_validate(
    config::QnpepsE2eConfig,
    peps::CuVector{ComplexF32},
    n_samples::Integer,
    dim_batch::Integer,
)
    Int(config.lx) >= 2 || throw(ArgumentError("lx must be at least two"))
    Int(config.ly) >= 2 || throw(ArgumentError("ly must be at least two"))
    Int(config.dim_phys) == 2 || throw(ArgumentError("dim_phys must equal two"))
    n_samples >= NODE_STEP_LANES ||
        throw(ArgumentError("n_samples must be at least the lane count"))
    1 <= dim_batch <= QNP.MAX_BATCH_SIZE ||
        throw(ArgumentError("dim_batch is outside the sampler batch range"))
    n_samples % dim_batch == 0 || throw(ArgumentError("n_samples must be divisible by dim_batch"))
    dense_count(config) == length(peps) ||
        throw(DimensionMismatch("PEPS storage does not match the configuration"))
    length(Threads.threadpooltids(:default)) >= 2 * NODE_STEP_LANES ||
        throw(ArgumentError("NodeStepHost requires at least eight Julia threads"))
    CUDA.device().handle == 0 || throw(ArgumentError("NodeStepHost must be created on GPU zero"))
    return nothing
end

function _node_step_row_partition(n_samples::Int64)
    quotient, remainder = divrem(n_samples, Int64(NODE_STEP_LANES))
    counts = ntuple(lane -> quotient + (lane <= remainder ? Int64(1) : Int64(0)), NODE_STEP_LANES)
    bases = ntuple(lane -> lane == 1 ? Int64(0) : sum(counts[1:(lane-1)]), NODE_STEP_LANES)
    return bases, counts
end

function NodeStepHost(
    peps::CuVector{ComplexF32},
    config::QnpepsE2eConfig,
    terms::HeisenbergTerms;
    n_samples::Integer,
    dim_batch::Integer=config.sample_batch == 0 ? n_samples : config.sample_batch,
    host_tile_bytes::Integer=0,
    capture::Symbol=:auto,
    eo_backend::Symbol=:c_abi,
    eo_selector::EoSelector=_EO_SELECTOR_DEFAULT,
    minsr_topology::MinsrHostTopology=MINSR_HOST_TOPOLOGY_JURECA,
)
    _node_step_validate(config, peps, n_samples, dim_batch)
    host_tile_bytes >= 0 || throw(ArgumentError("host_tile_bytes must be nonnegative"))
    eo_backend in (:c_abi, :julia) || throw(ArgumentError("eo_backend must be :c_abi or :julia"))
    sites = Int64(config.lx) * Int64(config.ly)
    compact = Int64(compact_count(config))
    dense = Int64(dense_count(config))
    wrapped =
        QNP.CuPeps(peps, Int(config.lx), Int(config.ly), Int(config.dim_phys), Int(config.dim_bond))
    source_peps_pointer = CUDA.CuPtr{Cvoid}(pointer(peps))
    source_peps_bytes = UInt64(length(peps) * sizeof(ComplexF32))
    dlenv =
        QNP.DlenvHost(wrapped; chi_s=Int(config.chi_s), chi_dl=Int(config.chi_dl), capture=capture)
    initial_dlenv = QNP.build_dlenv!(dlenv, wrapped, 0)
    sampler = QNP.SamplerHost(
        wrapped,
        initial_dlenv;
        chi_s=Int(config.chi_s),
        chi_dl=Int(config.chi_dl),
        seed=config.seed,
        sampling_mode=config.sampling_mode == 0 ? :fast : :full,
        chi_c=Int(config.contract_dim),
        dim_batch=dim_batch,
        generation=0,
    )
    samples_host = Vector{UInt8}(undef, n_samples * sites)
    logq_host = Vector{Float64}(undef, n_samples)
    log_gauge_host = Vector{Float64}(undef, n_samples)
    o_rows_host = Vector{ComplexF32}(undef, n_samples * compact)
    minsr_statistics = Vector{Float64}(undef, 4)
    CUDA.pin(samples_host)
    CUDA.pin(logq_host)
    CUDA.pin(log_gauge_host)
    QNP.sample_peps!(sampler, samples_host, logq_host, log_gauge_host; batch_base=UInt64(0))
    samples = CUDA.zeros(UInt8, n_samples * sites)
    logq = CUDA.zeros(Float64, n_samples)
    log_gauge = CUDA.zeros(Float64, n_samples)
    logpsi = CUDA.zeros(Float64, 2 * n_samples)
    e_loc = CUDA.zeros(Float64, 2 * n_samples)
    theta_dot = CUDA.zeros(ComplexF32, dense)
    samples_pointer = CUDA.CuPtr{Cvoid}(pointer(samples))
    logq_pointer = CUDA.CuPtr{Cvoid}(pointer(logq))
    log_gauge_pointer = CUDA.CuPtr{Cvoid}(pointer(log_gauge))
    logpsi_pointer = CUDA.CuPtr{Cvoid}(pointer(logpsi))
    e_loc_pointer = CUDA.CuPtr{Cvoid}(pointer(e_loc))
    theta_dot_pointer = CUDA.CuPtr{Cvoid}(pointer(theta_dot))
    bases, counts = _node_step_row_partition(Int64(n_samples))
    lconfig = QnpepsElocConfig(;
        lx=Int(config.lx),
        ly=Int(config.ly),
        dim_bond=Int(config.dim_bond),
        chi_eo=Int(config.chi_eo),
        meo=Int(config.meo),
        dim_phys=Int(config.dim_phys),
    )
    o_rows_registration = try
        GC.@preserve o_rows_host CUDA.register(
            CUDA.HostMemory,
            pointer(o_rows_host),
            sizeof(o_rows_host),
            CUDA.MEMHOSTREGISTER_PORTABLE,
        )
    catch
        close(sampler)
        close(dlenv)
        rethrow()
    end
    julia_eo = nothing
    if eo_backend === :julia
        try
            eo_arguments = EoHostArguments(;
                peps,
                config=lconfig,
                table=eo_term_table(lconfig, terms),
                n_samples,
                compact,
                samples,
                logpsi,
                e_loc,
                rows=o_rows_host,
            )
            julia_eo = EoHost(eo_arguments)
            set_eo_selector!(julia_eo, eo_selector)
            _node_step_precompile_eo_host!(dlenv, sampler)
        catch
            CUDA.context!(o_rows_registration.ctx)
            CUDA.unregister(o_rows_registration)
            close(sampler)
            close(dlenv)
            rethrow()
        end
    end
    minsr = try
        MinsrHost(;
            lx=Int(config.lx),
            ly=Int(config.ly),
            dim_bond=Int(config.dim_bond),
            dim_phys=Int(config.dim_phys),
            n_samples=n_samples,
            topology=minsr_topology,
        )
    catch
        julia_eo === nothing || close(julia_eo)
        CUDA.context!(o_rows_registration.ctx)
        CUDA.unregister(o_rows_registration)
        close(sampler)
        close(dlenv)
        rethrow()
    end
    caller_context = CUDA.context()
    eo_lanes =
        eo_backend === :c_abi ? Vector{_NodeStepEoLane}(undef, NODE_STEP_LANES) : _NodeStepEoLane[]
    built = 0
    try
        if eo_backend === :c_abi
            for lane_index in 1:NODE_STEP_LANES
                geometry = _NodeStepEoGeometry(;
                    peps_elements=length(peps),
                    sites,
                    compact,
                    row_base=bases[lane_index],
                    row_count=counts[lane_index],
                )
                outputs = _NodeStepEoOutputs(;
                    samples=samples_pointer,
                    logpsi=logpsi_pointer,
                    e_loc=e_loc_pointer,
                    rows=Ptr{Cvoid}(pointer(o_rows_host)),
                )
                arguments = _NodeStepEoLaneArguments(;
                    context=minsr.lanes[lane_index].context,
                    config=lconfig,
                    terms,
                    selector=eo_selector,
                    geometry,
                    outputs,
                )
                eo_lanes[lane_index] = _node_step_create_eo_lane(arguments)
                built += 1
            end
        else
            for lane_index in 1:NODE_STEP_LANES
                julia_eo.lanes[lane_index].context == minsr.lanes[lane_index].context ||
                    error("E/O and minSR primary contexts differ")
            end
        end
    catch
        for lane_index in 1:built
            _node_step_destroy_eo_lane!(eo_lanes[lane_index])
        end
        julia_eo === nothing || close(julia_eo)
        CUDA.context!(o_rows_registration.ctx)
        CUDA.unregister(o_rows_registration)
        close(minsr)
        close(sampler)
        close(dlenv)
        rethrow()
    finally
        CUDA.context!(caller_context)
    end
    max_rows = maximum(counts)
    gram_stage_bytes = UInt64(max_rows * compact * sizeof(ComplexF32))
    gram_tile_bytes = UInt64(max_rows * max_rows * sizeof(ComplexF32))
    gram_scratch = try
        CUDA.zeros(UInt8, 2 * gram_stage_bytes + gram_tile_bytes)
    catch
        for lane in eo_lanes
            _node_step_destroy_eo_lane!(lane)
        end
        julia_eo === nothing || close(julia_eo)
        CUDA.context!(o_rows_registration.ctx)
        CUDA.unregister(o_rows_registration)
        close(minsr)
        close(sampler)
        close(dlenv)
        rethrow()
    end
    gram_scratch_base = UInt(pointer(gram_scratch))
    gram_stage_a = CUDA.CuPtr{Cvoid}(gram_scratch_base)
    gram_stage_b = CUDA.CuPtr{Cvoid}(gram_scratch_base + UInt(gram_stage_bytes))
    gram_tile = CUDA.CuPtr{Cvoid}(gram_scratch_base + UInt(2 * gram_stage_bytes))
    gram_copy_args = [
        _NodeStepMemcpy2D(;
            src_x_bytes=UInt(0),
            src_y=UInt(0),
            src_memory_type=UInt32(0),
            src_host=Ptr{Cvoid}(0),
            src_device=UInt(0),
            src_array=Ptr{Cvoid}(0),
            src_pitch=UInt(0),
            dst_x_bytes=UInt(0),
            dst_y=UInt(0),
            dst_memory_type=UInt32(0),
            dst_host=Ptr{Cvoid}(0),
            dst_device=UInt(0),
            dst_array=Ptr{Cvoid}(0),
            dst_pitch=UInt(0),
            width_bytes=UInt(0),
            height=UInt(0),
        ),
    ]
    header_bytes = UInt(length(sampler.dims) * sizeof(Int32))
    dlenv_value_pointers = ntuple(lane -> UInt(pointer(dlenv.lanes[lane].data)) + header_bytes, 2)
    refresh_template = QNP.QnpepsSamplerHostRefreshArgs(;
        struct_size=UInt32(sizeof(QNP.QnpepsSamplerHostRefreshArgs)),
        peps_layout=Int32(0),
        peps=UInt(source_peps_pointer),
        dlenv_values=UInt(0),
        sampling=sampler.batch_args[1].sampling,
        sampling_bytes=sampler.batch_args[1].sampling_bytes,
    )
    sampler_refresh_lane_args = ntuple(
        lane -> QNP.QnpepsSamplerHostRefreshArgs(;
            struct_size=refresh_template.struct_size,
            peps_layout=refresh_template.peps_layout,
            peps=refresh_template.peps,
            dlenv_values=dlenv_value_pointers[lane],
            sampling=refresh_template.sampling,
            sampling_bytes=refresh_template.sampling_bytes,
        ),
        2,
    )
    batch_template = sampler.batch_args[1]
    sampler_batch_lane_args = ntuple(
        lane -> QNP.QnpepsSamplerHostBatchArgs(;
            struct_size=batch_template.struct_size,
            reserved=batch_template.reserved,
            peps=batch_template.peps,
            dlenv_dims=batch_template.dlenv_dims,
            dlenv_dims_count=batch_template.dlenv_dims_count,
            dlenv_values=dlenv_value_pointers[lane],
            scratch=batch_template.scratch,
            scratch_bytes=batch_template.scratch_bytes,
            sampling=batch_template.sampling,
            sampling_bytes=batch_template.sampling_bytes,
            dlenv_pointers=batch_template.dlenv_pointers,
            dlenv_pointer_count=batch_template.dlenv_pointer_count,
            samples_out=batch_template.samples_out,
            log_prob_config=batch_template.log_prob_config,
            log_gauge=batch_template.log_gauge,
            batch_seed=batch_template.batch_seed,
            batch_id=batch_template.batch_id,
            dim_batch=batch_template.dim_batch,
            peps_layout=batch_template.peps_layout,
            reserved2=batch_template.reserved2,
        ),
        2,
    )
    row_pointers =
        julia_eo === nothing ? ntuple(lane -> UInt(eo_lanes[lane].rows_pointer), NODE_STEP_LANES) :
        eo_host_row_pointers(julia_eo)
    row_bytes = ntuple(lane -> UInt64(counts[lane] * compact * sizeof(ComplexF32)), NODE_STEP_LANES)
    inputs = MinsrHostInputs(;
        samples=UInt(samples_pointer),
        samples_bytes=UInt64(n_samples * sites),
        logpsi=UInt(logpsi_pointer),
        logpsi_bytes=UInt64(2 * n_samples * sizeof(Float64)),
        e_loc=UInt(e_loc_pointer),
        e_loc_bytes=UInt64(2 * n_samples * sizeof(Float64)),
        logq=UInt(logq_pointer),
        logq_bytes=UInt64(n_samples * sizeof(Float64)),
        row_shards=row_pointers,
        row_shard_bytes=row_bytes,
        theta_dot=UInt(theta_dot_pointer),
        theta_dot_bytes=UInt64(dense * sizeof(ComplexF32)),
        relative_cut=0.0,
        absolute_cut=0.0,
    )
    julia_eo === nothing || seal_eo_host!(julia_eo)
    host = NodeStepHost(;
        config,
        config_ref=Ref(config),
        terms,
        peps=wrapped,
        source_peps_pointer,
        source_peps_bytes,
        peps_generation=Int64(0),
        dlenv,
        sampler,
        minsr,
        eo_lanes,
        eo_host=julia_eo,
        eo_backend,
        workers=Task[],
        eloc_config=lconfig,
        eloc_config_ref=Ref(lconfig),
        gram_scratch,
        gram_stage_a,
        gram_stage_b,
        gram_tile,
        gram_copy_args,
        samples_host,
        logq_host,
        log_gauge_host,
        o_rows_host,
        o_rows_registration,
        minsr_statistics,
        samples_host_pointer=Ptr{Cvoid}(pointer(samples_host)),
        logq_host_pointer=Ptr{Cvoid}(pointer(logq_host)),
        log_gauge_host_pointer=Ptr{Cvoid}(pointer(log_gauge_host)),
        o_rows_host_pointer=Ptr{Cvoid}(pointer(o_rows_host)),
        minsr_statistics_pointer=pointer(minsr_statistics),
        samples,
        logq,
        log_gauge,
        logpsi,
        e_loc,
        theta_dot,
        samples_pointer,
        logq_pointer,
        log_gauge_pointer,
        logpsi_pointer,
        e_loc_pointer,
        theta_dot_pointer,
        dlenv_value_pointers,
        sampler_refresh_args=[sampler_refresh_lane_args[1]],
        sampler_refresh_lane_args,
        sampler_batch_lane_args,
        minsr_inputs=inputs,
        telemetry=NodeStepTelemetry(),
        n_samples=Int64(n_samples),
        sites,
        compact,
        dense,
        dim_batch=Int64(dim_batch),
        host_tile_bytes=Int64(host_tile_bytes),
        next_batch=UInt64(0),
        epoch=Int64(0),
        stage=:ready,
        command=Threads.Atomic{Int32}(_NODE_STEP_EO_RUN),
        command_epoch=Threads.Atomic{Int}(0),
        ready=Threads.Atomic{Int}(0),
        done=Threads.Atomic{Int}(0),
        running=Threads.Atomic{Bool}(false),
        closed=Threads.Atomic{Bool}(false),
    )
    _node_step_start_workers!(host)
    finalizer(close, host)
    return host
end

Base.isopen(host::NodeStepHost) = !host.closed[]

function bind_node_peps!(host::NodeStepHost, peps::CuVector{ComplexF32})::NodeStepHost
    isopen(host) || throw(ArgumentError("NodeStepHost is closed"))
    length(peps) == host.dense || throw(DimensionMismatch("PEPS storage size changed"))
    peps === host.peps.data || throw(ArgumentError("NodeStepHost requires stable PEPS storage"))
    host.eo_host === nothing || bind_eo_host!(host.eo_host, peps)
    host.peps_generation += 1
    return host
end

function reset_node_step_schedule!(
    host::NodeStepHost;
    next_batch::Integer=0,
    epoch::Integer=0,
)::NodeStepHost
    next_batch >= 0 || throw(ArgumentError("next_batch must be nonnegative"))
    epoch >= 0 || throw(ArgumentError("epoch must be nonnegative"))
    host.next_batch = UInt64(next_batch)
    host.epoch = Int64(epoch)
    return host
end
