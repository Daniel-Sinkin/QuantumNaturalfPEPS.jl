
@inline function _node_step_h2d!(
    host::NodeStepHost,
    destination::CUDA.CuPtr{Cvoid},
    source::Ptr{Cvoid},
    bytes::UInt64,
)
    stream = host.minsr.lanes[1].stream
    status = FFI.cuda_memcpy_htod_async(destination, source, bytes, stream)
    status == CUDA.CUDA_SUCCESS || CUDA.throw_api_error(status)
    return nothing
end

function _node_step_gram!(host::NodeStepHost, lane)::Nothing
    stream = lane.stream
    stream_pointer = Ptr{Cvoid}(stream)
    zero_byte = UInt8(0)
    gram_bytes = UInt(host.n_samples * host.n_samples * sizeof(ComplexF32))
    memset_status = FFI.cuda_memset_d8_async(lane.raw_gram, zero_byte, gram_bytes, stream)
    memset_status == CUDA.CUDA_SUCCESS || CUDA.throw_api_error(memset_status)
    config = Base.unsafe_convert(Ptr{QnpepsElocConfig}, host.eloc_config_ref)
    zero_pointer = Ptr{Cvoid}(0)
    element_bytes = UInt(sizeof(ComplexF32))
    row_bytes = UInt(host.compact) * element_bytes
    for i in 1:NODE_STEP_LANES
        row_base_a = host.minsr.row_bases[i]
        row_count_a = host.minsr.row_counts[i]
        row_count_a <= 0 && continue
        source_a = Ptr{Cvoid}(UInt(host.o_rows_host_pointer) + UInt(row_base_a) * row_bytes)
        _node_step_h2d!(host, host.gram_stage_a, source_a, UInt64(row_count_a) * UInt64(row_bytes))
        samples_a = CUDA.CuPtr{UInt8}(UInt(lane.samples) + UInt(row_base_a * host.sites))
        for j in 1:NODE_STEP_LANES
            row_base_b = host.minsr.row_bases[j]
            row_count_b = host.minsr.row_counts[j]
            row_count_b <= 0 && continue
            rows_b = host.gram_stage_a
            samples_b = samples_a
            if j != i
                source_b = Ptr{Cvoid}(UInt(host.o_rows_host_pointer) + UInt(row_base_b) * row_bytes)
                _node_step_h2d!(
                    host,
                    host.gram_stage_b,
                    source_b,
                    UInt64(row_count_b) * UInt64(row_bytes),
                )
                rows_b = host.gram_stage_b
                samples_b = CUDA.CuPtr{UInt8}(UInt(lane.samples) + UInt(row_base_b * host.sites))
            end
            source = FFI._ElocGramTileRows(;
                rows=host.gram_stage_a,
                samples=samples_a,
                count=Int64(row_count_a),
            )
            target =
                FFI._ElocGramTileRows(; rows=rows_b, samples=samples_b, count=Int64(row_count_b))
            arguments = FFI._ElocGramTileArguments(;
                config,
                source,
                target,
                tile=host.gram_tile,
                stream=stream_pointer,
            )
            tile_status = GC.@preserve host FFI.eloc_gram_tile(arguments)
            tile_status == 0 || error("qnpeps_eloc_gram_tile failed with status $tile_status")
            source_pitch = UInt(row_count_b) * element_bytes
            destination_pitch = UInt(host.n_samples) * element_bytes
            destination =
                UInt(lane.raw_gram) + UInt(row_base_a * host.n_samples + row_base_b) * element_bytes
            host.gram_copy_args[1] = _NodeStepMemcpy2D(;
                src_x_bytes=UInt(0),
                src_y=UInt(0),
                src_memory_type=UInt32(2),
                src_host=zero_pointer,
                src_device=UInt(host.gram_tile),
                src_array=zero_pointer,
                src_pitch=source_pitch,
                dst_x_bytes=UInt(0),
                dst_y=UInt(0),
                dst_memory_type=UInt32(2),
                dst_host=zero_pointer,
                dst_device=destination,
                dst_array=zero_pointer,
                dst_pitch=destination_pitch,
                width_bytes=source_pitch,
                height=UInt(row_count_a),
            )
            copy_status = GC.@preserve host FFI.cuda_memcpy_2d_async(
                Ptr{Cvoid}(pointer(host.gram_copy_args)),
                stream,
            )
            copy_status == CUDA.CUDA_SUCCESS || CUDA.throw_api_error(copy_status)
        end
    end
    _minsr_host_stream_synchronize!(stream)
    return nothing
end

function _node_step_refresh_sampler!(host::NodeStepHost, lane_index::Int)::Nothing
    sampler = host.sampler
    sampler.initialized || throw(ArgumentError("sampler must be warm before refresh"))
    host.sampler_refresh_args[1] = host.sampler_refresh_lane_args[lane_index]
    status = GC.@preserve host sampler FFI.sampler_host_refresh(
        sampler.handle,
        pointer(host.sampler_refresh_args),
    )
    status == 0 || error("qnpeps_sampler_host_refresh failed with status $status")
    sampler.batch_args[1] = host.sampler_batch_lane_args[lane_index]
    sampler.generation = host.peps_generation
    return nothing
end

function _node_step_dlenv!(host::NodeStepHost)::Nothing
    host.stage = :dlenv
    QNP.build_dlenv!(host.dlenv, host.peps, host.peps_generation)
    lane_index = host.dlenv.active_lane
    _node_step_refresh_sampler!(host, lane_index)
    return nothing
end

function _node_step_draw!(host::NodeStepHost)::Nothing
    host.stage = :sampling
    QNP.sample_peps!(
        host.sampler,
        host.samples_host,
        host.logq_host,
        host.log_gauge_host;
        batch_base=host.next_batch,
    )
    host.next_batch += UInt64(host.n_samples ÷ host.dim_batch)
    return nothing
end

function _node_step_stage_samples!(host::NodeStepHost)::Nothing
    _node_step_h2d!(
        host,
        host.samples_pointer,
        host.samples_host_pointer,
        UInt64(host.n_samples * host.sites),
    )
    _node_step_h2d!(
        host,
        host.logq_pointer,
        host.logq_host_pointer,
        UInt64(host.n_samples * sizeof(Float64)),
    )
    _node_step_h2d!(
        host,
        host.log_gauge_pointer,
        host.log_gauge_host_pointer,
        UInt64(host.n_samples * sizeof(Float64)),
    )
    _minsr_host_stream_synchronize!(host.minsr.lanes[1].stream)
    return nothing
end

function _node_step_sampling!(host::NodeStepHost)::Nothing
    _node_step_draw!(host)
    _node_step_stage_samples!(host)
    return nothing
end
