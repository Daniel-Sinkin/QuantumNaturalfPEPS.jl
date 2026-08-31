
@inline function _minsr_host_arena_total(current::UInt64, bytes::UInt64)
    padding = mod(-current, _MINSR_HOST_ALIGNMENT)
    aligned = current + padding
    return aligned + bytes
end

@inline function _minsr_host_arena_take(base::UInt, offset::UInt64, bytes::UInt64)
    aligned = offset + mod(-offset, _MINSR_HOST_ALIGNMENT)
    return base + UInt(aligned), aligned + bytes
end

function _minsr_host_cpu_set(low::Int, high::Int)
    words = ntuple(Val(16)) do word
        value = UInt64(0)
        for cpu in low:high
            cpu ÷ 64 + 1 == word && (value |= UInt64(1) << (cpu % 64))
        end
        value
    end
    return _MinsrHostCpuSet(words)
end

function _minsr_host_pin_worker!(topology::MinsrHostTopology, lane_index::Int)
    original = Ref(_MinsrHostCpuSet(ntuple(_ -> UInt64(0), Val(16))))
    saved =
        ccall(
            :sched_getaffinity,
            Cint,
            (Cint, Csize_t, Ref{_MinsrHostCpuSet}),
            0,
            sizeof(_MinsrHostCpuSet),
            original,
        ) == 0
    saved || return false, false, original[]
    set = _minsr_host_cpu_set(topology.cpu_low[lane_index], topology.cpu_high[lane_index])
    status = ccall(
        :sched_setaffinity,
        Cint,
        (Cint, Csize_t, Ref{_MinsrHostCpuSet}),
        0,
        sizeof(_MinsrHostCpuSet),
        set,
    )
    return true, status == 0, original[]
end

function _minsr_host_restore_worker!(set::_MinsrHostCpuSet)
    return ccall(
        :sched_setaffinity,
        Cint,
        (Cint, Csize_t, Ref{_MinsrHostCpuSet}),
        0,
        sizeof(_MinsrHostCpuSet),
        set,
    ) == 0
end

@inline function _minsr_host_peer_copy!(
    destination::CUDA.CuPtr{Cvoid},
    destination_context::CUDA.CuContext,
    source::CUDA.CuPtr{Cvoid},
    source_context::CUDA.CuContext,
    bytes::UInt64,
    stream::CUDA.CUstream,
)
    status = if destination_context == source_context
        FFI.cuda_memcpy_dto_d_async(destination, source, bytes, stream)
    else
        FFI.cuda_memcpy_peer_async(
            destination,
            destination_context.handle,
            source,
            source_context.handle,
            bytes,
            stream,
        )
    end
    status == CUDA.CUDA_SUCCESS || CUDA.throw_api_error(status)
    return nothing
end

@inline function _minsr_host_stream_synchronize!(stream::CUDA.CUstream)
    status = FFI.cuda_stream_synchronize(stream)
    status == CUDA.CUDA_SUCCESS || CUDA.throw_api_error(status)
    return nothing
end

function _minsr_host_release_lane_resources!(
    context::CUDA.CuContext,
    stream::CUDA.CUstream,
    gram::Ptr{Cvoid},
    minsr::Ptr{Cvoid},
    arena,
)
    CUDA.context!(context)
    stream == C_NULL || CUDA.cuStreamSynchronize(stream)
    minsr == C_NULL || FFI.minsr_host_ctx_destroy(minsr)
    gram == C_NULL || FFI.gram_host_ctx_destroy(gram)
    arena === nothing || CUDA.free(arena)
    stream == C_NULL || CUDA.cuStreamDestroy_v2(stream)
    return nothing
end

function _minsr_host_validate_descriptor(
    lx::Integer,
    ly::Integer,
    dim_phys::Integer,
    dim_bond::Integer,
    n_samples::Integer,
    topology::MinsrHostTopology,
)
    lx >= 2 || throw(ArgumentError("lx must be at least two"))
    ly >= 2 || throw(ArgumentError("ly must be at least two"))
    dim_phys == 2 || throw(ArgumentError("dim_phys must equal two"))
    dim_bond >= 1 || throw(ArgumentError("dim_bond must be positive"))
    n_samples >= MINSR_HOST_LANES || throw(ArgumentError("n_samples must be at least four"))
    length(CUDA.devices()) == MINSR_HOST_LANES || throw(
        ArgumentError("the $(topology.name) minSR host requires exactly four visible devices"),
    )
    Threads.threadpoolsize(:default) >= MINSR_HOST_LANES ||
        throw(ArgumentError("the $(topology.name) minSR host requires at least four Julia threads"))
    sizeof(_MinsrHostGramDesc) == 40 || error("QnpepsGramDesc ABI mismatch")
    sizeof(_MinsrHostGramArgs) == 64 || error("QnpepsGramArgs ABI mismatch")
    sizeof(_MinsrHostGramFootprint) == 64 || error("QnpepsGramFootprint ABI mismatch")
    sizeof(_MinsrHostMinsrDesc) == 48 || error("QnpepsMinsrDesc ABI mismatch")
    sizeof(_MinsrHostMinsrArgs) == 176 || error("QnpepsMinsrArgs ABI mismatch")
    return nothing
end

function _minsr_host_enable_peers!(devices, contexts)
    enabled = 0
    for source in 1:MINSR_HOST_LANES
        CUDA.context!(contexts[source])
        for target in 1:MINSR_HOST_LANES
            source == target && continue
            try
                CUDA.enable_peer_access(contexts[target])
                enabled += 1
            catch error
                if error isa CUDA.CuError &&
                   error.code == CUDA.CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED
                    enabled += 1
                else
                    rethrow()
                end
            end
        end
    end
    enabled == 12 || error("all-pairs peer enable did not cover twelve directed pairs")
    return Int32(enabled)
end
