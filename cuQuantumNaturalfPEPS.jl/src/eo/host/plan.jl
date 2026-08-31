
@inline function _eo_host_arena_total(current::UInt64, bytes::UInt64)
    padding = mod(-current, _EO_HOST_ALIGNMENT)
    aligned = current + padding
    return aligned + bytes
end

@inline function _eo_host_arena_take(base::UInt, offset::UInt64, bytes::UInt64)
    aligned = offset + mod(-offset, _EO_HOST_ALIGNMENT)
    return base + UInt(aligned), aligned + bytes
end

function _eo_host_enable_peers!(contexts)
    enabled = 0
    for source in 1:EO_HOST_LANES
        CUDA.context!(contexts[source])
        for target in 1:EO_HOST_LANES
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

function _eo_host_validate(input::EoHostArguments)
    peps = input.peps
    config = input.config
    table = input.table
    n_samples = input.n_samples
    compact = input.compact
    samples = input.samples
    logpsi = input.logpsi
    e_loc = input.e_loc
    rows = input.rows
    Int(config.lx) >= 2 || throw(ArgumentError("lx must be at least two"))
    Int(config.ly) >= 2 || throw(ArgumentError("ly must be at least two"))
    Int(config.dim_phys) == 2 || throw(ArgumentError("dim_phys must equal two"))
    n_samples >= EO_HOST_LANES || throw(ArgumentError("n_samples must be at least four"))
    compact >= 1 || throw(ArgumentError("compact must be positive"))
    sites = Int(config.lx) * Int(config.ly)
    length(samples) == n_samples * sites ||
        throw(DimensionMismatch("sample storage does not match the E/O descriptor"))
    length(logpsi) == 2 * n_samples ||
        throw(DimensionMismatch("logpsi storage does not match the E/O descriptor"))
    length(e_loc) == 2 * n_samples ||
        throw(DimensionMismatch("E_loc storage does not match the E/O descriptor"))
    length(rows) == n_samples * compact ||
        throw(DimensionMismatch("O-row storage does not match the E/O descriptor"))
    length(peps) >= 1 || throw(ArgumentError("PEPS storage must be nonempty"))
    table.lx == config.lx || throw(DimensionMismatch("term-table lx does not match config"))
    table.ly == config.ly || throw(DimensionMismatch("term-table ly does not match config"))
    _eo_term_table_validate_pointers(table)
    length(CUDA.devices()) == EO_HOST_LANES ||
        throw(ArgumentError("EoHost requires exactly four visible devices"))
    CUDA.device().handle == 0 || throw(ArgumentError("EoHost must be created on GPU zero"))
    return nothing
end

function _eo_host_partition(n_samples::Int64)
    quotient, remainder = divrem(n_samples, Int64(EO_HOST_LANES))
    counts = ntuple(lane -> quotient + (lane <= remainder ? Int64(1) : Int64(0)), EO_HOST_LANES)
    bases = ntuple(lane -> lane == 1 ? Int64(0) : sum(counts[1:(lane-1)]), EO_HOST_LANES)
    return bases, counts
end

function _eo_host_create_handle(
    config_ref::Base.RefValue{QnpepsElocConfig},
    table::EoTermTable,
    row_count::Int64,
    stream::CUDA.CUstream,
)
    _eo_term_table_validate_pointers(table)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    table_ref = table.abi_ref
    status = GC.@preserve config_ref table table_ref FFI.eloc_ctx_create(
        config_ref,
        row_count,
        Base.unsafe_convert(Ptr{Cvoid}, table_ref),
        _EO_HOST_ROWS,
        Ptr{Cvoid}(stream),
        output,
    )
    _check(; status, what="qnpeps_eloc_ctx_create")
    output[] == C_NULL && error("qnpeps_eloc_ctx_create returned a null context")
    return output[]
end

function _eo_host_create_binding(
    peps_elements::Int,
    sites::Int64,
    compact::Int64,
    row_base::Int64,
    row_count::Int64,
    destination_rows::Ptr{Cvoid},
    epoch::UInt64,
    selector::EoSelector,
)
    sizes = (
        UInt64(peps_elements) * UInt64(sizeof(ComplexF32)),
        UInt64(row_count * sites),
        UInt64(2 * row_count) * UInt64(sizeof(Float64)),
        UInt64(2 * row_count) * UInt64(sizeof(Float64)),
        UInt64(row_count * compact) * UInt64(sizeof(ComplexF32)),
    )
    arena_bytes = UInt64(0)
    for bytes in sizes
        arena_bytes = _eo_host_arena_total(arena_bytes, bytes)
    end
    arena = CUDA.alloc(CUDA.DeviceMemory, Int(arena_bytes))
    base = UInt(pointer(arena))
    offset = UInt64(0)
    peps, offset = _eo_host_arena_take(base, offset, sizes[1])
    samples, offset = _eo_host_arena_take(base, offset, sizes[2])
    logpsi, offset = _eo_host_arena_take(base, offset, sizes[3])
    e_loc, offset = _eo_host_arena_take(base, offset, sizes[4])
    rows, offset = _eo_host_arena_take(base, offset, sizes[5])
    row_offset = UInt(row_base * compact * sizeof(ComplexF32))
    arguments = _EoHostRunArgs(;
        struct_size=UInt32(sizeof(_EoHostRunArgs)),
        padding=_EO_ABI_PADDING,
        device_peps=CUDA.CuPtr{Cvoid}(peps),
        device_samples=CUDA.CuPtr{UInt8}(samples),
        logpsi_out=CUDA.CuPtr{Float64}(logpsi),
        e_loc_out=CUDA.CuPtr{Float64}(e_loc),
        o_rows_dev=CUDA.CuPtr{Cvoid}(rows),
        o_rows_host=Ptr{Cvoid}(UInt(destination_rows) + row_offset),
        gram=CUDA.CuPtr{Cvoid}(0),
        lambda=0.0,
        j2_mode=selector.mode,
        j2_draw=selector.draw,
        j2_seed=selector.seed,
        j2_epoch=selector.epoch,
    )
    return _EoHostBinding(;
        arena,
        arena_bytes,
        peps=CUDA.CuPtr{Cvoid}(peps),
        samples=CUDA.CuPtr{Cvoid}(samples),
        logpsi=CUDA.CuPtr{Cvoid}(logpsi),
        e_loc=CUDA.CuPtr{Cvoid}(e_loc),
        rows=CUDA.CuPtr{Cvoid}(rows),
        args=Ref(arguments),
        epoch,
    )
end
