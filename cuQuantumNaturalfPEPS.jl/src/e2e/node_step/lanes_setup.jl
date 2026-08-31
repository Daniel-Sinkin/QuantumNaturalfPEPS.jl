
function _node_step_create_eo_lane(arguments::_NodeStepEoLaneArguments)
    context = arguments.context
    config = arguments.config
    terms = arguments.terms
    selector = arguments.selector
    geometry = arguments.geometry
    outputs = arguments.outputs
    peps_elements = geometry.peps_elements
    sites = geometry.sites
    compact = geometry.compact
    row_base = geometry.row_base
    row_count = geometry.row_count
    global_samples = outputs.samples
    global_logpsi = outputs.logpsi
    global_e_loc = outputs.e_loc
    global_rows_host = outputs.rows
    CUDA.context!(context)
    stream_ref = Ref{CUDA.CUstream}(CUDA.CUstream(C_NULL))
    CUDA.cuStreamCreate(stream_ref, CUDA.CU_STREAM_DEFAULT)
    stream = stream_ref[]
    handle = Ptr{Cvoid}(C_NULL)
    peps = CUDA.zeros(ComplexF32, peps_elements)
    samples = CUDA.zeros(UInt8, row_count * sites)
    logpsi = CUDA.zeros(Float64, 2 * row_count)
    e_loc = CUDA.zeros(Float64, 2 * row_count)
    rows = CUDA.zeros(ComplexF32, row_count * compact)
    try
        diag = terms.diag
        flip = terms.flip
        output = Ref{Ptr{Cvoid}}(C_NULL)
        status = GC.@preserve diag flip begin
            table = QnpepsElocTermTable(
                Int32(length(diag)),
                isempty(diag) ? Ptr{QnpepsElocDiagBond}(0) : pointer(diag),
                Int32(length(flip)),
                isempty(flip) ? Ptr{QnpepsElocFlipTerm}(0) : pointer(flip),
            )
            table_ref = Ref(table)
            GC.@preserve table_ref FFI.eloc_ctx_create(
                config,
                row_count,
                Base.unsafe_convert(Ptr{Cvoid}, table_ref),
                _NODE_STEP_EO_ROWS,
                Ptr{Cvoid}(stream),
                output,
            )
        end
        _check(; status, what="qnpeps_eloc_ctx_create")
        handle = output[]
        peps_pointer = CUDA.CuPtr{Cvoid}(pointer(peps))
        samples_pointer = CUDA.CuPtr{Cvoid}(pointer(samples))
        logpsi_pointer = CUDA.CuPtr{Cvoid}(pointer(logpsi))
        e_loc_pointer = CUDA.CuPtr{Cvoid}(pointer(e_loc))
        rows_pointer = CUDA.CuPtr{Cvoid}(pointer(rows))
        sample_offset = UInt(row_base * sites)
        scalar_offset = UInt(2 * row_base * sizeof(Float64))
        row_offset = UInt(row_base * compact * sizeof(ComplexF32))
        args = _NodeStepElocArgs(;
            struct_size=UInt32(sizeof(_NodeStepElocArgs)),
            device_peps=peps_pointer,
            device_samples=CUDA.CuPtr{UInt8}(samples_pointer),
            logpsi_out=CUDA.CuPtr{Float64}(logpsi_pointer),
            e_loc_out=CUDA.CuPtr{Float64}(e_loc_pointer),
            o_rows_dev=rows_pointer,
            o_rows_host=Ptr{Cvoid}(UInt(global_rows_host) + row_offset),
            gram=CUDA.CuPtr{Cvoid}(0),
            lambda=0.0,
            j2_mode=selector.mode,
            j2_draw=selector.draw,
            j2_seed=selector.seed,
            j2_epoch=selector.epoch,
        )
        return _NodeStepEoLane(;
            context,
            stream,
            handle,
            peps,
            samples,
            logpsi,
            e_loc,
            rows,
            peps_pointer,
            samples_pointer,
            logpsi_pointer,
            e_loc_pointer,
            rows_pointer,
            source_sample_pointer=global_samples + sample_offset,
            destination_logpsi_pointer=global_logpsi + scalar_offset,
            destination_e_loc_pointer=global_e_loc + scalar_offset,
            row_base,
            row_count,
            args=[args],
            original_affinity=ntuple(_ -> UInt64(0), Val(16)),
            affinity_saved=Int32(0),
            affinity_applied=Int32(0),
            status=Int32(0),
            failure_location="",
            failure_message="",
        )
    catch
        handle == C_NULL || FFI.eloc_ctx_destroy(handle)
        CUDA.unsafe_free!(rows)
        CUDA.unsafe_free!(e_loc)
        CUDA.unsafe_free!(logpsi)
        CUDA.unsafe_free!(samples)
        CUDA.unsafe_free!(peps)
        CUDA.cuStreamDestroy_v2(stream)
        rethrow()
    end
end
