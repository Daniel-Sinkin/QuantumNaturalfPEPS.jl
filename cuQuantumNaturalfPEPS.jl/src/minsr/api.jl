using CUDA

mutable struct GramContext{D,S}
    handle::Ptr{Cvoid}
    descriptor::QnpepsGramDesc
    footprint::QnpepsGramFootprint
    sites::Int
    compact_count::Int
    device::D
    stream::S
end

mutable struct MinsrContext{D,S}
    handle::Ptr{Cvoid}
    descriptor::QnpepsMinsrDesc
    sites::Int
    compact_count::Int
    dense_count::Int
    device::D
    stream::S
end

const _MinsrRowStore = Union{CuArray{ComplexF32},Array{ComplexF32}}

Base.isopen(context::GramContext)::Bool = context.handle != C_NULL
Base.isopen(context::MinsrContext)::Bool = context.handle != C_NULL

function Base.close(context::GramContext)::Nothing
    isopen(context) || return nothing
    handle = context.handle
    context.handle = C_NULL
    _ffi_gram_ctx_destroy(handle)
    return nothing
end

function Base.close(context::MinsrContext)::Nothing
    isopen(context) || return nothing
    handle = context.handle
    context.handle = C_NULL
    _ffi_minsr_ctx_destroy(handle)
    return nothing
end

Base.copy(::GramContext) = throw(ArgumentError("GramContext cannot be copied"))
Base.copy(::MinsrContext) = throw(ArgumentError("MinsrContext cannot be copied"))

function Base.show(io::IO, context::GramContext)::Nothing
    descriptor = context.descriptor
    return print(
        io,
        "GramContext(",
        descriptor.lx,
        "×",
        descriptor.ly,
        ", dim_bond=",
        descriptor.dim_bond,
        ", n_samples=",
        descriptor.n_samples,
        ")",
    )
end

function Base.show(io::IO, context::MinsrContext)::Nothing
    descriptor = context.descriptor
    return print(
        io,
        "MinsrContext(",
        descriptor.lx,
        "×",
        descriptor.ly,
        ", dim_bond=",
        descriptor.dim_bond,
        ", n_samples=",
        descriptor.n_samples,
        ", compact=",
        context.compact_count,
        ", dense=",
        context.dense_count,
        ")",
    )
end

function _minsr_validate_lattice(;
    lx::Integer,
    ly::Integer,
    dim_phys::Integer,
    dim_bond::Integer,
    n_samples::Integer,
)::Nothing
    lx >= 2 || throw(ArgumentError("lx must be at least two (got $lx)"))
    ly >= 2 || throw(ArgumentError("ly must be at least two (got $ly)"))
    dim_phys == 2 ||
        throw(ArgumentError("the minSR scatter supports dim_phys 2 only (got $dim_phys)"))
    dim_bond >= 1 || throw(ArgumentError("dim_bond must be positive (got $dim_bond)"))
    2 <= n_samples <= typemax(Int32) ||
        throw(ArgumentError("n_samples must be in 2:$(typemax(Int32)) (got $n_samples)"))
    return nothing
end

function minsr_dense_count(descriptor::QnpepsMinsrDesc)::Int64
    return _minsr_dense_count(; descriptor)
end

function minsr_dense_count(; lx, ly, dim_bond, n_samples, dim_phys=2)::Int64
    return _minsr_dense_count(; descriptor=QnpepsMinsrDesc(; lx, ly, dim_bond, n_samples, dim_phys))
end

function minsr_compact_count(descriptor::QnpepsMinsrDesc)::Int64
    return _minsr_compact_count(; descriptor)
end

function minsr_compact_count(; lx, ly, dim_bond, n_samples, dim_phys=2)::Int64
    return _minsr_compact_count(;
        descriptor=QnpepsMinsrDesc(; lx, ly, dim_bond, n_samples, dim_phys),
    )
end

function minsr_scratch_bytes(descriptor::QnpepsMinsrDesc)::Int64
    return _minsr_scratch_bytes(; descriptor)
end

function minsr_scratch_bytes(;
    lx,
    ly,
    dim_bond,
    n_samples,
    dim_phys=2,
    host_tile_bytes::Integer=0,
)::Int64
    return _minsr_scratch_bytes(;
        descriptor=QnpepsMinsrDesc(; lx, ly, dim_bond, n_samples, dim_phys, host_tile_bytes),
    )
end

function _minsr_counts(descriptor::QnpepsMinsrDesc)::NTuple{2,Int}
    compact = _minsr_compact_count(; descriptor)
    dense = _minsr_dense_count(; descriptor)
    compact > 0 && dense > 0 ||
        throw(ArgumentError("the CUDA library rejected the minSR descriptor"))
    return (Int(compact), Int(dense))
end

function GramContext(;
    lx::Integer,
    ly::Integer,
    dim_bond::Integer,
    n_samples::Integer,
    dim_phys::Integer=2,
    stream=CUDA.stream(),
)::GramContext
    _minsr_validate_lattice(; lx, ly, dim_phys, dim_bond, n_samples)
    descriptor = QnpepsGramDesc(; lx, ly, dim_phys, dim_bond, n_samples)
    sizing = QnpepsMinsrDesc(; lx, ly, dim_phys, dim_bond, n_samples)
    compact, _ = _minsr_counts(sizing)
    device = CUDA.device()
    handle = _ffi_gram_ctx_create(; descriptor, stream=Ptr{Cvoid}(stream.handle))
    context = GramContext(
        handle,
        descriptor,
        _empty_gram_footprint(),
        Int(lx) * Int(ly),
        compact,
        device,
        stream,
    )
    finalizer(close, context)
    try
        context.footprint = _ffi_gram_ctx_footprint(handle)
    catch
        close(context)
        rethrow()
    end
    return context
end

function MinsrContext(;
    lx::Integer,
    ly::Integer,
    dim_bond::Integer,
    n_samples::Integer,
    dim_phys::Integer=2,
    host_tile_bytes::Integer=0,
    diagnostics::Bool=false,
    stream=CUDA.stream(),
)::MinsrContext
    _minsr_validate_lattice(; lx, ly, dim_phys, dim_bond, n_samples)
    host_tile_bytes >= 0 ||
        throw(ArgumentError("host_tile_bytes must be nonnegative (got $host_tile_bytes)"))
    descriptor =
        QnpepsMinsrDesc(; lx, ly, dim_phys, dim_bond, n_samples, host_tile_bytes, diagnostics)
    compact, dense = _minsr_counts(descriptor)
    device = CUDA.device()
    handle = _ffi_minsr_ctx_create(; descriptor, stream=Ptr{Cvoid}(stream.handle))
    context = MinsrContext(handle, descriptor, Int(lx) * Int(ly), compact, dense, device, stream)
    finalizer(close, context)
    return context
end

function _validate_minsr_context(context)::Nothing
    isopen(context) || throw(ArgumentError("$(typeof(context).name.name) is closed"))
    CUDA.device() == context.device ||
        throw(ArgumentError("$(typeof(context).name.name) device mismatch"))
    CUDA.stream().handle == context.stream.handle ||
        throw(ArgumentError("$(typeof(context).name.name) stream mismatch"))
    return nothing
end

function _check_length(actual::Integer, expected::Integer, label::AbstractString)::Nothing
    actual == expected || throw(DimensionMismatch("$label has length $actual; expected $expected"))
    return nothing
end

function _minsr_row_source(o_rows::CuArray{ComplexF32})::NTuple{2,UInt}
    return (UInt(pointer(o_rows)), UInt(0))
end

function _minsr_row_source(o_rows::Array{ComplexF32})::NTuple{2,UInt}
    return (UInt(0), UInt(pointer(o_rows)))
end

function _minsr_row_source(o_rows)
    return throw(
        ArgumentError(
            "o_rows must be a CuArray{ComplexF32} device store or an Array{ComplexF32} host store",
        ),
    )
end

_minsr_interleaved_pointer(values::CuArray{ComplexF64})::UInt = UInt(pointer(values))
_minsr_interleaved_pointer(values::CuArray{Float64})::UInt = UInt(pointer(values))

function gram_footprint(context::GramContext)::QnpepsGramFootprint
    _validate_minsr_context(context)
    return _ffi_gram_ctx_footprint(context.handle)
end

function raw_gram!(
    context::GramContext,
    gram_out::CuArray{ComplexF32},
    samples::CuArray{UInt8},
    o_rows::CuArray{ComplexF32},
)::CuArray{ComplexF32}
    _validate_minsr_context(context)
    n_samples = Int(context.descriptor.n_samples)
    footprint = context.footprint
    _check_length(length(samples), n_samples * context.sites, "samples")
    _check_length(length(o_rows), n_samples * context.compact_count, "o_rows")
    _check_length(length(gram_out), n_samples * n_samples, "gram_out")
    UInt64(sizeof(UInt8) * length(samples)) == footprint.caller_samples_bytes ||
        throw(DimensionMismatch("samples byte count disagrees with the context footprint"))
    UInt64(sizeof(ComplexF32) * length(o_rows)) == footprint.caller_rows_bytes ||
        throw(DimensionMismatch("o_rows byte count disagrees with the context footprint"))
    UInt64(sizeof(ComplexF32) * length(gram_out)) == footprint.caller_gram_bytes ||
        throw(DimensionMismatch("gram_out byte count disagrees with the context footprint"))

    GC.@preserve gram_out samples o_rows begin
        args = QnpepsGramArgs(
            UInt32(sizeof(QnpepsGramArgs)),
            UInt32(0),
            UInt(pointer(samples)),
            UInt64(sizeof(UInt8) * length(samples)),
            UInt(pointer(o_rows)),
            UInt64(sizeof(ComplexF32) * length(o_rows)),
            UInt(pointer(gram_out)),
            UInt64(sizeof(ComplexF32) * length(gram_out)),
            UInt(context.stream.handle),
        )
        _ffi_gram_ctx_run(context.handle, args)
    end
    return gram_out
end

function _minsr_args(;
    theta_dot::CuArray{ComplexF32},
    samples::CuArray{UInt8},
    logpsi::CuArray{ComplexF64},
    e_loc::CuArray{ComplexF64},
    logq::CuArray{Float64},
    gram::CuArray{ComplexF32},
    o_rows,
    relative_cut::Real,
    absolute_cut::Real,
    e_mean::Vector{Float64},
    e_var::Base.RefValue{Float64},
    ess::Base.RefValue{Float64},
    stream,
)::QnpepsMinsrArgs
    device_rows, host_rows = _minsr_row_source(o_rows)
    return QnpepsMinsrArgs(
        UInt32(sizeof(QnpepsMinsrArgs)),
        UInt32(0),
        UInt(pointer(samples)),
        UInt64(sizeof(UInt8) * length(samples)),
        _minsr_interleaved_pointer(logpsi),
        UInt64(sizeof(ComplexF64) * length(logpsi)),
        _minsr_interleaved_pointer(e_loc),
        UInt64(sizeof(ComplexF64) * length(e_loc)),
        _minsr_interleaved_pointer(logq),
        UInt64(sizeof(Float64) * length(logq)),
        UInt(pointer(gram)),
        UInt64(sizeof(ComplexF32) * length(gram)),
        device_rows,
        host_rows,
        UInt64(sizeof(ComplexF32) * length(o_rows)),
        UInt(pointer(theta_dot)),
        UInt64(sizeof(ComplexF32) * length(theta_dot)),
        Float64(relative_cut),
        Float64(absolute_cut),
        UInt(pointer(e_mean)),
        UInt(Base.unsafe_convert(Ptr{Float64}, e_var)),
        UInt(Base.unsafe_convert(Ptr{Float64}, ess)),
        UInt(stream.handle),
    )
end

function _minsr_validate_buffers(;
    n_samples::Integer,
    sites::Integer,
    compact_count::Integer,
    dense_count::Integer,
    theta_dot,
    samples,
    logpsi,
    e_loc,
    logq,
    gram,
    o_rows,
    relative_cut::Real,
    absolute_cut::Real,
)::Nothing
    relative_cut >= 0 || throw(ArgumentError("relative_cut must be nonnegative"))
    absolute_cut >= 0 || throw(ArgumentError("absolute_cut must be nonnegative"))
    _check_length(length(samples), n_samples * sites, "samples")
    _check_length(length(logpsi), n_samples, "logpsi")
    _check_length(length(e_loc), n_samples, "e_loc")
    _check_length(length(logq), n_samples, "logq")
    _check_length(length(gram), n_samples * n_samples, "gram")
    _check_length(length(o_rows), n_samples * compact_count, "o_rows")
    _check_length(length(theta_dot), dense_count, "theta_dot")
    return nothing
end

function minsr_direction!(
    context::MinsrContext,
    theta_dot::CuArray{ComplexF32},
    samples::CuArray{UInt8},
    logpsi::CuArray{ComplexF64},
    e_loc::CuArray{ComplexF64},
    logq::CuArray{Float64},
    gram::CuArray{ComplexF32},
    o_rows::_MinsrRowStore;
    relative_cut::Real=1.0e-3,
    absolute_cut::Real=1.0e-8,
)::NamedTuple
    _validate_minsr_context(context)
    _minsr_validate_buffers(;
        n_samples=Int(context.descriptor.n_samples),
        sites=context.sites,
        compact_count=context.compact_count,
        dense_count=context.dense_count,
        theta_dot,
        samples,
        logpsi,
        e_loc,
        logq,
        gram,
        o_rows,
        relative_cut,
        absolute_cut,
    )

    e_mean = Vector{Float64}(undef, 2)
    e_var = Ref{Float64}(0.0)
    ess = Ref{Float64}(0.0)
    GC.@preserve theta_dot samples logpsi e_loc logq gram o_rows e_mean e_var ess begin
        args = _minsr_args(;
            theta_dot,
            samples,
            logpsi,
            e_loc,
            logq,
            gram,
            o_rows,
            relative_cut,
            absolute_cut,
            e_mean,
            e_var,
            ess,
            stream=context.stream,
        )
        _ffi_minsr_ctx_run(context.handle, args)
    end
    return (; e_mean=ComplexF64(e_mean[1], e_mean[2]), e_var=e_var[], ess=ess[])
end

function minsr_direction!(
    theta_dot::CuArray{ComplexF32},
    samples::CuArray{UInt8},
    logpsi::CuArray{ComplexF64},
    e_loc::CuArray{ComplexF64},
    logq::CuArray{Float64},
    gram::CuArray{ComplexF32},
    o_rows::_MinsrRowStore;
    lx::Integer,
    ly::Integer,
    dim_bond::Integer,
    dim_phys::Integer=2,
    n_samples::Integer=length(logpsi),
    host_tile_bytes::Integer=0,
    diagnostics::Bool=false,
    relative_cut::Real=1.0e-3,
    absolute_cut::Real=1.0e-8,
    stream=CUDA.stream(),
)::NamedTuple
    _minsr_validate_lattice(; lx, ly, dim_phys, dim_bond, n_samples)
    host_tile_bytes >= 0 ||
        throw(ArgumentError("host_tile_bytes must be nonnegative (got $host_tile_bytes)"))
    descriptor =
        QnpepsMinsrDesc(; lx, ly, dim_phys, dim_bond, n_samples, host_tile_bytes, diagnostics)
    compact, dense = _minsr_counts(descriptor)
    _minsr_validate_buffers(;
        n_samples,
        sites=Int(lx) * Int(ly),
        compact_count=compact,
        dense_count=dense,
        theta_dot,
        samples,
        logpsi,
        e_loc,
        logq,
        gram,
        o_rows,
        relative_cut,
        absolute_cut,
    )

    e_mean = Vector{Float64}(undef, 2)
    e_var = Ref{Float64}(0.0)
    ess = Ref{Float64}(0.0)
    GC.@preserve theta_dot samples logpsi e_loc logq gram o_rows e_mean e_var ess begin
        args = _minsr_args(;
            theta_dot,
            samples,
            logpsi,
            e_loc,
            logq,
            gram,
            o_rows,
            relative_cut,
            absolute_cut,
            e_mean,
            e_var,
            ess,
            stream,
        )
        _ffi_minsr(; descriptor, args)
    end
    return (; e_mean=ComplexF64(e_mean[1], e_mean[2]), e_var=e_var[], ess=ess[])
end

function minsr_direction(
    context::MinsrContext,
    samples::CuArray{UInt8},
    logpsi::CuArray{ComplexF64},
    e_loc::CuArray{ComplexF64},
    logq::CuArray{Float64},
    gram::CuArray{ComplexF32},
    o_rows::_MinsrRowStore;
    relative_cut::Real=1.0e-3,
    absolute_cut::Real=1.0e-8,
)::NamedTuple
    theta_dot = CUDA.zeros(ComplexF32, context.dense_count)
    statistics = minsr_direction!(
        context,
        theta_dot,
        samples,
        logpsi,
        e_loc,
        logq,
        gram,
        o_rows;
        relative_cut,
        absolute_cut,
    )
    return (; theta_dot, e_mean=statistics.e_mean, e_var=statistics.e_var, ess=statistics.ess)
end

function minsr_direction(
    samples::CuArray{UInt8},
    logpsi::CuArray{ComplexF64},
    e_loc::CuArray{ComplexF64},
    logq::CuArray{Float64},
    gram::CuArray{ComplexF32},
    o_rows::_MinsrRowStore;
    lx::Integer,
    ly::Integer,
    dim_bond::Integer,
    dim_phys::Integer=2,
    n_samples::Integer=length(logpsi),
    host_tile_bytes::Integer=0,
    diagnostics::Bool=false,
    relative_cut::Real=1.0e-3,
    absolute_cut::Real=1.0e-8,
    stream=CUDA.stream(),
)::NamedTuple
    _minsr_validate_lattice(; lx, ly, dim_phys, dim_bond, n_samples)
    dense =
        _minsr_dense_count(; descriptor=QnpepsMinsrDesc(; lx, ly, dim_phys, dim_bond, n_samples))
    dense > 0 || throw(ArgumentError("the CUDA library rejected the minSR descriptor"))
    theta_dot = CUDA.zeros(ComplexF32, dense)
    statistics = minsr_direction!(
        theta_dot,
        samples,
        logpsi,
        e_loc,
        logq,
        gram,
        o_rows;
        lx,
        ly,
        dim_bond,
        dim_phys,
        n_samples,
        host_tile_bytes,
        diagnostics,
        relative_cut,
        absolute_cut,
        stream,
    )
    return (; theta_dot, e_mean=statistics.e_mean, e_var=statistics.e_var, ess=statistics.ess)
end
