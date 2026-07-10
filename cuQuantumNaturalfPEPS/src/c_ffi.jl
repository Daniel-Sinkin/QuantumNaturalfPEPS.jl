using CUDA
using Libdl

const _LIB_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
const _SYM_CACHE = Dict{Symbol,Ptr{Cvoid}}()

const EXPECTED_CAPI_VERSION = "cuQuantumNaturalfPEPS 0.1 (2026-07-09)"

function _lib_path()::String
    override = get(ENV, "QNPEPS_LIB", "")
    isempty(override) || return override
    return normpath(joinpath(@__DIR__, "..", "cuda", "build", "libpeps_sampler.so"))
end

function _lib_missing_error(path::AbstractString)::Union{}
    error("cuQuantumNaturalfPEPS: libpeps_sampler.so not found. Checked: $path")
end

function _capi_version_mismatch_error(path::AbstractString, got::AbstractString)::Union{}
    error(
        "cuQuantumNaturalfPEPS: CUDA library version mismatch, " *
        "got \"$got\" expected \"$EXPECTED_CAPI_VERSION\" ($path)",
    )
end

function _lib_handle()::Ptr{Cvoid}
    _LIB_HANDLE[] == C_NULL || return _LIB_HANDLE[]
    path = _lib_path()
    isfile(path) || _lib_missing_error(path)
    handle = Libdl.dlopen(path)
    version_ptr = Libdl.dlsym(handle, :qnpeps_capi_version)
    got = unsafe_string(@ccall $version_ptr()::Cstring)
    got == EXPECTED_CAPI_VERSION || _capi_version_mismatch_error(path, got)
    _LIB_HANDLE[] = handle
    return _LIB_HANDLE[]
end

function _sym(name::Symbol)::Ptr{Cvoid}
    cached = get(_SYM_CACHE, name, C_NULL)
    cached == C_NULL || return cached
    ptr = Libdl.dlsym(_lib_handle(), name)
    _SYM_CACHE[name] = ptr
    return ptr
end

capi_version()::String = unsafe_string(@ccall $(_sym(:qnpeps_capi_version))()::Cstring)

function _strerror(status::Integer)::String
    return unsafe_string(@ccall $(_sym(:qnpeps_strerror))(status::Cint)::Cstring)
end

@inline function _check(status::Integer, what::AbstractString)::Nothing
    status == 0 && return
    error("cuQuantumNaturalfPEPS: $what failed (status $status: $(_strerror(status)))")
end

function _dlenv_bytes(config::QnpepsConfig)::Int64
    return @ccall $(_sym(:qnpeps_dlenv_bytes))(config::Ref{QnpepsConfig})::Int64
end

function _sample_bytes(config::QnpepsConfig, count::Integer)::Int64
    return @ccall $(_sym(:qnpeps_sample_bytes))(config::Ref{QnpepsConfig}, count::UInt64)::Int64
end

function _scratch_bytes(config::QnpepsConfig)::Int64
    return @ccall $(_sym(:qnpeps_sample_scratch_bytes))(config::Ref{QnpepsConfig})::Int64
end

function _sample_footprint_bytes(config::QnpepsConfig, count::Integer)::Int64
    fn = _sym(:qnpeps_sample_footprint_bytes)
    return @ccall $fn(config::Ref{QnpepsConfig}, count::UInt64)::Int64
end

function _peps_bytes(config::QnpepsConfig)::Int64
    return @ccall $(_sym(:qnpeps_peps_bytes))(config::Ref{QnpepsConfig})::Int64
end

function _ffi_build_dlenv(
    config::QnpepsConfig,
    peps::CuPtr,
    dlenv::CuPtr,
    cumulative_row_logs::CuPtr,
)::Nothing
    fn = _sym(:qnpeps_build_dlenv)
    status = GC.@preserve peps dlenv @ccall $fn(
        config::Ref{QnpepsConfig},
        peps::CuPtr{Cvoid},
        dlenv::CuPtr{Cvoid},
        cumulative_row_logs::CuPtr{Float64},
        CUDA.stream().handle::Ptr{Cvoid},
    )::Cint
    _check(status, "qnpeps_build_dlenv")
end

function _dlenv_row_bytes(config::QnpepsConfig, maxdim::Integer)::Int64
    return @ccall $(_sym(:qnpeps_dlenv_row_bytes))(config::Ref{QnpepsConfig}, maxdim::Cint)::Int64
end

function _ffi_double_layer_row(
    config::QnpepsConfig,
    row::Integer,
    maxdim::Integer,
    peps_row::CuPtr,
    env_below::CuPtr,
    out::CuPtr,
    row_log::Ref{Float64},
)::Nothing
    fn = _sym(:qnpeps_double_layer_row)
    status = GC.@preserve peps_row out row_log @ccall $fn(
        config::Ref{QnpepsConfig},
        Cint(row)::Cint,
        Cint(maxdim)::Cint,
        peps_row::CuPtr{Cvoid},
        env_below::CuPtr{Cvoid},
        out::CuPtr{Cvoid},
        row_log::Ptr{Float64},
        CUDA.stream().handle::Ptr{Cvoid},
    )::Cint
    _check(status, "qnpeps_double_layer_row")
end

function _ffi_sample(
    config::QnpepsConfig,
    peps::CuPtr,
    dlenv::CuPtr,
    gpus::Integer,
    scratch::CuPtr,
    scratch_bytes::Integer,
    samples::CuPtr,
    log_prob_config::CuPtr,
    log_gauge::CuPtr,
    count::Integer,
    batch_base::Integer,
    dim_batch::Integer,
)::Nothing
    args = QnpepsSampleArgs(
        UInt32(sizeof(QnpepsSampleArgs)),
        UInt(peps),
        UInt(dlenv),
        Int32(gpus),
        UInt(scratch),
        UInt64(scratch_bytes),
        UInt(samples),
        UInt(log_prob_config),
        UInt(log_gauge),
        UInt64(count),
        UInt64(batch_base),
        UInt64(dim_batch),
        UInt(CUDA.stream().handle),
    )
    fn = _sym(:qnpeps_sample)
    status = GC.@preserve peps dlenv scratch samples @ccall $fn(
        config::Ref{QnpepsConfig},
        args::Ref{QnpepsSampleArgs},
    )::Cint
    _check(status, "qnpeps_sample")
end

function _ffi_pool_release()::Nothing
    @ccall $(_sym(:qnpeps_sampler_pool_release))()::Cvoid
end

function _batched_rangefinder_scratch_bytes(
    rows::Integer,
    cols::Integer,
    rank::Integer,
    batch::Integer,
)::Int64
    fn = _sym(:qnpeps_batched_rangefinder_scratch_bytes)
    return @ccall $fn(rows::Cint, cols::Cint, rank::Cint, batch::Cint)::Int64
end

function _ffi_batched_rangefinder(
    input::CuPtr,
    rows::Integer,
    cols::Integer,
    rank::Integer,
    batch::Integer,
    input_stride::Integer,
    seed::Integer,
    q_out::CuPtr,
    q_stride::Integer,
    r_out::CuPtr,
    r_stride::Integer,
    scratch::CuPtr,
    scratch_bytes::Integer,
)::Nothing
    fn = _sym(:qnpeps_batched_rangefinder)
    status = GC.@preserve input q_out r_out scratch @ccall $fn(
        input::CuPtr{Cvoid},
        Cint(rows)::Cint,
        Cint(cols)::Cint,
        Cint(rank)::Cint,
        Cint(batch)::Cint,
        Int64(input_stride)::Int64,
        UInt64(seed)::UInt64,
        q_out::CuPtr{Cvoid},
        Int64(q_stride)::Int64,
        r_out::CuPtr{Cvoid},
        Int64(r_stride)::Int64,
        scratch::CuPtr{Cvoid},
        UInt64(scratch_bytes)::UInt64,
        CUDA.stream().handle::Ptr{Cvoid},
    )::Cint
    _check(status, "qnpeps_batched_rangefinder")
end
