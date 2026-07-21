using CUDA
using Libdl

# This file handles the .so and the raw function pointers and is purely plumbing work
# the ffi.jl contains the actual low level API endpoints to be used via Julia

# This holds a reference to the .so file which has the low level CUDA api functions
const _CUDA_LIB_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
# Each function is at a byte offset into the _CUDA_LIB_HANDLE and needs to be queried via
# the string name, this caches those offsets so every function has to only be found once
const _C_FFI_FUNCTION_PTR_CACHE = Dict{Symbol,Ptr{Cvoid}}()

const _C_API_VERSION_FILE = normpath(joinpath(@__DIR__, "..", "c_api_version.txt"))
Base.include_dependency(_C_API_VERSION_FILE)
const EXPECTED_CAPI_VERSION = let
    version = strip(read(_C_API_VERSION_FILE, String))
    occursin(r"^[0-9]+\.[0-9]+\.[0-9]+$", version) ||
        error("invalid C API version in $_C_API_VERSION_FILE: $version")
    version
end
const _COMPILED_CAPI_VERSION_PATTERN =
    r"^cuQuantumNaturalfPEPS ([0-9]+\.[0-9]+\.[0-9]+)(?: \([0-9]{4}-[0-9]{2}-[0-9]{2}\))?$"

function _lib_path()::String
    override = get(ENV, "QNPEPS_LIB", "")
    isempty(override) || return override
    return normpath(joinpath(@__DIR__, "..", "build", "cuda", "qnpeps.so"))
end

function _lib_missing_error(; path::AbstractString)::Union{}
    error("cuQuantumNaturalfPEPS: qnpeps.so not found. Checked: $path")
end

function _capi_version_mismatch_error(;
    path::AbstractString,
    got::AbstractString,
    got_version::Union{Nothing,AbstractString},
)::Union{}
    detail = if isnothing(got_version)
        "unrecognized CUDA library C API version \"$got\""
    else
        "CUDA library C API \"$got_version\" does not match expected \"$EXPECTED_CAPI_VERSION\""
    end
    error(
        "cuQuantumNaturalfPEPS: $detail ($path)",
    )
end

function _compiled_capi_version(version::AbstractString)::Union{Nothing,String}
    matched = match(_COMPILED_CAPI_VERSION_PATTERN, version)
    isnothing(matched) && return nothing
    return String(matched.captures[1])
end

function _lib_handle()::Ptr{Cvoid}
    _CUDA_LIB_HANDLE[] == C_NULL || return _CUDA_LIB_HANDLE[]
    path = _lib_path()
    isfile(path) || _lib_missing_error(; path)
    handle = Libdl.dlopen(path)

    # Calls the qnpepes_capi_version endpoint to get the compiled version string
    version_ptr = Libdl.dlsym(handle, :qnpeps_capi_version)
    # Copies a C-Style string (null terminated)
    got = unsafe_string(@ccall $version_ptr()::Cstring)
    got_version = _compiled_capi_version(got)
    got_version == EXPECTED_CAPI_VERSION ||
        _capi_version_mismatch_error(; path, got, got_version)
    _CUDA_LIB_HANDLE[] = handle
    return _CUDA_LIB_HANDLE[]
end

function _sym(; name::Symbol)::Ptr{Cvoid}
    cached = get(_C_FFI_FUNCTION_PTR_CACHE, name, C_NULL)
    cached == C_NULL || return cached
    ptr = Libdl.dlsym(_lib_handle(), name)
    _C_FFI_FUNCTION_PTR_CACHE[name] = ptr
    return ptr
end

capi_version()::String =
    unsafe_string(@ccall $(_sym(; name=:qnpeps_capi_version))()::Cstring)

function _strerror(; status::Integer)::String
    return unsafe_string(
        @ccall $(_sym(; name=:qnpeps_strerror))(status::Cint)::Cstring
    )
end

function _last_error_location()::String
    file_ptr = @ccall $(_sym(; name=:qnpeps_last_error_file))()::Cstring
    line = @ccall $(_sym(; name=:qnpeps_last_error_line))()::Cint
    (file_ptr == C_NULL || line <= 0) && return ""
    file = unsafe_string(file_ptr)
    isempty(file) && return ""
    return "$file:$line"
end

function _last_error_message()::String
    message_ptr = @ccall $(_sym(; name=:qnpeps_last_error_message))()::Cstring
    message_ptr == C_NULL && return ""
    return unsafe_string(message_ptr)
end

@inline function _check(; status::Integer, what::AbstractString)::Nothing
    status == 0 && return
    location = _last_error_location()
    at = isempty(location) ? "" : " at $location"
    message = _last_error_message()
    backend = isempty(message) ? "" : "; $message"
    error(
        "cuQuantumNaturalfPEPS: $what failed$at " *
        "(status $status: $(_strerror(; status))$backend)",
    )
end

function _dlenv_bytes(; config::QnpepsConfig)::Int64
    return @ccall $(_sym(; name=:qnpeps_dlenv_bytes))(config::Ref{QnpepsConfig})::Int64
end

function _sample_bytes(; config::QnpepsConfig, n_samples::Integer)::Int64
    return @ccall $(_sym(; name=:qnpeps_sample_bytes))(
        config::Ref{QnpepsConfig}, n_samples::UInt64
    )::Int64
end

function _scratch_bytes(;
    config::QnpepsConfig,
    dim_batch::Integer,
)::Int64
    fn = _sym(; name=:qnpeps_sample_scratch_bytes)
    return @ccall $fn(config::Ref{QnpepsConfig}, dim_batch::UInt64)::Int64
end

function _sample_footprint_bytes(;
    config::QnpepsConfig,
    n_samples::Integer,
    dim_batch::Integer,
)::Int64
    fn = _sym(; name=:qnpeps_sample_footprint_bytes)
    return @ccall $fn(
        config::Ref{QnpepsConfig},
        n_samples::UInt64,
        dim_batch::UInt64,
    )::Int64
end

function _peps_bytes(; config::QnpepsConfig)::Int64
    return @ccall $(_sym(; name=:qnpeps_peps_bytes))(config::Ref{QnpepsConfig})::Int64
end

function _ffi_build_dlenv(
    ;
    config::QnpepsConfig,
    peps::CuPtr,
    dlenv::CuPtr,
    cumulative_row_logs::CuPtr,
)::Nothing
    fn = _sym(; name=:qnpeps_build_dlenv)
    status = GC.@preserve peps dlenv @ccall $fn(
        config::Ref{QnpepsConfig},
        peps::CuPtr{Cvoid},
        dlenv::CuPtr{Cvoid},
        cumulative_row_logs::CuPtr{Float64},
        CUDA.stream().handle::Ptr{Cvoid},
    )::Cint
    _check(; status, what="qnpeps_build_dlenv")
end

function _dlenv_row_bytes(; config::QnpepsConfig, maxdim::Integer)::Int64
    return @ccall $(_sym(; name=:qnpeps_dlenv_row_bytes))(
        config::Ref{QnpepsConfig}, maxdim::Cint
    )::Int64
end

function _ffi_double_layer_row(
    ;
    config::QnpepsConfig,
    row::Integer,
    maxdim::Integer,
    peps_row::CuPtr,
    env_below::CuPtr,
    out::CuPtr,
    row_log::Ref{Float64},
)::Nothing
    fn = _sym(; name=:qnpeps_double_layer_row)
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
    _check(; status, what="qnpeps_double_layer_row")
end

function _ffi_sample(
    ;
    config::QnpepsConfig,
    peps::CuPtr,
    dlenv::CuPtr,
    gpus::Integer,
    scratch::CuPtr,
    scratch_bytes::Integer,
    samples::CuPtr,
    log_prob_config::CuPtr,
    log_gauge::CuPtr,
    n_samples::Integer,
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
        UInt64(n_samples),
        UInt64(batch_base),
        UInt64(dim_batch),
        UInt(CUDA.stream().handle),
    )
    fn = _sym(; name=:qnpeps_sample)
    status = GC.@preserve peps dlenv scratch samples @ccall $fn(
        config::Ref{QnpepsConfig},
        args::Ref{QnpepsSampleArgs},
    )::Cint
    _check(; status, what="qnpeps_sample")
end

function _ffi_pool_release()::Nothing
    @ccall $(_sym(; name=:qnpeps_sampler_pool_release))()::Cvoid
end

function _batched_rangefinder_scratch_bytes(
    ;
    rows::Integer,
    cols::Integer,
    rank::Integer,
    batch::Integer,
)::Int64
    fn = _sym(; name=:qnpeps_batched_rangefinder_scratch_bytes)
    return @ccall $fn(rows::Cint, cols::Cint, rank::Cint, batch::Cint)::Int64
end

function _ffi_batched_rangefinder(
    ;
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
    fn = _sym(; name=:qnpeps_batched_rangefinder)
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
    _check(; status, what="qnpeps_batched_rangefinder")
end
