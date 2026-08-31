using CUDA

const _C_FFI_LIBRARIES = Dict{String,String}()
const _C_FFI_BINDING_LOCK = ReentrantLock()

_is_version_digit(byte::UInt8)::Bool = UInt8('0') <= byte <= UInt8('9')

function _scan_capi_version(bytes, first::Integer)::Union{Nothing,Int}
    cursor = Int(first)
    for component in 1:3
        digits = 0
        while cursor <= length(bytes) && _is_version_digit(bytes[cursor]) && digits < 3
            cursor += 1
            digits += 1
        end
        digits > 0 || return nothing
        cursor <= length(bytes) && _is_version_digit(bytes[cursor]) && return nothing
        if component < 3
            cursor <= length(bytes) && bytes[cursor] == UInt8('.') || return nothing
            cursor += 1
        end
    end
    return cursor
end

function _capi_version_token(version::AbstractString)::Union{Nothing,String}
    bytes = codeunits(version)
    after = _scan_capi_version(bytes, 1)
    isnothing(after) && return nothing
    after == length(bytes) + 1 || return nothing
    return String(version)
end

function _is_build_date_suffix(bytes, first::Integer)::Bool
    cursor = Int(first)
    length(bytes) - cursor + 1 == 13 || return false
    bytes[cursor] == UInt8(' ') || return false
    bytes[cursor+1] == UInt8('(') || return false
    bytes[cursor+6] == UInt8('-') || return false
    bytes[cursor+9] == UInt8('-') || return false
    bytes[cursor+12] == UInt8(')') || return false
    for offset in (2, 3, 4, 5, 7, 8, 10, 11)
        _is_version_digit(bytes[cursor+offset]) || return false
    end
    return true
end

const _PACKAGE_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const _C_API_VERSION_FILE = normpath(joinpath(_PACKAGE_ROOT, "c_api_version.txt"))
Base.include_dependency(_C_API_VERSION_FILE)
const EXPECTED_CAPI_VERSION = let
    version = strip(read(_C_API_VERSION_FILE, String))
    parsed = _capi_version_token(version)
    isnothing(parsed) &&
        error("[c_ffi.jl] invalid C API version $version in $_C_API_VERSION_FILE")
    parsed
end
const _COMPILED_CAPI_VERSION_PREFIX = "cuQuantumNaturalfPEPS "

function _project_package_version(project::AbstractString)::Union{Nothing,String}
    versions = collect(eachmatch(r"(?m)^version = \"([0-9]+\.[0-9]+\.[0-9]+)\"$", project))
    length(versions) == 1 || return nothing
    return versions[1].captures[1]
end

const _PROJECT_FILE = normpath(joinpath(_PACKAGE_ROOT, "Project.toml"))
Base.include_dependency(_PROJECT_FILE)
const EXPECTED_PACKAGE_VERSION = let
    project = read(_PROJECT_FILE, String)
    version = _project_package_version(project)
    isnothing(version) && error("[c_ffi.jl] invalid package version in $_PROJECT_FILE")
    version
end
const EXPECTED_E2E_VERSION = "cuQuantumNaturalfPEPS e2e $EXPECTED_PACKAGE_VERSION"
const EXPECTED_ELOC_VERSION = "cuQuantumNaturalfPEPS eo $EXPECTED_PACKAGE_VERSION"

function _lib_path()::String
    override = get(ENV, "QNPEPS_LIB", "")
    isempty(override) || return override
    return normpath(joinpath(_PACKAGE_ROOT, "build", "cuda", "qnpeps.so"))
end

function _lib_missing_error(; path::AbstractString)::Union{}
    error("[cuQuantumNaturalfPEPS] qnpeps.so not found. Checked $path")
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
    error("[cuQuantumNaturalfPEPS] $detail ($path)")
end

function _component_version_mismatch_error(;
    path::AbstractString,
    component::AbstractString,
    got::AbstractString,
    expected::AbstractString,
)::Union{}
    error(
        "[cuQuantumNaturalfPEPS] CUDA library $component version \"$got\" " *
        "does not match expected \"$expected\" ($path)",
    )
end

function _compiled_capi_version(version::AbstractString)::Union{Nothing,String}
    startswith(version, _COMPILED_CAPI_VERSION_PREFIX) || return nothing
    bytes = codeunits(version)
    first = ncodeunits(_COMPILED_CAPI_VERSION_PREFIX) + 1
    after = _scan_capi_version(bytes, first)
    isnothing(after) && return nothing
    if after <= length(bytes)
        _is_build_date_suffix(bytes, after) || return nothing
    end
    return String(Vector{UInt8}(bytes[first:(after-1)]))
end

function _canonical_lib_path()::String
    path = abspath(_lib_path())
    isfile(path) || _lib_missing_error(; path)
    return realpath(path)
end

function _ffi_library()::String
    path = _canonical_lib_path()
    return lock(_C_FFI_BINDING_LOCK) do
        get!(_C_FFI_LIBRARIES, path) do
            FFI.validate_library()
            path
        end
    end
end

capi_version()::String = unsafe_string(FFI.capi_version())

function _strerror(; status::Integer)::String
    return unsafe_string(FFI.strerror(status))
end

function _last_error_location()::String
    file_ptr = FFI.last_error_file()
    line = FFI.last_error_line()
    (file_ptr == C_NULL || line <= 0) && return ""
    file = unsafe_string(file_ptr)
    isempty(file) && return ""
    return "$file:$line"
end

function _last_error_message()::String
    message_ptr = FFI.last_error_message()
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
        "[cuQuantumNaturalfPEPS] $what failed$at " *
        "(status $status, $(_strerror(; status))$backend)",
    )
end
