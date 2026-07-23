using CUDA
using ITensors

const SampleResult = @NamedTuple{
    configs::Vector{Matrix{UInt8}},
    log_prob_config::Vector{Float64},
    log_gauge::Vector{Float64},
}

const HostSampleResult = @NamedTuple{
    configs::AbstractArray{UInt8,3},
    log_prob_config::Vector{Float64},
    log_gauge::Vector{Float64},
}

function _checked_sample_count(n_samples::Integer)::Int
    n_samples >= 0 || throw(ArgumentError("n_samples must be nonnegative"))
    n_samples <= typemax(Int) || throw(OverflowError("n_samples does not fit in Int"))
    return Int(n_samples)
end

function _checked_batch_base(batch_base::Integer)::UInt64
    batch_base >= 0 || throw(ArgumentError("batch_base must be nonnegative"))
    batch_base <= typemax(UInt64) || throw(OverflowError("batch_base does not fit in UInt64"))
    return UInt64(batch_base)
end

function _sample_batch_size(n_samples::Int, batch_size::Integer)::Int
    0 <= batch_size <= MAX_BATCH_SIZE ||
        throw(ArgumentError("batch_size must be between 0 and $MAX_BATCH_SIZE"))
    return batch_size == 0 ? max(1, min(n_samples, MAX_BATCH_SIZE)) : Int(batch_size)
end

function _validate_sample_inputs(device_peps::CuPeps, dlenv::CuDlenv)::Nothing
    lx_match = device_peps.lx == dlenv.lx
    ly_match = device_peps.ly == dlenv.ly
    dim_phys_match = device_peps.dim_phys == dlenv.dim_phys
    dim_bond_match = device_peps.dim_bond == dlenv.dim_bond
    lx_match && ly_match && dim_phys_match && dim_bond_match ||
        throw(DimensionMismatch("$device_peps and $dlenv disagree"))
    return nothing
end

function _host_config_view(raw_configs::Array{UInt8,3})::AbstractArray{UInt8,3}
    return PermutedDimsArray(raw_configs, (2, 1, 3))
end

function _input_bond_dim(tensors::AbstractMatrix)::Int
    return tensors[1, 1] isa ITensor ? _itensor_bond_dim(tensors) : _dim_bond(load_peps(tensors))
end

function _unpack_samples(
    bytes::Vector{UInt8},
    config::QnpepsConfig,
    n_samples::Integer,
)::Vector{Matrix{UInt8}}
    lx, ly = Int(config.lx), Int(config.ly)
    out = Vector{Matrix{UInt8}}(undef, n_samples)
    @inbounds for sample in 1:n_samples
        config_matrix = Matrix{UInt8}(undef, lx, ly)
        base = (sample - 1) * lx * ly
        for row in 1:lx, col in 1:ly
            config_matrix[row, col] = bytes[base + (row-1) * ly + col]
        end
        out[sample] = config_matrix
    end
    return out
end

function _sample_all(
    device_peps::CuPeps,
    dlenv::CuDlenv,
    n_samples::Integer;
    gpus::Integer,
    seed::Integer,
    sampling_mode,
    chi_c::Integer,
    batch_size::Integer,
    batch_base::Integer,
)::SampleResult
    sample_count = _checked_sample_count(n_samples)
    _checked_batch_base(batch_base)
    available_gpus = CUDA.ndevices()
    1 <= gpus <= available_gpus ||
        throw(ArgumentError("gpus must be between 1 and $available_gpus"))
    dim_batch = _sample_batch_size(sample_count, batch_size)
    config = _cfg_of(dlenv; seed, sampling_mode, chi_c)
    samples = CUDA.zeros(UInt8, _sample_bytes(; config, n_samples=sample_count))
    log_prob_config = CUDA.zeros(Float64, sample_count)
    log_gauge = CUDA.zeros(Float64, sample_count)
    is_multigpu = gpus > 1
    scratch_bytes = is_multigpu ? 0 : _scratch_bytes(; config, dim_batch)
    scratch = CUDA.zeros(UInt8, scratch_bytes)
    scratch_pointer = is_multigpu ? CuPtr{Cvoid}(0) : pointer(scratch)
    GC.@preserve device_peps dlenv samples log_prob_config log_gauge scratch begin
        _ffi_sample(;
            config,
            peps=pointer(device_peps.data),
            dlenv=pointer(dlenv.data),
            gpus=is_multigpu ? gpus : 1,
            scratch=scratch_pointer,
            scratch_bytes,
            samples=pointer(samples),
            log_prob_config=pointer(log_prob_config),
            log_gauge=pointer(log_gauge),
            n_samples=sample_count,
            batch_base,
            dim_batch,
        )
    end
    CUDA.synchronize()
    return (
        configs=_unpack_samples(Array(samples), config, sample_count),
        log_prob_config=Array(log_prob_config),
        log_gauge=Array(log_gauge),
    )
end

function sample_peps(
    peps::Union{Peps,CuPeps},
    dlenv::CuDlenv,
    n_samples::Integer;
    gpus::Integer=CUDA.ndevices(),
    seed::Integer=0,
    sampling_mode=:fast,
    chi_c::Integer=3 * dlenv.dim_bond,
    batch_size::Integer=0,
    batch_base::Integer=0,
)::SampleResult
    device_peps = peps isa Peps ? upload_peps(peps) : peps
    owns_device_peps = peps isa Peps
    try
        _validate_sample_inputs(device_peps, dlenv)
        return _sample_all(
            device_peps,
            dlenv,
            n_samples;
            gpus,
            seed,
            sampling_mode,
            chi_c,
            batch_size,
            batch_base,
        )
    finally
        owns_device_peps && CUDA.unsafe_free!(device_peps.data)
    end
end

function _sample_host_all(
    device_peps::CuPeps,
    dlenv::CuDlenv,
    n_samples::Integer;
    gpus::Integer,
    seed::Integer,
    sampling_mode,
    chi_c::Integer,
    batch_size::Integer,
    batch_base::Integer,
)::HostSampleResult
    sample_count = _checked_sample_count(n_samples)
    _checked_batch_base(batch_base)
    available_gpus = CUDA.ndevices()
    1 <= gpus <= available_gpus ||
        throw(ArgumentError("gpus must be between 1 and $available_gpus"))
    dim_batch = _sample_batch_size(sample_count, batch_size)
    config = _cfg_of(dlenv; seed, sampling_mode, chi_c)
    raw_configs = Array{UInt8}(undef, Int(config.ly), Int(config.lx), sample_count)
    log_prob_config = Vector{Float64}(undef, sample_count)
    log_gauge = Vector{Float64}(undef, sample_count)
    if sample_count > 0
        GC.@preserve device_peps dlenv raw_configs log_prob_config log_gauge begin
            _ffi_sample_host(;
                config,
                peps=pointer(device_peps.data),
                dlenv=pointer(dlenv.data),
                gpus,
                samples=pointer(raw_configs),
                log_prob_config=pointer(log_prob_config),
                log_gauge=pointer(log_gauge),
                n_samples=sample_count,
                batch_base,
                dim_batch,
            )
        end
    end
    configs = _host_config_view(raw_configs)
    return (; configs, log_prob_config, log_gauge)
end

function sample_peps_host(
    peps::Union{Peps,CuPeps},
    dlenv::CuDlenv,
    n_samples::Integer;
    gpus::Integer=CUDA.ndevices(),
    seed::Integer=0,
    sampling_mode=:fast,
    chi_c::Integer=3 * dlenv.dim_bond,
    batch_size::Integer=0,
    batch_base::Integer=0,
)::HostSampleResult
    device_peps = peps isa Peps ? upload_peps(peps) : peps
    owns_device_peps = peps isa Peps
    try
        _validate_sample_inputs(device_peps, dlenv)
        return _sample_host_all(
            device_peps,
            dlenv,
            n_samples;
            gpus,
            seed,
            sampling_mode,
            chi_c,
            batch_size,
            batch_base,
        )
    finally
        owns_device_peps && CUDA.unsafe_free!(device_peps.data)
    end
end

function sample_peps(
    context::SamplerContext,
    n_samples::Integer;
    batch_base::Integer=0,
)::HostSampleResult
    _validate_sampler_context(context)
    sample_count = _checked_sample_count(n_samples)
    _checked_batch_base(batch_base)
    config = context.config
    raw_configs = Array{UInt8}(undef, Int(config.ly), Int(config.lx), sample_count)
    log_prob_config = Vector{Float64}(undef, sample_count)
    log_gauge = Vector{Float64}(undef, sample_count)
    if sample_count > 0
        dim_batch = min(sample_count, context.batch_size)
        GC.@preserve raw_configs log_prob_config log_gauge begin
            _ffi_ctx_sample_host(
                context.handle;
                samples=pointer(raw_configs),
                log_prob_config=pointer(log_prob_config),
                log_gauge=pointer(log_gauge),
                n_samples=sample_count,
                batch_base,
                dim_batch,
            )
        end
    end
    configs = _host_config_view(raw_configs)
    return (; configs, log_prob_config, log_gauge)
end

function sample_peps!(
    context::SamplerContext,
    samples::CuArray{UInt8};
    log_prob_config::Union{Nothing,CuArray{Float64}}=nothing,
    log_gauge::Union{Nothing,CuArray{Float64}}=nothing,
    batch_base::Integer=0,
)::CuArray{UInt8}
    _validate_sampler_context(context)
    _checked_batch_base(batch_base)
    num_sites = Int(context.config.lx) * Int(context.config.ly)
    length(samples) % num_sites == 0 ||
        throw(DimensionMismatch("sample buffer length is not divisible by the PEPS site count"))
    sample_count = _checked_sample_count(length(samples) ÷ num_sites)
    if log_prob_config !== nothing
        length(log_prob_config) >= sample_count ||
            throw(DimensionMismatch("configuration-log buffer is too small"))
    end
    if log_gauge !== nothing
        length(log_gauge) >= sample_count ||
            throw(DimensionMismatch("gauge-log buffer is too small"))
    end
    sample_count == 0 && return samples
    dim_batch = min(sample_count, context.batch_size)
    log_prob_config_ptr =
        log_prob_config === nothing ? CuPtr{Float64}(0) : pointer(log_prob_config)
    log_gauge_ptr = log_gauge === nothing ? CuPtr{Float64}(0) : pointer(log_gauge)
    GC.@preserve samples log_prob_config log_gauge begin
        _ffi_ctx_sample(
            context.handle;
            samples=pointer(samples),
            log_prob_config=log_prob_config_ptr,
            log_gauge=log_gauge_ptr,
            n_samples=sample_count,
            batch_base,
            dim_batch,
        )
    end
    return samples
end

function _sample_peps_one_shot(
    peps::Peps,
    tensors::Union{Nothing,AbstractMatrix},
    n_samples::Integer;
    chi_s::Integer,
    chi_dl::Integer,
    seed::Integer,
    sampling_mode,
    chi_c::Integer,
    batch_size::Integer,
    batch_base::Integer,
)::NamedTuple
    context = SamplerContext(
        peps;
        chi_s,
        chi_dl,
        seed,
        sampling_mode,
        chi_c,
        batch_size,
    )
    try
        samples = sample_peps(context, n_samples; batch_base)
        host_dlenv =
            tensors === nothing ? materialize_dlenv(context) :
            materialize_dlenv(context, tensors)
        return (
            dlenv=host_dlenv,
            cumulative_row_logs=copy(context.cumulative_row_logs),
            configs=samples.configs,
            log_prob_config=samples.log_prob_config,
            log_gauge=samples.log_gauge,
        )
    finally
        close(context)
    end
end

function sample_peps(
    peps::Peps,
    n_samples::Integer;
    chi_s::Integer=_dim_bond(peps),
    chi_dl::Integer=_dim_bond(peps),
    seed::Integer=0,
    sampling_mode=:fast,
    chi_c::Integer=3 * _dim_bond(peps),
    batch_size::Integer=MAX_BATCH_SIZE,
    batch_base::Integer=0,
)::NamedTuple
    return _sample_peps_one_shot(
        peps,
        nothing,
        n_samples;
        chi_s,
        chi_dl,
        seed,
        sampling_mode,
        chi_c,
        batch_size,
        batch_base,
    )
end

function sample_peps(
    tensors::AbstractMatrix,
    n_samples::Integer;
    chi_s::Integer=_input_bond_dim(tensors),
    chi_dl::Integer=_input_bond_dim(tensors),
    seed::Integer=0,
    sampling_mode=:fast,
    chi_c::Integer=3 * _input_bond_dim(tensors),
    batch_size::Integer=MAX_BATCH_SIZE,
    batch_base::Integer=0,
)::NamedTuple
    peps = load_peps(tensors)
    source_tensors = tensors[1, 1] isa ITensor ? tensors : nothing
    return _sample_peps_one_shot(
        peps,
        source_tensors,
        n_samples;
        chi_s,
        chi_dl,
        seed,
        sampling_mode,
        chi_c,
        batch_size,
        batch_base,
    )
end

sampler_pool_release()::Nothing = _ffi_pool_release()
