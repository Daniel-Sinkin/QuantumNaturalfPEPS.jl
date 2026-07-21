using CUDA

const SampleResult = @NamedTuple{
    configs::Vector{Matrix{UInt8}},
    log_prob_config::Vector{Float64},
    log_gauge::Vector{Float64},
}

function _unpack_samples(
    bytes::Vector{UInt8},
    config::QnpepsConfig,
    n_samples::Integer,
)::Vector{Matrix{UInt8}}
    lx, ly = Int(config.lx), Int(config.ly)
    out = [Matrix{UInt8}(undef, lx, ly) for _ in 1:n_samples]
    return _unpack_samples!(out, bytes, config, n_samples)
end

function _unpack_samples!(
    out::Vector{Matrix{UInt8}},
    bytes::Vector{UInt8},
    config::QnpepsConfig,
    n_samples::Integer,
)::Vector{Matrix{UInt8}}
    lx, ly = Int(config.lx), Int(config.ly)
    if length(out) != n_samples
        throw(DimensionMismatch("sample output has $(length(out)) entries, expected $n_samples"))
    end
    if length(bytes) != n_samples * lx * ly
        throw(DimensionMismatch("packed sample buffer has the wrong length"))
    end
    @inbounds for sample in 1:n_samples
        config_matrix = out[sample]
        if size(config_matrix) != (lx, ly)
            throw(
                DimensionMismatch("sample $sample has size $(size(config_matrix)), expected ($lx, $ly)"),
            )
        end
        base = (sample - 1) * lx * ly
        for row in 1:lx, col in 1:ly
            config_matrix[row, col] = bytes[base + (row-1) * ly + col]
        end
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
)::SampleResult
    n_samples >= 0 || throw(ArgumentError("n_samples must be nonnegative"))
    if batch_size < 0
        throw(ArgumentError("batch_size must be nonnegative"))
    end
    if batch_size > MAX_BATCH_SIZE
        throw(
            ArgumentError(
                "batch_size must not exceed $MAX_BATCH_SIZE (batch_size=$batch_size)",
            ),
        )
    end
    dim_batch = batch_size == 0 ? max(1, min(n_samples, MAX_BATCH_SIZE)) : batch_size
    config = _cfg_of(dlenv; seed, sampling_mode, chi_c)
    samples = CUDA.zeros(UInt8, _sample_bytes(; config, n_samples))
    log_prob_config = CUDA.zeros(Float64, n_samples)
    log_gauge = CUDA.zeros(Float64, n_samples)
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
            n_samples,
            batch_base=0,
            dim_batch,
        )
    end
    CUDA.synchronize()
    return (
        configs=_unpack_samples(Array(samples), config, n_samples),
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
)::SampleResult
    device_peps = peps isa Peps ? upload_peps(peps) : peps
    lx_match = device_peps.lx == dlenv.lx
    ly_match = device_peps.ly == dlenv.ly
    dim_phys_match = device_peps.dim_phys == dlenv.dim_phys
    dim_bond_match = device_peps.dim_bond == dlenv.dim_bond
    dims_match = lx_match && ly_match && dim_phys_match && dim_bond_match
    if !dims_match
        throw(DimensionMismatch("$device_peps and $dlenv disagree"))
    end
    return _sample_all(
        device_peps,
        dlenv,
        n_samples;
        gpus,
        seed,
        sampling_mode,
        chi_c,
        batch_size,
    )
end

function sample_peps!(
    peps_data::CuArray,
    dlenv_data::CuArray,
    samples::CuArray{UInt8},
    config::QnpepsConfig;
    seed::Integer=0,
    log_prob_config=nothing,
    log_gauge=nothing,
    batch_size::Integer=0,
)::CuArray{UInt8}
    0 <= batch_size <= MAX_BATCH_SIZE ||
        throw(ArgumentError("batch_size must be between 0 and $MAX_BATCH_SIZE"))
    seeded = _reseed(config, seed)
    n_samples = length(samples) ÷ (Int(seeded.lx) * Int(seeded.ly))
    dim_batch = batch_size == 0 ? max(1, min(n_samples, MAX_BATCH_SIZE)) : batch_size
    scratch = CUDA.zeros(UInt8, _scratch_bytes(; config=seeded, dim_batch))
    log_prob_config_ptr =
        log_prob_config !== nothing ? pointer(log_prob_config) : CuPtr{Float64}(0)
    log_gauge_ptr = log_gauge !== nothing ? pointer(log_gauge) : CuPtr{Float64}(0)
    GC.@preserve peps_data dlenv_data samples scratch log_prob_config log_gauge begin
        _ffi_sample(;
            config=seeded,
            peps=pointer(peps_data),
            dlenv=pointer(dlenv_data),
            gpus=1,
            scratch=pointer(scratch),
            scratch_bytes=length(scratch),
            samples=pointer(samples),
            log_prob_config=log_prob_config_ptr,
            log_gauge=log_gauge_ptr,
            n_samples,
            batch_base=0,
            dim_batch,
        )
    end
    return samples
end

sampler_pool_release()::Nothing = _ffi_pool_release()
