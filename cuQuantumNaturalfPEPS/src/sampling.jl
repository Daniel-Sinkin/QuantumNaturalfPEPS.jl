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
    out = Vector{Matrix{UInt8}}(undef, n_samples)
    @inbounds for sample in 1:n_samples
        config_matrix = Matrix{UInt8}(undef, lx, ly)
        base = (sample - 1) * lx * ly
        for row in 1:lx, col in 1:ly
            config_matrix[row, col] = bytes[base+(row-1)*ly+col]
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
)::SampleResult
    config = _cfg_of(dlenv; seed=seed)
    samples = CUDA.zeros(UInt8, _sample_bytes(config, n_samples))
    log_prob_config = CUDA.zeros(Float64, n_samples)
    log_gauge = CUDA.zeros(Float64, n_samples)
    if gpus > 1
        GC.@preserve device_peps dlenv samples log_prob_config log_gauge begin
            _ffi_sample(
                config,
                pointer(device_peps.data),
                pointer(dlenv.data),
                gpus,
                CuPtr{Cvoid}(0),
                0,
                pointer(samples),
                pointer(log_prob_config),
                pointer(log_gauge),
                n_samples,
                0,
                0,
            )
        end
    else
        scratch = CUDA.zeros(UInt8, _scratch_bytes(config))
        GC.@preserve device_peps dlenv samples log_prob_config log_gauge scratch begin
            _ffi_sample(
                config,
                pointer(device_peps.data),
                pointer(dlenv.data),
                1,
                pointer(scratch),
                length(scratch),
                pointer(samples),
                pointer(log_prob_config),
                pointer(log_gauge),
                n_samples,
                0,
                0,
            )
        end
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
)::SampleResult
    device_peps = peps isa Peps ? upload_peps(peps) : peps
    lx_match = device_peps.lx == dlenv.lx
    ly_match = device_peps.ly == dlenv.ly
    dim_phys_match = device_peps.dim_phys == dlenv.dim_phys
    dim_bond_match = device_peps.dim_bond == dlenv.dim_bond
    dims_match = lx_match && ly_match && dim_phys_match && dim_bond_match
    if !dims_match
        peps_desc =
            "$(device_peps.lx)x$(device_peps.ly), " *
            "dim_phys=$(device_peps.dim_phys), dim_bond=$(device_peps.dim_bond)"
        dlenv_desc =
            "$(dlenv.lx)x$(dlenv.ly), " * "dim_phys=$(dlenv.dim_phys), dim_bond=$(dlenv.dim_bond)"
        throw(PepsError("peps ($peps_desc) and dlenv ($dlenv_desc) disagree"))
    end
    return _sample_all(device_peps, dlenv, n_samples; gpus, seed)
end

function sample_peps!(
    peps_data::CuArray,
    dlenv_data::CuArray,
    samples::CuArray{UInt8},
    config::QnpepsConfig;
    seed::Integer=0,
    log_prob_config=nothing,
    log_gauge=nothing,
)::CuArray{UInt8}
    seeded = _reseed(config, seed)
    n_samples = length(samples) ÷ (Int(seeded.lx) * Int(seeded.ly))
    scratch = CUDA.zeros(UInt8, _scratch_bytes(seeded))
    log_prob_config_ptr = log_prob_config === nothing ? CuPtr{Float64}(0) : pointer(log_prob_config)
    log_gauge_ptr = log_gauge === nothing ? CuPtr{Float64}(0) : pointer(log_gauge)
    GC.@preserve peps_data dlenv_data samples scratch log_prob_config log_gauge begin
        _ffi_sample(
            seeded,
            pointer(peps_data),
            pointer(dlenv_data),
            1,
            pointer(scratch),
            length(scratch),
            pointer(samples),
            log_prob_config_ptr,
            log_gauge_ptr,
            n_samples,
            0,
            0,
        )
    end
    return samples
end

sampler_pool_release()::Nothing = _ffi_pool_release()
