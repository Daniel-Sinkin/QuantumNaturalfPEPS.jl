using CUDA

const _SAMPLER_SEED_MULTIPLIER = UInt64(1000003)

struct SamplerArenaPlan
    sampling_bytes::Int
end

Base.@kwdef mutable struct SamplerHost{B,P,D,S}
    handle::Ptr{Cvoid}
    config::QnpepsConfig
    plan::SamplerArenaPlan
    arena::B
    sampling::B
    peps_data::P
    dlenv_data::B
    dims::Vector{Int32}
    dlenv_pointers::Vector{UInt}
    staging_samples::Vector{UInt8}
    staging_logpc::Vector{Float64}
    staging_lognorm::Vector{Float64}
    batch_args::Vector{QnpepsSamplerHostBatchArgs}
    refresh_args::Vector{QnpepsSamplerHostRefreshArgs}
    sweep_selector::Symbol
    sweep_capture::Symbol
    sweep_host::Union{Nothing,SweepHost{S}}
    device::D
    stream::S
    dim_batch::Int
    num_sites::Int
    generation::Int
    initialized::Bool
    open::Bool
end

function _read_sampler_dims(dlenv::CuDlenv)::Vector{Int32}
    count = (dlenv.lx - 1) * dlenv.ly * 4
    dims = Vector{Int32}(undef, count)
    GC.@preserve dims dlenv begin
        unsafe_copyto!(
            pointer(dims),
            CuPtr{Int32}(pointer(dlenv.data)),
            count;
            stream=CUDA.stream(),
            async=true,
        )
    end
    CUDA.synchronize(CUDA.stream())
    return dims
end

function _validate_sampler_dims(config::QnpepsConfig, dims::Vector{Int32})::Int
    expected = (Int(config.lx) - 1) * Int(config.ly) * 4
    length(dims) == expected || throw(DimensionMismatch("invalid dl-env header length"))
    elements = 0
    bond_cap = min(Int(config.chi_dl), Int(config.dim_bond)^2)
    for site in 0:(expected÷4-1)
        base = 4 * site
        left = Int(dims[base+1])
        ket = Int(dims[base+2])
        bra = Int(dims[base+3])
        right = Int(dims[base+4])
        ket == Int(config.dim_bond) && bra == Int(config.dim_bond) ||
            throw(DimensionMismatch("dl-env vertical dimensions do not match the PEPS"))
        1 <= left <= bond_cap && 1 <= right <= bond_cap ||
            throw(DimensionMismatch("dl-env bond dimension exceeds the configured cap"))
        elements += (left * ket) * (bra * right)
    end
    return elements
end

function plan_sampler_host(
    config::QnpepsConfig,
    dims::Vector{Int32},
    dim_batch::Integer,
)::SamplerArenaPlan
    1 <= dim_batch <= MAX_BATCH_SIZE ||
        throw(ArgumentError("dim_batch must be between 1 and $MAX_BATCH_SIZE"))
    elements = _validate_sampler_dims(config, dims)
    sampling_bytes = 2 * sizeof(ComplexF32) * elements
    return SamplerArenaPlan(sampling_bytes)
end

@inline function _sampler_pointer_count(config::QnpepsConfig, dim_batch::Int)::Int
    return 2 * (Int(config.lx) - 1) * Int(config.ly) * dim_batch
end

function _prepare_sampler_pointers!(host::SamplerHost)::Nothing
    lane_capacity = host.dim_batch
    site_count = (Int(host.config.lx) - 1) * Int(host.config.ly)
    offsets = Vector{Int}(undef, site_count)
    cursor = 0
    for site in 1:site_count
        offsets[site] = cursor
        base = 4 * (site - 1)
        cursor +=
            Int(host.dims[base+1]) *
            Int(host.dims[base+2]) *
            Int(host.dims[base+3]) *
            Int(host.dims[base+4])
    end
    sampling_base = pointer(host.sampling)
    destination = 1
    for layout in 0:1
        layout_offset = layout * cursor
        for site in 1:site_count
            value = UInt(sampling_base + (layout_offset + offsets[site]) * sizeof(ComplexF32))
            for _ in 1:lane_capacity
                host.dlenv_pointers[destination] = value
                destination += 1
            end
        end
    end
    return nothing
end

function _pin_sampler_staging(dim_batch::Int, num_sites::Int)
    samples = Vector{UInt8}(undef, dim_batch * num_sites)
    logpc = Vector{Float64}(undef, dim_batch)
    lognorm = Vector{Float64}(undef, dim_batch)
    CUDA.pin(samples)
    CUDA.pin(logpc)
    CUDA.pin(lognorm)
    return samples, logpc, lognorm
end

function SamplerHost(
    device_peps::CuPeps,
    dlenv::CuDlenv;
    chi_s::Integer=dlenv.chi_s,
    chi_dl::Integer=dlenv.chi_dl,
    seed::Integer=0,
    sampling_mode=:fast,
    chi_c::Integer=3 * device_peps.dim_bond,
    dim_batch::Integer=MAX_BATCH_SIZE,
    generation::Integer=0,
    sweep=:compiled,
    sweep_capture=:auto,
)
    sweep_selector = _validate_sweep_selector(sweep)
    capture_policy = _validate_sweep_capture(sweep_capture)
    config = QnpepsConfig(;
        lx=device_peps.lx,
        ly=device_peps.ly,
        dim_phys=device_peps.dim_phys,
        dim_bond=device_peps.dim_bond,
        chi_s=chi_s,
        chi_dl=chi_dl,
        seed=seed,
        sampling_mode=sampling_mode,
        chi_c=chi_c,
    )
    dlenv.lx == device_peps.lx &&
    dlenv.ly == device_peps.ly &&
    dlenv.dim_phys == device_peps.dim_phys &&
    dlenv.dim_bond == device_peps.dim_bond ||
        throw(DimensionMismatch("CuDlenv and CuPeps geometry differ"))
    dims = _read_sampler_dims(dlenv)
    plan = plan_sampler_host(config, dims, dim_batch)
    arena = CUDA.zeros(UInt8, plan.sampling_bytes)
    sampling = arena
    device = CUDA.device()
    stream = CUDA.stream()
    handle = _ffi_ctx_create(; config=config, stream=Ptr{Cvoid}(stream.handle))
    num_sites = device_peps.lx * device_peps.ly
    staging_samples, staging_logpc, staging_lognorm =
        _pin_sampler_staging(Int(dim_batch), num_sites)
    pointers = Vector{UInt}(undef, _sampler_pointer_count(config, Int(dim_batch)))
    host = SamplerHost(;
        handle,
        config,
        plan,
        arena,
        sampling,
        peps_data=device_peps.data,
        dlenv_data=dlenv.data,
        dims,
        dlenv_pointers=pointers,
        staging_samples,
        staging_logpc,
        staging_lognorm,
        batch_args=Vector{QnpepsSamplerHostBatchArgs}(undef, 1),
        refresh_args=Vector{QnpepsSamplerHostRefreshArgs}(undef, 1),
        sweep_selector,
        sweep_capture=capture_policy,
        sweep_host=nothing,
        device,
        stream,
        dim_batch=Int(dim_batch),
        num_sites,
        generation=Int(generation),
        initialized=false,
        open=true,
    )
    _prepare_sampler_pointers!(host)
    _prepare_sampler_static_args!(host)
    if sweep_selector === :julia
        _run_compiled_sampler_batch!(host, UInt64(0), UInt64(0))
        host.sweep_host = SweepHost(host; capture=capture_policy)
    end
    finalizer(close, host)
    return host
end

Base.isopen(host::SamplerHost)::Bool = host.open

function Base.close(host::SamplerHost)::Nothing
    host.open || return nothing
    sweep_host = host.sweep_host
    if sweep_host !== nothing
        close(sweep_host)
        host.sweep_host = nothing
    end
    handle = host.handle
    host.handle = C_NULL
    _ffi_ctx_destroy(handle)
    CUDA.unsafe_free!(host.arena)
    host.open = false
    return nothing
end

Base.copy(::SamplerHost) = throw(ArgumentError("SamplerHost cannot be copied"))

sampler_sweep(host::SamplerHost)::Symbol = host.sweep_selector

function sweep_capture_mode(host::SamplerHost)::Symbol
    host.sweep_selector === :compiled && return :compiled
    driver = host.sweep_host
    driver === nothing && return :cold
    return sweep_capture_mode(driver)
end

function sweep_capture_reason(host::SamplerHost)::Symbol
    host.sweep_selector === :compiled && return :compiled_sweep
    driver = host.sweep_host
    driver === nothing && return :not_initialized
    return sweep_capture_reason(driver)
end

function _validate_sampler_host(host::SamplerHost)::Nothing
    host.open || throw(ArgumentError("SamplerHost is closed"))
    CUDA.device() == host.device || throw(ArgumentError("SamplerHost device mismatch"))
    CUDA.stream().handle == host.stream.handle ||
        throw(ArgumentError("SamplerHost stream mismatch"))
    return nothing
end

@inline function _sampler_batch_seed(host::SamplerHost, batch_base::UInt64, batch_id::UInt64)
    return host.config.seed * _SAMPLER_SEED_MULTIPLIER + batch_base + batch_id
end

function _prepare_sampler_static_args!(host::SamplerHost)::Nothing
    header_bytes = length(host.dims) * sizeof(Int32)
    host.batch_args[1] = QnpepsSamplerHostBatchArgs(;
        struct_size=UInt32(sizeof(QnpepsSamplerHostBatchArgs)),
        reserved=UInt32(0),
        peps=UInt(pointer(host.peps_data)),
        dlenv_dims=UInt(pointer(host.dims)),
        dlenv_dims_count=UInt64(length(host.dims)),
        dlenv_values=UInt(pointer(host.dlenv_data) + header_bytes),
        scratch=UInt(0),
        scratch_bytes=UInt64(0),
        sampling=UInt(pointer(host.sampling)),
        sampling_bytes=UInt64(host.plan.sampling_bytes),
        dlenv_pointers=UInt(pointer(host.dlenv_pointers)),
        dlenv_pointer_count=UInt64(length(host.dlenv_pointers)),
        samples_out=UInt(pointer(host.staging_samples)),
        log_prob_config=UInt(pointer(host.staging_logpc)),
        log_gauge=UInt(pointer(host.staging_lognorm)),
        batch_seed=UInt64(0),
        batch_id=UInt64(0),
        dim_batch=UInt64(host.dim_batch),
        peps_layout=Int32(0),
        reserved2=Int32(0),
    )
    return nothing
end

function _prepare_sampler_batch_args!(
    host::SamplerHost,
    batch_base::UInt64,
    batch_id::UInt64,
)::Nothing
    args = host.batch_args[1]
    host.batch_args[1] = QnpepsSamplerHostBatchArgs(;
        struct_size=args.struct_size,
        reserved=args.reserved,
        peps=args.peps,
        dlenv_dims=args.dlenv_dims,
        dlenv_dims_count=args.dlenv_dims_count,
        dlenv_values=args.dlenv_values,
        scratch=args.scratch,
        scratch_bytes=args.scratch_bytes,
        sampling=args.sampling,
        sampling_bytes=args.sampling_bytes,
        dlenv_pointers=args.dlenv_pointers,
        dlenv_pointer_count=args.dlenv_pointer_count,
        samples_out=args.samples_out,
        log_prob_config=args.log_prob_config,
        log_gauge=args.log_gauge,
        batch_seed=_sampler_batch_seed(host, batch_base, batch_id),
        batch_id,
        dim_batch=args.dim_batch,
        peps_layout=args.peps_layout,
        reserved2=args.reserved2,
    )
    return nothing
end

function _run_compiled_sampler_batch!(
    host::SamplerHost,
    batch_base::UInt64,
    batch_id::UInt64,
)::Nothing
    _prepare_sampler_batch_args!(host, batch_base, batch_id)
    status = GC.@preserve host FFI.sampler_host_batch(host.handle, pointer(host.batch_args))
    _check(; status, what="qnpeps_sampler_host_batch")
    host.initialized = true
    return nothing
end

function _run_sampler_batch!(host::SamplerHost, batch_base::UInt64, batch_id::UInt64)::Nothing
    if host.sweep_selector === :compiled
        return _run_compiled_sampler_batch!(host, batch_base, batch_id)
    end
    driver = host.sweep_host
    driver === nothing && throw(ArgumentError("Julia sweep host is not initialized"))
    _run_sweep!(driver, batch_base, batch_id)
    return nothing
end

function _copy_sampler_batch!(
    host::SamplerHost,
    samples::AbstractArray{UInt8},
    logpc::AbstractVector{Float64},
    lognorm::AbstractVector{Float64},
    destination_sample::Int,
    valid_samples::Int,
)::Nothing
    sample_count = valid_samples * host.num_sites
    GC.@preserve host samples logpc lognorm begin
        unsafe_copyto!(
            pointer(samples, destination_sample * host.num_sites + 1),
            pointer(host.staging_samples),
            sample_count,
        )
        unsafe_copyto!(
            pointer(logpc, destination_sample + 1),
            pointer(host.staging_logpc),
            valid_samples,
        )
        unsafe_copyto!(
            pointer(lognorm, destination_sample + 1),
            pointer(host.staging_lognorm),
            valid_samples,
        )
    end
    return nothing
end

function sample_peps!(
    host::SamplerHost,
    samples::AbstractArray{UInt8},
    log_prob_config::AbstractVector{Float64},
    log_gauge::AbstractVector{Float64};
    batch_base::Integer=0,
)::Nothing
    _validate_sampler_host(host)
    length(samples) % host.num_sites == 0 ||
        throw(DimensionMismatch("sample buffer length is not divisible by the site count"))
    sample_count = length(samples) ÷ host.num_sites
    length(log_prob_config) >= sample_count && length(log_gauge) >= sample_count ||
        throw(DimensionMismatch("sampler log buffers are too small"))
    batch_base_u = UInt64(batch_base)
    sample_count == 0 && return nothing
    batches = cld(sample_count, host.dim_batch)
    for batch in 0:(batches-1)
        _run_sampler_batch!(host, batch_base_u, UInt64(batch))
        destination = batch * host.dim_batch
        valid = min(host.dim_batch, sample_count - destination)
        _copy_sampler_batch!(host, samples, log_prob_config, log_gauge, destination, valid)
    end
    return nothing
end

function sample_peps(host::SamplerHost, n_samples::Integer; batch_base::Integer=0)
    sample_count = _sample_count(n_samples)
    raw = Array{UInt8}(undef, Int(host.config.ly), Int(host.config.lx), sample_count)
    logpc = Vector{Float64}(undef, sample_count)
    lognorm = Vector{Float64}(undef, sample_count)
    sample_count > 0 && sample_peps!(host, raw, logpc, lognorm; batch_base=batch_base)
    return (configs=_host_config_view(raw), log_prob_config=logpc, log_gauge=lognorm)
end

function refresh_sampler!(
    host::SamplerHost,
    device_peps::CuPeps,
    dlenv::CuDlenv;
    generation::Integer=(host.generation + 1),
)::SamplerHost
    _validate_sampler_host(host)
    Int(generation) == host.generation && return host
    host.initialized || throw(ArgumentError("SamplerHost must run one batch before refresh"))
    dims = _read_sampler_dims(dlenv)
    dims == host.dims || throw(DimensionMismatch("dl-env layout changed after setup"))
    host.peps_data = device_peps.data
    host.dlenv_data = dlenv.data
    header_bytes = length(host.dims) * sizeof(Int32)
    host.refresh_args[1] = QnpepsSamplerHostRefreshArgs(;
        struct_size=UInt32(sizeof(QnpepsSamplerHostRefreshArgs)),
        peps_layout=Int32(0),
        peps=UInt(pointer(host.peps_data)),
        dlenv_values=UInt(pointer(host.dlenv_data) + header_bytes),
        sampling=UInt(pointer(host.sampling)),
        sampling_bytes=UInt64(host.plan.sampling_bytes),
    )
    status = GC.@preserve host FFI.sampler_host_refresh(host.handle, pointer(host.refresh_args))
    _check(; status, what="qnpeps_sampler_host_refresh")
    _prepare_sampler_static_args!(host)
    host.generation = Int(generation)
    return host
end

function _sample_multigpu_lane!(
    host::SamplerHost,
    lane::Int,
    lanes::Int,
    sample_count::Int,
    samples::AbstractArray{UInt8},
    logpc::AbstractVector{Float64},
    lognorm::AbstractVector{Float64},
    batch_base::UInt64,
)::Nothing
    CUDA.device!(host.device) do
        CUDA.stream!(host.stream) do
            batches = cld(sample_count, host.dim_batch)
            for batch in lane:lanes:(batches-1)
                _run_sampler_batch!(host, batch_base, UInt64(batch))
                destination = batch * host.dim_batch
                valid = min(host.dim_batch, sample_count - destination)
                _copy_sampler_batch!(host, samples, logpc, lognorm, destination, valid)
            end
        end
    end
    return nothing
end

function sample_multigpu!(
    hosts::AbstractVector{<:SamplerHost},
    samples::AbstractArray{UInt8},
    log_prob_config::AbstractVector{Float64},
    log_gauge::AbstractVector{Float64};
    batch_base::Integer=0,
)::Nothing
    isempty(hosts) && throw(ArgumentError("at least one SamplerHost is required"))
    num_sites = hosts[1].num_sites
    dim_batch = hosts[1].dim_batch
    sweep_selector = hosts[1].sweep_selector
    sweep_capture = hosts[1].sweep_capture
    all(
        host ->
            host.num_sites == num_sites &&
            host.dim_batch == dim_batch &&
            host.sweep_selector === sweep_selector &&
            host.sweep_capture === sweep_capture,
        hosts,
    ) || throw(DimensionMismatch("multi-GPU sampler hosts differ"))
    length(samples) % num_sites == 0 ||
        throw(DimensionMismatch("sample buffer length is not divisible by the site count"))
    sample_count = length(samples) ÷ num_sites
    tasks = Vector{Task}(undef, length(hosts))
    for lane in eachindex(hosts)
        tasks[lane] = Threads.@spawn _sample_multigpu_lane!(
            hosts[lane],
            lane - 1,
            length(hosts),
            sample_count,
            samples,
            log_prob_config,
            log_gauge,
            UInt64(batch_base),
        )
    end
    foreach(fetch, tasks)
    return nothing
end
