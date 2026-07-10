using cuQuantumNaturalfPEPS
using CUDA

include(joinpath(@__DIR__, "common.jl"))

raw_void(a)::Ptr{Cvoid} = reinterpret(Ptr{Cvoid}, pointer(a))
raw_bytes(a)::Ptr{UInt8} = reinterpret(Ptr{UInt8}, pointer(a))
raw_f64(a)::Ptr{Float64} = reinterpret(Ptr{Float64}, pointer(a))

function ffi()::Nothing
    if !CUDA.functional()
        println("app/ffi Needs CUDA.")
        return nothing
    end
    println("capi_version $(unsafe_string(FFI.capi_version()))")

    n_samples = 8

    device_peps = upload_peps(load_peps(grid_peps(LX, LY, DIM_BOND)))
    peps_data = device_peps.data
    config = QnpepsConfig(
        lx=device_peps.lx,
        ly=device_peps.ly,
        dim_bond=device_peps.dim_bond,
        chi_s=device_peps.dim_bond,
        chi_dl=device_peps.dim_bond,
        dim_phys=device_peps.dim_phys,
        seed=SEED_SAMPLE,
    )
    stream = CUDA.stream().handle

    dlenv = CUDA.zeros(UInt8, FFI.dlenv_bytes(config))
    row_logs = CUDA.zeros(Float64, device_peps.lx - 1)
    build_status = GC.@preserve peps_data dlenv row_logs FFI.build_dlenv(
        config,
        raw_void(peps_data);
        dlenv_out=raw_void(dlenv),
        cumulative_row_logs=raw_f64(row_logs),
        stream=stream,
    )
    @assert build_status == 0

    scratch = CUDA.zeros(UInt8, FFI.sample_scratch_bytes(config))
    samples = CUDA.zeros(UInt8, FFI.sample_bytes(config, n_samples))
    log_prob_config = CUDA.zeros(Float64, n_samples)
    log_gauge = CUDA.zeros(Float64, n_samples)
    sample_bufs = (peps_data, dlenv, scratch, samples, log_prob_config, log_gauge)
    sample_status = GC.@preserve sample_bufs FFI.sample(
        config,
        raw_void(peps_data),
        raw_void(dlenv);
        gpus=1,
        scratch=raw_void(scratch),
        scratch_bytes=length(scratch),
        samples_out=raw_bytes(samples),
        log_prob_config=raw_f64(log_prob_config),
        log_gauge=raw_f64(log_gauge),
        n_samples=n_samples,
        batch_base=0,
        dim_batch=0,
        stream=stream,
    )
    @assert sample_status == 0
    CUDA.synchronize()
    oneshot = Array(samples)
    println("oneshot_first_bytes $(oneshot[1:min(length(oneshot), 16)])")

    ctx_handle = Ref{Ptr{Cvoid}}(C_NULL)
    create_status = FFI.ctx_create(config, ctx_handle; stream=stream)
    @assert create_status == 0
    ctx = ctx_handle[]
    @assert ctx != C_NULL

    ctx_row_logs = CUDA.zeros(Float64, device_peps.lx - 1)
    ctx_build_status = GC.@preserve peps_data ctx_row_logs FFI.ctx_build_dlenv(
        ctx,
        raw_void(peps_data);
        cumulative_row_logs=raw_f64(ctx_row_logs),
    )
    @assert ctx_build_status == 0

    samples_base0 = CUDA.zeros(UInt8, FFI.sample_bytes(config, n_samples))
    log_prob_base0 = CUDA.zeros(Float64, n_samples)
    status_base0 = GC.@preserve samples_base0 log_prob_base0 FFI.ctx_sample(
        ctx,
        raw_bytes(samples_base0);
        log_prob_config=raw_f64(log_prob_base0),
        log_gauge=C_NULL,
        n_samples=n_samples,
        batch_base=0,
    )
    @assert status_base0 == 0

    samples_base4096 = CUDA.zeros(UInt8, FFI.sample_bytes(config, n_samples))
    log_prob_base4096 = CUDA.zeros(Float64, n_samples)
    status_base4096 = GC.@preserve samples_base4096 log_prob_base4096 FFI.ctx_sample(
        ctx,
        raw_bytes(samples_base4096);
        log_prob_config=raw_f64(log_prob_base4096),
        log_gauge=C_NULL,
        n_samples=n_samples,
        batch_base=4096,
    )
    @assert status_base4096 == 0
    CUDA.synchronize()

    host_base0 = Array(samples_base0)
    host_base4096 = Array(samples_base4096)
    println("ctx_base0_first_bytes $(host_base0[1:min(length(host_base0), 16)])")
    println("ctx_base4096_first_bytes $(host_base4096[1:min(length(host_base4096), 16)])")
    println("base0_equals_oneshot $(host_base0 == oneshot)")
    println("base4096_equals_base0 $(host_base4096 == host_base0)")
    @assert host_base0 != host_base4096

    FFI.ctx_destroy(ctx)
    FFI.sampler_pool_release()
    return nothing
end

ffi()
