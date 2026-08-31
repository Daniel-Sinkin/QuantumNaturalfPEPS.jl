using CUDA

function _validate_gpus(gpus::Integer)
    gpus >= 1 || throw(ArgumentError("gpus must be >= 1 (got $gpus)"))
    return Int(gpus)
end

function _upload_peps_arrays(tensors::AbstractMatrix)
    lx, ly = size(tensors)
    A11 = tensors[1, 1]
    ndims(A11) == 5 || throw(
        ArgumentError(
            "PEPS site tensors must be rank-5 [p,u,r,d,l]; got rank $(ndims(A11)). " *
            "For an ITensor grid, convert it to the package tensor-array layout first.",
        ),
    )
    dim_phys = size(A11, 1)
    dim_bond = 1
    for row in 1:lx, col in 1:ly
        A = tensors[row, col]
        for ax in 2:5
            dim_bond = max(dim_bond, size(A, ax))
        end
    end
    buf = ComplexF32[]
    for row in 1:lx, col in 1:ly
        A = tensors[row, col]
        append!(buf, vec(ComplexF32.(permutedims(A, (5, 4, 3, 2, 1)))))
    end
    return CuArray(buf), Int(lx), Int(ly), Int(dim_phys), Int(dim_bond)
end

function vmc_step!(peps::AbstractMatrix; kwargs...)
    data, lx, ly, dim_phys, dim_bond = _upload_peps_arrays(peps)
    return _vmc_step_device!(data, lx, ly, dim_phys, dim_bond; kwargs...)
end

function vmc_step!(
    peps::CuArray{ComplexF32};
    lx::Integer,
    ly::Integer,
    dim_bond::Integer,
    dim_phys::Integer=2,
    kwargs...,
)
    return _vmc_step_device!(peps, Int(lx), Int(ly), Int(dim_phys), Int(dim_bond); kwargs...)
end

function _vmc_step_device!(
    data::CuArray{ComplexF32},
    lx::Integer,
    ly::Integer,
    dim_phys::Integer,
    dim_bond::Integer;
    ns::Integer,
    chi_s::Integer,
    chi_dl::Integer=dim_bond,
    chi_eo::Integer=dim_bond,
    meo::Integer,
    seed::Integer=0,
    sampling_mode=:fast,
    contract_dim::Integer=3 * dim_bond,
    sample_batch::Integer=0,
    gpus::Integer=1,
    J1::Real=1.0,
    J2::Real=0.0,
    terms::HeisenbergTerms=heisenberg_terms(lx, ly; J1=J1, J2=J2),
    relative_cut::Real=1.0e-3,
    absolute_cut::Real=1.0e-8,
    host_tile_bytes::Integer=0,
    want_samples::Bool=false,
    want_logq::Bool=false,
    want_log_gauge::Bool=false,
    want_logpsi::Bool=false,
    want_e_loc::Bool=false,
    want_o_rows::Bool=false,
    stream::CuStream=CUDA.stream(),
)
    ngpu = _validate_gpus(gpus)
    cfg = QnpepsE2eConfig(;
        lx=lx,
        ly=ly,
        dim_bond=dim_bond,
        chi_s=chi_s,
        chi_dl=chi_dl,
        chi_eo=chi_eo,
        meo=meo,
        dim_phys=dim_phys,
        seed=seed,
        sampling_mode=sampling_mode,
        contract_dim=contract_dim,
        sample_batch=sample_batch,
    )
    if ngpu > 1
        return _vmc_step_multigpu_composed!(
            data,
            cfg,
            terms,
            ngpu,
            ns;
            host_tile_bytes,
            relative_cut,
            absolute_cut,
            want_samples,
            want_logq,
            want_log_gauge,
            want_logpsi,
            want_e_loc,
            want_o_rows,
        )
    end
    dense = dense_count(cfg)
    compact = compact_count(cfg)
    sites = Int(lx) * Int(ly)

    theta = CUDA.zeros(ComplexF32, dense)
    emean = Vector{Float64}(undef, 2)
    evar = Vector{Float64}(undef, 1)
    ess = Vector{Float64}(undef, 1)

    samples = want_samples ? CUDA.zeros(UInt8, ns * sites) : nothing
    logq = want_logq ? CUDA.zeros(Float64, ns) : nothing
    loggauge = want_log_gauge ? CUDA.zeros(Float64, ns) : nothing
    logpsi = want_logpsi ? CUDA.zeros(Float64, 2 * ns) : nothing
    eloc = want_e_loc ? CUDA.zeros(Float64, 2 * ns) : nothing
    orows = want_o_rows ? Vector{ComplexF32}(undef, ns * compact) : nothing

    samples_ptr = samples === nothing ? CuPtr{UInt8}(0) : pointer(samples)
    logq_ptr = logq === nothing ? CuPtr{Float64}(0) : pointer(logq)
    loggauge_ptr = loggauge === nothing ? CuPtr{Float64}(0) : pointer(loggauge)
    logpsi_ptr = logpsi === nothing ? CuPtr{Float64}(0) : pointer(logpsi)
    eloc_ptr = eloc === nothing ? CuPtr{Float64}(0) : pointer(eloc)

    diag = terms.diag
    flip = terms.flip
    GC.@preserve data theta emean evar ess samples logq loggauge logpsi eloc orows diag flip begin
        table = QnpepsElocTermTable(
            Int32(length(diag)),
            length(diag) == 0 ? Ptr{QnpepsElocDiagBond}(0) : pointer(diag),
            Int32(length(flip)),
            length(flip) == 0 ? Ptr{QnpepsElocFlipTerm}(0) : pointer(flip),
        )
        table_ref = Ref(table)
        orows_ptr = orows === nothing ? Ptr{Cvoid}(0) : Ptr{Cvoid}(pointer(orows))
        GC.@preserve table_ref begin
            cuts = FFI._E2eCuts(;
                relative_cut=Float64(relative_cut),
                absolute_cut=Float64(absolute_cut),
            )
            minsr_outputs = FFI._E2eMinsrOutputs(;
                theta_dot=CUDA.CuPtr{Cvoid}(pointer(theta)),
                e_mean=pointer(emean),
                e_var=pointer(evar),
                ess=pointer(ess),
            )
            sample_outputs = FFI._E2eSampleOutputs(;
                samples=CUDA.CuPtr{UInt8}(samples_ptr),
                logq=CUDA.CuPtr{Float64}(logq_ptr),
                log_gauge=CUDA.CuPtr{Float64}(loggauge_ptr),
                logpsi=CUDA.CuPtr{Float64}(logpsi_ptr),
                e_loc=CUDA.CuPtr{Float64}(eloc_ptr),
                o_rows_host=orows_ptr,
            )
            inputs = FFI._E2eStepInputs(;
                device_peps=CUDA.CuPtr{Cvoid}(pointer(data)),
                terms=Base.unsafe_convert(Ptr{Cvoid}, table_ref),
            )
            arguments = FFI._E2eStepArguments(;
                config=cfg,
                inputs,
                n_samples=Int64(ns),
                host_tile_bytes=Int64(host_tile_bytes),
                cuts,
                minsr_outputs,
                sample_outputs,
                stream=Ptr{Cvoid}(stream.handle),
            )
            _ffi_step(arguments)
        end
    end

    base = (theta_dot=Array(theta), e_mean=complex(emean[1], emean[2]), e_var=evar[1], ess=ess[1])
    extra = NamedTuple()
    want_samples && (extra = merge(extra, (samples=Array(samples),)))
    want_logq && (extra = merge(extra, (logq=Array(logq),)))
    want_log_gauge && (extra = merge(extra, (log_gauge=Array(loggauge),)))
    want_logpsi && (extra = merge(extra, (logpsi=_pack_complex(Array(logpsi)),)))
    want_e_loc && (extra = merge(extra, (e_loc=_pack_complex(Array(eloc)),)))
    want_o_rows && (extra = merge(extra, (o_rows=orows,)))
    return merge(base, extra)
end

function _pack_complex(interleaved::Vector{Float64})
    n = length(interleaved) ÷ 2
    out = Vector{ComplexF64}(undef, n)
    @inbounds for j in 1:n
        out[j] = complex(interleaved[2j-1], interleaved[2j])
    end
    return out
end
