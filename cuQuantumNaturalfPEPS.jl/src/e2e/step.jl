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
    return _vmc_step_node!(
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

function _pack_complex(interleaved::Vector{Float64})
    n = length(interleaved) ÷ 2
    out = Vector{ComplexF64}(undef, n)
    @inbounds for j in 1:n
        out[j] = complex(interleaved[2j-1], interleaved[2j])
    end
    return out
end
