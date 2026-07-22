import cuQuantumNaturalfPEPS

function Ok_and_Ek_cuda(
    peps,
    ham_op,
    S,
    logpc;
    timer=TimerOutput(),
    Ok=nothing,
)
    env_top = Array{Environment}(undef, size(S, 1) - 1)
    logψ, env_top, env_down, max_bond = @timeit timer "vertical_envs" get_logψ_and_envs(peps, S, env_top; overwrite=true)
    h_envs_r, h_envs_l = @timeit timer "horizontal_envs" get_all_horizontal_envs(peps, env_top, env_down, S)
    logψ_flipped = Dict{Any,Number}()
    Ek_terms = @timeit timer "precomp_sHψ_elems" QuantumNaturalGradient.get_precomp_sOψ_elems(ham_op, S; get_flip_sites=true)
    E_loc = @timeit timer "energy" get_Ek(peps, ham_op, env_top, env_down, S, logψ; h_envs_r, h_envs_l, logψ_flipped, Ek_terms, timer)
    grad = @timeit timer "log_gradients" get_Ok(peps, env_top, env_down, S, logψ; h_envs_r, h_envs_l, Ok)
    return grad, E_loc, logψ, S, logpc, max_bond
end

function Oks_and_Eks_cuda(peps, ham_op, samples, logpcs)
    sample_nr = length(samples)
    eltype_ = eltype(peps)
    eltype_real = real(eltype_)
    Oks = Matrix{eltype_}(undef, length(peps), sample_nr)
    Eks = Vector{eltype_}(undef, sample_nr)
    logψs = Vector{Complex{eltype_real}}(undef, sample_nr)
    evaluated_samples = Vector{Matrix{Int64}}(undef, sample_nr)
    evaluated_logpcs = Vector{eltype_real}(undef, sample_nr)
    contract_dims = Vector{Int}(undef, sample_nr)

    Threads.@threads for i in eachindex(samples)
        Ok = @view Oks[:, i]
        _, E_loc, logψs[i], evaluated_samples[i], evaluated_logpcs[i], contract_dims[i] = Ok_and_Ek_cuda(
            peps,
            ham_op,
            Matrix{Int64}(samples[i]),
            logpcs[i];
            Ok,
        )
        if eltype(E_loc) != eltype_
            if abs(imag(E_loc)) > 1e-6
                @warn "Large imaginary part detected"
            end
            Eks[i] = real(E_loc)
        else
            Eks[i] = E_loc
        end
    end

    return (
        Oks=Matrix(transpose(Oks)),
        Eks,
        logψs,
        samples=evaluated_samples,
        logpcs=evaluated_logpcs,
        contract_dims,
    )
end

function Oks_and_Eks_cuda_multiproc(peps, ham_op, samples, logpcs)
    width = cld(length(samples), length(workers()))
    ranges = [first:min(first + width - 1, length(samples)) for first in 1:width:length(samples)]
    futures = map(zip(workers(), ranges)) do (worker, range)
        samples_chunk = samples[range]
        logpcs_chunk = logpcs[range]
        Distributed.remotecall(worker) do
            Oks_and_Eks_cuda(peps, ham_op, samples_chunk, logpcs_chunk)
        end
    end
    chunks = fetch.(futures)
    logψs = reduce(vcat, (chunk.logψs for chunk in chunks))
    logpcs = reduce(vcat, (chunk.logpcs for chunk in chunks))
    return Dict(
        :Oks => reduce(vcat, (chunk.Oks for chunk in chunks)),
        :Eks => reduce(vcat, (chunk.Eks for chunk in chunks)),
        :logψs => logψs,
        :samples => reduce(vcat, (chunk.samples for chunk in chunks)),
        :weights => compute_importance_weights(logψs, logpcs),
        :contract_dims => reduce(vcat, (chunk.contract_dims for chunk in chunks)),
    )
end

function generate_Oks_and_Eks_cuda(peps::AbstractPEPS, ham::OpSum; timer=TimerOutput())
    return generate_Oks_and_Eks_cuda(peps, TensorOperatorSum(ham, siteinds(peps)); timer)
end

function generate_Oks_and_Eks_cuda(peps::AbstractPEPS, ham_op::TensorOperatorSum; timer=TimerOutput())
    context = nothing

    function Oks_and_Eks_(Θ::Vector{T}, sample_nr::Integer; reset_double_layer=true) where T
        write!(peps, Θ; reset_double_layer)
        return Oks_and_Eks_(Parameters(peps), sample_nr)
    end

    function Oks_and_Eks_(peps_::Parameters{<:AbstractPEPS}, sample_nr::Integer)
        peps_ = peps_.obj
        host_peps = @timeit timer "cuda_pack" cuQuantumNaturalfPEPS.load_peps(peps_.tensors)
        if isnothing(context)
            context = @timeit timer "cuda_context_create" cuQuantumNaturalfPEPS.CuSamplingContext(
                host_peps,
                sample_nr;
                seed=1,
                sampling_mode=:full,
                chi_s=peps_.bond_dim,
                chi_dl=peps_.double_contract_dim,
                chi_c=3 * peps_.bond_dim,
            )
        else
            @timeit timer "cuda_upload" cuQuantumNaturalfPEPS.update_peps!(context, host_peps)
        end
        @timeit timer "cuda_double_layer_envs" cuQuantumNaturalfPEPS.build_dlenv!(context)
        proposal = @timeit timer "cuda_sampling" cuQuantumNaturalfPEPS.sample_peps!(context)
        return @timeit timer "Oks_and_Eks" Oks_and_Eks_cuda_multiproc(
            peps_,
            ham_op,
            proposal.configs,
            proposal.log_prob_config,
        )
    end

    return Oks_and_Eks_
end
