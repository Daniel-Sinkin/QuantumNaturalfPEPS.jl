using CUDA
using Distributed
using ITensors
using JLD2
using LinearAlgebra
using QuantumNaturalGradient
using QuantumNaturalfPEPS
using Random
using TimerOutputs

const QNG = QuantumNaturalGradient

const NR_PROCS = 9
const NR_THREADS = 8
addprocs(NR_PROCS, exeflags="-t $NR_THREADS,0")

@everywhere begin
    using LinearAlgebra
    using QuantumNaturalfPEPS
    BLAS.set_num_threads(1)
end
BLAS.set_num_threads(16)

L = 4
J1 = 2 * cos(0.06π) * cos(0.14π)
J2 = 2 * cos(0.06π) * sin(0.14π)
λ = 2 * sin(0.06π)

params = Dict{Symbol,Any}(
    :T => ComplexF64,
    :seed => 1,
    :Lx => L,
    :Ly => L,
    :bond_dim => 2,
    :sample_nr => 1000,
    :lr => 0.05,
    :J1 => J1,
    :J2 => J2,
    :lambda => λ,
    :eigencut => 1e-4,
    :contract_cutoff => 1e-4,
    :sample_cutoff => 1e-3,
    :contract_dim => 100,
    :maxiter => 10,
    :α_init => 3.0,
    :sampling_mode => :full,
)
println("Simulation parameters: ", params)
Random.seed!(params[:seed])

save_file = "cu_CSL.jld2"
hilbert = siteinds("S=1/2", params[:Lx], params[:Ly])
bond_dim = params[:bond_dim]
peps = PEPS(
    params[:T],
    hilbert;
    bond_dim,
    sample_dim=params[:contract_dim],
    sample_cutoff=params[:sample_cutoff],
    double_contract_dim=bond_dim,
    contract_dim=params[:contract_dim],
    contract_cutoff=params[:contract_cutoff],
    show_warning=true,
)
QuantumNaturalfPEPS.multiply_algebraic_spectrum!(peps, params[:α_init])

function P_matrix(factor)
    P = zeros(ComplexF64, 2, 2, 2, 2, 2, 2, 2, 2)
    for i1 in 1:2, i2 in 1:2, i3 in 1:2, i4 in 1:2
        P[i1, i2, i2, i3, i3, i4, i4, i1] = factor
    end
    return P
end

function P_operator(hilbert, spins; P=nothing, factor=1)
    isnothing(P) && (P = P_matrix(factor))
    inds = [hilbert[site]' for site in spins]
    append!(inds, [hilbert[site] for site in spins])
    return ITensor(P, inds)
end

function add_P_operators!(ham_op, hilbert, Lx, Ly, factor)
    for i in 1:(Lx-1), j in 1:(Ly-1)
        sites = [
            i + (j - 1) * Lx,
            i + 1 + (j - 1) * Lx,
            i + 1 + j * Lx,
            i + j * Lx,
        ]
        push!(ham_op.tensors, P_operator(hilbert, sites; factor))
        push!(ham_op.sites, sites)
    end
    return nothing
end

ham_J1J2 = QuantumNaturalfPEPS.hamiltonain_J1J2(
    params[:J1] / 4,
    params[:J2] / 4,
    size(peps)...,
)
ham_op = QNG.TensorOperatorSum(ham_J1J2, hilbert)
add_P_operators!(ham_op, hilbert, params[:Lx], params[:Ly], im * params[:lambda])
add_P_operators!(ham_op, hilbert, params[:Lx], params[:Ly], -im * params[:lambda])

timer = TimerOutput()
Oks_and_Eks = QuantumNaturalfPEPS.generate_Oks_and_Eks_cuda(
    peps,
    ham_op;
    timer,
)

integrator = QNG.Euler(
    lr=params[:lr],
    use_clipping=false,
    clip_norm=0.03 * params[:Lx] * params[:Ly] * params[:bond_dim]^4,
)
solver = QNG.EigenSolver(params[:eigencut], verbose=true)
θ = QNG.Parameters(peps)

history_contract_dims(; contract_dims) = contract_dims
logger_funcs = [history_contract_dims]

function callback(; niter=1, kwargs...)
    data = Dict{String,Any}(string(key) => value for (key, value) in kwargs)
    data["params"] = params
    data["solver"] = solver
    data["integrator"] = integrator
    data["peps"] = peps
    data["timer"] = timer
    save(save_file, data)
end

@time loss_value, trained_θ, misc = QNG.evolve(
    Oks_and_Eks,
    θ;
    integrator,
    verbosity=2,
    solver,
    sample_nr=params[:sample_nr],
    maxiter=params[:maxiter],
    callback,
    timer,
    logger_funcs,
)
rm(save_file; force=true)
