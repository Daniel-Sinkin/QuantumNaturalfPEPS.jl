using CUDA
using Distributed
using ITensors
using JLD2
using LinearAlgebra
using QuantumNaturalGradient
using QuantumNaturalfPEPS
using Statistics
using TimerOutputs

const NR_PROCS = 2
const NR_THREADS = 8
addprocs(NR_PROCS, exeflags="-t $NR_THREADS,0")

@everywhere begin
    using LinearAlgebra
    using QuantumNaturalfPEPS
    BLAS.set_num_threads(1)
end
BLAS.set_num_threads(16)

params = Dict{Symbol,Any}(
    :T => Float64,
    :seed => 1,
    :Lx => 4,
    :Ly => 4,
    :bdim => 2,
    :lr => 0.05,
    :J1 => 1,
    :eigencut => 1e-4,
    :contract_cutoff => 1e-4,
    :sample_cutoff => 1e-3,
    :contract_dim => 200,
    :maxiter => 10,
    :α_init => 3.0,
    :sample_nr => 1000,
    :sampling_mode => :full,
)

hilbert = siteinds("S=1/2", params[:Lx], params[:Ly])
bond_dim = params[:bdim]
peps = PEPS(
    params[:T],
    hilbert;
    bond_dim,
    show_warning=true,
    contract_cutoff=params[:contract_cutoff],
    contract_dim=params[:contract_dim],
    sample_cutoff=params[:sample_cutoff],
    double_contract_dim=bond_dim,
)
QuantumNaturalfPEPS.multiply_algebraic_spectrum!(peps, params[:α_init])

ham_J1 = OpSum()
for i in 1:params[:Lx], j in 1:params[:Ly], operator in ("X", "Y", "Z")
    if j < params[:Ly]
        ham_J1 .+= (params[:J1], operator, (i, j), operator, (i, j + 1))
    end
    if i < params[:Lx]
        ham_J1 .+= (params[:J1], operator, (i, j), operator, (i + 1, j))
    end
end

timer = TimerOutput()
Oks_and_Eks = QuantumNaturalfPEPS.generate_Oks_and_Eks_cuda(
    peps,
    ham_J1;
    timer,
)
integrator = QuantumNaturalGradient.Euler(lr=params[:lr])
solver = QuantumNaturalGradient.EigenSolver(params[:eigencut], verbose=true)
θ = QuantumNaturalGradient.Parameters(peps)

contract_dim(; contract_dims) = mean(contract_dims)
logger_funcs = [contract_dim]
save_file = "cu_heisenberg_multithreaded.jld2"

function callback(; niter=1, kwargs...)
    data = Dict{String,Any}(string(key) => value for (key, value) in kwargs)
    data["params"] = params
    data["solver"] = solver
    data["integrator"] = integrator
    data["peps"] = peps
    data["timer"] = timer
    save(save_file, data)
end

@time loss_value, trained_θ, misc = QuantumNaturalGradient.evolve(
    Oks_and_Eks,
    θ;
    integrator,
    verbosity=2,
    solver,
    sample_nr=params[:sample_nr],
    maxiter=params[:maxiter],
    logger_funcs,
    callback,
    timer,
)
rm(save_file; force=true)
