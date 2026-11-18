using cuQuantumNaturalfPEPS
using CUDA

include(joinpath(@__DIR__, "common.jl"))

const OPERATOR_TOLERANCE = 1.0e-9
const TRANSVERSE_FIELD = 0.4
const DM_COUPLING = 0.3
const LONG_RANGE_COUPLING = 0.2

const PAULI_X = ComplexF64[0 1; 1 0]
const PAULI_Y = ComplexF64[0 -im; im 0]
const PAULI_Z = ComplexF64[1 0; 0 -1]

const EXCHANGE = kron(PAULI_X, PAULI_X) .+ kron(PAULI_Y, PAULI_Y) .+ kron(PAULI_Z, PAULI_Z)
const DM_Z = kron(PAULI_X, PAULI_Y) .- kron(PAULI_Y, PAULI_X)

function nearest_neighbor_bonds(lx, ly)
    horizontal = [((row, col), (row, col + 1)) for row in 1:lx for col in 1:(ly-1)]
    vertical = [((row, col), (row + 1, col)) for row in 1:(lx-1) for col in 1:ly]
    return vcat(horizontal, vertical)
end

function long_range_pair(lx, ly)
    ly >= 3 && return ((1, 1), (1, ly))
    lx >= 3 && return ((1, 1), (lx, 1))
    return error("custom_operators.jl needs a lattice side of at least three")
end

function hand_built_heisenberg(lx, ly)::OperatorTerms
    terms = OperatorTerms(lx, ly)
    for bond in nearest_neighbor_bonds(lx, ly)
        add_operator!(terms, bond, EXCHANGE)
    end
    return terms
end

function perturbed_heisenberg(lx, ly, strength)::OperatorTerms
    terms = heisenberg_terms(lx, ly; J1=1.0, J2=0.0)
    for row in 1:lx, col in 1:ly
        add_operator!(terms, (row, col), (strength * TRANSVERSE_FIELD) .* PAULI_X)
    end
    for bond in nearest_neighbor_bonds(lx, ly)
        add_operator!(terms, bond, (strength * DM_COUPLING) .* DM_Z)
    end
    long_range = (strength * LONG_RANGE_COUPLING) .* kron(PAULI_X, PAULI_X)
    add_operator!(terms, long_range_pair(lx, ly), long_range)
    return terms
end

function custom_operators_example(arguments=ARGS)::Nothing
    options = parse_app_options("custom_operators.jl", arguments)
    app_dry_run("custom_operators.jl", options) && return nothing
    CUDA.functional() || error("custom_operators.jl requires CUDA")
    lx = options[:lx]
    ly = options[:ly]
    dim_bond = options[:dim_bond]
    n_samples = options[:samples]
    tensors = array_peps(lx, ly, dim_bond, 2; seed=options[:seed])
    device_peps = upload_peps(load_peps(tensors))
    dlenv = double_layer(device_peps; chi_s=dim_bond, chi_dl=dim_bond)
    sampled = sample_peps(device_peps, dlenv, n_samples; gpus=1, seed=options[:seed] + 1)
    samples = CuArray(pack_sample_configs(sampled.configs))
    config = QnpepsElocConfig(; lx, ly, dim_bond, chi_eo=dim_bond, meo=32, dim_phys=2)
    energies(terms) = Array(local_energies(device_peps.data, config, terms, samples).e_loc)

    built_in = energies(heisenberg_terms(lx, ly; J1=1.0, J2=0.0))
    hand_built = energies(hand_built_heisenberg(lx, ly))
    heisenberg_deviation = maximum(abs.(hand_built .- built_in))
    println("[custom_operators] samples $(length(built_in)) lattice $((lx, ly))")
    println("[custom_operators] built-in energies   $(value_preview(built_in))")
    println("[custom_operators] hand-built energies $(value_preview(hand_built))")
    println("[custom_operators] maximum deviation $heisenberg_deviation")
    heisenberg_deviation <= OPERATOR_TOLERANCE ||
        error("hand-built Heisenberg energies deviate by $heisenberg_deviation")

    single = energies(perturbed_heisenberg(lx, ly, 1.0)) .- built_in
    double = energies(perturbed_heisenberg(lx, ly, 2.0)) .- built_in
    linearity_deviation = maximum(abs.(double .- 2 .* single))
    println("[custom_operators] perturbation energies $(value_preview(single))")
    println("[custom_operators] linearity deviation $linearity_deviation")
    linearity_deviation <= OPERATOR_TOLERANCE ||
        error("perturbation energies deviate from linearity by $linearity_deviation")

    step = vmc_step!(
        tensors;
        ns=n_samples,
        chi_s=dim_bond,
        chi_dl=dim_bond,
        chi_eo=dim_bond,
        meo=32,
        seed=options[:seed] + 1,
        gpus=options[:gpus],
        terms=perturbed_heisenberg(lx, ly, 1.0),
    )
    println("[custom_operators] step energy $(step.e_mean)")
    isfinite(step.e_mean) || error("step energy $(step.e_mean) is not finite")
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && custom_operators_example()
