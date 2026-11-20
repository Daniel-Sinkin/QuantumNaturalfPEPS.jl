using cuQuantumNaturalfPEPS
using CUDA

include(joinpath(@__DIR__, "common.jl"))

const OPERATOR_TOLERANCE = 1.0e-9
const TERM_SHIFT_FLOOR = 1.0e-6
const TRANSVERSE_FIELD = 0.4
const DM_COUPLING = 0.3
const LONG_RANGE_COUPLING = 0.2

const PAULI_X = ComplexF64[0 1; 1 0]
const PAULI_Y = ComplexF64[0 -im; im 0]
const PAULI_Z = ComplexF64[1 0; 0 -1]

const EXCHANGE = kron(PAULI_X, PAULI_X) .+ kron(PAULI_Y, PAULI_Y) .+ kron(PAULI_Z, PAULI_Z)
const DM_Z = kron(PAULI_X, PAULI_Y) .- kron(PAULI_Y, PAULI_X)

const PAULIS = (PAULI_X, PAULI_Y, PAULI_Z)
const LEVI_CIVITA = (
    (1, 2, 3, 1.0),
    (2, 3, 1, 1.0),
    (3, 1, 2, 1.0),
    (1, 3, 2, -1.0),
    (3, 2, 1, -1.0),
    (2, 1, 3, -1.0),
)
const CHIRALITY =
    sum(sign .* kron(PAULIS[i], PAULIS[j], PAULIS[k]) for (i, j, k, sign) in LEVI_CIVITA)

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

function lone_operator(lx, ly, sites, matrix)::OperatorTerms
    return add_operator!(OperatorTerms(lx, ly), sites, matrix)
end

function separate_perturbation_terms(lx, ly)::Vector{Tuple{Float64,OperatorTerms}}
    field = [
        (TRANSVERSE_FIELD, lone_operator(lx, ly, (row, col), PAULI_X)) for row in 1:lx for
        col in 1:ly
    ]
    dm = [
        (DM_COUPLING, lone_operator(lx, ly, bond, DM_Z)) for
        bond in nearest_neighbor_bonds(lx, ly)
    ]
    long_range = (
        LONG_RANGE_COUPLING,
        lone_operator(lx, ly, long_range_pair(lx, ly), kron(PAULI_X, PAULI_X)),
    )
    return vcat(field, dm, [long_range])
end

function plaquette_triangles(lx, ly)
    triangles = NTuple{3,NTuple{2,Int}}[]
    for row in 1:(lx-1), col in 1:(ly-1)
        corners = ((row, col), (row + 1, col), (row + 1, col + 1), (row, col + 1))
        for omitted in 0:3
            push!(triangles, ntuple(k -> corners[mod(omitted + k, 4)+1], 3))
        end
    end
    return triangles
end

function chirality_terms(lx, ly; exchanged::Bool=false)::OperatorTerms
    terms = OperatorTerms(lx, ly)
    for (a, b, c) in plaquette_triangles(lx, ly)
        add_operator!(terms, exchanged ? (b, a, c) : (a, b, c), CHIRALITY)
    end
    return terms
end

function chirality_from_products(lx, ly)::OperatorTerms
    terms = OperatorTerms(lx, ly)
    for (a, b, c) in plaquette_triangles(lx, ly), (i, j, k, sign) in LEVI_CIVITA
        add_operator!(terms, (c, a, b), sign .* kron(PAULIS[k], PAULIS[i], PAULIS[j]))
    end
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

    term_shifts = [
        coefficient .* energies(terms) for
        (coefficient, terms) in separate_perturbation_terms(lx, ly)
    ]
    decomposition_deviation = maximum(abs.(single .- sum(term_shifts)))
    smallest_term_shift = minimum(maximum(abs.(shift)) for shift in term_shifts)
    println("[custom_operators] decomposition deviation $decomposition_deviation")
    term_count = length(term_shifts)
    println("[custom_operators] smallest term shift $smallest_term_shift over $term_count terms")
    decomposition_deviation <= OPERATOR_TOLERANCE || error(
        "perturbation energies deviate from the sum of separately evaluated terms by " *
        "$decomposition_deviation",
    )
    smallest_term_shift >= TERM_SHIFT_FLOOR || error(
        "a perturbation term shifts the energies by at most $smallest_term_shift, " *
        "below $TERM_SHIFT_FLOOR, the decomposition cannot see it",
    )

    chirality = energies(chirality_terms(lx, ly))
    exchanged = energies(chirality_terms(lx, ly; exchanged=true))
    from_products = energies(chirality_from_products(lx, ly))
    chirality_deviation =
        max(maximum(abs.(exchanged .+ chirality)), maximum(abs.(from_products .- chirality)))
    println("[custom_operators] chirality energies $(value_preview(chirality))")
    println("[custom_operators] chirality deviation $chirality_deviation")
    chirality_deviation <= OPERATOR_TOLERANCE ||
        error("chirality energies deviate from their exchange or products by $chirality_deviation")

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
