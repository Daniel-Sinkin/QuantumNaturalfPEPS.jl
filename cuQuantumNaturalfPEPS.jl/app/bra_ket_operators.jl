using cuQuantumNaturalfPEPS
using CUDA

include(joinpath(@__DIR__, "common.jl"))

const OPERATOR_TOLERANCE = 1.0e-9
const TERM_SHIFT_FLOOR = 1.0e-6

const PAULI_X = ComplexF64[0 1; 1 0]
const PAULI_Y = ComplexF64[0 -im; im 0]
const PAULI_Z = ComplexF64[1 0; 0 -1]
const IDENTITY_TWO = ComplexF64[1 0; 0 1]

const SYMMETRIC_EXCHANGE = kron(PAULI_X, PAULI_X) .+ kron(PAULI_Y, PAULI_Y)
const DM_Z = kron(PAULI_X, PAULI_Y) .- kron(PAULI_Y, PAULI_X)

const SINGLET_AMPLITUDES = ComplexF64[0.6-0.2im, -0.3+0.5im, 0.4+0.7im]
const SITE_ORDER = (3, 1, 2)

function ket(bits::Integer...)::Vector{ComplexF64}
    isempty(bits) && throw(ArgumentError("a ket needs at least one occupation bit"))
    all(bit -> bit in (0, 1), bits) || throw(ArgumentError("occupation bits are 0 or 1"))
    state = zeros(ComplexF64, 2^length(bits))
    state[foldl((index, bit) -> 2 * index + bit, bits; init=0)+1] = 1
    return state
end

const SINGLET_FROM_PAULIS =
    (
        kron(IDENTITY_TWO, IDENTITY_TWO) .- kron(PAULI_X, PAULI_X) .- kron(PAULI_Y, PAULI_Y) .-
        kron(PAULI_Z, PAULI_Z)
    ) ./ 4
const SINGLET = ket(1, 0) - ket(0, 1)
const SINGLET_PROJECTOR = SINGLET * SINGLET' / 2
const FORWARD_HOP = ket(1, 0) * ket(0, 1)'
const BACKWARD_HOP = ket(0, 1) * ket(1, 0)'

function nearest_neighbor_bonds(lx, ly)
    horizontal = [((row, col), (row, col + 1)) for row in 1:lx for col in 1:(ly-1)]
    vertical = [((row, col), (row + 1, col)) for row in 1:(lx-1) for col in 1:ly]
    return vcat(horizontal, vertical)
end

function bond_amplitude(index::Integer)::ComplexF64
    return complex(0.4 + 0.1 * mod(index, 3), (-1)^index * (0.3 + 0.05 * mod(index, 4)))
end

function bond_terms(lx, ly, operator_of_bond)::OperatorTerms
    terms = OperatorTerms(lx, ly)
    for (index, bond) in enumerate(nearest_neighbor_bonds(lx, ly))
        add_operator!(terms, bond, operator_of_bond(index))
    end
    return terms
end

singlet_terms(lx, ly) = bond_terms(lx, ly, index -> SINGLET_PROJECTOR)

function hopping_matrix(amplitude)::Matrix{ComplexF64}
    return amplitude .* FORWARD_HOP .+ conj(amplitude) .* BACKWARD_HOP
end

function pauli_hopping_matrix(amplitude; sign=-1)::Matrix{ComplexF64}
    return (real(amplitude) / 2) .* SYMMETRIC_EXCHANGE .+
           (sign * imag(amplitude) / 2) .* DM_Z
end

hermitian_hopping_terms(lx, ly) =
    bond_terms(lx, ly, index -> hopping_matrix(bond_amplitude(index)))
pauli_hopping_terms(lx, ly) =
    bond_terms(lx, ly, index -> pauli_hopping_matrix(bond_amplitude(index)))
forward_hopping_terms(lx, ly) =
    bond_terms(lx, ly, index -> bond_amplitude(index) .* FORWARD_HOP)
backward_hopping_terms(lx, ly) =
    bond_terms(lx, ly, index -> conj(bond_amplitude(index)) .* BACKWARD_HOP)

function singlet_and_hopping_terms(lx, ly)::OperatorTerms
    terms = singlet_terms(lx, ly)
    for (index, bond) in enumerate(nearest_neighbor_bonds(lx, ly))
        add_operator!(terms, bond, hopping_matrix(bond_amplitude(index)))
    end
    return terms
end

function triple_sites(lx, ly, shape::Symbol)
    shape == :row && return [((row, 1), (row, 2), (row, 3)) for row in 1:lx]
    shape == :corner &&
        return [((r, c), (r + 1, c), (r + 1, c + 1)) for r in 1:(lx-1) for c in 1:(ly-1)]
    throw(ArgumentError("unknown triple shape $shape"))
end

function weighted_state(order=(1, 2, 3))::Vector{ComplexF64}
    state = zeros(ComplexF64, 8)
    for position in 1:3
        bits = zeros(Int, 3)
        bits[position] = 1
        state .+= SINGLET_AMPLITUDES[position] .* ket(bits[collect(order)]...)
    end
    return state
end

function whole_triple_terms(lx, ly, shape)::OperatorTerms
    state = weighted_state()
    terms = OperatorTerms(lx, ly)
    for sites in triple_sites(lx, ly, shape)
        add_operator!(terms, sites, state * state')
    end
    return terms
end

function product_triple_terms(lx, ly, shape)::OperatorTerms
    terms = OperatorTerms(lx, ly)
    for sites in triple_sites(lx, ly, shape), row in 1:3, col in 1:3
        bra_bits = zeros(Int, 3)
        ket_bits = zeros(Int, 3)
        bra_bits[row] = 1
        ket_bits[col] = 1
        coefficient = SINGLET_AMPLITUDES[row] * conj(SINGLET_AMPLITUDES[col])
        add_operator!(terms, sites, coefficient .* (ket(bra_bits...) * ket(ket_bits...)'))
    end
    return terms
end

function permuted_triple_terms(lx, ly, shape)::OperatorTerms
    state = weighted_state(SITE_ORDER)
    terms = OperatorTerms(lx, ly)
    for sites in triple_sites(lx, ly, shape)
        add_operator!(terms, ntuple(k -> sites[SITE_ORDER[k]], 3), state * state')
    end
    return terms
end

function bra_ket_operators_example(arguments=ARGS)::Nothing
    options = parse_app_options("bra_ket_operators.jl", arguments)
    app_dry_run("bra_ket_operators.jl", options) && return nothing
    CUDA.functional() || error("bra_ket_operators.jl requires CUDA")
    lx = options[:lx]
    ly = options[:ly]
    dim_bond = options[:dim_bond]
    n_samples = options[:samples]
    ly >= 3 || error("bra_ket_operators.jl needs a lattice with at least three columns")
    bonds = length(nearest_neighbor_bonds(lx, ly))
    amplitudes = [bond_amplitude(index) for index in 1:bonds]
    all(a -> real(a) != 0 && imag(a) != 0, amplitudes) ||
        error("a bond amplitude has a vanishing real or imaginary part")
    length(unique(amplitudes)) > 1 || error("the bond amplitudes do not differ")

    ket(1, 0) == ComplexF64[0, 0, 1, 0] || error("ket(1, 0) is not the third basis vector")
    ket(0, 1) == ComplexF64[0, 1, 0, 0] || error("ket(0, 1) is not the second basis vector")
    identity_deviation = maximum(
        maximum(abs.(hopping_matrix(a) .- pauli_hopping_matrix(a))) for a in amplitudes
    )
    flipped_deviation = minimum(
        maximum(abs.(hopping_matrix(a) .- pauli_hopping_matrix(a; sign=1))) for a in amplitudes
    )
    hermitian_deviation = maximum(
        maximum(abs.(hopping_matrix(a) .- hopping_matrix(a)')) for a in amplitudes
    )
    singlet_matrix_deviation = maximum(abs.(SINGLET_PROJECTOR .- SINGLET_FROM_PAULIS))
    println("[bra_ket_operators] pauli identity deviation $identity_deviation")
    println("[bra_ket_operators] flipped sign deviation $flipped_deviation")
    println("[bra_ket_operators] hermiticity deviation $hermitian_deviation")
    println("[bra_ket_operators] singlet matrix deviation $singlet_matrix_deviation")
    identity_deviation <= OPERATOR_TOLERANCE ||
        error("the Pauli form of the hopping matrix deviates by $identity_deviation")
    flipped_deviation >= TERM_SHIFT_FLOOR ||
        error("the opposite sign of the Pauli form deviates by only $flipped_deviation")
    hermitian_deviation <= OPERATOR_TOLERANCE ||
        error("the hopping matrix is not Hermitian, deviation $hermitian_deviation")
    singlet_matrix_deviation <= OPERATOR_TOLERANCE ||
        error("the singlet projector deviates from its Pauli form by $singlet_matrix_deviation")

    tensors = array_peps(lx, ly, dim_bond, 2; seed=options[:seed])
    device_peps = upload_peps(load_peps(tensors))
    dlenv = double_layer(device_peps; chi_s=dim_bond, chi_dl=dim_bond)
    sampled = sample_peps(device_peps, dlenv, n_samples; gpus=1, seed=options[:seed] + 1)
    samples = CuArray(pack_sample_configs(sampled.configs))
    config = QnpepsElocConfig(; lx, ly, dim_bond, chi_eo=dim_bond, meo=32, dim_phys=2)
    energies(terms) = Array(local_energies(device_peps.data, config, terms, samples).e_loc)

    built_in = energies(heisenberg_terms(lx, ly; J1=1.0, J2=0.0))
    println("[bra_ket_operators] samples $(length(built_in)) lattice $((lx, ly)) bonds $bonds")
    println("[bra_ket_operators] built-in energies $(value_preview(built_in))")

    singlet = energies(singlet_terms(lx, ly))
    singlet_expected = (bonds .- built_in) ./ 4
    singlet_deviation = maximum(abs.(singlet .- singlet_expected))
    println("[bra_ket_operators] singlet energies $(value_preview(singlet))")
    println("[bra_ket_operators] singlet deviation $singlet_deviation")
    singlet_deviation <= OPERATOR_TOLERANCE ||
        error("singlet energies deviate from the Heisenberg energies by $singlet_deviation")

    hermitian = energies(hermitian_hopping_terms(lx, ly))
    from_paulis = energies(pauli_hopping_terms(lx, ly))
    pauli_deviation = maximum(abs.(hermitian .- from_paulis))
    println("[bra_ket_operators] mixed amplitude energies $(value_preview(hermitian))")
    println("[bra_ket_operators] mixed amplitude deviation $pauli_deviation")
    pauli_deviation <= OPERATOR_TOLERANCE ||
        error("mixed amplitude energies deviate from their Pauli form by $pauli_deviation")

    forward = energies(forward_hopping_terms(lx, ly))
    backward = energies(backward_hopping_terms(lx, ly))
    halves_deviation = maximum(abs.(forward .+ backward .- hermitian))
    forward_shift = maximum(abs.(forward .- hermitian))
    backward_shift = maximum(abs.(backward .- hermitian))
    println("[bra_ket_operators] non-Hermitian half energies $(value_preview(forward))")
    println("[bra_ket_operators] halves deviation $halves_deviation")
    println("[bra_ket_operators] half shifts $forward_shift $backward_shift")
    halves_deviation <= OPERATOR_TOLERANCE ||
        error("the two non-Hermitian halves deviate from the Hermitian sum by $halves_deviation")
    min(forward_shift, backward_shift) >= TERM_SHIFT_FLOOR || error(
        "a non-Hermitian half shifts the energies by at most " *
        "$(min(forward_shift, backward_shift)), below $TERM_SHIFT_FLOOR",
    )

    for shape in (:row, :corner)
        whole = energies(whole_triple_terms(lx, ly, shape))
        products = energies(product_triple_terms(lx, ly, shape))
        permuted = energies(permuted_triple_terms(lx, ly, shape))
        size_reading = maximum(abs.(whole))
        product_deviation = maximum(abs.(whole .- products))
        permuted_deviation = maximum(abs.(whole .- permuted))
        println("[bra_ket_operators] $shape triple energies $(value_preview(whole))")
        println("[bra_ket_operators] $shape triple size $size_reading")
        println("[bra_ket_operators] $shape triple products deviation $product_deviation")
        println("[bra_ket_operators] $shape triple permuted deviation $permuted_deviation")
        size_reading >= TERM_SHIFT_FLOOR ||
            error("$shape triple energies are at most $size_reading, the checks cannot see them")
        product_deviation <= OPERATOR_TOLERANCE ||
            error("$shape triple energies deviate from their outer products by $product_deviation")
        permuted_deviation <= OPERATOR_TOLERANCE ||
            error("$shape triple energies deviate under permuted sites by $permuted_deviation")
    end

    step = vmc_step!(
        tensors;
        ns=n_samples,
        chi_s=dim_bond,
        chi_dl=dim_bond,
        chi_eo=dim_bond,
        meo=32,
        seed=options[:seed] + 1,
        gpus=options[:gpus],
        terms=singlet_and_hopping_terms(lx, ly),
    )
    println("[bra_ket_operators] step energy $(step.e_mean)")
    isfinite(step.e_mean) || error("step energy $(step.e_mean) is not finite")
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && bra_ket_operators_example()
