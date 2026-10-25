using cuQuantumNaturalfPEPS
using CUDA

include(joinpath(@__DIR__, "common.jl"))

const FIELD_STRENGTH = 0.7
const FIELD_TOLERANCE = 1.0e-9

staggered_field(row, col) = isodd(row + col) ? -FIELD_STRENGTH : FIELD_STRENGTH

pauli_z(spin) = spin == 0 ? 1.0 : -1.0

site_index(row, col, ly) = Int32((row - 1) * ly + (col - 1))

function heisenberg_with_field(lx, ly)::HeisenbergTerms
    heisenberg = heisenberg_terms(lx, ly; J1=1.0, J2=0.0)
    field = [
        QnpepsElocDiagBond(
            site_index(row, col, ly),
            ELOC_DIAG_ONE_SITE_Z,
            staggered_field(row, col),
        ) for row in 1:lx for col in 1:ly
    ]
    return HeisenbergTerms(vcat(heisenberg.diag, field), heisenberg.flip)
end

function field_energy(config)::Float64
    lx, ly = size(config)
    return sum(staggered_field(row, col) * pauli_z(config[row, col]) for row in 1:lx, col in 1:ly)
end

function eo_one_site_z_example(arguments=ARGS)::Nothing
    options = parse_app_options("eo_one_site_z.jl", arguments)
    app_dry_run("eo_one_site_z.jl", options) && return nothing
    CUDA.functional() || error("eo_one_site_z.jl requires CUDA")
    lx = options[:lx]
    ly = options[:ly]
    dim_bond = options[:dim_bond]
    device_peps = upload_peps(load_peps(array_peps(lx, ly, dim_bond, 2; seed=options[:seed])))
    dlenv = double_layer(device_peps; chi_s=dim_bond, chi_dl=dim_bond)
    sampled = sample_peps(device_peps, dlenv, options[:samples]; gpus=1, seed=options[:seed] + 1)
    samples = CuArray(pack_sample_configs(sampled.configs))
    config = QnpepsElocConfig(; lx, ly, dim_bond, chi_eo=dim_bond, meo=32, dim_phys=2)

    heisenberg = local_energies(device_peps.data, config, heisenberg_terms(lx, ly), samples)
    with_field = local_energies(device_peps.data, config, heisenberg_with_field(lx, ly), samples)
    device_field = Array(with_field.e_loc) .- Array(heisenberg.e_loc)
    host_field = field_energy.(sampled.configs)
    deviation = maximum(abs.(device_field .- host_field))

    println("[eo_one_site_z] samples $(length(host_field)) lattice $((lx, ly))")
    println("[eo_one_site_z] staggered field strength $FIELD_STRENGTH")
    println("[eo_one_site_z] device field energy $(value_preview(device_field))")
    println("[eo_one_site_z] host field energy   $(value_preview(host_field))")
    println("[eo_one_site_z] maximum deviation $deviation")
    deviation < FIELD_TOLERANCE || error("one-site Z field energy deviates by $deviation")
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && eo_one_site_z_example()
