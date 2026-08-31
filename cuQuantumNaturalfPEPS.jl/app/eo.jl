using cuQuantumNaturalfPEPS
using CUDA

include(joinpath(@__DIR__, "common.jl"))

function eo_example(arguments=ARGS)::Nothing
    options = parse_app_options("eo.jl", arguments)
    app_dry_run("eo.jl", options) && return nothing
    CUDA.functional() || error("eo.jl requires CUDA")
    CUDA.ndevices() == 4 || error("eo.jl requires exactly four visible GPUs")
    lx = options[:lx]
    ly = options[:ly]
    dim_bond = options[:dim_bond]
    n_samples = options[:samples]
    host_peps = array_peps(lx, ly, dim_bond, 2; seed=options[:seed])
    device_peps = upload_peps(load_peps(host_peps))
    dlenv = double_layer(device_peps; chi_s=dim_bond, chi_dl=dim_bond)
    sampled = sample_peps(
        device_peps,
        dlenv,
        n_samples;
        gpus=1,
        seed=options[:seed] + 1,
    )
    samples = CuArray(pack_sample_configs(sampled.configs))
    config = QnpepsElocConfig(
        lx=lx,
        ly=ly,
        dim_bond=dim_bond,
        chi_eo=dim_bond,
        meo=32,
        dim_phys=2,
    )
    terms = heisenberg_terms(lx, ly; J1=1.0, J2=0.0)
    compact = eloc_compact_count(config)
    logpsi = CUDA.zeros(Float64, 2 * n_samples)
    e_loc = CUDA.zeros(Float64, 2 * n_samples)
    rows = Vector{ComplexF32}(undef, n_samples * compact)
    registration = GC.@preserve rows CUDA.register(
        CUDA.HostMemory,
        pointer(rows),
        sizeof(rows),
        CUDA.MEMHOSTREGISTER_PORTABLE,
    )
    host = nothing
    try
        arguments = EoHostArguments(
            ;
            peps=device_peps.data,
            config,
            table=eo_term_table(config, terms),
            n_samples,
            compact,
            samples,
            logpsi,
            e_loc,
            rows,
        )
        host = EoHost(arguments)
        for lane in 1:4
            status = CUDA.device!(lane - 1) do
                eo_host_execute!(host, lane)
            end
            status == 0 || error("E O lane $lane failed with status $status")
        end
        CUDA.device!(0)
        logpsi_host = copy(reinterpret(ComplexF64, Array(logpsi)))
        e_loc_host = copy(reinterpret(ComplexF64, Array(e_loc)))
        println("[eo] samples_shape $((n_samples, lx, ly))")
        println("[eo] logpsi_shape $(size(logpsi_host)) values $(value_preview(logpsi_host))")
        println("[eo] local_energy_shape $(size(e_loc_host)) values $(value_preview(e_loc_host))")
        println("[eo] ok_layout compact_by_sample $((compact, n_samples))")
        println("[eo] ok_values $(value_preview(rows))")
    finally
        host === nothing || close(host)
        CUDA.device!(0) do
            CUDA.unregister(registration)
        end
    end
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && eo_example()
