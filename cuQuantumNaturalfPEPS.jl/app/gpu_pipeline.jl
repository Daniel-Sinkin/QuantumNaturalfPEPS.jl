using cuQuantumNaturalfPEPS
using ITensors
using CUDA
using Random

include(joinpath(@__DIR__, "common.jl"))

function random_peps(lx, ly, dim_bond, dim_phys; seed=1)
    Random.seed!(seed)
    sites = [Index(dim_phys, "p$row$col") for row in 1:lx, col in 1:ly]
    horizontal = [Index(dim_bond, "h$row$col") for row in 1:lx, col in 1:(ly-1)]
    vertical = [Index(dim_bond, "v$row$col") for row in 1:(lx-1), col in 1:ly]
    tensors = Matrix{ITensor}(undef, lx, ly)
    for row in 1:lx, col in 1:ly
        legs = Index[sites[row, col]]
        row > 1 && push!(legs, vertical[row-1, col])
        col < ly && push!(legs, horizontal[row, col])
        row < lx && push!(legs, vertical[row, col])
        col > 1 && push!(legs, horizontal[row, col-1])
        tensors[row, col] = random_itensor(ComplexF64, legs...)
    end
    return tensors
end

function gpu_pipeline(arguments=ARGS)::Nothing
    options = parse_app_options("gpu_pipeline.jl", arguments)
    app_dry_run("gpu_pipeline.jl", options) && return nothing
    CUDA.functional() || error("gpu_pipeline.jl requires a functional CUDA GPU")
    lx = options[:lx]
    ly = options[:ly]
    dim_bond = options[:dim_bond]
    n_samples = options[:samples]
    tensors = random_peps(lx, ly, dim_bond, DIM_PHYS; seed=options[:seed])
    device_peps = upload_peps(load_peps(tensors))
    dlenv = double_layer(device_peps)
    config = QnpepsConfig(
        lx=dlenv.lx,
        ly=dlenv.ly,
        dim_bond=dlenv.dim_bond,
        chi_s=dlenv.chi_s,
        chi_dl=dlenv.chi_dl,
        dim_phys=dlenv.dim_phys,
    )
    samples = CUDA.zeros(UInt8, n_samples * dlenv.lx * dlenv.ly)
    log_prob_config = CUDA.zeros(Float64, n_samples)
    log_gauge = CUDA.zeros(Float64, n_samples)
    sample_peps!(
        device_peps.data,
        dlenv.data,
        samples,
        config;
        log_prob_config=log_prob_config,
        log_gauge=log_gauge,
    )
    CUDA.synchronize()
    mean_spin = sum(Float32.(samples)) / length(samples)
    mean_log_prob_config = sum(log_prob_config) / length(log_prob_config)
    mean_log_gauge = sum(log_gauge) / length(log_gauge)
    println("drew $n_samples configs through the on-GPU pipeline ($lx x $ly, dim_bond=$dim_bond)")
    println("samples ", typeof(samples), " ", length(samples), " bytes, device-resident")
    println("log_prob_config ", typeof(log_prob_config), " device-resident")
    println("mean spin ", mean_spin)
    println("mean log p(config) ", mean_log_prob_config)
    println("mean log gauge ", mean_log_gauge)
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && gpu_pipeline()
