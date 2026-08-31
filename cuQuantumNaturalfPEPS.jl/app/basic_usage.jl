# Goes through the basic endpoints one by one.
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

function tiny_array_peps(dim_bond, dim_phys; seed=2)
    Random.seed!(seed)
    site(dims...) = rand(ComplexF64, dims...)
    arrays = Matrix{Array{ComplexF64,5}}(undef, 2, 2)
    arrays[1, 1] = site(dim_phys, 1, dim_bond, dim_bond, 1)
    arrays[1, 2] = site(dim_phys, 1, 1, dim_bond, dim_bond)
    arrays[2, 1] = site(dim_phys, dim_bond, dim_bond, 1, 1)
    arrays[2, 2] = site(dim_phys, dim_bond, 1, 1, dim_bond)
    return arrays
end

function basic_usage(arguments=ARGS)::Nothing
    options = parse_app_options("basic_usage.jl", arguments)
    app_dry_run("basic_usage.jl", options) && return nothing
    lx = options[:lx]
    ly = options[:ly]
    dim_bond = options[:dim_bond]
    tensors = random_peps(lx, ly, dim_bond, DIM_PHYS; seed=options[:seed])
    peps = load_peps(tensors)
    println("load_peps (ITensor grid) -> ", peps)
    peps_from_arrays = load_peps(tiny_array_peps(dim_bond, DIM_PHYS; seed=options[:seed]))
    println("load_peps (rank-5 [p,u,r,d,l] arrays) -> ", peps_from_arrays)
    if !CUDA.functional()
        println("no functional GPU; device endpoints skipped")
        return nothing
    end
    mpss, logs = double_layer(tensors; maxdim=dim_bond * dim_bond)
    println(
        "double_layer (Julia row loop) -> ",
        length(mpss),
        " boundary-MPS rows; cumulative logs ",
        round.(logs; digits=3),
    )
    env_row, row_log = double_layer_step(tensors, 2, mpss[3]; maxdim=dim_bond * dim_bond)
    println(
        "double_layer_step (CUDA row FFI) -> env row of ",
        length(env_row),
        " sites, log ",
        round(row_log; digits=3),
    )
    device_peps = upload_peps(peps)
    println("upload_peps -> ", typeof(device_peps), " on the current GPU")
    dlenv = double_layer(device_peps; chi_s=dim_bond)
    println("double_layer -> device-resident double-layer env (chi_s=", dlenv.chi_s, ")")
    dlenv_wide = double_layer(device_peps; chi_s=dim_bond, chi_dl=dim_bond * dim_bond)
    println("double_layer (chi_dl=", dlenv_wide.chi_dl, ") -> wider double-layer env")
    result = sample_peps(device_peps, dlenv, 64; gpus=1)
    println("sample_peps -> ", length(result.configs), " configs of size ", size(result.configs[1]))
    show(stdout, "text/plain", result.configs[1])
    println()
    config = QnpepsConfig(lx=lx, ly=ly, dim_bond=dim_bond, chi_s=dim_bond, dim_phys=DIM_PHYS)
    samples = CUDA.zeros(UInt8, 64 * lx * ly)
    sample_peps!(device_peps.data, dlenv.data, samples, config)
    println(
        "sample_peps! -> drew ",
        length(samples) ÷ (lx * ly),
        " configs into the caller's device buffer",
    )
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && basic_usage()
