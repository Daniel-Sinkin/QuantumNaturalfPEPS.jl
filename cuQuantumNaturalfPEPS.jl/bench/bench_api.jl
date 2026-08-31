using cuQuantumNaturalfPEPS
using ITensors
using CUDA
using Random

function grid_peps(lx, ly, dim_bond, dim_phys; seed=1)
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

function mean_sd(samples)
    n = length(samples)
    mean = sum(samples) / n
    sd = sqrt(sum((value - mean)^2 for value in samples) / (n - 1))
    return (mean, sd)
end

function bench(lx, dim_bond, chi, n_samples, iters)
    tensors = grid_peps(lx, lx, dim_bond, 2)
    peps = load_peps(tensors)
    dp = upload_peps(peps)
    CUDA.synchronize()

    dlenv = double_layer(dp; chi_s=chi, chi_dl=dim_bond)
    CUDA.synchronize()

    build_samples = Float64[]
    GC.gc()
    for i in 1:iters
        t = time_ns()
        double_layer(dp; chi_s=chi, chi_dl=dim_bond)
        CUDA.synchronize()
        push!(build_samples, (time_ns() - t) / 1e6)
    end
    build_ms, build_sd = mean_sd(build_samples)

    sample_peps(dp, dlenv, n_samples; gpus=1, seed=7)
    sample_peps(dp, dlenv, n_samples; gpus=1, seed=7)
    CUDA.synchronize()

    sample_samples = Float64[]
    GC.gc()
    for i in 1:iters
        t = time_ns()
        sample_peps(dp, dlenv, n_samples; gpus=1, seed=7)
        CUDA.synchronize()
        push!(sample_samples, (time_ns() - t) / 1e6)
    end
    sample_ms, sample_sd = mean_sd(sample_samples)

    println(
        "[bench_api] api=julia L=$(lx) D=$(dim_bond) chi=$(chi) n_samples=$(n_samples) iters=$(iters) build_ms=$(round(build_ms, digits=3)) build_sd=$(round(build_sd, digits=3)) sample_ms=$(round(sample_ms, digits=3)) sample_sd=$(round(sample_sd, digits=3))",
    )
    return nothing
end

function main()
    if !CUDA.functional()
        println(stderr, "[bench_api] no functional GPU")
        exit(1)
    end
    bench_case = parse(Int, ARGS[1])
    bench_case == 1 ? bench(8, 4, 4, 1024, 100) : bench(16, 7, 7, 512, 40)
    println("[bench_api] DONE")
    return nothing
end

main()
