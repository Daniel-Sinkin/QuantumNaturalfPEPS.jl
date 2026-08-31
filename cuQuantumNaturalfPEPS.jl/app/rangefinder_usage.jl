using cuQuantumNaturalfPEPS
using CUDA
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "common.jl"))

const ROWS = 64
const COLS = 48
const BATCH = 32
const RANK = 16

function low_rank_batch(rows, cols, rank, batch; seed=0)
    Random.seed!(seed)
    input = Array{ComplexF32}(undef, rows, cols, batch)
    for b in 1:batch
        u = randn(ComplexF32, rows, rank)
        v = randn(ComplexF32, rank, cols)
        input[:, :, b] = u * v
    end
    return input
end

function rangefinder_usage(arguments=ARGS)::Nothing
    options = parse_app_options("rangefinder_usage.jl", arguments)
    app_dry_run("rangefinder_usage.jl", options) && return nothing
    if !CUDA.functional()
        println("no functional GPU; batched_rangefinder is a device endpoint, skipped")
        return nothing
    end
    input_host = low_rank_batch(ROWS, COLS, RANK, BATCH)
    input = CuArray(input_host)
    println("input -> ", size(input), " ComplexF32 (rows, cols, batch)")

    q, r = batched_rangefinder(input, RANK; seed=0)
    CUDA.synchronize()
    println("batched_rangefinder -> q ", size(q), "  r ", size(r), "   (input ≈ q * r per lane)")

    q1 = Array(q[:, :, 1])
    r1 = Array(r[:, :, 1])
    p1 = input_host[:, :, 1]
    rel_recon = maximum(abs.(q1 * r1 .- p1)) / maximum(abs.(p1))
    ortho = maximum(abs.(q1' * q1 - I))
    println(
        "lane 1: rel|q*r - input| = ",
        round(rel_recon; sigdigits=3),
        "   max|q'q - I| = ",
        round(ortho; sigdigits=3),
    )

    dense = ROWS * COLS
    factored = ROWS * RANK + RANK * COLS
    println(
        "compression: ",
        dense,
        " -> ",
        factored,
        " entries/lane (",
        round(dense / factored; digits=2),
        "x smaller)",
    )
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && rangefinder_usage()
