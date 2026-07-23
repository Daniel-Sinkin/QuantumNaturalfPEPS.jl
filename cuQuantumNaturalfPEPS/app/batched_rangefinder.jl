using cuQuantumNaturalfPEPS
using CUDA
using LinearAlgebra

include(joinpath(@__DIR__, "common.jl"))

const ROWS, COLS, RANK, BATCH_SIZE = 32, 24, 8, 4

function batched_rangefinder()::Nothing
    if !CUDA.functional()
        println("app/batched_rangefinder needs CUDA.")
        return nothing
    end
    Random.seed!(0) # TODO: This should use the shared app/ seed

    # TODO: Add some explanations here

    host = Array{ComplexF32}(undef, ROWS, COLS, BATCH_SIZE)
    for batch in 1:BATCH_SIZE
        host[:, :, batch] = randn(ComplexF32, ROWS, RANK) * randn(ComplexF32, RANK, COLS)
    end
    input = CuArray(host)

    qs, rs = cuQuantumNaturalfPEPS.batched_rangefinder(input, RANK; seed=0)
    CUDA.synchronize()

    println("input_dims $(size(input)) rank $RANK")
    println("q_dims $(size(qs)) r_dims $(size(rs))")
    q1 = Array(qs[:, :, 1])
    r1 = Array(rs[:, :, 1])
    rel = norm(q1 * r1 - host[:, :, 1]) / norm(host[:, :, 1])
    println("rel_error $rel")
    return nothing
end
