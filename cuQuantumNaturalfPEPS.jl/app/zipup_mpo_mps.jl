using cuQuantumNaturalfPEPS
using CUDA
using Random

include(joinpath(@__DIR__, "common.jl"))

function zipup_mpo_mps_example(arguments=ARGS)::Nothing
    options = parse_app_options("zipup_mpo_mps.jl", arguments)
    app_dry_run("zipup_mpo_mps.jl", options) && return nothing
    CUDA.functional() || error("zipup_mpo_mps.jl requires CUDA")
    Random.seed!(options[:seed])
    sites = 3
    physical = 2
    bond = 2
    mpo = [
        CUDA.CuArray(
            reshape(
                ComplexF32[
                    input == output ? 1 : 0 for left in 1:1, input in 1:physical,
                    output in 1:physical, right in 1:1
                ],
                1,
                physical,
                physical,
                1,
            ),
        ) for _ in 1:sites
    ]
    mps = [
        CUDA.CuArray(rand(ComplexF32, site == 1 ? 1 : bond, physical, site == sites ? 1 : bond)) for site in 1:sites
    ]
    result = zipup_mpo_mps(mpo, mps; maxdim=bond)
    # zipup_mpo_mps dispatches the instruction (basically launches a kernel, i.e., enqueues the operation into a task queue which at some point in the future is going to be done)
    # This allows us to do some other work (on the host (= cpu) while we wait for the zipup to finish, or we can also launch multiple operations on the GPU like this without
    # having the overhead of always synchronizing with the CPU.

    # This has the disadvantage that we can only be certain the the operations are done once we synchronize, this blocks the CPU (process that is executing it) until it is done.
    CUDA.synchronize()

    println("[zipup] input_mpo_shapes $(size.(mpo))")
    println("[zipup] input_mps_shapes $(size.(mps))")
    println("[zipup] output_mps_shapes $(size.(result.mps))")
    println("[zipup] log_gauge $(result.log_gauge)")
    println("[zipup] first_values $(value_preview(Array(result.mps[1])))")
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && zipup_mpo_mps_example()
