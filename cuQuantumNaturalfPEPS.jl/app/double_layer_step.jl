# Does the double layer orchestrated on the Julia side, i.e., Julia is responsible for calling into the more specialized double_layer_step instead of it being handed internally by double_layer.
using cuQuantumNaturalfPEPS
using ITensorMPS
using CUDA

include(joinpath(@__DIR__, "common.jl"))

function double_layer_step_example(arguments=ARGS)::Nothing
    options = parse_app_options("double_layer_step.jl", arguments)
    app_dry_run("double_layer_step.jl", options) && return nothing
    if !CUDA.functional()
        println("app/double_layer_step Needs CUDA.")
        return nothing
    end
    tensors = grid_peps(LX, LY, DIM_BOND)

    row_logs = zeros(Float64, LX - 1)
    env_below = nothing
    for row in (LX-1):-1:1
        mps_row, row_log =
            cuQuantumNaturalfPEPS.double_layer_step(tensors, row, env_below; maxdim=CHI)
        row_logs[row] = row_log
        env_below = mps_row
        println(
            "row $row sites $(length(mps_row)) maxlinkdim $(maxlinkdim(mps_row)) row_log $row_log",
        )
    end

    step_cumulative_row_logs = zeros(Float64, LX - 1)
    acc = 0.0
    for row in (LX-1):-1:1
        acc += row_logs[row]
        step_cumulative_row_logs[row] = acc
    end

    device_peps = upload_peps(load_peps(tensors))
    dlenv = double_layer(device_peps; chi_dl=CHI)

    println("step_cumulative_row_logs $step_cumulative_row_logs")
    println("oneshot_cumulative_row_logs $(dlenv.cumulative_row_logs)")
    println("max_abs_diff $(maximum(abs.(step_cumulative_row_logs .- dlenv.cumulative_row_logs)))")
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && double_layer_step_example()
