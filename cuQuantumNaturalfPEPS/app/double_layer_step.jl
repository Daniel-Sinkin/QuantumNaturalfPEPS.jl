using cuQuantumNaturalfPEPS
using ITensorMPS
using CUDA

include(joinpath(@__DIR__, "common.jl"))

function double_layer_step()::Nothing
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

double_layer_step()
