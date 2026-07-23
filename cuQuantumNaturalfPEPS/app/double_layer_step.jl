import cuQuantumNaturalfPEPS
using ITensorMPS
using CUDA

include(joinpath(@__DIR__, "common.jl"))

# TODO: Replace this with the ZipUp API 
function double_layer_step()::Nothing
    if !CUDA.functional()
        println("app/double_layer_step needs CUDA.")
        return nothing
    end
    tensors = peps_as_itensors(LX, LY, DIM_BOND)

    row_logs = zeros(Float64, LX - 1)
    env_below = nothing
    for row in (LX-1):-1:1
        mps_row, row_log = cuQuantumNaturalfPEPS.double_layer_step(
            tensors,
            row,
            env_below;
            maxdim=DIM_BOND * DIM_BOND,
        )
        row_logs[row] = row_log
        env_below = mps_row
    end

    step_norms = zeros(Float64, LX - 1)
    acc = 0.0
    for row in (LX-1):-1:1
        acc += row_logs[row]
        step_norms[row] = acc
    end

    device_peps = cuQuantumNaturalfPEPS.upload_peps(cuQuantumNaturalfPEPS.load_peps(tensors))
    dlenv = cuQuantumNaturalfPEPS.double_layer(
        device_peps;
        chi_dl=DIM_BOND * DIM_BOND,
    )

    println("step_cumulative_row_logs $step_norms")
    println("oneshot_cumulative_row_logs $(dlenv.cumulative_row_logs)")
    println("max_abs_diff $(maximum(abs.(step_norms .- dlenv.cumulative_row_logs)))")
    return nothing
end
