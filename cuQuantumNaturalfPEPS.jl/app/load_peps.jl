using cuQuantumNaturalfPEPS
using CUDA

include(joinpath(@__DIR__, "common.jl"))

function load_peps_example(arguments=ARGS)::Nothing
    options = parse_app_options("load_peps.jl", arguments)
    app_dry_run("load_peps.jl", options) && return nothing
    if !CUDA.functional()
        println("app/load_peps Needs CUDA.")
        return nothing
    end
    peps_arrays = cuQuantumNaturalfPEPS.load_peps(array_peps(LX, LY, DIM_BOND, DIM_PHYS))
    peps_grid = cuQuantumNaturalfPEPS.load_peps(grid_peps(LX, LY, DIM_BOND))
    println("arrays_grid $(size(peps_arrays)) dim_bond $DIM_BOND")
    println("itensor_grid $(size(peps_grid)) dim_bond $DIM_BOND")
    println("corner_dims $(size(peps_arrays.tensors[1, 1]))")
    println("center_dims $(size(peps_arrays.tensors[2, 2]))")
    device_peps = upload_peps(peps_arrays)
    println("device_buffer_elements $(length(device_peps.data)) device $(CUDA.device())")
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && load_peps_example()
