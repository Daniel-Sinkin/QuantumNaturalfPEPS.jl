using cuQuantumNaturalfPEPS
using CUDA

include(joinpath(@__DIR__, "common.jl"))

function double_layer()::Nothing
    if !CUDA.functional()
        println("app/double_layer Needs CUDA.")
        return nothing
    end
    device_peps = upload_peps(load_peps(grid_peps(LX, LY, DIM_BOND)))
    println("peps: $LX x $LY lattice, dim_bond $DIM_BOND, dim_phys $DIM_PHYS")

    chi_dl_exact = DIM_BOND * DIM_BOND
    println("chi_dl caps the bond dimension of the double-layer row MPS.")
    println("chi_dl = dim_bond^2 = $chi_dl_exact is exact, while default chi_dl = dim_bond truncates.")

    untruncated = cuQuantumNaturalfPEPS.double_layer(device_peps; chi_dl=chi_dl_exact)
    truncated = cuQuantumNaturalfPEPS.double_layer(device_peps)
    println("untruncated: chi_dl $(untruncated.chi_dl), env bytes $(length(untruncated.data))")
    println("truncated: chi_dl $(truncated.chi_dl), env bytes $(length(truncated.data))")
    println("cumulative row log-norms untruncated: $(untruncated.cumulative_row_logs)")
    println("cumulative row log-norms truncated: $(truncated.cumulative_row_logs)")

    deviation = maximum(abs.(untruncated.cumulative_row_logs .- truncated.cumulative_row_logs))
    memory_percent = round(100 * length(truncated.data) / length(untruncated.data); digits=1)
    println("max row-log deviation $deviation at $memory_percent% of the env memory")
    return nothing
end

double_layer()
