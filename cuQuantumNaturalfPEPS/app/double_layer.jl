import cuQuantumNaturalfPEPS
using CUDA

include(joinpath(@__DIR__, "common.jl"))

function double_layer()::Nothing
    if !CUDA.functional()
        println("app/double_layer needs CUDA.")
        return nothing
    end
    println("Invoked app/double_layer()")
    # Here we create a peps in the existing style, transform to the cuda aligned style
    # and then upload it to the GPU.
    device_peps = cuQuantumNaturalfPEPS.upload_peps(
        cuQuantumNaturalfPEPS.load_peps(peps_as_itensors(LX, LY, DIM_BOND)),
    )
    println("$(device_peps)")

    # This means we don't do any truncations
    chi_dl_exact = DIM_BOND * DIM_BOND
    println("We compare the behaviour of the double layer with and without truncations (chi_dl = D, D^2)")

    untruncated = cuQuantumNaturalfPEPS.double_layer(device_peps; chi_dl=chi_dl_exact)
    truncated = cuQuantumNaturalfPEPS.double_layer(device_peps)

    ut_bytes = length(untruncated.data)
    ut_norms = untruncated.cumulative_row_logs
    tr_bytes = length(truncated.data)
    tr_norms = truncated.cumulative_row_logs

    bytes_pct = round(100 * tr_bytes / ut_bytes; digits = 1)

    println("untruncated takes $ut_bytes while truncated takes $tr_bytes ($bytes_pct%)")

    deviation = maximum(abs.(ut_norms.- tr_norms))
    println("Maxium Row Log-Norm deviation is $deviation")
    return nothing
end
