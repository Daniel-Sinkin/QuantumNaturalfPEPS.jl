import cuQuantumNaturalfPEPS
using CUDA

include(joinpath(@__DIR__, "common.jl"))

function load_peps()::Nothing
    if !CUDA.functional()
        println("app/load_peps needs CUDA.")
        return nothing
    end
    println("Invoked app/load_peps()")
    println()

    println("This example shows off the two types of PEPS which are supported for device upload")
    array_peps_host = peps_as_arrays(LX, LY, DIM_BOND, DIM_PHYS)
    println(" (1) CUDA Implementation aligned ArrayPeps which are just $(typeof(array_peps_host))")
    itensors_peps_host = peps_as_itensors(LX, LY, DIM_BOND)
    println(" (2) Existing Julia aligned ITensorPeps which are just $(typeof(itensors_peps_host))")
    println()

    println("Both get first loaded to to unified transferrable format")
    array_peps_host_cu = cuQuantumNaturalfPEPS.load_peps(array_peps_host)
    itensors_peps_host_cu = cuQuantumNaturalfPEPS.load_peps(itensors_peps_host)

    # Currently this always just copies that data over to GPU0 which then broadcasts it to the other GPUs
    # as needed, might make sense to be able to specify the GPU for H2D (host to device) transfer directly
    println("Those representations can then be uploaded to the device using the upload_peps() function")
    array_peps_device = cuQuantumNaturalfPEPS.upload_peps(array_peps_host_cu)
    #=
    The upload_peps() function returns a cuda array reference and holds some metadata.
    struct CuPeps
        data::CuArray{ComplexF32,1} # Host reference to GPU memory
        lx::Int       # pure cpu object
        ly::Int       # pure cpu object
        dim_phys::Int # pure cpu object
        dim_bond::Int # pure cpu object
    end
    =#

    # TODO: Do some kind of CUDA memory transaction to show that these two are the same
    # TODO: Show that the uploaded peps is row major while the two host repr are col major

    return nothing
end
