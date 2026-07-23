using cuQuantumNaturalfPEPS
using CUDA

include(joinpath(@__DIR__, "common.jl"))

function sampling()::Nothing
    if !CUDA.functional()
        println("app/sampling needs CUDA.")
        return nothing
    end
    host_peps = cuQuantumNaturalfPEPS.load_peps(peps_as_itensors(LX, LY, DIM_BOND))
    device_peps = cuQuantumNaturalfPEPS.upload_peps(host_peps)
    dlenv = cuQuantumNaturalfPEPS.double_layer(device_peps)

    # TODO: Replace this with the sugar API (cpu peps only, result retval)
    result = cuQuantumNaturalfPEPS.sample_peps(
        device_peps,
        dlenv,
        NS;
        gpus=1,
        seed=SEED_SAMPLE,
    )

    num_returned = length(result.configs)
    if num_returned != NS
        println("sampling returned $num_returned samples but we queried $NS!")
        return nothing
    end
    println("The sampling returned $num_returned samples as expected")

    log_prob = result.log_prob_config[1]
    log_gauge = result.log_gauge[1]
    println("The first sample is $(result.configs[1]), log_prob=$log_prob, log_gauge=$log_gauge")

    # TODO: Remove this and replace with the context preserving one, this mem prealloc julia path is deprecated
    config = QnpepsConfig(
        lx=dlenv.lx,
        ly=dlenv.ly,
        dim_bond=dlenv.dim_bond,
        chi_s=dlenv.chi_s,
        chi_dl=dlenv.chi_dl,
        dim_phys=dlenv.dim_phys,
    )
    samples = CUDA.zeros(UInt8, NS * dlenv.lx * dlenv.ly)
    log_prob_config = CUDA.zeros(Float64, NS)
    log_gauge = CUDA.zeros(Float64, NS)
    sample_peps!(
        device_peps.data,
        dlenv.data,
        samples,
        config;
        seed=SEED_SAMPLE,
        log_prob_config=log_prob_config,
        log_gauge=log_gauge,
    )
    CUDA.synchronize()

    flat = Array(samples)
    log_prob_equal = Array(log_prob_config) == result.log_prob_config
    log_gauge_equal = Array(log_gauge) == result.log_gauge
    configs_equal = all(
        result.configs[s][row, col] == flat[(s-1)*LX*LY+(row-1)*LY+col] for
        s in 1:NS, row in 1:LX, col in 1:LY
    )
    equal = log_prob_equal && log_gauge_equal && configs_equal
    println("sample_peps!_equal $equal")
    @assert log_prob_equal
    @assert log_gauge_equal
    @assert configs_equal
    return nothing
end
