using cuQuantumNaturalfPEPS
using CUDA

include(joinpath(@__DIR__, "common.jl"))

function sampling()::Nothing
    if !CUDA.functional()
        println("app/sampling Needs CUDA.")
        return nothing
    end
    device_peps = upload_peps(load_peps(grid_peps(LX, LY, DIM_BOND)))
    dlenv = double_layer(device_peps)

    result = sample_peps(device_peps, dlenv, NS; gpus=1, seed=SEED_SAMPLE)
    println("count $NS config_dims $(size(result.configs[1]))")
    println("first_config $(result.configs[1])")
    println("first_log_prob_config $(result.log_prob_config[1])")
    println("first_log_gauge $(result.log_gauge[1])")

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

sampling()
