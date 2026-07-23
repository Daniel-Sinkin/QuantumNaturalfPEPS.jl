using cuQuantumNaturalfPEPS
using CUDA

include(joinpath(@__DIR__, "common.jl"))

function sampling_multigpu()::Nothing
    if !CUDA.functional()
        println("app/sampling_multigpu needs CUDA.")
        return nothing
    end
    device_peps = upload_peps(load_peps(peps_as_itensors(LX, LY, DIM_BOND)))
    dlenv = double_layer(device_peps)

    # TODO: Rewrite this, use this to show the sampling parallel efficiency instead
    sample_peps(device_peps, dlenv, 64; gpus=1, seed=SEED_SAMPLE)
    reference = sample_peps(device_peps, dlenv, NS; gpus=1, seed=SEED_SAMPLE)
    t1 = @elapsed sample_peps(device_peps, dlenv, NS; gpus=1, seed=SEED_SAMPLE)
    println("gpus 1 elapsed_s $t1")

    for gpu in 2:4
        gpu <= CUDA.ndevices() || continue
        res = sample_peps(device_peps, dlenv, NS; gpus=gpu, seed=SEED_SAMPLE)
        configs_equal = res.configs == reference.configs
        log_prob_equal = res.log_prob_config == reference.log_prob_config
        equal = configs_equal && log_prob_equal
        @assert configs_equal
        @assert log_prob_equal
        tg = @elapsed sample_peps(device_peps, dlenv, NS; gpus=gpu, seed=SEED_SAMPLE)
        println("gpus $gpu elapsed_s $tg equal_gpus1 $equal")
    end
    return nothing
end
