# VMC = Variational Monte Carlo
using cuQuantumNaturalfPEPS
using CUDA

include(joinpath(@__DIR__, "common.jl"))

function vmc_step_multigpu_example(arguments=ARGS)::Nothing
    options = parse_app_options("vmc_step_multigpu.jl", arguments)
    app_dry_run("vmc_step_multigpu.jl", options) && return nothing
    CUDA.functional() || error("vmc_step_multigpu.jl requires CUDA")
    CUDA.ndevices() >= options[:gpus] || error("requested GPU count is not visible")
    lx = options[:lx]
    ly = options[:ly]
    dim_bond = options[:dim_bond]
    n_samples = options[:samples]
    peps = array_peps(lx, ly, dim_bond, 2; seed=options[:seed])
    result = vmc_step!(
        peps;
        ns=n_samples,
        chi_s=dim_bond,
        chi_dl=dim_bond,
        chi_eo=dim_bond,
        meo=32,
        seed=options[:seed] + 1,
        gpus=options[:gpus],
        want_logq=true,
        want_logpsi=true,
    )
    println("[vmc-multigpu] gpus $(options[:gpus]) samples $n_samples")
    println("[vmc-multigpu] theta_dot_shape $(size(result.theta_dot))")
    println("[vmc-multigpu] theta_dot_values $(value_preview(result.theta_dot))")
    println("[vmc-multigpu] e_mean $(result.e_mean) e_var $(result.e_var) ess $(result.ess)")
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && vmc_step_multigpu_example()
