using cuQuantumNaturalfPEPS
using CUDA

include(joinpath(@__DIR__, "common.jl"))

function vmc_step_single_gpu_example(arguments=ARGS)::Nothing
    options = parse_app_options("vmc_step_single_gpu.jl", arguments)
    app_dry_run("vmc_step_single_gpu.jl", options) && return nothing
    CUDA.functional() || error("vmc_step_single_gpu.jl requires CUDA")
    lx = options[:lx]
    ly = options[:ly]
    dim_bond = options[:dim_bond]
    n_samples = options[:samples]
    peps = CuArray(pack_peps_arrays(array_peps(lx, ly, dim_bond, 2; seed=options[:seed])))
    config = QnpepsE2eConfig(
        lx=lx,
        ly=ly,
        dim_bond=dim_bond,
        chi_s=dim_bond,
        chi_dl=dim_bond,
        chi_eo=dim_bond,
        meo=32,
        dim_phys=2,
        seed=options[:seed] + 1,
    )
    terms = heisenberg_terms(lx, ly; J1=1.0, J2=0.0)
    context = VMCContext(
        config,
        terms;
        gpus=1,
        ns_capacity=n_samples,
        ns_ahead=0,
        dim_batch=min(n_samples, 2048),
    )
    try
        submit_peps!(context, peps)
        result = vmc_euler_step!(
            context,
            peps;
            n_samples=n_samples,
            learning_rate=options[:learning_rate],
            want_theta=true,
            want_logq=true,
            want_logpsi=true,
            want_e_loc=true,
        )
        CUDA.synchronize()
        println("[vmc-single] theta_dot_shape $(size(result.theta_dot))")
        println("[vmc-single] theta_dot_values $(value_preview(Array(result.theta_dot)))")
        println("[vmc-single] e_mean $(result.e_mean) e_var $(result.e_var) ess $(result.ess)")
        println("[vmc-single] updated_peps_values $(value_preview(Array(result.peps)))")
    finally
        close(context)
    end
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && vmc_step_single_gpu_example()
