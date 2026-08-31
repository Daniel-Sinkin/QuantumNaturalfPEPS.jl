using cuQuantumNaturalfPEPS
using CUDA

include(joinpath(@__DIR__, "common.jl"))

function minsr_example(arguments=ARGS)::Nothing
    options = parse_app_options("minsr.jl", arguments)
    app_dry_run("minsr.jl", options) && return nothing
    CUDA.functional() || error("minsr.jl requires CUDA")
    lx = options[:lx]
    ly = options[:ly]
    dim_bond = options[:dim_bond]
    n_samples = options[:samples]
    host_peps = array_peps(lx, ly, dim_bond, 2; seed=options[:seed])
    outputs = vmc_step!(
        host_peps;
        ns=n_samples,
        chi_s=dim_bond,
        chi_dl=dim_bond,
        chi_eo=dim_bond,
        meo=32,
        seed=options[:seed] + 1,
        gpus=1,
        want_samples=true,
        want_logq=true,
        want_logpsi=true,
        want_e_loc=true,
        want_o_rows=true,
    )
    samples = CuArray(outputs.samples)
    logpsi = CuArray(outputs.logpsi)
    e_loc = CuArray(outputs.e_loc)
    logq = CuArray(outputs.logq)
    o_rows = CuArray(outputs.o_rows)
    gram_context = GramContext(
        lx=lx,
        ly=ly,
        dim_bond=dim_bond,
        n_samples=n_samples,
        dim_phys=2,
    )
    minsr_context = MinsrContext(
        lx=lx,
        ly=ly,
        dim_bond=dim_bond,
        n_samples=n_samples,
        dim_phys=2,
    )
    try
        gram = CUDA.zeros(ComplexF32, n_samples * n_samples)
        raw_gram!(gram_context, gram, samples, o_rows)
        result = minsr_direction(
            minsr_context,
            samples,
            logpsi,
            e_loc,
            logq,
            gram,
            o_rows,
        )
        CUDA.synchronize()
        footprint = gram_footprint(gram_context)
        println("[minsr] gram_shape $((n_samples, n_samples))")
        println("[minsr] gram_caller_bytes $(footprint.caller_gram_bytes)")
        println("[minsr] theta_dot_shape $(size(result.theta_dot))")
        println("[minsr] theta_dot_values $(value_preview(Array(result.theta_dot)))")
        println("[minsr] e_mean $(result.e_mean) e_var $(result.e_var) ess $(result.ess)")
    finally
        close(minsr_context)
        close(gram_context)
    end
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && minsr_example()
