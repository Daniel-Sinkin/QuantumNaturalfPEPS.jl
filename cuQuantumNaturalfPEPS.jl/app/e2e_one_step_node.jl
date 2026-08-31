using cuQuantumNaturalfPEPS
using CUDA
using LinearAlgebra

include(joinpath(@__DIR__, "common.jl"))

const NODE_RELATIVE_CUT = 1.0e-3
const NODE_ABSOLUTE_CUT = 1.0e-8

function resolve_node_entrypoints()::Nothing
    for symbol in (
        :QnpepsE2eConfig,
        :heisenberg_terms,
        :VMCContext,
        :submit_peps!,
        :vmc_euler_step!,
        :vmc_step!,
    )
        isdefined(cuQuantumNaturalfPEPS, symbol) || error("unresolved package entry point $symbol")
    end
    return nothing
end

function print_node_updated_peps(tensors, lx, ly, dim_bond)::Nothing
    if lx == 2 && ly == 2 && dim_bond == 2
        for row in 1:lx, col in 1:ly
            println("[update] tensor $row $col shape $(size(tensors[row, col]))")
            show(stdout, "text/plain", tensors[row, col])
            println()
        end
    else
        for row in 1:lx, col in 1:ly
            tensor = tensors[row, col]
            println(
                "[update] tensor $row $col shape $(size(tensor)) digest $(complex_digest(tensor))",
            )
        end
    end
    return nothing
end

function e2e_one_step_node(arguments=ARGS)::Nothing
    options = parse_app_options("e2e_one_step_node.jl", arguments)
    lx = options[:lx]
    ly = options[:ly]
    dim_bond = options[:dim_bond]
    n_samples = options[:samples]
    seed = options[:seed]
    learning_rate = options[:learning_rate]
    dim_batch = min(n_samples, 2048)
    config = QnpepsE2eConfig(
        lx=lx,
        ly=ly,
        dim_bond=dim_bond,
        chi_s=dim_bond,
        chi_dl=dim_bond,
        chi_eo=dim_bond,
        meo=32,
        dim_phys=2,
        seed=seed + 1,
    )
    terms = heisenberg_terms(lx, ly; J1=1.0, J2=0.0)
    resolve_node_entrypoints()
    if options[:dry_run]
        println(
            "[app] e2e_one_step_node.jl config gpus $(options[:gpus]) samples $n_samples ns_ahead $(options[:ns_ahead]) ns_capacity $(options[:ns_capacity])",
        )
    end
    app_dry_run("e2e_one_step_node.jl", options) && return nothing

    CUDA.functional() || error("e2e_one_step_node.jl requires CUDA")
    CUDA.ndevices() >= options[:gpus] || error("requested GPU count is not visible")
    tensors = array_peps(lx, ly, dim_bond, 2; seed)
    packed = pack_peps_arrays(tensors)
    peps = CuArray(packed)
    println("[host] lattice $lx $ly dim_bond $dim_bond seed $seed")
    println("[host] site_shapes $(size.(tensors)) packed_shape $(size(packed))")
    println("[host] packed_values $(value_preview(packed))")
    println(
        "[node] gpus $(options[:gpus]) samples $n_samples ns_ahead $(options[:ns_ahead]) ns_capacity $(options[:ns_capacity])",
    )

    context = VMCContext(
        config,
        terms;
        gpus=options[:gpus],
        ns_capacity=options[:ns_capacity],
        ns_ahead=options[:ns_ahead],
        dim_batch,
    )
    result = try
        submit_peps!(context, peps)
        vmc_euler_step!(
            context,
            peps;
            n_samples,
            learning_rate,
            relative_cut=NODE_RELATIVE_CUT,
            absolute_cut=NODE_ABSOLUTE_CUT,
            want_theta=true,
        )
    finally
        close(context)
    end
    CUDA.synchronize()

    one_gpu = vmc_step!(
        tensors;
        ns=n_samples,
        chi_s=dim_bond,
        chi_dl=dim_bond,
        chi_eo=dim_bond,
        meo=32,
        seed=seed + 1,
        gpus=1,
        relative_cut=NODE_RELATIVE_CUT,
        absolute_cut=NODE_ABSOLUTE_CUT,
    )
    theta_dot = Array(result.theta_dot)
    max_deviation = maximum(abs.(theta_dot .- one_gpu.theta_dot))
    if options[:gpus] == 1
        theta_dot == one_gpu.theta_dot || error("persistent and one-GPU directions differ")
        println("[compare] one_gpu_direction_equal true max_abs_deviation $max_deviation")
    else
        println(
            "[compare] $(options[:gpus])_gpu_vs_one_gpu_theta_dot_max_abs_deviation $max_deviation",
        )
    end

    println("[node] e_mean $(result.e_mean) e_var $(result.e_var) ess $(result.ess)")
    println("[node] theta_dot_norm $(norm(theta_dot))")
    updated_packed = Array(result.peps)
    updated = unpack_peps_arrays(updated_packed, tensors)
    println("[update] learning_rate $learning_rate packed_shape $(size(updated_packed))")
    println("[update] packed_values $(value_preview(updated_packed))")
    print_node_updated_peps(updated, lx, ly, dim_bond)
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && e2e_one_step_node()
