using cuQuantumNaturalfPEPS
using CUDA

include(joinpath(@__DIR__, "common.jl"))

function print_updated_peps(tensors, lx, ly, dim_bond)::Nothing
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

function e2e_one_step(arguments=ARGS)::Nothing
    options = parse_app_options("e2e_one_step.jl", arguments)
    app_dry_run("e2e_one_step.jl", options) && return nothing
    CUDA.functional() || error("e2e_one_step.jl requires CUDA")
    CUDA.ndevices() >= options[:gpus] || error("requested GPU count is not visible")
    lx = options[:lx]
    ly = options[:ly]
    dim_bond = options[:dim_bond]
    n_samples = options[:samples]
    seed = options[:seed]
    learning_rate = options[:learning_rate]

    tensors = array_peps(lx, ly, dim_bond, 2; seed)
    packed = pack_peps_arrays(tensors)
    println("[host] lattice $lx $ly dim_bond $dim_bond seed $seed")
    println("[host] site_shapes $(size.(tensors)) packed_shape $(size(packed))")
    println("[host] packed_values $(value_preview(packed))")

    peps = load_peps(tensors)
    device_peps = upload_peps(peps)
    println("[upload] device_shape $(size(device_peps.data))")
    println("[upload] values $(value_preview(Array(device_peps.data)))")

    dlenv = double_layer(device_peps; chi_s=dim_bond, chi_dl=dim_bond)
    println("[dlenv] byte_shape $(size(dlenv.data)) chi_s $(dlenv.chi_s) chi_dl $(dlenv.chi_dl)")
    println("[dlenv] row_logs $(dlenv.cumulative_row_logs)")

    sampled = sample_peps(
        device_peps,
        dlenv,
        n_samples;
        gpus=options[:gpus],
        seed=seed + 1,
    )
    println("[sample] batch_shape $((n_samples, lx, ly))")
    println("[sample] first_config $(sampled.configs[1])")
    println("[sample] log_q_values $(value_preview(sampled.log_prob_config))")

    result = vmc_step!(
        tensors;
        ns=n_samples,
        chi_s=dim_bond,
        chi_dl=dim_bond,
        chi_eo=dim_bond,
        meo=32,
        seed=seed + 1,
        gpus=options[:gpus],
        want_samples=true,
        want_logq=true,
        want_log_gauge=true,
        want_logpsi=true,
        want_e_loc=true,
        want_o_rows=true,
    )
    sample_match = result.samples == pack_sample_configs(sampled.configs)
    proposal_match = result.logq == sampled.log_prob_config
    options[:gpus] == 1 && !sample_match && error("sample stage and full-step samples differ")
    options[:gpus] == 1 && !proposal_match &&
        error("sample stage and full-step proposal logs differ")
    println("[sample] full_step_samples_equal $sample_match proposal_logs_equal $proposal_match")
    compact = length(result.o_rows) ÷ n_samples
    println("[eo] logpsi_shape $(size(result.logpsi)) values $(value_preview(result.logpsi))")
    println("[eo] local_energy_shape $(size(result.e_loc)) values $(value_preview(result.e_loc))")
    println("[eo] ok_layout compact_by_sample $((compact, n_samples))")
    println("[eo] ok_values $(value_preview(result.o_rows))")
    println("[minsr] theta_dot_shape $(size(result.theta_dot))")
    println("[minsr] theta_dot_values $(value_preview(result.theta_dot))")
    println("[minsr] e_mean $(result.e_mean) e_var $(result.e_var) ess $(result.ess)")

    length(packed) == length(result.theta_dot) ||
        throw(DimensionMismatch("theta_dot and PEPS lengths differ"))
    updated_packed = packed .+ ComplexF32(learning_rate) .* result.theta_dot
    updated = unpack_peps_arrays(updated_packed, tensors)
    println("[update] learning_rate $learning_rate packed_shape $(size(updated_packed))")
    println("[update] packed_values $(value_preview(updated_packed))")
    print_updated_peps(updated, lx, ly, dim_bond)
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && e2e_one_step()
