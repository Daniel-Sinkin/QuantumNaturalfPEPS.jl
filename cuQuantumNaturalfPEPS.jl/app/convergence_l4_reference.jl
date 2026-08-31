if "--figure-only" ∉ ARGS
    using cuQuantumNaturalfPEPS
    using CUDA
end
using LinearAlgebra
using Libdl
using Printf
using Statistics

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "convergence_l4_reference", "io.jl"))
include(joinpath(@__DIR__, "convergence_l4_reference", "state.jl"))
include(joinpath(@__DIR__, "convergence_l4_reference", "ranks.jl"))

const CONVERGENCE_EO_ROUTES = (
    "cholqr",
    "householder",
    "qb_svd",
    "gesvda_cutoff",
)
const CONVERGENCE_TRAJECTORY_HEADER = join(
    (
        "step",
        "seed",
        "route",
        "precision",
        "chi",
        "samples",
        "library_energy_re",
        "library_energy_im",
        "library_energy_per_site",
        "library_energy_error",
        "library_variance",
        "library_ess",
        "reference_energy_re",
        "reference_energy_per_site",
        "reference_energy_error",
        "exact_energy",
        "library_gap",
        "reference_gap",
        "theta_norm",
        "update_norm",
        "state_norm",
        "wall_seconds",
    ),
    ',',
)
const CONVERGENCE_FIRST_SEED = 2_560_101
const CONVERGENCE_NATIVE_SYMBOLS = (
    :qnpeps_capi_version,
    :qnpeps_e2e_version,
    :qnpeps_eloc_version,
    :qnpeps_e2e_strerror,
    :qnpeps_e2e_dense_count,
    :qnpeps_e2e_compact_count,
    :qnpeps_e2e_node_create,
    :qnpeps_e2e_node_submit_theta,
    :qnpeps_e2e_node_step_euler,
    :qnpeps_e2e_node_error_stage,
    :qnpeps_e2e_node_destroy,
)

function convergence_help()
    println(
        "usage convergence_l4_reference.jl [--steps N] [--route ROUTE] [--chi N] " *
        "[--precision fp32|fp64] [--output DIR] [--figure-only]",
    )
    println(
        "composite routes rangefinder, rf, svd, and density are accepted with every E O selector",
    )
    return nothing
end

function configure_convergence_route!(route::AbstractString)
    if route == "svd"
        ENV["QNPEPS_SAMPLER_TRUNC"] = "svd"
        ENV["QNPEPS_DLENV_TRUNC"] = "svd"
        ENV["QNPEPS_E0191_RF_ROUTE"] = "gesvda_cutoff"
        ENV["QNPEPS_ELOC_GESVDA_CUTOFF"] = "0"
    elseif route in ("rf", "rangefinder")
        ENV["QNPEPS_SAMPLER_TRUNC"] = "rf"
        ENV["QNPEPS_DLENV_TRUNC"] = "rf"
        ENV["QNPEPS_E0191_RF_ROUTE"] = "cholqr"
    elseif route == "density"
        ENV["QNPEPS_SAMPLER_TRUNC"] = "density"
        ENV["QNPEPS_DLENV_TRUNC"] = "density"
        ENV["QNPEPS_E0191_RF_ROUTE"] = "density"
    elseif route in CONVERGENCE_EO_ROUTES
        ENV["QNPEPS_SAMPLER_TRUNC"] = "rf"
        ENV["QNPEPS_DLENV_TRUNC"] = "rf"
        ENV["QNPEPS_E0191_RF_ROUTE"] = route
        route == "gesvda_cutoff" && (ENV["QNPEPS_ELOC_GESVDA_CUTOFF"] = "1e-13")
    else
        throw(ArgumentError("unknown truncation route $(repr(route))"))
    end
    return nothing
end

function convergence_output_directory(options)
    options[:output] !== nothing && return abspath(options[:output])
    root = normpath(joinpath(@__DIR__, "..", ".."))
    name =
        "$(options[:route])-$(options[:precision])-chi$(options[:chi])-steps$(options[:steps])"
    return joinpath(root, "private", "convergence_l4_reference", name)
end

function stationary_statistics(energies, errors, variances, exact)
    count = min(50, length(energies))
    selected = (length(energies) - count + 1):length(energies)
    mean_energy = mean(@view energies[selected])
    pooled_error = sqrt(sum(abs2, @view errors[selected])) / count
    gap = mean_energy - exact
    tolerance = max(2 * pooled_error, 1.0e-3 * abs(exact))
    return (
        count,
        first_step=first(selected),
        last_step=last(selected),
        mean_energy,
        pooled_error,
        gap,
        tolerance,
        mean_variance=mean(@view variances[selected]),
        r1=count == 50 && abs(gap) <= tolerance,
    )
end

function write_convergence_summary(
    path,
    options,
    statistics,
    reference_statistics,
    overlap,
    exact,
)
    open(path, "w") do io
        println(
            io,
            "route,precision,chi,steps,samples,window_first,window_last,window_count," *
            "exact_energy,library_mean_energy,library_pooled_error,library_gap," *
            "library_tolerance,library_mean_variance,library_r1,reference_mean_energy," *
            "reference_pooled_error,reference_gap,reference_tolerance,reference_r1," *
            "final_state_overlap,ok_comparison,reference_ok_order,library_ok_order",
        )
        write_csv_row(
            io,
            (
                options[:route],
                options[:precision],
                options[:chi],
                options[:steps],
                1008,
                statistics.first_step,
                statistics.last_step,
                statistics.count,
                exact,
                statistics.mean_energy,
                statistics.pooled_error,
                statistics.gap,
                statistics.tolerance,
                statistics.mean_variance,
                statistics.r1,
                reference_statistics.mean_energy,
                reference_statistics.pooled_error,
                reference_statistics.gap,
                reference_statistics.tolerance,
                reference_statistics.r1,
                overlap,
                false,
                "W-S-E-N",
                "W-N-S-E",
            ),
        )
    end
    return nothing
end

function make_convergence_figure(trajectory, ranks, output)
    root = normpath(joinpath(@__DIR__, "..", ".."))
    script = joinpath(root, "documentation", "dossier_figures", "make_figures.py")
    python = Sys.which("python3")
    python === nothing && error("python3 is required to write the PNG")
    run(`$python $script --convergence-l4 $trajectory $ranks $output`)
    return nothing
end

function convergence_config(options, seed)
    return QnpepsE2eConfig(
        lx=4,
        ly=4,
        dim_bond=6,
        chi_s=options[:chi],
        chi_dl=options[:chi],
        chi_eo=options[:chi],
        meo=63,
        dim_phys=2,
        seed=seed,
        sampling_mode=:full,
        contract_dim=options[:chi],
        sample_batch=63,
    )
end

function convergence_library_path()
    configured = get(ENV, "QNPEPS_LIB", "")
    isempty(configured) || return abspath(configured)
    return normpath(joinpath(@__DIR__, "..", "build", "cuda", "qnpeps.so"))
end

function convergence_dry_run(options)
    options[:dry_run] || return false
    options[:chi] >= 2 || throw(ArgumentError("chi must be at least 2"))
    options[:steps] <= 2000 || throw(ArgumentError("steps exceeds the E0261 budget"))
    configure_convergence_route!(options[:route])
    config = convergence_config(options, CONVERGENCE_FIRST_SEED)
    terms = heisenberg_terms(4, 4; J1=1.0, J2=0.58)
    path = convergence_library_path()
    isfile(path) || error("qnpeps.so is absent at $path")
    handle = Libdl.dlopen(path)
    try
        for symbol in CONVERGENCE_NATIVE_SYMBOLS
            Libdl.dlsym(handle, symbol)
        end
    finally
        Libdl.dlclose(handle)
    end
    println(
        "[convergence-dry-run] config=$(config.lx)x$(config.ly) D=$(config.dim_bond) " *
        "chi=$(config.chi_s) seed=$(config.seed) terms=$(length(terms.diag)+length(terms.flip))",
    )
    println(
        "[convergence-dry-run] library=$(realpath(path)) symbols=$(length(CONVERGENCE_NATIVE_SYMBOLS))",
    )
    return true
end

function convergence_native_step!(peps, state_f64, seed, options, terms)
    config = convergence_config(options, seed)
    context = VMCContext(
        config,
        terms;
        gpus=1,
        ns_capacity=1008,
        ns_ahead=0,
        dim_batch=63,
    )
    try
        submit_peps!(context, peps)
        result = vmc_euler_step!(
            context,
            peps;
            state_f64,
            precision=options[:precision] == "fp64" ? :f64 : :f32,
            n_samples=1008,
            learning_rate=0.05,
            relative_cut=1.0e-4,
            absolute_cut=0.0,
            want_theta=true,
        )
        theta_dot = Array(result.theta_dot)
        packed = Array(result.peps)
        return (
            theta_dot,
            packed,
            e_mean=result.e_mean,
            e_var=result.e_var,
            ess=result.ess,
        )
    finally
        close(context)
    end
end

function render_convergence_figure(output)
    trajectory = joinpath(output, "convergence_l4_reference.csv")
    ranks = joinpath(output, "convergence_l4_reference_ranks.csv")
    figure = joinpath(output, "convergence_l4_reference.png")
    isfile(trajectory) || error("trajectory CSV is absent at $trajectory")
    isfile(ranks) || error("rank CSV is absent at $ranks")
    make_convergence_figure(trajectory, ranks, figure)
    println("[convergence] figure=$figure")
    return nothing
end

function convergence_l4_reference(arguments=ARGS)::Nothing
    options = parse_app_options("convergence_l4_reference.jl", arguments)
    options[:help] && return convergence_help()
    convergence_dry_run(options) && return nothing
    if options[:figure_only]
        options[:output] === nothing &&
            throw(ArgumentError("--figure-only requires --output DIR"))
        render_convergence_figure(abspath(options[:output]))
        return nothing
    end
    CUDA.functional() || error("convergence_l4_reference.jl requires CUDA")
    options[:chi] >= 2 || throw(ArgumentError("chi must be at least 2"))
    options[:steps] <= 2000 || throw(ArgumentError("steps exceeds the E0261 budget"))
    configure_convergence_route!(options[:route])

    fixture_path = joinpath(CONVERGENCE_REFERENCE_ROOT, "fixtures", "l4_d6_a1.5_s1.qnf")
    final_path = joinpath(CONVERGENCE_REFERENCE_WINDOW, "final_states", "reference_J.qnf")
    fixture = read_convergence_qnf(fixture_path)
    reference_final = read_convergence_qnf(final_path)
    (fixture.lx, fixture.ly, fixture.bond) == (4, 4, 6) || error("E0261 fixture differs")
    fixture.seed == 1 || error("E0261 fixture seed differs")
    reference = reference_trajectory()
    exact = parse(Float64, reference[1]["exact_energy"])
    output = convergence_output_directory(options)
    ispath(output) && error("output path already exists at $output")
    mkpath(output)
    trajectory_path = joinpath(output, "convergence_l4_reference.csv")
    rank_path = joinpath(output, "convergence_l4_reference_ranks.csv")
    raw_profile = joinpath(output, "eo_rank_telemetry_raw.csv")
    summary_path = joinpath(output, "convergence_l4_reference_summary.csv")
    figure_path = joinpath(output, "convergence_l4_reference.png")
    ENV["QNPEPS_ELOC_PROF"] = raw_profile

    tensors = map(tensor -> ComplexF32.(real.(tensor)), fixture.tensors)
    packed = pack_convergence_peps(tensors)
    peps = CuArray(packed)
    state_f64 = options[:precision] == "fp64" ? CuArray(ComplexF64.(packed)) : nothing
    terms = heisenberg_terms(4, 4; J1=1.0, J2=0.58)
    energies = Float64[]
    errors = Float64[]
    variances = Float64[]
    telemetry_offset = Int64(0)
    trajectory_io = open(trajectory_path, "w")
    ranks_io = open(rank_path, "w")
    println(trajectory_io, CONVERGENCE_TRAJECTORY_HEADER)
    println(ranks_io, "step,phase,cut,lane,retained_rank,cap,batch,route,source")
    try
        for step in 1:options[:steps]
            seed = CONVERGENCE_FIRST_SEED + step - 1
            started = time_ns()
            result = convergence_native_step!(peps, state_f64, seed, options, terms)
            wall = (time_ns() - started) * 1.0e-9
            energy = result.e_mean
            energy_error = sqrt(max(result.e_var, 0.0) / result.ess)
            theta_norm = norm(ComplexF64.(result.theta_dot))
            update_norm = 0.05 * theta_norm
            packed = result.packed
            state_norm = state_f64 === nothing ?
                         norm(ComplexF64.(packed)) : norm(Array(state_f64))
            ref = reference[step]
            ref_energy = parse(Float64, ref["energy_re"])
            push!(energies, real(energy))
            push!(errors, energy_error)
            push!(variances, result.e_var)
            write_csv_row(
                trajectory_io,
                (
                    step,
                    seed,
                    options[:route],
                    options[:precision],
                    options[:chi],
                    1008,
                    @sprintf("%.17g", real(energy)),
                    @sprintf("%.17g", imag(energy)),
                    @sprintf("%.17g", real(energy) / 16),
                    @sprintf("%.17g", energy_error),
                    @sprintf("%.17g", result.e_var),
                    @sprintf("%.17g", result.ess),
                    ref["energy_re"],
                    ref["energy_per_site"],
                    ref["energy_error"],
                    ref["exact_energy"],
                    @sprintf("%.17g", real(energy) - exact),
                    ref["gap"],
                    @sprintf("%.17g", theta_norm),
                    @sprintf("%.17g", update_norm),
                    @sprintf("%.17g", state_norm),
                    @sprintf("%.9f", wall),
                ),
            )
            telemetry, telemetry_offset = telemetry_rows(raw_profile, telemetry_offset, step)
            write_rank_rows(ranks_io, telemetry)
            println(
                "[convergence] step=$step energy=$(real(energy)) reference=$ref_energy " *
                "ess=$(result.ess) wall_s=$wall",
            )
        end
    finally
        close(trajectory_io)
        close(ranks_io)
    end

    final_packed = state_f64 === nothing ? packed : Array(state_f64)
    tensors = unpack_convergence_peps(final_packed, tensors)
    overlap = convergence_overlap(tensors, reference_final.device_tensors)
    statistics = stationary_statistics(energies, errors, variances, exact)
    reference_statistics = stationary_statistics(
        parse.(Float64, getindex.(reference, "energy_re")),
        parse.(Float64, getindex.(reference, "energy_error")),
        parse.(Float64, getindex.(reference, "local_energy_variance")),
        exact,
    )
    write_convergence_summary(
        summary_path,
        options,
        statistics,
        reference_statistics,
        overlap,
        exact,
    )
    println("[convergence] trajectory=$trajectory_path")
    println("[convergence] ranks=$rank_path")
    println("[convergence] summary=$summary_path")
    println("[convergence] final_state_overlap=$overlap r1=$(statistics.r1)")
    try
        make_convergence_figure(trajectory_path, rank_path, figure_path)
        println("[convergence] figure=$figure_path")
    catch exception
        print(stderr, "[convergence] figure unavailable ")
        showerror(stderr, exception)
        println(stderr)
    end
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && convergence_l4_reference()
