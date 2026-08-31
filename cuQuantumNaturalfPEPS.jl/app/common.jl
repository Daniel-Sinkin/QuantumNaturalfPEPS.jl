# Contains shared data and definitions for app/, should not be used outside of app/
using ITensors
using Random

const LX, LY, DIM_BOND, DIM_PHYS = 4, 4, 4, 2
const CHI = DIM_BOND * DIM_BOND
const SEED = 1
const SEED_SAMPLE = 7
const NS = 4096

const APP_FILES = (
    "convergence_l4_reference.jl",
    "e2e_one_step.jl",
    "e2e_one_step_node.jl",
    "walkthrough_examples.jl",
    "basic_usage.jl",
    "load_peps.jl",
    "double_layer.jl",
    "double_layer_step.jl",
    "sampling.jl",
    "sampling_multigpu.jl",
    "gpu_pipeline.jl",
    "batched_rangefinder.jl",
    "rangefinder_usage.jl",
    "zipup_mpo_mps.jl",
    "eo.jl",
    "minsr.jl",
    "vmc_step_single_gpu.jl",
    "vmc_step_multigpu.jl",
    "ffi.jl",
    "runner_demo.jl",
)

const APP_PUBLIC_SYMBOLS = Dict(
    "convergence_l4_reference.jl" =>
        (:QnpepsE2eConfig, :heisenberg_terms, :VMCContext, :submit_peps!, :vmc_euler_step!),
    "e2e_one_step.jl" => (:load_peps, :upload_peps, :double_layer, :sample_peps, :vmc_step!),
    "e2e_one_step_node.jl" => (
        :QnpepsE2eConfig,
        :heisenberg_terms,
        :VMCContext,
        :submit_peps!,
        :vmc_euler_step!,
        :vmc_step!,
    ),
    "walkthrough_examples.jl" => (
        :FFI,
        :e2e_version,
        :eloc_version,
        :sampler_version,
        :batched_rangefinder,
        :zipup_mpo_mps,
        :QnpepsElocConfig,
        :QnpepsElocTermTable,
        :eloc_compact_count,
        :heisenberg_terms,
        :minsr_compact_count,
        :minsr_dense_count,
        :GramContext,
        :gram_footprint,
        :raw_gram!,
        :minsr_direction,
    ),
    "basic_usage.jl" => (
        :load_peps,
        :double_layer,
        :double_layer_step,
        :upload_peps,
        :sample_peps,
        :sample_peps!,
        :QnpepsConfig,
    ),
    "load_peps.jl" => (:load_peps, :upload_peps),
    "double_layer.jl" => (:load_peps, :upload_peps, :double_layer),
    "double_layer_step.jl" => (:double_layer_step, :load_peps, :upload_peps, :double_layer),
    "sampling.jl" =>
        (:load_peps, :upload_peps, :double_layer, :sample_peps, :sample_peps!, :QnpepsConfig),
    "sampling_multigpu.jl" => (:load_peps, :upload_peps, :double_layer, :sample_peps),
    "gpu_pipeline.jl" =>
        (:load_peps, :upload_peps, :double_layer, :sample_peps!, :QnpepsConfig),
    "batched_rangefinder.jl" => (:batched_rangefinder,),
    "rangefinder_usage.jl" => (:batched_rangefinder,),
    "zipup_mpo_mps.jl" => (:zipup_mpo_mps,),
    "eo.jl" => (
        :load_peps,
        :upload_peps,
        :double_layer,
        :sample_peps,
        :QnpepsElocConfig,
        :heisenberg_terms,
        :eloc_compact_count,
        :EoHost,
        :eo_host_execute!,
    ),
    "minsr.jl" => (
        :vmc_step!,
        :GramContext,
        :MinsrContext,
        :raw_gram!,
        :gram_footprint,
        :minsr_direction,
    ),
    "vmc_step_single_gpu.jl" =>
        (:QnpepsE2eConfig, :heisenberg_terms, :VMCContext, :submit_peps!, :vmc_euler_step!),
    "vmc_step_multigpu.jl" => (:vmc_step!,),
    "ffi.jl" => (:FFI, :MAX_BATCH_SIZE, :QnpepsConfig, :load_peps, :upload_peps),
    "runner_demo.jl" => (
        :SyntheticAdapter,
        :StreamEventSink,
        :JsonlEventSink,
        :TeeEventSink,
        :RunnerControl,
        :RunnerConfig,
        :start_stdin_control!,
        :run_iterations!,
        :close_event_sink!,
    ),
)

function _app_option_value(arguments, index, name)
    index < length(arguments) || throw(ArgumentError("missing value for $(name)"))
    return arguments[index+1]
end

function parse_app_options(name::AbstractString, arguments)
    name in APP_FILES || throw(ArgumentError("unknown app $(repr(name))"))
    options = Dict{Symbol,Any}(
        :dry_run => false,
        :gpus => name in ("vmc_step_multigpu.jl", "e2e_one_step_node.jl") ? 4 : 1,
        :lx => name in ("e2e_one_step.jl", "e2e_one_step_node.jl") ? 2 : 4,
        :ly => name in ("e2e_one_step.jl", "e2e_one_step_node.jl") ? 2 : 4,
        :dim_bond => name == "e2e_one_step.jl" ? 2 : 2,
        :samples => name in ("e2e_one_step.jl", "e2e_one_step_node.jl") ? 64 : 256,
        :seed => 17,
        :learning_rate => 1.0e-3,
        :ns_ahead => 0,
        :ns_capacity => nothing,
        :iterations => 20,
        :steps => name == "convergence_l4_reference.jl" ? 2000 : 20,
        :route => "svd",
        :precision => name == "convergence_l4_reference.jl" ? "fp64" : "fp32",
        :chi => 36,
        :output => nothing,
        :figure_only => false,
        :jsonl => nothing,
        :checkpoint => joinpath("private", "runner_demo", "checkpoint.json"),
        :resume => false,
        :interactive => false,
        :checkpoint_every => 0,
        :sleep_ms => 0,
        :warn_at => Int[],
        :fail_at => nothing,
        :debug => false,
        :help => false,
    )
    integer_options = Dict(
        "--gpus" => :gpus,
        "--lx" => :lx,
        "--ly" => :ly,
        "--dim-bond" => :dim_bond,
        "--samples" => :samples,
        "--seed" => :seed,
        "--ns-ahead" => :ns_ahead,
        "--ns-capacity" => :ns_capacity,
        "--iterations" => :iterations,
        "--steps" => :steps,
        "--chi" => :chi,
        "--checkpoint-every" => :checkpoint_every,
        "--sleep-ms" => :sleep_ms,
        "--fail-at" => :fail_at,
    )
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--dry-run"
            options[:dry_run] = true
            index += 1
        elseif argument == "--help"
            options[:help] = true
            index += 1
        elseif argument == "--figure-only"
            options[:figure_only] = true
            index += 1
        elseif argument == "--learning-rate"
            options[:learning_rate] = parse(Float64, _app_option_value(arguments, index, argument))
            index += 2
        elseif haskey(integer_options, argument)
            options[integer_options[argument]] =
                parse(Int, _app_option_value(arguments, index, argument))
            index += 2
        elseif argument == "--jsonl"
            options[:jsonl] = _app_option_value(arguments, index, argument)
            index += 2
        elseif argument == "--route"
            options[:route] = _app_option_value(arguments, index, argument)
            index += 2
        elseif argument == "--precision"
            options[:precision] = _app_option_value(arguments, index, argument)
            index += 2
        elseif argument == "--output"
            options[:output] = _app_option_value(arguments, index, argument)
            index += 2
        elseif argument == "--checkpoint"
            options[:checkpoint] = _app_option_value(arguments, index, argument)
            index += 2
        elseif argument == "--warn-at"
            push!(options[:warn_at], parse(Int, _app_option_value(arguments, index, argument)))
            index += 2
        elseif argument == "--resume"
            options[:resume] = true
            index += 1
        elseif argument == "--interactive"
            options[:interactive] = true
            index += 1
        elseif argument == "--debug"
            options[:debug] = true
            index += 1
        else
            throw(ArgumentError("unknown argument $(repr(argument)) for $name"))
        end
    end
    options[:gpus] >= 1 || throw(ArgumentError("gpus must be positive"))
    options[:lx] >= 2 || throw(ArgumentError("lx must be at least two"))
    options[:ly] >= 2 || throw(ArgumentError("ly must be at least two"))
    options[:dim_bond] >= 1 || throw(ArgumentError("dim-bond must be positive"))
    options[:samples] >= 2 || throw(ArgumentError("samples must be at least two"))
    if name == "e2e_one_step_node.jl"
        options[:ns_capacity] === nothing && (options[:ns_capacity] = options[:samples])
        options[:ns_ahead] >= 0 || throw(ArgumentError("ns-ahead must be nonnegative"))
        options[:ns_capacity] >= options[:samples] ||
            throw(ArgumentError("ns-capacity must be at least samples"))
        options[:ns_ahead] <= options[:ns_capacity] ||
            throw(ArgumentError("ns-ahead must not exceed ns-capacity"))
    end
    options[:steps] >= 1 || throw(ArgumentError("steps must be positive"))
    options[:chi] >= 1 || throw(ArgumentError("chi must be positive"))
    options[:precision] in ("fp32", "fp64") ||
        throw(ArgumentError("precision must be fp32 or fp64"))
    isfinite(options[:learning_rate]) || throw(ArgumentError("learning-rate must be finite"))
    return options
end

function app_dry_run(name::AbstractString, options)::Bool
    options[:dry_run] || return false
    println("[app] $name arguments resolved")
    return true
end

function pack_peps_arrays(tensors::AbstractMatrix)::Vector{ComplexF32}
    packed = ComplexF32[]
    for tensor in tensors
        append!(packed, vec(ComplexF32.(permutedims(tensor, (5, 4, 3, 2, 1)))))
    end
    return packed
end

function unpack_peps_arrays(
    packed::AbstractVector{ComplexF32},
    template::AbstractMatrix,
)::Matrix{Array{ComplexF32,5}}
    output = Matrix{Array{ComplexF32,5}}(undef, size(template))
    offset = 1
    for index in eachindex(template)
        dims = size(template[index])
        count = prod(dims)
        reversed = reshape(packed[offset:(offset+count-1)], reverse(dims))
        output[index] = permutedims(reversed, (5, 4, 3, 2, 1))
        offset += count
    end
    offset == length(packed) + 1 || throw(DimensionMismatch("packed PEPS length mismatch"))
    return output
end

function complex_digest(values)::String
    real_sum = sum(Float64(real(value)) for value in values)
    imag_sum = sum(Float64(imag(value)) for value in values)
    norm_sum = sum(Float64(abs2(value)) for value in values)
    return "count=$(length(values)) real_sum=$real_sum imag_sum=$imag_sum norm2=$norm_sum"
end

function pack_sample_configs(configs)::Vector{UInt8}
    isempty(configs) && return UInt8[]
    lx, ly = size(first(configs))
    packed = Vector{UInt8}(undef, length(configs) * lx * ly)
    offset = 1
    for config in configs, row in 1:lx, col in 1:ly
        packed[offset] = config[row, col]
        offset += 1
    end
    return packed
end

function value_preview(values, count::Integer=6)
    host = collect(Iterators.take(values, count))
    return host
end

function array_peps(lx, ly, dim_bond, dim_phys; seed=SEED)::Matrix{Array{ComplexF32,5}}
    Random.seed!(seed)
    arrays = Matrix{Array{ComplexF32,5}}(undef, lx, ly)
    for row in 1:lx, col in 1:ly
        u = row == 1 ? 1 : dim_bond
        r = col == ly ? 1 : dim_bond
        d = row == lx ? 1 : dim_bond
        l = col == 1 ? 1 : dim_bond
        arrays[row, col] = rand(ComplexF32, dim_phys, u, r, d, l)
    end
    return arrays
end

function random_unitary(::Type{S}, ingoing, outgoing)::ITensor where {S<:Number}
    t = ITensors.NDTensors.random_unitary(S, dim(ingoing), dim(outgoing))
    return ITensor(t, ingoing..., outgoing...)
end

function grid_peps(lx, ly, dim_bond; seed=SEED)::Matrix{ITensor}
    Random.seed!(seed)
    hilbert = [siteind("S=1/2"; addtags="nx=$row,ny=$col") for row in 1:lx, col in 1:ly]
    h_links = Matrix{Index{Int64}}(undef, lx, ly - 1)
    v_links = Matrix{Index{Int64}}(undef, lx - 1, ly)
    for row in 1:lx, col in 1:(ly-1)
        h_links[row, col] = Index(dim_bond, "h_link, $(row);$(col) -> $(row);$(col+1)")
    end
    for row in 1:(lx-1), col in 1:ly
        v_links[row, col] = Index(dim_bond, "v_link, $(row);$(col) -> $(row+1);$(col)")
    end
    tensors = Matrix{ITensor}(undef, lx, ly)
    for row in 1:lx, col in 1:ly
        ingoing = Index{Int64}[hilbert[row, col]]
        outgoing = Index{Int64}[]
        col != ly && push!(outgoing, h_links[row, col])
        row != lx && push!(outgoing, v_links[row, col])
        col != 1 && push!(ingoing, h_links[row, col-1])
        row != 1 && push!(ingoing, v_links[row-1, col])
        tensors[row, col] = random_unitary(ComplexF64, ingoing, outgoing)
    end
    return tensors
end
