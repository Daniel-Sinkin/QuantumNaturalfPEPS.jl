using Pkg
using TOML

const package_dir = abspath(@__DIR__)
const cuda_version = get(ENV, "QNPEPS_CUDA_VERSION", "13.0")
const cuda_runtime_name = "CUDA_Runtime_jll"
const cuda_runtime_uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"

occursin(r"^[0-9]+\.[0-9]+$", cuda_version) ||
    error("QNPEPS_CUDA_VERSION must have major.minor form, got: $cuda_version")

function project_dirs()
    projects = String[]
    parent = dirname(package_dir)
    isfile(joinpath(parent, "Project.toml")) && push!(projects, parent)
    push!(projects, package_dir)
    return unique(projects)
end

function register_cuda_runtime(project::String)
    path = joinpath(project, "Project.toml")
    project_toml = TOML.parsefile(path)
    for section in ("deps", "extras")
        packages = get(project_toml, section, Dict{String,Any}())
        if haskey(packages, cuda_runtime_name)
            packages[cuda_runtime_name] == cuda_runtime_uuid ||
                error("$path registers $cuda_runtime_name with the wrong UUID")
            return
        end
    end

    lines = readlines(path; keep = true)
    extras = findfirst(line -> strip(line) == "[extras]", lines)
    entry = "$cuda_runtime_name = \"$cuda_runtime_uuid\"\n"
    if extras === nothing
        !isempty(lines) && !endswith(lines[end], "\n") && (lines[end] *= "\n")
        !isempty(lines) && !isempty(strip(lines[end])) && push!(lines, "\n")
        append!(lines, ["[extras]\n", entry])
    else
        insert!(lines, extras + 1, entry)
    end
    open(path, "w") do io
        write(io, join(lines))
    end
    TOML.parsefile(path)
    @info "Registered CUDA runtime preferences" project
end

function configure_cuda_runtime(project::String)
    path = joinpath(project, "LocalPreferences.toml")
    preferences = isfile(path) ? TOML.parsefile(path) : Dict{String,Any}()
    runtime = get!(preferences, "CUDA_Runtime_jll", Dict{String,Any}())
    runtime isa AbstractDict || error("$path has a non-table CUDA_Runtime_jll entry")

    wanted = Dict("local" => "true", "version" => cuda_version)
    if all(get(runtime, key, nothing) == value for (key, value) in wanted)
        @info "CUDA.jl runtime already configured" project version = cuda_version local_toolkit = true
        return
    end

    merge!(runtime, wanted)
    temporary, io = mktemp(dirname(path))
    try
        TOML.print(io, preferences; sorted = true)
        close(io)
        mv(temporary, path; force = true)
    catch
        isopen(io) && close(io)
        rm(temporary; force = true)
        rethrow()
    end
    @info "Configured CUDA.jl to use the module-provided toolkit" project version = cuda_version local_toolkit = true
end

projects = project_dirs()
foreach(register_cuda_runtime, projects)
foreach(configure_cuda_runtime, projects)

for project in projects
    Pkg.activate(project)
    Pkg.instantiate(; allow_autoprecomp = false)
end

for project in projects
    Pkg.activate(project)
    Pkg.precompile()
end

Pkg.activate(package_dir)
@info "cuQuantumNaturalfPEPS Julia setup complete" projects cuda_version depot = DEPOT_PATH
