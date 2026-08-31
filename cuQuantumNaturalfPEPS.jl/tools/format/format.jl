using JuliaFormatter

const OPTIONS = (margin=100, indent=4, always_for_in=true, whitespace_in_kwargs=false)

function main()
    isempty(ARGS) &&
        error("[format.jl] usage julia --project=tools/format tools/format/format.jl <path>...")
    all_clean = true
    for path in ARGS
        clean = format(path; OPTIONS...)
        clean || (all_clean = false)
        println(clean ? "clean      " : "reformatted", " ", path)
    end
    return all_clean
end

main()
