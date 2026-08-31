using cuQuantumNaturalfPEPS
using ITensors
using CUDA
using Random
using Printf

function _fixture_peps()
    Random.seed!(1)
    lx, ly, dim_bond, dim_phys = 4, 4, 2, 2
    phys_indices = [Index(dim_phys, "p$i$j") for i in 1:lx, j in 1:ly]
    horizontal_bonds = [Index(dim_bond, "h$i$j") for i in 1:lx, j in 1:(ly-1)]
    vertical_bonds = [Index(dim_bond, "v$i$j") for i in 1:(lx-1), j in 1:ly]
    tensors = Matrix{ITensor}(undef, lx, ly)
    for i in 1:lx, j in 1:ly
        legs = Index[phys_indices[i, j]]
        i > 1 && push!(legs, vertical_bonds[i-1, j])
        j < ly && push!(legs, horizontal_bonds[i, j])
        i < lx && push!(legs, vertical_bonds[i, j])
        j > 1 && push!(legs, horizontal_bonds[i, j-1])
        tensors[i, j] = random_itensor(ComplexF64, legs...)
    end
    return tensors
end

_fmt(x) = @sprintf("%.17g", x)

function main()
    tensors = _fixture_peps()
    device_peps = upload_peps(load_peps(tensors))
    dlenv = double_layer(device_peps; chi_s=4, chi_dl=2)
    result = sample_peps(device_peps, dlenv, 256; gpus=1, seed=11)
    logs = dlenv.cumulative_row_logs
    head = reduce(vcat, vec.(result.configs))[1:32]
    logpc = result.log_prob_config[1:8]
    abi = cuQuantumNaturalfPEPS.capi_version()
    println("const FIXTURE_ROW_LOGS = Float64[", join(_fmt.(logs), ", "), "]")
    println("const FIXTURE_SAMPLE_HEAD = UInt8[", join(Int.(head), ", "), "]")
    println("const FIXTURE_LOGPC_HEAD = Float64[", join(_fmt.(logpc), ", "), "]")
    println("const FIXTURE_ABI = \"", abi, "\"")
end

main()
