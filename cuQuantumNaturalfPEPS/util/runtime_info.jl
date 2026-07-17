using CUDA
using Downloads
using Libdl
using cuQuantumNaturalfPEPS

println("cuQuantumNaturalfPEPS runtime")
println("  native library: ", realpath(cuQuantumNaturalfPEPS._lib_path()))
println("  loaded C API: ", cuQuantumNaturalfPEPS.capi_version())

CUDA.versioninfo()

library_pattern = r"^(libcurl|libcuda|libcudart|libcublas|libcublasLt|libcusolver|libcusolverMg|libcurand|libcusparse|libnvJitLink|libcutensor)\.so"
libraries = filter(Libdl.dllist()) do library
    occursin(library_pattern, basename(library))
end

println("Loaded native library paths:")
for library in sort!(unique!(libraries))
    resolved = ispath(library) ? realpath(library) : library
    if resolved == library
        println("- $library")
    else
        println("- $library => $resolved")
    end
end
