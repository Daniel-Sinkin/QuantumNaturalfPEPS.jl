using CUDA
using Downloads
using Libdl
using cuQuantumNaturalfPEPS

const show_paths = "--show-path" in ARGS
all(argument == "--show-path" for argument in ARGS) || error("unknown argument")

println("cuQuantumNaturalfPEPS runtime")
native_library = realpath(cuQuantumNaturalfPEPS._lib_path())
println("  native library: ", show_paths ? native_library : "build/cuda/qnpeps.so")
println("  loaded C API: ", cuQuantumNaturalfPEPS.capi_version())

CUDA.versioninfo()

library_pattern = r"^(libcurl|libcuda|libcudart|libcublas|libcublasLt|libcusolver|libcusolverMg|libcurand|libcusparse|libnvJitLink|libcutensor)\.so"
libraries = filter(Libdl.dllist()) do library
    occursin(library_pattern, basename(library))
end

println("Loaded native library paths:")
for library in sort!(unique!(libraries))
    resolved = ispath(library) ? realpath(library) : library
    if show_paths
        println(resolved == library ? "- $library" : "- $library => $resolved")
    else
        println("- ", basename(resolved))
    end
end
