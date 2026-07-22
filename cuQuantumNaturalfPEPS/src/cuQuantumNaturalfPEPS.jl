module cuQuantumNaturalfPEPS

include("config.jl")
include("c_ffi.jl")
include("ffi.jl")
include("peps.jl")
include("double_layer.jl")
include("rangefinder.jl")
include("sampling.jl")
include("context.jl")

export QnpepsConfig, Peps, CuPeps, CuDlenv, CuSamplingContext
export MAX_BATCH_SIZE
export load_peps, upload_peps, upload_peps!, double_layer, double_layer_step, build_dlenv!
export sample_peps, sample_peps!, batched_rangefinder, sampler_pool_release
export update_peps!, close!
export FFI

end
