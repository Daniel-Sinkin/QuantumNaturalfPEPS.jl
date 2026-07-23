module JuliaReference

using LinearAlgebra
using Random

include("tensor_core.jl")
include("double_layer.jl")
include("sampler.jl")

export RangefinderConfig, RangefinderDiagnostic, RangefinderFactorizer
export RangefinderResult, FixedRankRangefinderError
export contract_plan, contract_arrays, normalize_log!, batched_rangefinder, factorize_matrix
export qr_factorize, zipup_mpo_mps

export DoubleLayerStack, apply_fused_double_layer_row, build_double_layer_stack, double_layer

export SamplerConfig, SamplerContext, ctx_sample_refresh!, ctx_sample_run
export proposal_log_probability, sample_peps

end
