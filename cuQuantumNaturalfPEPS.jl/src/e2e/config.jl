struct QnpepsE2eConfig
    struct_size::UInt32
    lx::Int32
    ly::Int32
    dim_phys::Int32
    dim_bond::Int32
    chi_s::Int32
    chi_dl::Int32
    chi_eo::Int32
    meo::Int32
    seed::UInt64
    sampling_mode::Int32
    contract_dim::Int32
    sample_batch::Int32
end

Base.@kwdef struct QnpepsE2eEulerStepArgs
    struct_size::UInt32
    precision::Int32
    n_samples::Int64
    relative_cut::Float64
    absolute_cut::Float64
    learning_rate::Float64
    state_f64_io::CuPtr{Cvoid}
    state_f64_bytes::UInt64
    peps_f32_io::CuPtr{Cvoid}
    peps_f32_bytes::UInt64
    theta_dot_out::CuPtr{Cvoid}
    theta_dot_bytes::UInt64
    e_mean_out::Ptr{Float64}
    e_var_out::Ptr{Float64}
    ess_out::Ptr{Float64}
    samples_out::CuPtr{UInt8}
    logq_out::CuPtr{Float64}
    log_gauge_out::CuPtr{Float64}
    logpsi_out::CuPtr{Float64}
    e_loc_out::CuPtr{Float64}
    o_rows_host::Ptr{Cvoid}
    epoch_out::Ptr{Int64}
end

struct QnpepsE2eGramTimings
    struct_size::UInt32
    slabs::Int32
    virtual_shards::Int32
    block_calls::Int32
    slab_width::UInt64
    compute_s::Float64
    complete_s::Float64
end

Base.@kwdef struct QnpepsE2eGramFootprint
    struct_size::UInt32
    reserved::UInt32
    context_device_bytes::UInt64
    geometry_device_bytes::UInt64
    dense_a_device_bytes::UInt64
    dense_b_device_bytes::UInt64
    caller_samples_bytes::UInt64
    caller_rows_bytes::UInt64
    caller_gram_bytes::UInt64
end

function QnpepsE2eConfig(;
    lx,
    ly,
    dim_bond,
    chi_s,
    chi_dl=dim_bond,
    chi_eo=dim_bond,
    meo,
    dim_phys=2,
    seed::Integer=0,
    sampling_mode=:fast,
    contract_dim::Integer=3 * dim_bond,
    sample_batch::Integer=0,
)
    mode =
        sampling_mode === :fast ? Int32(0) :
        sampling_mode === :full ? Int32(1) : Int32(sampling_mode)
    mode in (Int32(0), Int32(1)) || throw(ArgumentError("sampling_mode must be :fast or :full"))
    mode == Int32(1) &&
        contract_dim < 1 &&
        throw(ArgumentError("contract_dim must be positive in full sampling mode"))
    0 <= sample_batch <= 2048 || throw(ArgumentError("sample_batch must be between 0 and 2048"))
    return QnpepsE2eConfig(
        UInt32(sizeof(QnpepsE2eConfig)),
        Int32(lx),
        Int32(ly),
        Int32(dim_phys),
        Int32(dim_bond),
        Int32(chi_s),
        Int32(chi_dl),
        Int32(chi_eo),
        Int32(meo),
        UInt64(seed),
        mode,
        Int32(contract_dim),
        Int32(sample_batch),
    )
end
