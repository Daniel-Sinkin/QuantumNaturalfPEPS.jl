
Base.@kwdef struct _MinsrHostGramDesc
    struct_size::UInt32
    lx::Int32
    ly::Int32
    dim_phys::Int32
    dim_bond::Int32
    consumer::Int32
    reserved::Int32
    n_samples::Int64
end

Base.@kwdef struct _MinsrHostGramArgs
    struct_size::UInt32
    reserved::UInt32
    samples::UInt
    samples_bytes::UInt64
    o_rows::UInt
    o_rows_bytes::UInt64
    gram_out::UInt
    gram_out_bytes::UInt64
    stream::UInt
end

Base.@kwdef struct _MinsrHostGramFootprint
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

Base.@kwdef struct _MinsrHostMinsrDesc
    struct_size::UInt32
    lx::Int32
    ly::Int32
    dim_phys::Int32
    dim_bond::Int32
    diagnostics::Int32
    reserved::Int32
    tail_padding::UInt32
    n_samples::Int64
    host_tile_bytes::Int64
end

Base.@kwdef struct _MinsrHostMinsrArgs
    struct_size::UInt32
    reserved::UInt32
    samples::UInt
    samples_bytes::UInt64
    logpsi::UInt
    logpsi_bytes::UInt64
    e_loc::UInt
    e_loc_bytes::UInt64
    logq::UInt
    logq_bytes::UInt64
    gram::UInt
    gram_bytes::UInt64
    o_rows_device::UInt
    o_rows_host::UInt
    o_rows_bytes::UInt64
    theta_dot_out::UInt
    theta_dot_out_bytes::UInt64
    relative_cut::Float64
    absolute_cut::Float64
    e_mean_out::UInt
    e_var_out::UInt
    ess_out::UInt
    stream::UInt
end
