
Base.@kwdef struct _EoHostRunArgs
    struct_size::UInt32
    padding::NTuple{4,UInt8}
    device_peps::CUDA.CuPtr{Cvoid}
    device_samples::CUDA.CuPtr{UInt8}
    logpsi_out::CUDA.CuPtr{Float64}
    e_loc_out::CUDA.CuPtr{Float64}
    o_rows_dev::CUDA.CuPtr{Cvoid}
    o_rows_host::Ptr{Cvoid}
    gram::CUDA.CuPtr{Cvoid}
    lambda::Float64
    j2_mode::UInt32
    j2_draw::UInt32
    j2_seed::UInt64
    j2_epoch::UInt64
end

Base.@kwdef struct EoHostStats
    struct_size::UInt32
    graph_enabled::UInt32
    runs::UInt64
    bindings::UInt64
    graph_captures::UInt64
    graph_replays::UInt64
    graph_capture_failures::UInt64
    graph_nodes::UInt64
    graph_edges::UInt64
    graph_introspection_failures::UInt64
    j2_mode_last::UInt32
    j2_draw_last::UInt32
    j2_seed_last::UInt64
    j2_epoch_last::UInt64
    j2_waves::UInt64
    j2_group0_waves::UInt64
    j2_group1_waves::UInt64
    j2_row_groups_total::UInt64
    j2_row_groups_retained::UInt64
    j2_column_groups_total::UInt64
    j2_column_groups_retained::UInt64
    j2_diag_terms_total::UInt64
    j2_diag_terms_retained::UInt64
    j2_flip_terms_total::UInt64
    j2_flip_terms_retained::UInt64
end
