@enum DensityPrecision::UInt32 begin
    density_precision_full = 1
end

Base.@kwdef struct ZipupDims
    num_sites::Int
    dim_phys::Int
    dim_bond::Int
    chi::Int
end

Base.@kwdef struct ZipupSettings
    graph_capture_enabled::Bool = false
    truncation_route::UInt32 = UInt32(2)
    rangefinder_oversampling::Int = 8
    rangefinder_seed::UInt64 = UInt64(777)
    density_precision::DensityPrecision = density_precision_full
    density_cutoff::Float64 = 1.0e-13
end

Base.@kwdef struct ZipupConfig
    dims::ZipupDims
    settings::ZipupSettings = ZipupSettings()
end

struct DensityMatrixProductState
    dims::CUDA.CuVector{Int32}
    values::CUDA.CuVector{ComplexF64}
end

struct DensityMatrixProductOperator
    dims::CUDA.CuVector{Int32}
    values::CUDA.CuVector{ComplexF64}
end

Base.@kwdef struct QnpepsDensitySettings
    struct_size::UInt32 = UInt32(sizeof(QnpepsDensitySettings))
    precision::UInt32
    mindim::UInt32 = UInt32(1)
    reserved::UInt32 = UInt32(0)
    relative_cutoff::Float64
end

Base.@kwdef struct QnpepsDensityRankRecord
    struct_size::UInt32 = UInt32(sizeof(QnpepsDensityRankRecord))
    flags::UInt32 = UInt32(0)
    matrix_order::Int64
    natural_cap::Int64
    applied_cap::Int64
    rank_after_cap::Int64
    active_rank::Int64
    hard_discarded::Int64
    cutoff_discarded::Int64
    truncation_error::Float64
    docut::Float64
    retained_sum::Float64
    discarded_weight::Float64
end

Base.@kwdef struct QnpepsDeviceBuffer
    struct_size::UInt32 = UInt32(sizeof(QnpepsDeviceBuffer))
    reserved::UInt32 = UInt32(0)
    values::UInt64
    bytes::UInt64
end

Base.@kwdef struct QnpepsDensitySitePlan
    struct_size::UInt32 = UInt32(sizeof(QnpepsDensitySitePlan))
    site::UInt32
    num_sites::UInt32
    reserved::UInt32 = UInt32(0)
    state_left::Int64
    physical_input::Int64
    state_right::Int64
    operator_left::Int64
    operator_input::Int64
    physical_output::Int64
    operator_right::Int64
    state_offset::UInt64
    operator_offset::UInt64
    output_offset::UInt64
end

Base.@kwdef struct QnpepsDensityWorkspace
    struct_size::UInt32 = UInt32(sizeof(QnpepsDensityWorkspace))
    reserved::UInt32 = UInt32(0)
    left_environments::QnpepsDeviceBuffer
    right_blocks::QnpepsDeviceBuffer
    temporaries::QnpepsDeviceBuffer
    density::QnpepsDeviceBuffer
    eigenvalues::QnpepsDeviceBuffer
    sorted_eigenvalues::QnpepsDeviceBuffer
    sort_indices::QnpepsDeviceBuffer
    active_ranks::QnpepsDeviceBuffer
    truncation_records::QnpepsDeviceBuffer
    solver_workspace::QnpepsDeviceBuffer
    solver_information::QnpepsDeviceBuffer
end

Base.@kwdef struct QnpepsDensityTrace
    struct_size::UInt32 = UInt32(sizeof(QnpepsDensityTrace))
    enabled::UInt32
    left_environments::QnpepsDeviceBuffer
    right_blocks::QnpepsDeviceBuffer
    conjugate_right_blocks::QnpepsDeviceBuffer
    density_matrices::QnpepsDeviceBuffer
    solver_spectra::QnpepsDeviceBuffer
    sorted_spectra::QnpepsDeviceBuffer
    retained_spectra::QnpepsDeviceBuffer
    right_bases::QnpepsDeviceBuffer
    left_bases::QnpepsDeviceBuffer
    projectors::QnpepsDeviceBuffer
    accumulated_gauges::QnpepsDeviceBuffer
    matrix_orders::QnpepsDeviceBuffer
    combined_inputs::QnpepsDeviceBuffer
end

Base.@kwdef struct QnpepsDensityApplyArgs
    struct_size::UInt32 = UInt32(sizeof(QnpepsDensityApplyArgs))
    normalize::UInt32
    settings::QnpepsDensitySettings
    upper_bond::Int64
    num_sites::UInt64
    sites::UInt64
    sites_bytes::UInt64
    input_gauge::Float64
    workspace::QnpepsDensityWorkspace
    trace::QnpepsDensityTrace
    state_values::QnpepsDeviceBuffer
    operator_values::QnpepsDeviceBuffer
    result_dimensions::QnpepsDeviceBuffer
    result_values::QnpepsDeviceBuffer
    normalization_log::QnpepsDeviceBuffer
    output_gauge::QnpepsDeviceBuffer
end

Base.@kwdef struct QnpepsDensityFilterArgs
    struct_size::UInt32 = UInt32(sizeof(QnpepsDensityFilterArgs))
    reserved::UInt32 = UInt32(0)
    settings::QnpepsDensitySettings
    order::Int64
    natural_cap::Int64
    applied_cap::Int64
    solver_values::QnpepsDeviceBuffer
    sorted_values::QnpepsDeviceBuffer
    retained_values::QnpepsDeviceBuffer
    sorted_indices::QnpepsDeviceBuffer
    active_rank::QnpepsDeviceBuffer
    record::QnpepsDeviceBuffer
end

Base.@kwdef struct QnpepsDensityWorkspaceSizes
    struct_size::UInt32 = UInt32(sizeof(QnpepsDensityWorkspaceSizes))
    reserved::UInt32 = UInt32(0)
    combined_input::UInt64 = UInt64(0)
    capped_output::UInt64 = UInt64(0)
    left_environment_bytes::UInt64 = UInt64(0)
    right_block_bytes::UInt64 = UInt64(0)
    temporary_bytes::UInt64 = UInt64(0)
    density_bytes::UInt64 = UInt64(0)
    eigenvalues_bytes::UInt64 = UInt64(0)
    sort_keys_bytes::UInt64 = UInt64(0)
    sort_indices_bytes::UInt64 = UInt64(0)
    active_ranks_bytes::UInt64 = UInt64(0)
    truncation_records_bytes::UInt64 = UInt64(0)
    solver_workspace_bytes::UInt64 = UInt64(0)
    solver_information_bytes::UInt64 = UInt64(0)
    arena_bytes::UInt64 = UInt64(0)
end

const DensityMatrixWorkspaceSizes = QnpepsDensityWorkspaceSizes

Base.@kwdef struct QnpepsDensityWorkspaceQuery
    struct_size::UInt32 = UInt32(sizeof(QnpepsDensityWorkspaceQuery))
    reserved::UInt32 = UInt32(0)
    num_sites::Int64
    input_bond::Int64
    operator_bond::Int64
    output_dimension::Int64
    upper_bond::Int64
    lanes::Int64
    sizes_out::UInt64
end

function protocol_buffer(array::CUDA.CuArray)::QnpepsDeviceBuffer
    return QnpepsDeviceBuffer(
        values=UInt64(UInt(pointer(array))),
        bytes=UInt64(sizeof(eltype(array)) * length(array)),
    )
end

function protocol_buffer(::Nothing)::QnpepsDeviceBuffer
    return QnpepsDeviceBuffer(values=UInt64(0), bytes=UInt64(0))
end

function _density_protocol_settings(settings::ZipupSettings)::QnpepsDensitySettings
    return QnpepsDensitySettings(
        precision=UInt32(settings.density_precision),
        mindim=UInt32(1),
        relative_cutoff=settings.density_cutoff,
    )
end

function _density_product(values::Integer...)::UInt64
    result = UInt64(1)
    for value in values
        result *= UInt64(value)
    end
    return result
end
