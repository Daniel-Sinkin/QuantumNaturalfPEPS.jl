Base.@kwdef struct DensityMatrixWorkspaceGeometry
    num_sites::Int
    input_bond::Int
    operator_bond::Int
    output_dimension::Int
    upper_bond::Int
    lanes::Int = 1
end

function _densitymatrix_validate(config::ZipupConfig)::Nothing
    dims = config.dims
    settings = config.settings
    dims.num_sites >= 2 || throw(ArgumentError("density num_sites must be at least two"))
    min(dims.dim_phys, dims.dim_bond, dims.chi) >= 1 ||
        throw(ArgumentError("density dimensions must be positive"))
    settings.truncation_route == UInt32(TRUNCATION_DENSITY) ||
        throw(ArgumentError("density plan requires route six"))
    settings.rangefinder_oversampling == 0 ||
        throw(ArgumentError("density rangefinder oversampling must be zero"))
    settings.density_precision == density_precision_full ||
        throw(ArgumentError("density precision must be full"))
    isfinite(settings.density_cutoff) && settings.density_cutoff >= 0 ||
        throw(ArgumentError("density cutoff must be finite and nonnegative"))
    return nothing
end

function _densitymatrix_site_plans(
    state_dimensions::Vector{Int32},
    operator_dimensions::Vector{Int32},
    num_sites::Int,
    upper_bond::Int,
)::NamedTuple
    length(state_dimensions) == 3 * num_sites ||
        throw(DimensionMismatch("density state dimensions differ"))
    length(operator_dimensions) == 4 * num_sites ||
        throw(DimensionMismatch("density operator dimensions differ"))
    sites = Vector{QnpepsDensitySitePlan}(undef, num_sites)
    state_offset = UInt64(0)
    operator_offset = UInt64(0)
    output_offset = UInt64(0)
    state_bond = 1
    operator_bond = 1
    maximum_state_bond = 1
    maximum_operator_bond = 1
    maximum_output = 1
    for site in 0:(num_sites-1)
        state_base = 3 * site
        operator_base = 4 * site
        state_left = Int(state_dimensions[state_base+1])
        physical_input = Int(state_dimensions[state_base+2])
        state_right = Int(state_dimensions[state_base+3])
        operator_left = Int(operator_dimensions[operator_base+1])
        operator_input = Int(operator_dimensions[operator_base+2])
        physical_output = Int(operator_dimensions[operator_base+3])
        operator_right = Int(operator_dimensions[operator_base+4])
        minimum((
            state_left,
            physical_input,
            state_right,
            operator_left,
            operator_input,
            physical_output,
            operator_right,
        )) >= 1 || throw(DimensionMismatch("density site dimensions must be positive"))
        state_left == state_bond ||
            throw(DimensionMismatch("density state bond differs at site $(site + 1)"))
        operator_left == operator_bond ||
            throw(DimensionMismatch("density operator bond differs at site $(site + 1)"))
        physical_input == operator_input ||
            throw(DimensionMismatch("density physical input differs at site $(site + 1)"))
        sites[site+1] = QnpepsDensitySitePlan(
            site=UInt32(site),
            num_sites=UInt32(num_sites),
            state_left=Int64(state_left),
            physical_input=Int64(physical_input),
            state_right=Int64(state_right),
            operator_left=Int64(operator_left),
            operator_input=Int64(operator_input),
            physical_output=Int64(physical_output),
            operator_right=Int64(operator_right),
            state_offset=state_offset,
            operator_offset=operator_offset,
            output_offset=output_offset,
        )
        state_offset += _density_product(state_left, physical_input, state_right)
        operator_offset +=
            _density_product(operator_left, operator_input, physical_output, operator_right)
        output_offset += _density_product(upper_bond, physical_output, upper_bond)
        state_bond = state_right
        operator_bond = operator_right
        maximum_state_bond = max(maximum_state_bond, state_left, state_right)
        maximum_operator_bond = max(maximum_operator_bond, operator_left, operator_right)
        maximum_output = max(maximum_output, physical_output)
    end
    state_bond == 1 || throw(DimensionMismatch("density state right boundary differs"))
    operator_bond == 1 || throw(DimensionMismatch("density operator right boundary differs"))
    return (;
        sites,
        state_count=state_offset,
        operator_count=operator_offset,
        output_count=output_offset,
        maximum_state_bond,
        maximum_operator_bond,
        maximum_output,
    )
end

function _density_context_config(geometry::DensityMatrixWorkspaceGeometry)::QnpepsConfig
    return QnpepsConfig(
        lx=2,
        ly=2,
        dim_phys=geometry.output_dimension,
        dim_bond=max(geometry.input_bond, geometry.operator_bond),
        chi_s=geometry.upper_bond,
        chi_dl=geometry.upper_bond,
    )
end
