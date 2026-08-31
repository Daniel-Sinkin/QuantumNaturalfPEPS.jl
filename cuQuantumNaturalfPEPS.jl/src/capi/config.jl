struct QnpepsConfig
    struct_size::UInt32
    lx::Int32
    ly::Int32
    dim_phys::Int32
    dim_bond::Int32
    chi_s::Int32
    chi_dl::Int32
    seed::UInt64
    sampling_mode::Int32
    chi_c::Int32
    dlenv_truncation_route::Int32
    sampler_truncation_route::Int32
    dlenv_density_cutoff::Float64
    sampler_density_cutoff::Float64
    projected_density_cutoff::Float64
end

const SAMPLING_FAST = Int32(0)
const SAMPLING_FULL = Int32(1)
const TRUNCATION_DEFAULT = Int32(0)
const TRUNCATION_DENSITY = Int32(6)
const MAX_BATCH_SIZE = 2048

function _sampling_mode_value(mode)::Int32
    if mode === :fast || mode == SAMPLING_FAST
        return SAMPLING_FAST
    end
    if mode === :full || mode == SAMPLING_FULL
        return SAMPLING_FULL
    end
    throw(ArgumentError("sampling_mode must be :fast or :full"))
end

function _truncation_route_value(route)::Int32
    if route === :default || route == TRUNCATION_DEFAULT
        return TRUNCATION_DEFAULT
    end
    if route === :density || route == TRUNCATION_DENSITY
        return TRUNCATION_DENSITY
    end
    throw(ArgumentError("truncation route must be :default or :density"))
end

function QnpepsConfig(;
    lx,
    ly,
    dim_bond,
    chi_s,
    chi_dl=dim_bond,
    dim_phys=2,
    seed::Integer=0,
    sampling_mode=:fast,
    chi_c::Integer=3 * dim_bond,
    dlenv_truncation_route=:default,
    sampler_truncation_route=:default,
    dlenv_density_cutoff::Real=1.0e-13,
    sampler_density_cutoff::Real=1.0e-3,
    projected_density_cutoff::Real=1.0e-4,
)::QnpepsConfig
    sampling_mode_value = _sampling_mode_value(sampling_mode)
    dlenv_route_value = _truncation_route_value(dlenv_truncation_route)
    sampler_route_value = _truncation_route_value(sampler_truncation_route)
    if sampling_mode_value == SAMPLING_FULL && chi_c < 1
        throw(ArgumentError("chi_c must be positive in full sampling mode"))
    end
    all(isfinite, (dlenv_density_cutoff, sampler_density_cutoff, projected_density_cutoff)) ||
        throw(ArgumentError("density cutoffs must be finite"))
    min(dlenv_density_cutoff, sampler_density_cutoff, projected_density_cutoff) >= 0 ||
        throw(ArgumentError("density cutoffs must be nonnegative"))
    return QnpepsConfig(
        UInt32(sizeof(QnpepsConfig)),
        Int32(lx),
        Int32(ly),
        Int32(dim_phys),
        Int32(dim_bond),
        Int32(chi_s),
        Int32(chi_dl),
        UInt64(seed),
        sampling_mode_value,
        Int32(chi_c),
        dlenv_route_value,
        sampler_route_value,
        Float64(dlenv_density_cutoff),
        Float64(sampler_density_cutoff),
        Float64(projected_density_cutoff),
    )
end

function _reseed(config::QnpepsConfig, seed::Integer)::QnpepsConfig
    return QnpepsConfig(
        config.struct_size,
        config.lx,
        config.ly,
        config.dim_phys,
        config.dim_bond,
        config.chi_s,
        config.chi_dl,
        UInt64(seed),
        config.sampling_mode,
        config.chi_c,
        config.dlenv_truncation_route,
        config.sampler_truncation_route,
        config.dlenv_density_cutoff,
        config.sampler_density_cutoff,
        config.projected_density_cutoff,
    )
end

struct QnpepsSampleArgs
    struct_size::UInt32
    peps::UInt
    dlenv::UInt
    gpus::Int32
    scratch::UInt
    scratch_bytes::UInt64
    samples_out::UInt
    log_prob_config::UInt
    log_gauge::UInt
    n_samples::UInt64
    batch_base::UInt64
    dim_batch::UInt64
    stream::UInt
end

struct QnpepsCtxSampleArgs
    struct_size::UInt32
    samples_out::UInt
    log_prob_config::UInt
    log_gauge::UInt
    n_samples::UInt64
    batch_base::UInt64
    dim_batch::UInt64
end

Base.@kwdef struct QnpepsSamplerHostBatchArgs
    struct_size::UInt32
    reserved::UInt32
    peps::UInt
    dlenv_dims::UInt
    dlenv_dims_count::UInt64
    dlenv_values::UInt
    scratch::UInt
    scratch_bytes::UInt64
    sampling::UInt
    sampling_bytes::UInt64
    dlenv_pointers::UInt
    dlenv_pointer_count::UInt64
    samples_out::UInt
    log_prob_config::UInt
    log_gauge::UInt
    batch_seed::UInt64
    batch_id::UInt64
    dim_batch::UInt64
    peps_layout::Int32
    reserved2::Int32
end

Base.@kwdef struct QnpepsSamplerHostRefreshArgs
    struct_size::UInt32
    peps_layout::Int32
    peps::UInt
    dlenv_values::UInt
    sampling::UInt
    sampling_bytes::UInt64
end

Base.@kwdef struct QnpepsSampleHostArgs
    struct_size::UInt32
    gpus::Int32
    peps::UInt
    dlenv::UInt
    samples_out::UInt
    log_prob_config::UInt
    log_gauge::UInt
    n_samples::UInt64
    batch_base::UInt64
    dim_batch::UInt64
    stream::UInt
end

Base.@kwdef struct QnpepsZipupPepsRowArgs
    struct_size::UInt32
    row::Int32
    peps_row::UInt
    peps_row_bytes::UInt64
    mps_dims::UInt
    mps_values::UInt
    mps_bytes::UInt64
    output_dims::UInt
    output_values::UInt
    output_bytes::UInt64
end

struct QnpepsZipupMpoMpsDesc
    struct_size::UInt32
    num_sites::Int32
    maxdim::Int32
    reserved::Int32
    mpo_dims::UInt
    mps_dims::UInt
end

struct QnpepsZipupMpoMpsArgs
    struct_size::UInt32
    reserved::UInt32
    mpo::UInt
    mpo_bytes::UInt64
    mps::UInt
    mps_bytes::UInt64
    output::UInt
    output_bytes::UInt64
    log_gauge::UInt
    stream::UInt
end

const GRAM_CONSUMER_SLAB = Int32(0)
const GRAM_CONSUMER_CUSTOM = Int32(1)
const GRAM_CONSUMER_DENSE = Int32(2)

struct QnpepsGramDesc
    struct_size::UInt32
    lx::Int32
    ly::Int32
    dim_phys::Int32
    dim_bond::Int32
    consumer::Int32
    reserved::Int32
    n_samples::Int64
end

struct QnpepsGramArgs
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

struct QnpepsGramFootprint
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

struct QnpepsMinsrDesc
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

struct QnpepsMinsrArgs
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

function _gram_consumer_value(consumer)::Int32
    if consumer === :slab || consumer == GRAM_CONSUMER_SLAB
        return GRAM_CONSUMER_SLAB
    end
    if consumer === :custom || consumer == GRAM_CONSUMER_CUSTOM
        return GRAM_CONSUMER_CUSTOM
    end
    if consumer === :dense || consumer == GRAM_CONSUMER_DENSE
        return GRAM_CONSUMER_DENSE
    end
    throw(ArgumentError("consumer must be :slab, :custom, or :dense"))
end

function QnpepsGramDesc(;
    lx,
    ly,
    dim_bond,
    n_samples,
    dim_phys=2,
    consumer=:slab,
)::QnpepsGramDesc
    return QnpepsGramDesc(
        UInt32(sizeof(QnpepsGramDesc)),
        Int32(lx),
        Int32(ly),
        Int32(dim_phys),
        Int32(dim_bond),
        _gram_consumer_value(consumer),
        Int32(0),
        Int64(n_samples),
    )
end

function QnpepsMinsrDesc(;
    lx,
    ly,
    dim_bond,
    n_samples,
    dim_phys=2,
    host_tile_bytes::Integer=0,
    diagnostics=false,
)::QnpepsMinsrDesc
    return QnpepsMinsrDesc(
        UInt32(sizeof(QnpepsMinsrDesc)),
        Int32(lx),
        Int32(ly),
        Int32(dim_phys),
        Int32(dim_bond),
        Int32(diagnostics),
        Int32(0),
        UInt32(0),
        Int64(n_samples),
        Int64(host_tile_bytes),
    )
end

function _empty_gram_footprint()::QnpepsGramFootprint
    return QnpepsGramFootprint(
        UInt32(sizeof(QnpepsGramFootprint)),
        UInt32(0),
        UInt64(0),
        UInt64(0),
        UInt64(0),
        UInt64(0),
        UInt64(0),
        UInt64(0),
        UInt64(0),
    )
end
