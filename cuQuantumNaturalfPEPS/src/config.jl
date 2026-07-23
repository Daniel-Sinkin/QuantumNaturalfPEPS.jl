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
end

const SAMPLING_FAST = Int32(0)
const SAMPLING_FULL = Int32(1)
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
)::QnpepsConfig
    sampling_mode_value = _sampling_mode_value(sampling_mode)
    if sampling_mode_value == SAMPLING_FULL && chi_c < 1
        throw(ArgumentError("chi_c must be positive in full sampling mode"))
    end
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

struct QnpepsSampleHostArgs
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

struct QnpepsZipupPepsRowArgs
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
