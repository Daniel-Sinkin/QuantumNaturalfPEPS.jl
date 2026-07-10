struct QnpepsConfig
    struct_size::UInt32
    lx::Int32
    ly::Int32
    dim_phys::Int32
    dim_bond::Int32
    chi_s::Int32
    chi_dl::Int32
    seed::UInt64
end

function QnpepsConfig(;
    lx,
    ly,
    dim_bond,
    chi_s,
    chi_dl=dim_bond,
    dim_phys=2,
    seed::Integer=0,
)::QnpepsConfig
    return QnpepsConfig(
        UInt32(sizeof(QnpepsConfig)),
        Int32(lx),
        Int32(ly),
        Int32(dim_phys),
        Int32(dim_bond),
        Int32(chi_s),
        Int32(chi_dl),
        UInt64(seed),
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
end
