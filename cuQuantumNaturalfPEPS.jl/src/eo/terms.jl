using Libdl

struct QnpepsElocConfig
    struct_size::UInt32
    lx::Int32
    ly::Int32
    dim_phys::Int32
    dim_bond::Int32
    chi_eo::Int32
    meo::Int32
    truncation_route::Int32
    density_cutoff::Float64
    svd_cutoff::Float64
end

function QnpepsElocConfig(;
    lx,
    ly,
    dim_bond,
    chi_eo,
    meo,
    dim_phys=2,
    route=:default,
    density_cutoff::Real=1.0e-13,
    svd_cutoff::Real=1.0e-13,
)
    isfinite(density_cutoff) || throw(ArgumentError("density cutoff must be finite"))
    density_cutoff >= 0 || throw(ArgumentError("density cutoff must be nonnegative"))
    0 <= svd_cutoff <= 1 || throw(ArgumentError("svd cutoff must lie in [0, 1]"))
    return QnpepsElocConfig(
        UInt32(sizeof(QnpepsElocConfig)),
        Int32(lx),
        Int32(ly),
        Int32(dim_phys),
        Int32(dim_bond),
        Int32(chi_eo),
        Int32(meo),
        _truncation_route_value(route),
        Float64(density_cutoff),
        Float64(svd_cutoff),
    )
end

struct QnpepsElocDiagBond
    site_a::Int32
    site_b::Int32
    coeff::Float64
end

const ELOC_DIAG_ONE_SITE_Z = Int32(-1)
const ELOC_CAP_ONE_SITE_Z = UInt64(1)

struct QnpepsElocFlipTerm
    n_flips::Int32
    flip_site::NTuple{4,Int32}
    flip_value::NTuple{4,Int32}
    mask_a::Int32
    mask_b::Int32
    coeff_re::Float64
    coeff_im::Float64
end

struct QnpepsElocTermTable
    n_diag::Int32
    diag::Ptr{QnpepsElocDiagBond}
    n_flip::Int32
    flip::Ptr{QnpepsElocFlipTerm}
end

struct HeisenbergTerms
    diag::Vector{QnpepsElocDiagBond}
    flip::Vector{QnpepsElocFlipTerm}
end

function _eloc_capabilities()::UInt64
    handle = Libdl.dlopen(_ffi_library())
    try
        symbol = Libdl.dlsym_e(handle, :qnpeps_eloc_capabilities)
        (isnothing(symbol) || symbol == C_NULL) && return UInt64(0)
        return ccall(symbol, UInt64, ())
    finally
        Libdl.dlclose(handle)
    end
end

function _require_eloc_term_capabilities(terms::HeisenbergTerms)::Nothing
    requires_one_site = any(term -> term.site_b == ELOC_DIAG_ONE_SITE_Z, terms.diag)
    requires_one_site || return nothing
    (_eloc_capabilities() & ELOC_CAP_ONE_SITE_Z) != 0 || throw(
        ArgumentError("the loaded CUDA library does not support one-site Pauli-Z diagonal terms"),
    )
    return nothing
end

function _eloc_term_table(terms::HeisenbergTerms)::QnpepsElocTermTable
    _require_eloc_term_capabilities(terms)
    diag = terms.diag
    flip = terms.flip
    length(diag) <= typemax(Int32) || throw(ArgumentError("too many diagonal terms"))
    length(flip) <= typemax(Int32) || throw(ArgumentError("too many flip terms"))
    return QnpepsElocTermTable(
        Int32(length(diag)),
        isempty(diag) ? Ptr{QnpepsElocDiagBond}(0) : pointer(diag),
        Int32(length(flip)),
        isempty(flip) ? Ptr{QnpepsElocFlipTerm}(0) : pointer(flip),
    )
end

_site0(i::Integer, j::Integer, ly::Integer) = Int32((i - 1) * ly + (j - 1))

function _push_bond!(diag, flip, a::Int32, b::Int32, J::Real)
    push!(diag, QnpepsElocDiagBond(a, b, Float64(J)))
    push!(
        flip,
        QnpepsElocFlipTerm(
            Int32(2),
            (a, b, Int32(0), Int32(0)),
            (Int32(-1), Int32(-1), Int32(0), Int32(0)),
            a,
            b,
            2.0 * Float64(J),
            0.0,
        ),
    )
    return nothing
end

function heisenberg_terms(lx::Integer, ly::Integer; J1::Real=1.0, J2::Real=0.0)
    diag = QnpepsElocDiagBond[]
    flip = QnpepsElocFlipTerm[]
    for i in 1:lx, j in 1:(ly-1)
        _push_bond!(diag, flip, _site0(i, j, ly), _site0(i, j + 1, ly), J1)
    end
    for i in 1:(lx-1), j in 1:ly
        _push_bond!(diag, flip, _site0(i, j, ly), _site0(i + 1, j, ly), J1)
    end
    if J2 != 0
        for i in 1:(lx-1), j in 1:(ly-1)
            _push_bond!(diag, flip, _site0(i, j, ly), _site0(i + 1, j + 1, ly), J2)
            _push_bond!(diag, flip, _site0(i + 1, j, ly), _site0(i, j + 1, ly), J2)
        end
    end
    return HeisenbergTerms(diag, flip)
end

function truncated_rydberg_terms(
    lx::Integer,
    ly::Integer;
    Omega::Real=1.0,
    delta::Real=0.0,
    V_nn::Real=2.0,
    V_diag::Real=V_nn / 8,
)
    lx >= 2 || throw(ArgumentError("lx must be at least two"))
    ly >= 2 || throw(ArgumentError("ly must be at least two"))
    all(isfinite, (Omega, delta, V_nn, V_diag)) ||
        throw(ArgumentError("Rydberg coefficients must be finite"))
    interactions = Tuple{Int32,Int32,Float64}[]
    flip = QnpepsElocFlipTerm[]
    for i in 1:lx, j in 1:ly
        site = _site0(i, j, ly)
        push!(
            flip,
            QnpepsElocFlipTerm(
                Int32(1),
                (site, Int32(0), Int32(0), Int32(0)),
                (Int32(-1), Int32(0), Int32(0), Int32(0)),
                Int32(-1),
                Int32(-1),
                0.5 * Float64(Omega),
                0.0,
            ),
        )
    end
    for i in 1:lx, j in 1:(ly-1)
        push!(interactions, (_site0(i, j, ly), _site0(i, j + 1, ly), Float64(V_nn)))
    end
    for i in 1:(lx-1), j in 1:ly
        push!(interactions, (_site0(i, j, ly), _site0(i + 1, j, ly), Float64(V_nn)))
    end
    for i in 1:(lx-1), j in 1:(ly-1)
        push!(interactions, (_site0(i, j, ly), _site0(i + 1, j + 1, ly), Float64(V_diag)))
        push!(interactions, (_site0(i + 1, j, ly), _site0(i, j + 1, ly), Float64(V_diag)))
    end
    fields = fill(0.5 * Float64(delta), lx * ly)
    constant = -0.5 * lx * ly * Float64(delta)
    diag = QnpepsElocDiagBond[]
    for (site_a, site_b, interaction) in interactions
        quarter = 0.25 * interaction
        constant += quarter
        fields[Int(site_a)+1] -= quarter
        fields[Int(site_b)+1] -= quarter
    end
    constant == 0.0 || push!(diag, QnpepsElocDiagBond(Int32(0), Int32(0), constant))
    for (index, field) in pairs(fields)
        field == 0.0 ||
            push!(diag, QnpepsElocDiagBond(Int32(index - 1), ELOC_DIAG_ONE_SITE_Z, field))
    end
    for (site_a, site_b, interaction) in interactions
        interaction == 0.0 || push!(diag, QnpepsElocDiagBond(site_a, site_b, 0.25 * interaction))
    end
    return HeisenbergTerms(diag, flip)
end
