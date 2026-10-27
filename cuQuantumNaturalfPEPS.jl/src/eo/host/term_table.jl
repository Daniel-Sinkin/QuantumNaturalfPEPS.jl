const EO_HOST_LANES = 4
const _EO_HOST_ALIGNMENT = UInt64(256)
const _EO_HOST_ROWS = UInt32(1)
const _EO_HOST_OK = Int32(0)
const _EO_TERM_HORIZONTAL = UInt32(0)
const _EO_TERM_FOURBODY = UInt32(1)
const _EO_TERM_LONGER_HORIZONTAL = UInt32(2)
const _EO_ABI_PADDING = (UInt8(0), UInt8(0), UInt8(0), UInt8(0))

Base.@kwdef struct EoFlipClassification
    source_index::Int32
    pass::Int32
    bucket::UInt32
    active_slot::Int32
    n_flips::Int32
    flip_site::NTuple{4,Int32}
    flip_value::NTuple{4,Int32}
    mask_a::Int32
    mask_b::Int32
    coeff_re::Float64
    coeff_im::Float64
end

Base.@kwdef struct EoTermClassification
    original::Vector{EoFlipClassification}
    transposed::Vector{EoFlipClassification}
    original_active_slots::Int32
    transposed_active_slots::Int32
end

struct _EoAbiDiagBond
    site_a::Int32
    site_b::Int32
    coeff::Float64
end

struct _EoAbiFlipTerm
    n_flips::Int32
    flip_site::NTuple{4,Int32}
    flip_value::NTuple{4,Int32}
    mask_a::Int32
    mask_b::Int32
    padding::NTuple{4,UInt8}
    coeff_re::Float64
    coeff_im::Float64
end

struct _EoAbiTermTable
    n_diag::Int32
    padding_diag::NTuple{4,UInt8}
    diag::Ptr{_EoAbiDiagBond}
    n_flip::Int32
    padding_flip::NTuple{4,UInt8}
    flip::Ptr{_EoAbiFlipTerm}
end

Base.@kwdef struct EoTermTable
    lx::Int32
    ly::Int32
    active_compact::Bool
    diag::Vector{_EoAbiDiagBond}
    flip::Vector{_EoAbiFlipTerm}
    classification::EoTermClassification
    diag_pointer::Ptr{_EoAbiDiagBond}
    flip_pointer::Ptr{_EoAbiFlipTerm}
    abi::_EoAbiTermTable
    abi_ref::Base.RefValue{_EoAbiTermTable}
end

@inline function _eo_term_transpose_site(site::Int32, lx::Int32, ly::Int32)::Int32
    return (site % ly) * lx + site ÷ ly
end

function _eo_term_classify_flip(
    config::QnpepsElocConfig,
    term::QnpepsElocFlipTerm,
    source_index::Int32,
    active_compact::Bool,
    original_slot::Int32,
    transposed_slot::Int32,
)
    n_flips = term.n_flips
    1 <= n_flips <= 4 || throw(ArgumentError("unsupported E/O flip count"))
    lx = config.lx
    ly = config.ly
    sites = lx * ly
    for index in 1:Int(n_flips)
        site = term.flip_site[index]
        0 <= site < sites || throw(ArgumentError("E/O flip site is outside the lattice"))
    end
    masks_present = term.mask_a >= 0 || term.mask_b >= 0
    if masks_present
        0 <= term.mask_a < sites || throw(ArgumentError("E/O mask site is outside the lattice"))
        0 <= term.mask_b < sites || throw(ArgumentError("E/O mask site is outside the lattice"))
    end
    min_row = term.flip_site[1] ÷ ly
    max_row = min_row
    min_column = term.flip_site[1] % ly
    max_column = min_column
    for index in 1:Int(n_flips)
        row = term.flip_site[index] ÷ ly
        column = term.flip_site[index] % ly
        min_row = min(min_row, row)
        max_row = max(max_row, row)
        min_column = min(min_column, column)
        max_column = max(max_column, column)
    end
    dx = max_row - min_row
    dy = max_column - min_column
    pass = Int32(0)
    bucket = _EO_TERM_HORIZONTAL
    if n_flips == 1
        bucket = _EO_TERM_HORIZONTAL
    elseif dx == 0
        bucket = dy <= 1 ? _EO_TERM_HORIZONTAL : _EO_TERM_LONGER_HORIZONTAL
    elseif dy == 0
        pass = Int32(1)
        bucket = dx <= 1 ? _EO_TERM_HORIZONTAL : _EO_TERM_LONGER_HORIZONTAL
    elseif dx == 1 && dy == 1 && n_flips == 2
        bucket = _EO_TERM_FOURBODY
    else
        throw(ArgumentError("unsupported E/O flip geometry"))
    end
    sites_tuple = if pass == 0
        term.flip_site
    else
        ntuple(
            index ->
                index <= n_flips ? _eo_term_transpose_site(term.flip_site[index], lx, ly) :
                term.flip_site[index],
            Val(4),
        )
    end
    mask_a =
        pass == 1 && term.mask_a >= 0 ? _eo_term_transpose_site(term.mask_a, lx, ly) : term.mask_a
    mask_b =
        pass == 1 && term.mask_b >= 0 ? _eo_term_transpose_site(term.mask_b, lx, ly) : term.mask_b
    active_slot = if active_compact && mask_a >= 0 && mask_b >= 0
        pass == 0 ? original_slot : transposed_slot
    else
        Int32(-1)
    end
    classified = EoFlipClassification(;
        source_index,
        pass,
        bucket,
        active_slot,
        n_flips,
        flip_site=sites_tuple,
        flip_value=term.flip_value,
        mask_a,
        mask_b,
        coeff_re=term.coeff_re,
        coeff_im=term.coeff_im,
    )
    next_original = original_slot + (active_slot >= 0 && pass == 0 ? Int32(1) : Int32(0))
    next_transposed = transposed_slot + (active_slot >= 0 && pass == 1 ? Int32(1) : Int32(0))
    return classified, next_original, next_transposed
end

function eo_term_table(
    config::QnpepsElocConfig,
    terms::HeisenbergTerms,
    active_compact::Bool,
)::EoTermTable
    length(terms.diag) <= typemax(Int32) || throw(ArgumentError("too many diagonal terms"))
    length(terms.flip) <= typemax(Int32) || throw(ArgumentError("too many flip terms"))
    config.lx >= 2 || throw(ArgumentError("lx must be at least two"))
    config.ly >= 2 || throw(ArgumentError("ly must be at least two"))
    sites = config.lx * config.ly
    source_diag = terms.diag
    source_flip = terms.flip
    _require_eloc_term_capabilities(terms)
    for term in source_diag
        0 <= term.site_a < sites || throw(ArgumentError("E/O diagonal site is outside the lattice"))
        term.site_b == ELOC_DIAG_ONE_SITE_Z ||
            0 <= term.site_b < sites ||
            throw(ArgumentError("E/O diagonal site is outside the lattice"))
        isfinite(term.coeff) || throw(ArgumentError("E/O diagonal coefficient must be finite"))
    end
    original = EoFlipClassification[]
    transposed = EoFlipClassification[]
    sizehint!(original, length(source_flip))
    sizehint!(transposed, length(source_flip))
    original_slot = Int32(0)
    transposed_slot = Int32(0)
    for (source_index, term) in pairs(source_flip)
        classified, original_slot, transposed_slot = _eo_term_classify_flip(
            config,
            term,
            Int32(source_index - 1),
            active_compact,
            original_slot,
            transposed_slot,
        )
        push!(classified.pass == 0 ? original : transposed, classified)
    end
    classification = EoTermClassification(;
        original,
        transposed,
        original_active_slots=original_slot,
        transposed_active_slots=transposed_slot,
    )
    diag =
        _EoAbiDiagBond[_EoAbiDiagBond(term.site_a, term.site_b, term.coeff) for term in source_diag]
    flip = _EoAbiFlipTerm[
        _EoAbiFlipTerm(
            term.n_flips,
            term.flip_site,
            term.flip_value,
            term.mask_a,
            term.mask_b,
            _EO_ABI_PADDING,
            term.coeff_re,
            term.coeff_im,
        ) for term in source_flip
    ]
    diag_pointer = isempty(diag) ? Ptr{_EoAbiDiagBond}(0) : pointer(diag)
    flip_pointer = isempty(flip) ? Ptr{_EoAbiFlipTerm}(0) : pointer(flip)
    abi = _EoAbiTermTable(
        Int32(length(diag)),
        _EO_ABI_PADDING,
        diag_pointer,
        Int32(length(flip)),
        _EO_ABI_PADDING,
        flip_pointer,
    )
    return EoTermTable(;
        lx=config.lx,
        ly=config.ly,
        active_compact,
        diag,
        flip,
        classification,
        diag_pointer,
        flip_pointer,
        abi,
        abi_ref=Ref(abi),
    )
end

function _eo_term_active_compact_from_env()::Bool
    value = get(ENV, "QNPEPS_ELOC_ACTIVE_COMPACT", "")
    return isempty(value) || first(value) != '0'
end

function eo_term_table(config::QnpepsElocConfig, terms::HeisenbergTerms)::EoTermTable
    return eo_term_table(config, terms, _eo_term_active_compact_from_env())
end

function _eo_term_table_validate_pointers(table::EoTermTable)::Nothing
    diag_pointer = isempty(table.diag) ? Ptr{_EoAbiDiagBond}(0) : pointer(table.diag)
    flip_pointer = isempty(table.flip) ? Ptr{_EoAbiFlipTerm}(0) : pointer(table.flip)
    diag_pointer == table.diag_pointer || throw(ArgumentError("E/O diagonal storage changed"))
    flip_pointer == table.flip_pointer || throw(ArgumentError("E/O flip storage changed"))
    Int32(length(table.diag)) == table.abi.n_diag ||
        throw(ArgumentError("E/O diagonal table length changed"))
    Int32(length(table.flip)) == table.abi.n_flip ||
        throw(ArgumentError("E/O flip table length changed"))
    table.abi.padding_diag == _EO_ABI_PADDING && table.abi.padding_flip == _EO_ABI_PADDING ||
        throw(ArgumentError("E/O ABI table padding changed"))
    all(term -> term.padding == _EO_ABI_PADDING, table.flip) ||
        throw(ArgumentError("E/O flip-term padding changed"))
    abi = table.abi_ref[]
    abi.n_diag == table.abi.n_diag &&
    abi.padding_diag == table.abi.padding_diag &&
    abi.diag == table.abi.diag &&
    abi.n_flip == table.abi.n_flip &&
    abi.padding_flip == table.abi.padding_flip &&
    abi.flip == table.abi.flip || throw(ArgumentError("E/O ABI table changed"))
    table.active_compact == _eo_term_active_compact_from_env() ||
        throw(ArgumentError("E/O active-compaction policy changed after table construction"))
    return nothing
end
