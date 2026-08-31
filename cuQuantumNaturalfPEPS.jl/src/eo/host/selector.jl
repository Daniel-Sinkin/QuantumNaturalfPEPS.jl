const EO_HOST_LANES = 4
const _EO_HOST_ALIGNMENT = UInt64(256)
const _EO_HOST_ROWS = UInt32(1)
const _EO_HOST_OK = Int32(0)
const EO_SELECTOR_EXACT = UInt32(0)
const EO_SELECTOR_HALF_ROW = UInt32(1)
const EO_SELECTOR_HALF_COLUMN = UInt32(2)
const EO_SELECTOR_BALANCED = UInt32(0)
const EO_SELECTOR_FORCE_GROUP_0 = UInt32(1)
const EO_SELECTOR_FORCE_GROUP_1 = UInt32(2)
const EO_SELECTOR_FORCE_ALL = UInt32(3)
const _EO_TERM_HORIZONTAL = UInt32(0)
const _EO_TERM_FOURBODY = UInt32(1)
const _EO_TERM_LONGER_HORIZONTAL = UInt32(2)
const _EO_ABI_PADDING = (UInt8(0), UInt8(0), UInt8(0), UInt8(0))

Base.@kwdef struct EoFlipClassification
    source_index::Int32
    pass::Int32
    bucket::UInt32
    active_slot::Int32
    j2_row_group::Int32
    j2_column_group::Int32
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
    j2_row_groups::NTuple{2,UInt64}
    j2_column_groups::NTuple{2,UInt64}
    j2_row_diag::NTuple{2,UInt64}
    j2_column_diag::NTuple{2,UInt64}
    j2_row_flip::NTuple{2,UInt64}
    j2_column_flip::NTuple{2,UInt64}
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

struct EoSelector
    mode::UInt32
    draw::UInt32
    seed::UInt64
    epoch::UInt64
    function EoSelector(mode::UInt32, draw::UInt32, seed::UInt64, epoch::UInt64)
        mode <= EO_SELECTOR_HALF_COLUMN || throw(ArgumentError("unknown E/O selector mode"))
        draw <= EO_SELECTOR_FORCE_ALL || throw(ArgumentError("unknown E/O selector draw"))
        mode == EO_SELECTOR_EXACT &&
            draw != EO_SELECTOR_BALANCED &&
            throw(ArgumentError("exact E/O selection requires the balanced draw code"))
        return new(mode, draw, seed, epoch)
    end
end

struct EoSelectorDecision
    group::Int32
    weight::Float64
end

Base.@kwdef struct EoSelectorAccounting
    mode::UInt32
    draw::UInt32
    seed::UInt64
    epoch::UInt64
    waves::UInt64
    group0_waves::UInt64
    group1_waves::UInt64
    row_groups_total::UInt64
    row_groups_retained::UInt64
    column_groups_total::UInt64
    column_groups_retained::UInt64
    diag_terms_total::UInt64
    diag_terms_retained::UInt64
    flip_terms_total::UInt64
    flip_terms_retained::UInt64
end

const _EO_SELECTOR_DEFAULT =
    EoSelector(EO_SELECTOR_EXACT, EO_SELECTOR_BALANCED, UInt64(0), UInt64(0))

function eo_selector(mode::Symbol, draw::Symbol, seed::Integer, epoch::Integer)::EoSelector
    seed >= 0 || throw(ArgumentError("selector seed must be nonnegative"))
    epoch >= 0 || throw(ArgumentError("selector epoch must be nonnegative"))
    mode_code =
        mode === :exact ? EO_SELECTOR_EXACT :
        mode === :half_row ? EO_SELECTOR_HALF_ROW :
        mode === :half_column ? EO_SELECTOR_HALF_COLUMN :
        throw(ArgumentError("unknown E/O selector mode"))
    draw_code =
        draw === :balanced ? EO_SELECTOR_BALANCED :
        draw === :group0 ? EO_SELECTOR_FORCE_GROUP_0 :
        draw === :group1 ? EO_SELECTOR_FORCE_GROUP_1 :
        draw === :all ? EO_SELECTOR_FORCE_ALL : throw(ArgumentError("unknown E/O selector draw"))
    return EoSelector(mode_code, draw_code, UInt64(seed), UInt64(epoch))
end

@inline function _eo_selector_splitmix64(value::UInt64)::UInt64
    value += UInt64(0x9e3779b97f4a7c15)
    value = xor(value, value >> 30) * UInt64(0xbf58476d1ce4e5b9)
    value = xor(value, value >> 27) * UInt64(0x94d049bb133111eb)
    return xor(value, value >> 31)
end

@inline function eo_selector_decision(selector::EoSelector, wave_index::Int64)::EoSelectorDecision
    wave_index >= 0 || throw(ArgumentError("wave index must be nonnegative"))
    group = if selector.mode == EO_SELECTOR_EXACT
        Int32(-1)
    elseif selector.draw == EO_SELECTOR_FORCE_ALL
        Int32(-2)
    elseif selector.draw == EO_SELECTOR_FORCE_GROUP_0
        Int32(0)
    elseif selector.draw == EO_SELECTOR_FORCE_GROUP_1
        Int32(1)
    else
        first = Int32(
            _eo_selector_splitmix64(xor(selector.seed, _eo_selector_splitmix64(selector.epoch))) & UInt64(1),
        )
        xor(first, Int32(wave_index & 1))
    end
    return EoSelectorDecision(group, group >= 0 ? 2.0 : 1.0)
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
    row_group = Int32(-1)
    column_group = Int32(-1)
    if n_flips == 1
        bucket = _EO_TERM_HORIZONTAL
    elseif dx == 0
        bucket = dy <= 1 ? _EO_TERM_HORIZONTAL : _EO_TERM_LONGER_HORIZONTAL
    elseif dy == 0
        pass = Int32(1)
        bucket = dx <= 1 ? _EO_TERM_HORIZONTAL : _EO_TERM_LONGER_HORIZONTAL
    elseif dx == 1 && dy == 1 && n_flips == 2
        bucket = _EO_TERM_FOURBODY
        row_group = min_row & Int32(1)
        column_group = min_column & Int32(1)
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
        j2_row_group=row_group,
        j2_column_group=column_group,
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
    row_diag = UInt64[0, 0]
    column_diag = UInt64[0, 0]
    for term in source_diag
        0 <= term.site_a < sites || throw(ArgumentError("E/O diagonal site is outside the lattice"))
        0 <= term.site_b < sites || throw(ArgumentError("E/O diagonal site is outside the lattice"))
        row_a = term.site_a ÷ config.ly
        row_b = term.site_b ÷ config.ly
        column_a = term.site_a % config.ly
        column_b = term.site_b % config.ly
        if abs(row_a - row_b) == 1 && abs(column_a - column_b) == 1
            row_diag[Int((min(row_a, row_b) & Int32(1)) + 1)] += UInt64(1)
            column_diag[Int((min(column_a, column_b) & Int32(1)) + 1)] += UInt64(1)
        end
    end
    original = EoFlipClassification[]
    transposed = EoFlipClassification[]
    sizehint!(original, length(source_flip))
    sizehint!(transposed, length(source_flip))
    row_flip = UInt64[0, 0]
    column_flip = UInt64[0, 0]
    original_slot = Int32(0)
    transposed_slot = Int32(0)
    original_fb = false
    for (source_index, term) in pairs(source_flip)
        classified, original_slot, transposed_slot = _eo_term_classify_flip(
            config,
            term,
            Int32(source_index - 1),
            active_compact,
            original_slot,
            transposed_slot,
        )
        if classified.bucket == _EO_TERM_FOURBODY
            row_flip[Int(classified.j2_row_group + 1)] += UInt64(1)
            column_flip[Int(classified.j2_column_group + 1)] += UInt64(1)
            original_fb = true
        end
        push!(classified.pass == 0 ? original : transposed, classified)
    end
    row_groups = UInt64[0, 0]
    column_groups = UInt64[0, 0]
    if original_fb
        for upper_row in Int32(0):(config.lx-Int32(2))
            row_groups[Int((upper_row & Int32(1)) + 1)] += UInt64(1)
            for left_column in Int32(0):(config.ly-Int32(2))
                column_groups[Int((left_column & Int32(1)) + 1)] += UInt64(1)
            end
        end
    end
    classification = EoTermClassification(;
        original,
        transposed,
        j2_row_groups=(row_groups[1], row_groups[2]),
        j2_column_groups=(column_groups[1], column_groups[2]),
        j2_row_diag=(row_diag[1], row_diag[2]),
        j2_column_diag=(column_diag[1], column_diag[2]),
        j2_row_flip=(row_flip[1], row_flip[2]),
        j2_column_flip=(column_flip[1], column_flip[2]),
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

function eo_selector_accounting(
    table::EoTermTable,
    selector::EoSelector,
    waves::Int64,
)::EoSelectorAccounting
    waves >= 0 || throw(ArgumentError("wave count must be nonnegative"))
    classification = table.classification
    column = selector.mode == EO_SELECTOR_HALF_COLUMN
    diag = column ? classification.j2_column_diag : classification.j2_row_diag
    flip = column ? classification.j2_column_flip : classification.j2_row_flip
    row_total = classification.j2_row_groups[1] + classification.j2_row_groups[2]
    column_total = classification.j2_column_groups[1] + classification.j2_column_groups[2]
    diag_total = diag[1] + diag[2]
    flip_total = flip[1] + flip[2]
    group0_waves = UInt64(0)
    group1_waves = UInt64(0)
    row_retained = UInt64(0)
    column_retained = UInt64(0)
    diag_retained = UInt64(0)
    flip_retained = UInt64(0)
    for wave_index in Int64(0):(waves-Int64(1))
        group = eo_selector_decision(selector, wave_index).group
        if group < 0
            row_retained += row_total
            column && (column_retained += column_total)
            diag_retained += diag_total
            flip_retained += flip_total
        elseif group == 0
            group0_waves += UInt64(1)
            if column
                row_retained += row_total
                column_retained += classification.j2_column_groups[1]
            else
                row_retained += classification.j2_row_groups[1]
            end
            diag_retained += diag[1]
            flip_retained += flip[1]
        else
            group1_waves += UInt64(1)
            if column
                row_retained += row_total
                column_retained += classification.j2_column_groups[2]
            else
                row_retained += classification.j2_row_groups[2]
            end
            diag_retained += diag[2]
            flip_retained += flip[2]
        end
    end
    wave_count = UInt64(waves)
    return EoSelectorAccounting(;
        mode=selector.mode,
        draw=selector.draw,
        seed=selector.seed,
        epoch=selector.epoch,
        waves=wave_count,
        group0_waves,
        group1_waves,
        row_groups_total=row_total * wave_count,
        row_groups_retained=row_retained,
        column_groups_total=column ? column_total * wave_count : UInt64(0),
        column_groups_retained=column_retained,
        diag_terms_total=diag_total * wave_count,
        diag_terms_retained=diag_retained,
        flip_terms_total=flip_total * wave_count,
        flip_terms_retained=flip_retained,
    )
end
