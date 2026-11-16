const EO_HOST_LANES = 4
const _EO_HOST_ALIGNMENT = UInt64(256)
const _EO_HOST_ROWS = UInt32(1)
const _EO_HOST_OK = Int32(0)
const _EO_ABI_PADDING = (UInt8(0), UInt8(0), UInt8(0), UInt8(0))

Base.@kwdef struct EoTermTable
    lx::Int32
    ly::Int32
    buffers::ElocTermBuffers
    abi::QnpepsElocTermTable
    abi_ref::Base.RefValue{QnpepsElocTermTable}
end

function eo_term_table(config::QnpepsElocConfig, terms::OperatorTerms)::EoTermTable
    buffers = _eloc_term_buffers(terms, config)
    abi = _eloc_term_table(buffers)
    return EoTermTable(; lx=config.lx, ly=config.ly, buffers, abi, abi_ref=Ref(abi))
end

function _eo_term_table_validate_pointers(table::EoTermTable)::Nothing
    _eloc_abi_table(table.buffers) == table.abi || throw(ArgumentError("E/O term storage changed"))
    table.abi_ref[] == table.abi || throw(ArgumentError("E/O ABI table changed"))
    return nothing
end
