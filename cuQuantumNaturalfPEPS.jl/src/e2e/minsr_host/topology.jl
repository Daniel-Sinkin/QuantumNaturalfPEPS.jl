

const MINSR_HOST_LANES = 4
const MINSR_HOST_PEER_TILE_BYTES = UInt64(1) << 30

struct MinsrHostTopology
    name::Symbol
    numa_nodes::NTuple{MINSR_HOST_LANES,Int32}
    cpu_low::NTuple{MINSR_HOST_LANES,Int}
    cpu_high::NTuple{MINSR_HOST_LANES,Int}
end

const MINSR_HOST_TOPOLOGY_JURECA = MinsrHostTopology(
    :JURECA,
    (Int32(3), Int32(1), Int32(7), Int32(5)),
    (48, 16, 112, 80),
    (63, 31, 127, 95),
)

const MINSR_HOST_TOPOLOGY_JUPITER = MinsrHostTopology(
    :JUPITER,
    (Int32(0), Int32(1), Int32(2), Int32(3)),
    (0, 72, 144, 216),
    (71, 143, 215, 287),
)

const _MINSR_HOST_ALIGNMENT = UInt64(256)
const _MINSR_HOST_OK = Int32(0)
const _MINSR_HOST_ERR_NULL = Int32(1)
const _MINSR_HOST_ERR_CONFIG = Int32(2)
const _MINSR_HOST_ERR_VERSION = Int32(3)
const _MINSR_HOST_ERR_CUDA = Int32(4)
const _MINSR_HOST_ERR_OOM = Int32(5)
const _MINSR_HOST_ERR_INTERNAL = Int32(6)
const _MINSR_HOST_COMMAND_FULL = Int32(1)
const _MINSR_HOST_COMMAND_COPY = Int32(2)
const _MINSR_HOST_COMMAND_NOOP = Int32(3)
const _MINSR_HOST_COMMAND_STOP = Int32(4)
