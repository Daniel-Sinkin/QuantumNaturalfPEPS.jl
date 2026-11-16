#ifndef QNPEPS_CAPI_ARENA_READING_CUH
#define QNPEPS_CAPI_ARENA_READING_CUH

#include "capi/qnpeps.h"
#include "core/arena_cursor.cuh"
#include "core/error.hpp"

#include <cstdint>

namespace qnpeps::capi
{
[[nodiscard]] inline auto arena_reading_record(int32_t device, const ArenaReading& reading) noexcept
    -> QnpepsArenaReading
{
    return {
        .struct_size = static_cast<uint32_t>(sizeof(QnpepsArenaReading)),
        .device = device,
        .capacity_bytes = reading.capacity,
        .in_use_bytes = reading.in_use,
        .top_bytes = reading.top,
        .largest_free_bytes = reading.largest_free,
        .reservation_count = reading.reservation_count,
        .quarantined_count = reading.quarantined_count,
    };
}

inline auto read_device_arena(int32_t device, QnpepsArenaReading* out) -> void
{
    require(out != nullptr, Status::null_argument, "arena reading output is null");
    require_struct_size(*out, "arena reading");
    require(device >= 0, Status::bad_config, "arena reading device {} is negative", device);
    const auto arena = find_device_arena(device);
    const auto reading = arena ? arena->reading() : ArenaReading{};
    *out = arena_reading_record(device, reading);
}
}

#endif
