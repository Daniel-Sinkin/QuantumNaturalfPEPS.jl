#include "capi/internal/arena_reading.cuh"
#include "capi/qnpeps.h"
#include "core/arena_cursor.cuh"
#include "core/error.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <format>
#include <string>
#include <string_view>

namespace
{
using qnpeps::Status;
using qnpeps::usize;
using qnpeps::operator""_uz;

static_assert(sizeof(QnpepsArenaReading) == 56);
static_assert(offsetof(QnpepsArenaReading, device) == 4);
static_assert(offsetof(QnpepsArenaReading, capacity_bytes) == 8);
static_assert(offsetof(QnpepsArenaReading, in_use_bytes) == 16);
static_assert(offsetof(QnpepsArenaReading, top_bytes) == 24);
static_assert(offsetof(QnpepsArenaReading, largest_free_bytes) == 32);
static_assert(offsetof(QnpepsArenaReading, reservation_count) == 40);
static_assert(offsetof(QnpepsArenaReading, quarantined_count) == 48);

constexpr usize k_mebibyte{1024_uz * 1024_uz};
constexpr usize k_first_bytes{3_uz * k_mebibyte};
constexpr usize k_second_bytes{5_uz * k_mebibyte};
constexpr usize k_quarantined_bytes{7_uz * k_mebibyte};
constexpr int32_t k_arena_device{0};
constexpr int32_t k_untouched_device{3};

int failures{};

auto expect(bool condition, std::string_view name, const std::string& detail = {}) -> void
{
    std::printf(
        "[arena_reading_test] %s %.*s",
        condition ? "ok" : "FAIL",
        static_cast<int>(name.size()),
        name.data()
    );
    if (not condition and not detail.empty()) std::printf(" (%s)", detail.c_str());
    std::printf("\n");
    if (not condition) ++failures;
}

struct ExpectedReading
{
    usize capacity{};
    usize in_use{};
    usize top{};
    usize largest_free{};
    usize reservation_count{};
    usize quarantined_count{};
};

[[nodiscard]] auto describe(const QnpepsArenaReading& reading) -> std::string
{
    return std::format(
        "device={} capacity={} in_use={} top={} largest_free={} reservations={} quarantined={}",
        reading.device,
        reading.capacity_bytes,
        reading.in_use_bytes,
        reading.top_bytes,
        reading.largest_free_bytes,
        reading.reservation_count,
        reading.quarantined_count
    );
}

[[nodiscard]] auto read(int32_t device) -> QnpepsArenaReading
{
    QnpepsArenaReading reading{};
    reading.struct_size = static_cast<uint32_t>(sizeof(QnpepsArenaReading));
    const auto status = qnpeps::guarded([&] { qnpeps::capi::read_device_arena(device, &reading); });
    expect(status == QNPEPS_OK, "reading returns the ok status");
    return reading;
}

auto expect_reading(int32_t device, const ExpectedReading& expected, std::string_view name) -> void
{
    const auto reading = read(device);
    const auto matches = reading.device == device and reading.capacity_bytes == expected.capacity
                         and reading.in_use_bytes == expected.in_use
                         and reading.top_bytes == expected.top
                         and reading.largest_free_bytes == expected.largest_free
                         and reading.reservation_count == expected.reservation_count
                         and reading.quarantined_count == expected.quarantined_count;
    expect(matches, name, describe(reading));
}

auto argument_validation() -> void
{
    const auto null_status =
        qnpeps::guarded([] { qnpeps::capi::read_device_arena(k_arena_device, nullptr); });
    expect(null_status == QNPEPS_ERR_NULL_ARG, "a null output is rejected");

    QnpepsArenaReading unsized{};
    const auto size_status =
        qnpeps::guarded([&] { qnpeps::capi::read_device_arena(k_arena_device, &unsized); });
    expect(size_status == QNPEPS_ERR_BAD_VERSION, "a wrong struct size is rejected");

    QnpepsArenaReading negative{};
    negative.struct_size = static_cast<uint32_t>(sizeof(QnpepsArenaReading));
    const auto device_status =
        qnpeps::guarded([&] { qnpeps::capi::read_device_arena(-1, &negative); });
    expect(device_status == QNPEPS_ERR_BAD_CONFIG, "a negative device is rejected");
}

auto absent_arena() -> void
{
    expect_reading(k_untouched_device, {}, "a device without an arena reads zero");
    expect(
        qnpeps::find_device_arena(k_untouched_device) == nullptr,
        "reading a device does not create its arena"
    );
}

auto lease_sequence() -> void
{
    expect(
        qnpeps::find_device_arena(k_arena_device) == nullptr,
        "the arena device starts without an arena"
    );
    const auto arena = qnpeps::shared_device_arena(k_arena_device);
    const auto capacity = arena->capacity();
    cudaStream_t stream{};
    qnpeps::check(cudaStreamCreate(&stream));

    expect_reading(
        k_arena_device, {.capacity = capacity, .largest_free = capacity}, "a new arena is empty"
    );

    auto first = arena->acquire_lease(stream);
    auto second = arena->acquire_lease(stream);
    static_cast<void>(first->take(k_first_bytes));
    static_cast<void>(second->take(k_second_bytes));
    const auto both = k_first_bytes + k_second_bytes;
    expect_reading(
        k_arena_device,
        {.capacity = capacity,
         .in_use = both,
         .top = both,
         .largest_free = capacity - both,
         .reservation_count = 2},
        "two leases stack from the base"
    );

    first->release();
    expect_reading(
        k_arena_device,
        {.capacity = capacity,
         .in_use = k_second_bytes,
         .top = both,
         .largest_free = capacity - both,
         .reservation_count = 1},
        "a released lower lease leaves the top in place"
    );

    const auto stacked = both + k_quarantined_bytes;
    {
        auto streamless = arena->acquire_lease(nullptr);
        static_cast<void>(streamless->take(k_quarantined_bytes));
    }
    expect_reading(
        k_arena_device,
        {.capacity = capacity,
         .in_use = k_second_bytes + k_quarantined_bytes,
         .top = stacked,
         .largest_free = capacity - stacked,
         .reservation_count = 2,
         .quarantined_count = 1},
        "a lease without a stream is quarantined above the hole"
    );

    second->release();
    expect_reading(
        k_arena_device,
        {.capacity = capacity,
         .in_use = k_quarantined_bytes,
         .top = stacked,
         .largest_free = capacity - stacked,
         .reservation_count = 1,
         .quarantined_count = 1},
        "a quarantined block holds the top after every lease is released"
    );
    qnpeps::unchecked(cudaStreamDestroy(stream));
}
}

auto main() -> int
{
    argument_validation();
    absent_arena();
    const auto sequence_status = qnpeps::guarded([] { lease_sequence(); });
    expect(
        sequence_status == QNPEPS_OK, "the lease sequence runs", std::string{qnpeps_last_error()}
    );
    std::printf("[arena_reading_test] failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
