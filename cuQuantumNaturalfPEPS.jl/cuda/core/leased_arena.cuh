#ifndef QNPEPS_LEASED_ARENA_CUH
#define QNPEPS_LEASED_ARENA_CUH

#include "core/arena_cursor.cuh"

#include <cuda_runtime.h>
#include <memory>
#include <type_traits>
#include <utility>

namespace qnpeps
{
class LeasedArena
{
  public:
    LeasedArena() = default;

    LeasedArena(int device, cudaStream_t stream)
    {
        const auto arena = shared_device_arena(device);
        if (not arena) return;
        lease_ = arena->acquire_lease(stream);
        if (not lease_) return;
        cursor_ = ArenaCursor::persistent(*lease_);
    }

    LeasedArena(const LeasedArena&) = delete;
    auto operator=(const LeasedArena&) -> LeasedArena& = delete;

    LeasedArena(LeasedArena&& other) noexcept
        : lease_(std::move(other.lease_)), cursor_(std::exchange(other.cursor_, ArenaCursor{}))
    {
    }

    auto operator=(LeasedArena&& other) noexcept -> LeasedArena&
    {
        if (this == &other) return *this;
        lease_ = std::move(other.lease_);
        cursor_ = std::exchange(other.cursor_, ArenaCursor{});
        return *this;
    }

    ~LeasedArena() = default;

    [[nodiscard]] auto cursor() noexcept -> ArenaCursor& { return cursor_; }

  private:
    std::unique_ptr<ArenaLease> lease_{};
    ArenaCursor cursor_{};
};

static_assert(not std::is_copy_constructible_v<LeasedArena>);
static_assert(std::is_nothrow_move_assignable_v<LeasedArena>);
}

#endif
