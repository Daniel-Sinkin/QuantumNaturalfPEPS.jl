#ifndef QNPEPS_ARENA_BOOK_HPP
#define QNPEPS_ARENA_BOOK_HPP

#include "core/arena_intervals.hpp"

#include <cstddef>
#include <limits>
#include <optional>
#include <utility>

namespace qnpeps
{
template <typename Synchronize>
[[nodiscard]] auto is_lease_reusable(bool has_reservations, Synchronize&& synchronize) -> bool
{
    if (not has_reservations) return true;
    return std::forward<Synchronize>(synchronize)();
}

class ArenaBook
{
  public:
    using Owner = ArenaIntervals::Owner;

    [[nodiscard]] auto reset(std::size_t capacity, std::size_t alignment) noexcept -> bool
    {
        next_owner_ = 1;
        return intervals_.reset(capacity, alignment);
    }

    [[nodiscard]] auto open_lease() noexcept -> std::optional<Owner>
    {
        if (next_owner_ == std::numeric_limits<Owner>::max()) return std::nullopt;
        return next_owner_++;
    }

    [[nodiscard]] auto take(Owner owner, std::size_t bytes) -> std::optional<ArenaInterval>
    {
        return intervals_.acquire(owner, bytes);
    }

    auto finish_lease(Owner owner, bool reusable) -> void
    {
        if (reusable)
            intervals_.release(owner);
        else
            intervals_.quarantine(owner);
    }

    [[nodiscard]] auto intervals() const noexcept -> const ArenaIntervals& { return intervals_; }

    [[nodiscard]] auto intervals() noexcept -> ArenaIntervals& { return intervals_; }

  private:
    ArenaIntervals intervals_{};
    Owner next_owner_{1};
};
}

#endif
