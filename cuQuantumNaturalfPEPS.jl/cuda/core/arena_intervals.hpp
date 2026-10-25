#ifndef QNPEPS_ARENA_INTERVALS_HPP
#define QNPEPS_ARENA_INTERVALS_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <utility>
#include <vector>

namespace qnpeps
{
struct ArenaInterval
{
    std::size_t begin{};
    std::size_t end{};

    [[nodiscard]] constexpr auto size() const noexcept -> std::size_t { return end - begin; }
    auto operator==(const ArenaInterval&) const -> bool = default;
};

class ArenaIntervals
{
  public:
    using Owner = std::uint64_t;
    using Checkpoint = std::uint64_t;

    ArenaIntervals() = default;
    ArenaIntervals(std::size_t capacity, std::size_t alignment) { reset(capacity, alignment); }

    auto reset(std::size_t capacity, std::size_t alignment) noexcept -> bool
    {
        if (alignment == 0 or (alignment & (alignment - 1)) != 0 or capacity % alignment != 0)
            return false;
        capacity_ = capacity;
        alignment_ = alignment;
        next_sequence_ = 1;
        occupied_.clear();
        return true;
    }

    [[nodiscard]] auto acquire(Owner owner, std::size_t bytes) -> std::optional<ArenaInterval>
    {
        if (owner == 0 or alignment_ == 0) return std::nullopt;
        if (bytes == 0) return ArenaInterval{};
        constexpr auto maximum = std::numeric_limits<std::size_t>::max();
        constexpr auto maximum_sequence = std::numeric_limits<Checkpoint>::max();
        if (bytes > maximum - (alignment_ - 1) or next_sequence_ == maximum_sequence)
            return std::nullopt;
        const auto reserved = (bytes + alignment_ - 1) & ~(alignment_ - 1);
        auto begin = std::size_t{};
        auto position = occupied_.begin();
        for (; position != occupied_.end(); ++position)
        {
            if (begin <= position->interval.begin and reserved <= position->interval.begin - begin)
                break;
            begin = position->interval.end;
        }
        if (begin > capacity_ or reserved > capacity_ - begin) return std::nullopt;
        const auto end = begin + reserved;
        try
        {
            occupied_.insert(position, Occupied{ArenaInterval{begin, end}, owner, next_sequence_});
        }
        catch (const std::bad_alloc&)
        {
            return std::nullopt;
        }
        ++next_sequence_;
        return ArenaInterval{begin, end};
    }

    [[nodiscard]] auto checkpoint() const noexcept -> Checkpoint { return next_sequence_; }

    auto release_since(Owner owner, Checkpoint checkpoint) noexcept -> void
    {
        occupied_.erase(
            std::remove_if(
                occupied_.begin(),
                occupied_.end(),
                [owner, checkpoint](const Occupied& region)
                { return region.owner == owner and region.sequence >= checkpoint; }
            ),
            occupied_.end()
        );
    }

    auto release(Owner owner) noexcept -> void
    {
        occupied_.erase(
            std::remove_if(
                occupied_.begin(),
                occupied_.end(),
                [owner](const Occupied& region) { return region.owner == owner; }
            ),
            occupied_.end()
        );
    }

    auto quarantine(Owner owner) noexcept -> void
    {
        for (auto& region : occupied_)
        {
            if (region.owner != owner) continue;
            region.owner = 0;
            region.sequence = 0;
        }
    }

    [[nodiscard]] auto total(Owner owner) const noexcept -> std::size_t
    {
        auto bytes = std::size_t{};
        for (const auto& region : occupied_)
            if (region.owner == owner) bytes += region.interval.size();
        return bytes;
    }

    [[nodiscard]] auto reservation_count(Owner owner) const noexcept -> std::size_t
    {
        return static_cast<std::size_t>(std::count_if(
            occupied_.begin(),
            occupied_.end(),
            [owner](const Occupied& region) { return region.owner == owner; }
        ));
    }

    [[nodiscard]] auto reservation_count() const noexcept -> std::size_t
    {
        return occupied_.size();
    }

    [[nodiscard]] auto quarantine_count() const noexcept -> std::size_t
    {
        return static_cast<std::size_t>(std::count_if(
            occupied_.begin(),
            occupied_.end(),
            [](const Occupied& region) { return region.owner == 0; }
        ));
    }

    [[nodiscard]] auto high_water() const noexcept -> std::size_t
    {
        return occupied_.empty() ? 0 : occupied_.back().interval.end;
    }

    [[nodiscard]] auto largest_free() const noexcept -> std::size_t
    {
        auto largest = std::size_t{};
        auto begin = std::size_t{};
        for (const auto& region : occupied_)
        {
            largest = std::max(largest, region.interval.begin - begin);
            begin = region.interval.end;
        }
        return std::max(largest, capacity_ - begin);
    }

    [[nodiscard]] auto free_bytes() const noexcept -> std::size_t
    {
        auto occupied = std::size_t{};
        for (const auto& region : occupied_)
            occupied += region.interval.size();
        return capacity_ - occupied;
    }

    [[nodiscard]] auto capacity() const noexcept -> std::size_t { return capacity_; }

    [[nodiscard]] auto free_intervals() const -> std::vector<ArenaInterval>
    {
        auto free = std::vector<ArenaInterval>{};
        auto begin = std::size_t{};
        for (const auto& region : occupied_)
        {
            if (begin < region.interval.begin) free.push_back({begin, region.interval.begin});
            begin = region.interval.end;
        }
        if (begin < capacity_) free.push_back({begin, capacity_});
        return free;
    }

  private:
    struct Occupied
    {
        ArenaInterval interval{};
        Owner owner{};
        Checkpoint sequence{};
    };

    std::size_t capacity_{};
    std::size_t alignment_{};
    Checkpoint next_sequence_{1};
    std::vector<Occupied> occupied_{};
};
}

#endif
