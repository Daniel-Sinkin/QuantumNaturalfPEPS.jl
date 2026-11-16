#include "core/arena_book.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace
{
using qnpeps::ArenaBook;
using qnpeps::ArenaInterval;
using qnpeps::ArenaIntervals;
using Owner = ArenaBook::Owner;
using Checkpoint = ArenaIntervals::Checkpoint;

constexpr std::size_t k_operations_per_run{1500uz};
constexpr std::size_t k_fixed_seed_count{16uz};
constexpr std::size_t k_max_live_leases{12uz};
constexpr std::size_t k_max_alignment_exponent{12uz};
constexpr std::size_t k_max_capacity_blocks{4096uz};
constexpr std::size_t k_closed_owner_window{64uz};
constexpr auto k_budget = std::chrono::seconds{3};

enum class Operation
{
    open,
    take,
    checkpoint,
    release_since,
    close_synchronized,
    close_unsynchronized,
};

[[nodiscard]] auto operation_name(Operation operation) -> const char*
{
    switch (operation)
    {
        case Operation::open:
            return "open";
        case Operation::take:
            return "take";
        case Operation::checkpoint:
            return "checkpoint";
        case Operation::release_since:
            return "release_since";
        case Operation::close_synchronized:
            return "close_synchronized";
        case Operation::close_unsynchronized:
            return "close_unsynchronized";
    }
    return "unknown";
}

struct ModelRange
{
    std::size_t begin{};
    std::size_t end{};
    Owner owner{};
    Checkpoint sequence{};
};

struct LiveLease
{
    Owner owner{};
    std::vector<Checkpoint> checkpoints{};
};

class ReferenceModel
{
  public:
    ReferenceModel(std::size_t capacity, std::size_t alignment)
        : capacity_{capacity}, alignment_{alignment}
    {
    }

    [[nodiscard]] auto reserved_bytes(std::size_t bytes) const noexcept -> std::size_t
    {
        return (bytes + alignment_ - 1) & ~(alignment_ - 1);
    }

    [[nodiscard]] auto gaps() const -> std::vector<ArenaInterval>
    {
        auto free = std::vector<ArenaInterval>{};
        auto begin = 0uz;
        for (const auto& range : ranges_)
        {
            if (begin < range.begin) free.push_back({begin, range.begin});
            begin = range.end;
        }
        if (begin < capacity_) free.push_back({begin, capacity_});
        return free;
    }

    [[nodiscard]] auto first_fit(std::size_t reserved) const -> std::optional<std::size_t>
    {
        for (const auto& gap : gaps())
            if (gap.size() >= reserved) return gap.begin;
        return std::nullopt;
    }

    auto insert(ArenaInterval interval, Owner owner) -> void
    {
        const auto position = std::lower_bound(
            ranges_.begin(),
            ranges_.end(),
            interval.begin,
            [](const ModelRange& range, std::size_t begin) { return range.begin < begin; }
        );
        ranges_.insert(position, ModelRange{interval.begin, interval.end, owner, next_sequence_});
        ++next_sequence_;
    }

    auto release(Owner owner) -> void
    {
        std::erase_if(ranges_, [owner](const ModelRange& range) { return range.owner == owner; });
    }

    auto release_since(Owner owner, Checkpoint checkpoint) -> void
    {
        std::erase_if(
            ranges_,
            [owner, checkpoint](const ModelRange& range)
            { return range.owner == owner and range.sequence >= checkpoint; }
        );
    }

    auto quarantine(Owner owner) -> void
    {
        for (auto& range : ranges_)
        {
            if (range.owner != owner) continue;
            quarantined_bytes_ += range.end - range.begin;
            range.owner = 0;
            range.sequence = 0;
        }
    }

    [[nodiscard]] auto total(Owner owner) const noexcept -> std::size_t
    {
        auto bytes = 0uz;
        for (const auto& range : ranges_)
            if (range.owner == owner) bytes += range.end - range.begin;
        return bytes;
    }

    [[nodiscard]] auto count(Owner owner) const noexcept -> std::size_t
    {
        return static_cast<std::size_t>(std::count_if(
            ranges_.begin(),
            ranges_.end(),
            [owner](const ModelRange& range) { return range.owner == owner; }
        ));
    }

    [[nodiscard]] auto occupied_bytes() const noexcept -> std::size_t
    {
        auto bytes = 0uz;
        for (const auto& range : ranges_)
            bytes += range.end - range.begin;
        return bytes;
    }

    [[nodiscard]] auto high_water() const noexcept -> std::size_t
    {
        return ranges_.empty() ? 0uz : ranges_.back().end;
    }

    [[nodiscard]] auto largest_gap() const -> std::size_t
    {
        auto largest = 0uz;
        for (const auto& gap : gaps())
            largest = std::max(largest, gap.size());
        return largest;
    }

    [[nodiscard]] auto range_count() const noexcept -> std::size_t { return ranges_.size(); }
    [[nodiscard]] auto next_sequence() const noexcept -> Checkpoint { return next_sequence_; }
    [[nodiscard]] auto quarantined_bytes() const noexcept -> std::size_t
    {
        return quarantined_bytes_;
    }
    [[nodiscard]] auto capacity() const noexcept -> std::size_t { return capacity_; }
    [[nodiscard]] auto alignment() const noexcept -> std::size_t { return alignment_; }

  private:
    std::size_t capacity_{};
    std::size_t alignment_{};
    std::vector<ModelRange> ranges_{};
    Checkpoint next_sequence_{1};
    std::size_t quarantined_bytes_{};
};

struct Failure
{
    std::string message{};
};

class FuzzRun
{
  public:
    FuzzRun(std::uint64_t seed, std::size_t capacity, std::size_t alignment)
        : random_{seed}, model_{capacity, alignment}
    {
        book_ready_ = book_.reset(capacity, alignment);
    }

    [[nodiscard]] auto is_ready() const noexcept -> bool { return book_ready_; }
    [[nodiscard]] auto operations_done() const noexcept -> std::size_t { return operations_done_; }
    [[nodiscard]] auto last_operation() const noexcept -> Operation { return last_operation_; }

    [[nodiscard]] auto step() -> std::optional<Failure>
    {
        last_operation_ = pick_operation();
        ++operations_done_;
        if (auto failure = apply(last_operation_)) return failure;
        return check_invariants();
    }

    [[nodiscard]] auto drain() -> std::optional<Failure>
    {
        while (not live_.empty())
        {
            last_operation_ = Operation::close_synchronized;
            ++operations_done_;
            if (auto failure = close(0uz, true)) return failure;
            if (auto failure = check_invariants()) return failure;
        }
        const auto& intervals = book_.intervals();
        const auto quarantined = model_.quarantined_bytes();
        const auto drained = intervals.free_bytes() == model_.capacity() - quarantined
                             and intervals.reservation_count() == intervals.quarantine_count();
        if (not drained) return Failure{"leases closed with working sync left live intervals"};
        const auto is_empty = intervals.high_water() == 0 and intervals.reservation_count() == 0;
        if (quarantined == 0 and not is_empty)
            return Failure{"arena without quarantine is not empty after every lease closed"};
        return std::nullopt;
    }

  private:
    [[nodiscard]] auto uniform(std::size_t low, std::size_t high) -> std::size_t
    {
        return std::uniform_int_distribution<std::size_t>{low, high}(random_);
    }

    [[nodiscard]] auto pick_operation() -> Operation
    {
        if (live_.empty()) return Operation::open;
        const auto roll = uniform(0uz, 99uz);
        if (roll < 12uz) return Operation::open;
        if (roll < 70uz) return Operation::take;
        if (roll < 76uz) return Operation::checkpoint;
        if (roll < 82uz) return Operation::release_since;
        if (roll < 94uz) return Operation::close_synchronized;
        return Operation::close_unsynchronized;
    }

    [[nodiscard]] auto pick_bytes() -> std::size_t
    {
        const auto capacity = model_.capacity();
        const auto alignment = model_.alignment();
        const auto roll = uniform(0uz, 99uz);
        if (roll < 5uz) return 0uz;
        if (roll < 60uz) return uniform(1uz, 4uz * alignment);
        if (roll < 95uz) return uniform(1uz, std::max(1uz, capacity / 8uz));
        return uniform(capacity / 2uz, 2uz * capacity);
    }

    [[nodiscard]] auto apply(Operation operation) -> std::optional<Failure>
    {
        switch (operation)
        {
            case Operation::open:
                return open();
            case Operation::take:
                return take();
            case Operation::checkpoint:
                return checkpoint();
            case Operation::release_since:
                return release_since();
            case Operation::close_synchronized:
                return close(uniform(0uz, live_.size() - 1uz), true);
            case Operation::close_unsynchronized:
                return close(uniform(0uz, live_.size() - 1uz), false);
        }
        return Failure{"unknown operation"};
    }

    [[nodiscard]] auto open() -> std::optional<Failure>
    {
        if (live_.size() >= k_max_live_leases) return std::nullopt;
        const auto owner = book_.open_lease();
        if (not owner) return Failure{"lease owners ran out"};
        const auto is_fresh = std::none_of(
            closed_.begin(), closed_.end(), [&](Owner closed) { return closed == *owner; }
        );
        if (not is_fresh) return Failure{"lease owner reused"};
        live_.push_back(LiveLease{*owner, {}});
        return std::nullopt;
    }

    [[nodiscard]] auto take() -> std::optional<Failure>
    {
        const auto owner = live_[uniform(0uz, live_.size() - 1uz)].owner;
        const auto bytes = pick_bytes();
        const auto interval = book_.take(owner, bytes);
        if (bytes == 0)
        {
            const auto is_empty = interval and interval->size() == 0;
            return is_empty ? std::nullopt : std::optional{Failure{"zero-byte take was not empty"}};
        }
        const auto reserved = model_.reserved_bytes(bytes);
        const auto expected_begin = model_.first_fit(reserved);
        if (not interval)
        {
            if (expected_begin) return Failure{"take failed although a gap fits"};
            return std::nullopt;
        }
        if (not expected_begin) return Failure{"take succeeded although no gap fits"};
        if (interval->begin != *expected_begin) return Failure{"take is not first fit"};
        if (interval->size() != reserved) return Failure{"take size is not the aligned request"};
        if (interval->begin % model_.alignment() != 0) return Failure{"take is misaligned"};
        model_.insert(*interval, owner);
        return std::nullopt;
    }

    [[nodiscard]] auto checkpoint() -> std::optional<Failure>
    {
        auto& lease = live_[uniform(0uz, live_.size() - 1uz)];
        const auto checkpoint = book_.intervals().checkpoint();
        if (checkpoint != model_.next_sequence()) return Failure{"checkpoint differs from model"};
        lease.checkpoints.push_back(checkpoint);
        return std::nullopt;
    }

    [[nodiscard]] auto release_since() -> std::optional<Failure>
    {
        auto& lease = live_[uniform(0uz, live_.size() - 1uz)];
        if (lease.checkpoints.empty()) return std::nullopt;
        const auto checkpoint = lease.checkpoints.back();
        lease.checkpoints.pop_back();
        book_.intervals().release_since(lease.owner, checkpoint);
        model_.release_since(lease.owner, checkpoint);
        return std::nullopt;
    }

    [[nodiscard]] auto close(std::size_t index, bool synchronized) -> std::optional<Failure>
    {
        const auto owner = live_[index].owner;
        live_.erase(live_.begin() + static_cast<std::ptrdiff_t>(index));
        const auto has_reservations = book_.intervals().reservation_count(owner) != 0;
        auto synchronize_calls = 0uz;
        const auto reusable = qnpeps::is_lease_reusable(
            has_reservations,
            [&]
            {
                ++synchronize_calls;
                return synchronized;
            }
        );
        const auto expected_calls = has_reservations ? 1uz : 0uz;
        if (synchronize_calls != expected_calls) return Failure{"release synchronized needlessly"};
        book_.finish_lease(owner, reusable);
        if (reusable)
            model_.release(owner);
        else
            model_.quarantine(owner);
        closed_.push_back(owner);
        if (closed_.size() > k_closed_owner_window) closed_.erase(closed_.begin());
        return std::nullopt;
    }

    [[nodiscard]] auto check_invariants() -> std::optional<Failure>
    {
        const auto& intervals = book_.intervals();
        const auto book_gaps = intervals.free_intervals();
        const auto model_gaps = model_.gaps();
        const auto same_gaps =
            std::equal(book_gaps.begin(), book_gaps.end(), model_gaps.begin(), model_gaps.end());
        if (not same_gaps) return Failure{"free intervals differ from the model"};
        auto owned_bytes = intervals.total(0);
        for (const auto& lease : live_)
        {
            const auto total = intervals.total(lease.owner);
            if (total != model_.total(lease.owner)) return Failure{"lease total differs"};
            if (intervals.reservation_count(lease.owner) != model_.count(lease.owner))
                return Failure{"lease reservation count differs"};
            owned_bytes += total;
        }
        for (const auto owner : closed_)
            if (intervals.reservation_count(owner) != 0) return Failure{"closed lease holds space"};
        if (owned_bytes + intervals.free_bytes() != model_.capacity())
            return Failure{"bytes are neither free, leased, nor quarantined"};
        if (owned_bytes != model_.occupied_bytes()) return Failure{"occupied bytes differ"};
        if (intervals.total(0) != model_.quarantined_bytes())
            return Failure{"quarantined bytes differ"};
        if (model_.quarantined_bytes() < quarantined_floor_)
            return Failure{"quarantined bytes returned"};
        quarantined_floor_ = model_.quarantined_bytes();
        if (intervals.quarantine_count() != model_.count(0))
            return Failure{"quarantined interval count differs"};
        if (intervals.reservation_count() != model_.range_count())
            return Failure{"reservation count differs"};
        if (intervals.high_water() != model_.high_water()) return Failure{"high water differs"};
        if (intervals.largest_free() != model_.largest_gap())
            return Failure{"largest free block differs"};
        return std::nullopt;
    }

    std::mt19937_64 random_;
    ArenaBook book_{};
    ReferenceModel model_;
    bool book_ready_{};
    std::vector<LiveLease> live_{};
    std::vector<Owner> closed_{};
    std::size_t quarantined_floor_{};
    std::size_t operations_done_{};
    Operation last_operation_{Operation::open};
};

struct RunOutcome
{
    std::size_t operations{};
    std::optional<Failure> failure{};
    Operation operation{Operation::open};
};

[[nodiscard]] auto run_seed(std::uint64_t seed) -> RunOutcome
{
    auto shape_random = std::mt19937_64{seed};
    const auto exponent =
        std::uniform_int_distribution<std::size_t>{0uz, k_max_alignment_exponent}(shape_random);
    const auto alignment = 1uz << exponent;
    const auto blocks =
        std::uniform_int_distribution<std::size_t>{1uz, k_max_capacity_blocks}(shape_random);
    auto run = FuzzRun{seed, alignment * blocks, alignment};
    if (not run.is_ready()) return {0uz, Failure{"book rejected the arena shape"}};
    for (auto operation = 0uz; operation < k_operations_per_run; ++operation)
    {
        if (auto failure = run.step())
            return {run.operations_done(), std::move(failure), run.last_operation()};
    }
    auto failure = run.drain();
    return {run.operations_done(), std::move(failure), run.last_operation()};
}

[[nodiscard]] auto parse_seed(const char* text) -> std::optional<std::uint64_t>
{
    char* end{};
    const auto value = std::strtoull(text, &end, 10);
    if (end == text or *end != '\0') return std::nullopt;
    return value;
}
}

auto main(int argc, char** argv) -> int
{
    if (argc > 2)
    {
        std::fprintf(stderr, "[arena_book_fuzz] usage arena_book_fuzz [seed]\n");
        return 2;
    }
    const auto single_seed = argc == 2 ? parse_seed(argv[1]) : std::optional<std::uint64_t>{};
    if (argc == 2 and not single_seed)
    {
        std::fprintf(stderr, "[arena_book_fuzz] seed is not a number %s\n", argv[1]);
        return 2;
    }
    const auto start = std::chrono::steady_clock::now();
    auto seed = single_seed.value_or(1);
    auto last_seed = seed;
    auto runs = 0uz;
    auto operations = 0uz;
    while (true)
    {
        last_seed = seed;
        const auto outcome = run_seed(seed);
        operations += outcome.operations;
        ++runs;
        if (outcome.failure)
        {
            std::printf(
                "[arena_book_fuzz] FAIL seed=%llu operation=%zu kind=%s total_operations=%zu "
                "message=%s\n",
                static_cast<unsigned long long>(seed),
                outcome.operations,
                operation_name(outcome.operation),
                operations,
                outcome.failure->message.c_str()
            );
            return 1;
        }
        if (single_seed) break;
        ++seed;
        const auto past_fixed = runs >= k_fixed_seed_count;
        if (past_fixed and std::chrono::steady_clock::now() - start >= k_budget) break;
    }
    std::printf(
        "[arena_book_fuzz] PASS runs=%zu operations=%zu last_seed=%llu\n",
        runs,
        operations,
        static_cast<unsigned long long>(last_seed)
    );
    return 0;
}
