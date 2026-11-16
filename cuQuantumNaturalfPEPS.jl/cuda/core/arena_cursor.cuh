#ifndef QNPEPS_ARENA_CURSOR_CUH
#define QNPEPS_ARENA_CURSOR_CUH

#include "core/arena_book.hpp"
#include "core/cuda_utils.cuh"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <utility>

namespace qnpeps
{
class DeviceArena;

struct ArenaReading
{
    usize capacity{};
    usize in_use{};
    usize top{};
    usize largest_free{};
    usize reservation_count{};
    usize quarantined_count{};
};

class ArenaLease
{
  public:
    ~ArenaLease();

    ArenaLease(const ArenaLease&) = delete;
    ArenaLease(ArenaLease&&) = delete;
    auto operator=(const ArenaLease&) -> ArenaLease& = delete;
    auto operator=(ArenaLease&&) -> ArenaLease& = delete;

    [[nodiscard]] auto take(usize bytes) -> void*;
    [[nodiscard]] auto total() const noexcept -> usize;
    [[nodiscard]] auto remaining() const noexcept -> usize;
    [[nodiscard]] auto high_water() const noexcept -> usize;
    [[nodiscard]] auto reservation_count() const noexcept -> usize;
    auto release() -> void;

  private:
    friend class DeviceArena;

    ArenaLease(
        std::shared_ptr<DeviceArena> arena, ArenaBook::Owner owner, int device, cudaStream_t stream
    )
        : arena_(std::move(arena)), owner_(owner), device_(device), stream_(stream)
    {
    }

    std::shared_ptr<DeviceArena> arena_{};
    ArenaBook::Owner owner_{};
    int device_{-1};
    cudaStream_t stream_{};
};

class ArenaCursor
{
  public:
    ArenaCursor() = default;

    [[nodiscard]] static auto carve(void* base, usize capacity) -> ArenaCursor
    {
        require(base != nullptr, "arena of {} bytes has no base pointer", capacity);
        return ArenaCursor{Mode::carve, static_cast<char*>(base), capacity};
    }

    [[nodiscard]] static auto persistent(ArenaLease& lease) noexcept -> ArenaCursor
    {
        return ArenaCursor{lease};
    }

    [[nodiscard]] static auto measuring() noexcept -> ArenaCursor
    {
        return ArenaCursor{Mode::measuring, nullptr, std::numeric_limits<usize>::max()};
    }

    [[nodiscard]] auto is_persistent() const noexcept -> bool { return mode_ == Mode::persistent; }

    template <typename T>
    auto take(usize count) -> T*
    {
        require(mode_ != Mode::unbound, "arena take from an unbound cursor");
        require(
            count <= std::numeric_limits<usize>::max() / sizeof(T),
            Status::out_of_memory,
            "arena take of {} elements of {} bytes overflows the size range",
            count,
            sizeof(T)
        );
        const auto bytes = count * sizeof(T);
        if (mode_ == Mode::persistent) return reinterpret_cast<T*>(lease_->take(bytes));

        constexpr auto alignment_padding = k_device_malloc_align - 1;
        constexpr auto max_safe_end = std::numeric_limits<usize>::max() - alignment_padding;
        const auto begin = device_align(offset_);
        const auto end_limit = std::min(capacity_, max_safe_end);
        if (begin > end_limit or count > (end_limit - begin) / sizeof(T))
        {
            fail(
                Status::out_of_memory,
                "arena needs {} bytes, {} of {} in use",
                bytes,
                begin,
                capacity_
            );
        }
        const auto end = begin + bytes;

        offset_ = end;
        if (mode_ == Mode::measuring) return reinterpret_cast<T*>(k_measured_address + begin);
        return reinterpret_cast<T*>(base_ + begin);
    }

    template <typename T>
    auto take_product(std::initializer_list<u64> factors) -> T*
    {
        usize count{1};
        for (const auto factor : factors)
        {
            if (factor == 0)
            {
                count = 0;
                break;
            }
            require(
                factor <= std::numeric_limits<usize>::max() / count,
                Status::out_of_memory,
                "arena take factor {} overflows the element count {}",
                factor,
                count
            );
            count *= static_cast<usize>(factor);
        }
        return take<T>(count);
    }

    [[nodiscard]] auto take_subarena(usize bytes) -> ArenaCursor
    {
        auto* subarena = take<char>(bytes);
        return carve(subarena, bytes);
    }

    [[nodiscard]] auto total() const noexcept -> usize
    {
        if (mode_ == Mode::persistent) return lease_->total();
        return total_unlocked();
    }

    [[nodiscard]] auto remaining() const noexcept -> usize
    {
        if (mode_ == Mode::unbound) return 0;
        if (mode_ == Mode::persistent) return lease_->remaining();
        const auto begin = device_align(offset_);
        return begin < capacity_ ? capacity_ - begin : 0;
    }

    [[nodiscard]] auto high_water() const noexcept -> usize
    {
        return mode_ == Mode::persistent ? lease_->high_water() : total_unlocked();
    }

    [[nodiscard]] auto reservation_count() const noexcept -> usize
    {
        return mode_ == Mode::persistent ? lease_->reservation_count() : 0;
    }

    auto rewind() -> void
    {
        require(mode_ == Mode::carve, "arena rewind of a cursor that does not carve");
        offset_ = 0;
    }

  private:
    friend class DeviceArena;

    enum class Mode : u8
    {
        unbound,
        carve,
        persistent,
        measuring
    };

    static constexpr std::uintptr_t k_measured_address{k_device_malloc_align};

    ArenaCursor(Mode mode, char* base, usize capacity)
        : mode_(mode), base_(base), capacity_(capacity)
    {
    }

    explicit ArenaCursor(ArenaLease& lease) : mode_(Mode::persistent), lease_(&lease) {}

    [[nodiscard]] auto total_unlocked() const noexcept -> usize
    {
        if (mode_ == Mode::unbound) return 0;
        return device_align(offset_);
    }

    Mode mode_{Mode::unbound};
    char* base_{};
    usize capacity_{};
    usize offset_{};
    ArenaLease* lease_{};
};

class TransientArenaCursor
{
  public:
    ~TransientArenaCursor()
    {
        if (stream_ and completion_event_ and completion_pending_)
        {
            const auto record_status = cudaEventRecord(*completion_event_, stream_);
            if (record_status == cudaSuccess)
                *completion_pending_ = true;
            else
            {
                unchecked(cudaStreamSynchronize(stream_));
                *completion_pending_ = false;
            }
        }
    }

    TransientArenaCursor(const TransientArenaCursor&) = delete;
    TransientArenaCursor(TransientArenaCursor&& other) noexcept
        : stream_(std::exchange(other.stream_, nullptr)),
          completion_event_(std::exchange(other.completion_event_, nullptr)),
          completion_pending_(std::exchange(other.completion_pending_, nullptr)),
          allocation_lock_(std::move(other.allocation_lock_)), cursor_(other.cursor_)
    {
    }
    auto operator=(const TransientArenaCursor&) -> TransientArenaCursor& = delete;
    auto operator=(TransientArenaCursor&&) -> TransientArenaCursor& = delete;

    [[nodiscard]] auto cursor() noexcept -> ArenaCursor& { return cursor_; }

  private:
    friend class DeviceArena;

    TransientArenaCursor(
        cudaStream_t stream,
        cudaEvent_t* completion_event,
        bool* completion_pending,
        std::unique_lock<std::mutex>&& allocation_lock,
        ArenaCursor cursor
    )
        : stream_(stream), completion_event_(completion_event),
          completion_pending_(completion_pending), allocation_lock_(std::move(allocation_lock)),
          cursor_(cursor)
    {
    }

    cudaStream_t stream_{};
    cudaEvent_t* completion_event_{};
    bool* completion_pending_{};
    std::unique_lock<std::mutex> allocation_lock_{};
    ArenaCursor cursor_{};
};

inline constexpr usize k_device_arena_minimum_headroom{2_uz * 1024_uz * 1024_uz * 1024_uz};

[[nodiscard]] inline auto arena_reservation_bytes() -> usize
{
    usize free_bytes{};
    usize total_bytes{};
    check(cudaMemGetInfo(&free_bytes, &total_bytes));
    const auto headroom = std::max(free_bytes / 10, k_device_arena_minimum_headroom);
    if (free_bytes <= headroom) return 0;
    const auto requested = free_bytes - headroom;
    return requested & ~(k_device_malloc_align - 1);
}

inline auto reserve_device_arena(char*& base, usize& capacity) -> void
{
    require(base == nullptr, "device arena reserved twice");
    capacity = arena_reservation_bytes();
    require(
        capacity != 0,
        Status::out_of_memory,
        "device arena reservation found no free memory above the headroom"
    );
    check(cudaMalloc(reinterpret_cast<void**>(&base), capacity));
}

template <typename T>
inline auto maybe_free_device(T*& pointer) -> void
{
    if (not pointer) return;
    unchecked(cudaFree(pointer));
    pointer = nullptr;
}

template <typename T>
inline auto maybe_free_host(T*& pointer) -> void
{
    if (not pointer) return;
    unchecked(cudaFreeHost(pointer));
    pointer = nullptr;
}

inline auto release_device_arena(char*& base, usize& capacity) -> void
{
    maybe_free_device(base);
    capacity = 0;
}

class DeviceArena : public std::enable_shared_from_this<DeviceArena>
{
  public:
    [[nodiscard]] static auto create(int device) -> std::shared_ptr<DeviceArena>
    {
        std::shared_ptr<DeviceArena> arena{new DeviceArena{device}};
        DeviceScope scope{device};
        check(cudaEventCreateWithFlags(&arena->transient_completion_, cudaEventDisableTiming));
        reserve_device_arena(arena->base_, arena->capacity_);
        scope.restore();
        const auto book_ready = arena->book_.reset(arena->capacity_, k_device_malloc_align);
        require(book_ready, "device arena interval map rejects capacity {}", arena->capacity_);
        return arena;
    }

    ~DeviceArena()
    {
        int caller_device{};
        if (cudaGetDevice(&caller_device) != cudaSuccess) return;
        const auto switched = caller_device != device_;
        if (switched and cudaSetDevice(device_) != cudaSuccess) return;
        if (transient_completion_) unchecked(cudaEventDestroy(transient_completion_));
        release_device_arena(base_, capacity_);
        if (switched) unchecked(cudaSetDevice(caller_device));
    }

    DeviceArena(const DeviceArena&) = delete;
    DeviceArena(DeviceArena&&) = delete;
    auto operator=(const DeviceArena&) -> DeviceArena& = delete;
    auto operator=(DeviceArena&&) -> DeviceArena& = delete;

    [[nodiscard]] auto acquire_lease(cudaStream_t stream) -> std::unique_ptr<ArenaLease>
    {
        std::lock_guard<std::mutex> lock{allocation_mutex_};
        const auto owner = book_.open_lease();
        require(owner.has_value(), Status::out_of_memory, "device arena ran out of lease owners");
        return std::unique_ptr<ArenaLease>{
            new ArenaLease{shared_from_this(), *owner, device_, stream}
        };
    }

    [[nodiscard]] auto transient_cursor(cudaStream_t stream) -> TransientArenaCursor
    {
        std::unique_lock<std::mutex> allocation_lock{allocation_mutex_};
        const auto persistent_bytes = book_.intervals().high_water();
        if (transient_pending_)
        {
            const auto wait_status = cudaStreamWaitEvent(stream, transient_completion_, 0);
            if (wait_status != cudaSuccess)
            {
                unchecked(cudaEventSynchronize(transient_completion_));
                transient_pending_ = false;
                check(wait_status);
            }
        }
        return TransientArenaCursor{
            stream,
            &transient_completion_,
            &transient_pending_,
            std::move(allocation_lock),
            ArenaCursor::carve(base_ + persistent_bytes, capacity_ - persistent_bytes),
        };
    }

    [[nodiscard]] auto capacity() const noexcept -> usize { return capacity_; }

    [[nodiscard]] auto reading() const noexcept -> ArenaReading
    {
        std::lock_guard<std::mutex> lock{allocation_mutex_};
        return {
            .capacity = capacity_,
            .in_use = capacity_ - book_.intervals().free_bytes(),
            .top = book_.intervals().high_water(),
            .largest_free = book_.intervals().largest_free(),
            .reservation_count = book_.intervals().reservation_count(),
            .quarantined_count = book_.intervals().quarantine_count(),
        };
    }

  private:
    friend class ArenaLease;

    explicit DeviceArena(int device) : device_(device) {}

    auto wait_for_transient_unlocked() -> void
    {
        if (not transient_pending_) return;
        check(cudaEventSynchronize(transient_completion_));
        transient_pending_ = false;
    }

    [[nodiscard]] auto lease_take(ArenaBook::Owner owner, usize bytes) -> void*
    {
        std::lock_guard<std::mutex> lock{allocation_mutex_};
        wait_for_transient_unlocked();
        const auto interval = book_.take(owner, bytes);
        if (not interval)
        {
            fail(
                Status::out_of_memory,
                "arena needs {} bytes, {} of {} in use, largest free block {}",
                bytes,
                capacity_ - book_.intervals().free_bytes(),
                capacity_,
                book_.intervals().largest_free()
            );
        }
        return base_ + interval->begin;
    }

    [[nodiscard]] auto lease_total(ArenaBook::Owner owner) const noexcept -> usize
    {
        std::lock_guard<std::mutex> lock{allocation_mutex_};
        return book_.intervals().total(owner);
    }

    [[nodiscard]] auto lease_remaining() const noexcept -> usize
    {
        std::lock_guard<std::mutex> lock{allocation_mutex_};
        return book_.intervals().largest_free();
    }

    [[nodiscard]] auto lease_has_reservations(ArenaBook::Owner owner) const noexcept -> bool
    {
        std::lock_guard<std::mutex> lock{allocation_mutex_};
        return book_.intervals().reservation_count(owner) != 0;
    }

    [[nodiscard]] auto high_water() const noexcept -> usize
    {
        std::lock_guard<std::mutex> lock{allocation_mutex_};
        return book_.intervals().high_water();
    }

    [[nodiscard]] auto reservation_count() const noexcept -> usize
    {
        std::lock_guard<std::mutex> lock{allocation_mutex_};
        return book_.intervals().reservation_count();
    }

    auto finish_lease(ArenaBook::Owner owner, bool reusable) -> void
    {
        std::lock_guard<std::mutex> lock{allocation_mutex_};
        book_.finish_lease(owner, reusable);
    }

    int device_{-1};
    char* base_{};
    usize capacity_{};
    mutable std::mutex allocation_mutex_{};
    cudaEvent_t transient_completion_{};
    bool transient_pending_{};
    ArenaBook book_{};
};

inline ArenaLease::~ArenaLease()
{
    release();
}

[[nodiscard]] inline auto ArenaLease::take(usize bytes) -> void*
{
    require(arena_ != nullptr and owner_ != 0, "arena take from a released lease");
    return arena_->lease_take(owner_, bytes);
}

[[nodiscard]] inline auto ArenaLease::total() const noexcept -> usize
{
    return arena_ and owner_ != 0 ? arena_->lease_total(owner_) : 0;
}

[[nodiscard]] inline auto ArenaLease::remaining() const noexcept -> usize
{
    return arena_ and owner_ != 0 ? arena_->lease_remaining() : 0;
}

[[nodiscard]] inline auto ArenaLease::high_water() const noexcept -> usize
{
    return arena_ and owner_ != 0 ? arena_->high_water() : 0;
}

[[nodiscard]] inline auto ArenaLease::reservation_count() const noexcept -> usize
{
    return arena_ and owner_ != 0 ? arena_->reservation_count() : 0;
}

inline auto ArenaLease::release() -> void
{
    if (not arena_ or owner_ == 0) return;
    auto caller_device = int{};
    auto switched = false;
    const auto synchronize = [&]
    {
        if (cudaGetDevice(&caller_device) != cudaSuccess) return false;
        if (caller_device != device_)
        {
            if (cudaSetDevice(device_) != cudaSuccess) return false;
            switched = true;
        }
        return stream_ != nullptr and cudaStreamSynchronize(stream_) == cudaSuccess;
    };
    const auto reusable = is_lease_reusable(arena_->lease_has_reservations(owner_), synchronize);
    arena_->finish_lease(owner_, reusable);
    if (switched) unchecked(cudaSetDevice(caller_device));
    owner_ = 0;
    stream_ = nullptr;
    device_ = -1;
    arena_.reset();
}

struct DeviceArenaRegistry
{
    std::mutex mutex{};
    std::map<int, std::shared_ptr<DeviceArena>> arenas{};
};

[[nodiscard]] inline auto device_arena_registry() noexcept -> DeviceArenaRegistry&
{
    static DeviceArenaRegistry registry{};
    return registry;
}

[[nodiscard]] inline auto shared_device_arena(int device) -> std::shared_ptr<DeviceArena>
{
    auto& registry = device_arena_registry();
    const std::lock_guard<std::mutex> lock{registry.mutex};
    const auto existing = registry.arenas.find(device);
    if (existing != registry.arenas.end()) return existing->second;
    auto arena = DeviceArena::create(device);
    registry.arenas.emplace(device, arena);
    return arena;
}

[[nodiscard]] inline auto find_device_arena(int device) -> std::shared_ptr<DeviceArena>
{
    auto& registry = device_arena_registry();
    const std::lock_guard<std::mutex> lock{registry.mutex};
    const auto existing = registry.arenas.find(device);
    if (existing == registry.arenas.end()) return nullptr;
    return existing->second;
}
}

#endif
