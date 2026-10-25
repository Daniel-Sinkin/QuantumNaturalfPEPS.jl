#ifndef QNPEPS_ARENA_CURSOR_CUH
#define QNPEPS_ARENA_CURSOR_CUH

#include "core/arena_intervals.hpp"
#include "core/cuda_utils.cuh"

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <utility>

namespace qnpeps
{
class DeviceArena;

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
    auto release() noexcept -> void;

  private:
    friend class DeviceArena;

    ArenaLease(
        std::shared_ptr<DeviceArena> arena,
        ArenaIntervals::Owner owner,
        int device,
        cudaStream_t stream
    ) noexcept
        : arena_(std::move(arena)), owner_(owner), device_(device), stream_(stream)
    {
    }

    std::shared_ptr<DeviceArena> arena_{};
    ArenaIntervals::Owner owner_{};
    int device_{-1};
    cudaStream_t stream_{};
};

class ArenaCursor
{
  public:
    ArenaCursor() = default;

    [[nodiscard]] static auto carve(void* base, usize capacity) noexcept -> ArenaCursor
    {
        if (not base)
        {
            assert(false);
            set_err(QNPEPS_ERR_INTERNAL);
            return {};
        }
        return ArenaCursor{Mode::carve, static_cast<char*>(base), capacity};
    }

    [[nodiscard]] static auto persistent(ArenaLease& lease) noexcept -> ArenaCursor
    {
        return ArenaCursor{lease};
    }

    template <typename T>
    auto take(usize count) -> T*
    {
        if (mode_ == Mode::unbound)
        {
            assert(false);
            set_err(QNPEPS_ERR_INTERNAL);
            return nullptr;
        }

        if (count > std::numeric_limits<usize>::max() / sizeof(T))
        {
            set_err(QNPEPS_ERR_OOM);
            return nullptr;
        }
        const auto bytes = count * sizeof(T);
        if (mode_ == Mode::persistent) return reinterpret_cast<T*>(lease_->take(bytes));

        constexpr auto alignment_padding = k_device_malloc_align - 1;
        constexpr auto max_safe_end = std::numeric_limits<usize>::max() - alignment_padding;
        const auto begin = device_align(offset_);
        const auto end_limit = std::min(capacity_, max_safe_end);
        if (begin > end_limit or count > (end_limit - begin) / sizeof(T))
        {
            set_err(QNPEPS_ERR_OOM);
            return nullptr;
        }
        const auto end = begin + count * sizeof(T);

        offset_ = end;
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
            if (factor > std::numeric_limits<usize>::max() / count)
            {
                set_err(QNPEPS_ERR_OOM);
                return nullptr;
            }
            count *= static_cast<usize>(factor);
        }
        return take<T>(count);
    }

    [[nodiscard]] auto take_subarena(usize bytes) -> ArenaCursor
    {
        auto* subarena = take<char>(bytes);
        if (err_state() != QNPEPS_OK) return {};
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
        if (mode_ != Mode::carve)
        {
            assert(false);
            set_err(QNPEPS_ERR_INTERNAL);
            return;
        }
        offset_ = 0;
    }

  private:
    friend class DeviceArena;

    enum class Mode : u8
    {
        unbound,
        carve,
        persistent
    };

    ArenaCursor(Mode mode, char* base, usize capacity) noexcept
        : mode_(mode), base_(base), capacity_(capacity)
    {
    }

    explicit ArenaCursor(ArenaLease& lease) noexcept : mode_(Mode::persistent), lease_(&lease) {}

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
                CUDA_NOCHECK(cudaStreamSynchronize(stream_));
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
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (err_state() != QNPEPS_OK) return 0;
    const auto headroom = std::max(free_bytes / 10, k_device_arena_minimum_headroom);
    if (free_bytes <= headroom) return 0;
    const auto requested = free_bytes - headroom;
    return requested & ~(k_device_malloc_align - 1);
}

inline auto reserve_device_arena(char*& base, usize& capacity) -> void
{
    if (base)
    {
        set_err(QNPEPS_ERR_INTERNAL);
        return;
    }
    capacity = arena_reservation_bytes();
    if (capacity == 0)
    {
        set_err(QNPEPS_ERR_OOM);
        return;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&base), capacity));
}

template <typename T>
inline auto maybe_free_device(T*& pointer) -> void
{
    if (not pointer) return;
    CUDA_NOCHECK(cudaFree(pointer));
    pointer = nullptr;
}

template <typename T>
inline auto maybe_free_host(T*& pointer) -> void
{
    if (not pointer) return;
    CUDA_NOCHECK(cudaFreeHost(pointer));
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
        std::shared_ptr<DeviceArena> arena{new (std::nothrow) DeviceArena{device}};
        if (not arena)
        {
            set_err(QNPEPS_ERR_OOM);
            return nullptr;
        }
        int caller_device{};
        CUDA_CHECK(cudaGetDevice(&caller_device));
        if (err_state() != QNPEPS_OK) return nullptr;
        const auto switched = caller_device != device;
        if (switched) CUDA_CHECK(cudaSetDevice(device));
        if (err_state() == QNPEPS_OK)
            CUDA_CHECK(
                cudaEventCreateWithFlags(&arena->transient_completion_, cudaEventDisableTiming)
            );
        if (err_state() == QNPEPS_OK) reserve_device_arena(arena->base_, arena->capacity_);
        if (switched)
        {
            const auto restore_status = cudaSetDevice(caller_device);
            if (restore_status != cudaSuccess and err_state() == QNPEPS_OK)
                set_cuda_err(restore_status);
        }
        if (err_state() != QNPEPS_OK) return nullptr;
        if (not arena->intervals_.reset(arena->capacity_, k_device_malloc_align))
        {
            set_err(QNPEPS_ERR_INTERNAL);
            return nullptr;
        }
        return arena;
    }

    ~DeviceArena()
    {
        int caller_device{};
        if (cudaGetDevice(&caller_device) != cudaSuccess) return;
        const auto switched = caller_device != device_;
        if (switched and cudaSetDevice(device_) != cudaSuccess) return;
        if (transient_completion_) CUDA_NOCHECK(cudaEventDestroy(transient_completion_));
        release_device_arena(base_, capacity_);
        if (switched) CUDA_NOCHECK(cudaSetDevice(caller_device));
    }

    DeviceArena(const DeviceArena&) = delete;
    DeviceArena(DeviceArena&&) = delete;
    auto operator=(const DeviceArena&) -> DeviceArena& = delete;
    auto operator=(DeviceArena&&) -> DeviceArena& = delete;

    [[nodiscard]] auto acquire_lease(cudaStream_t stream) -> std::unique_ptr<ArenaLease>
    {
        std::lock_guard<std::mutex> lock{allocation_mutex_};
        if (next_owner_ == std::numeric_limits<ArenaIntervals::Owner>::max())
        {
            set_err(QNPEPS_ERR_OOM);
            return nullptr;
        }
        const auto owner = next_owner_++;
        auto* lease = new (std::nothrow) ArenaLease{shared_from_this(), owner, device_, stream};
        if (not lease)
        {
            set_err(QNPEPS_ERR_OOM);
            return nullptr;
        }
        return std::unique_ptr<ArenaLease>{lease};
    }

    [[nodiscard]] auto transient_cursor(cudaStream_t stream) -> TransientArenaCursor
    {
        std::unique_lock<std::mutex> allocation_lock{allocation_mutex_};
        const auto persistent_bytes = intervals_.high_water();
        if (transient_pending_)
        {
            CUDA_CHECK(cudaStreamWaitEvent(stream, transient_completion_, 0));
            if (err_state() != QNPEPS_OK)
            {
                CUDA_NOCHECK(cudaEventSynchronize(transient_completion_));
                transient_pending_ = false;
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

  private:
    friend class ArenaLease;

    explicit DeviceArena(int device) : device_(device) {}

    [[nodiscard]] auto wait_for_transient_unlocked() -> bool
    {
        if (not transient_pending_) return true;
        const auto status = cudaEventSynchronize(transient_completion_);
        if (status != cudaSuccess)
        {
            set_cuda_err(status);
            return false;
        }
        transient_pending_ = false;
        return true;
    }

    [[nodiscard]] auto lease_take(ArenaIntervals::Owner owner, usize bytes) -> void*
    {
        std::lock_guard<std::mutex> lock{allocation_mutex_};
        if (not wait_for_transient_unlocked()) return nullptr;
        const auto interval = intervals_.acquire(owner, bytes);
        if (not interval)
        {
            set_err(QNPEPS_ERR_OOM);
            return nullptr;
        }
        return base_ + interval->begin;
    }

    [[nodiscard]] auto lease_total(ArenaIntervals::Owner owner) const noexcept -> usize
    {
        std::lock_guard<std::mutex> lock{allocation_mutex_};
        return intervals_.total(owner);
    }

    [[nodiscard]] auto lease_remaining() const noexcept -> usize
    {
        std::lock_guard<std::mutex> lock{allocation_mutex_};
        return intervals_.largest_free();
    }

    [[nodiscard]] auto lease_has_reservations(ArenaIntervals::Owner owner) const noexcept -> bool
    {
        std::lock_guard<std::mutex> lock{allocation_mutex_};
        return intervals_.reservation_count(owner) != 0;
    }

    [[nodiscard]] auto high_water() const noexcept -> usize
    {
        std::lock_guard<std::mutex> lock{allocation_mutex_};
        return intervals_.high_water();
    }

    [[nodiscard]] auto reservation_count() const noexcept -> usize
    {
        std::lock_guard<std::mutex> lock{allocation_mutex_};
        return intervals_.reservation_count();
    }

    auto finish_lease(ArenaIntervals::Owner owner, bool reusable) noexcept -> void
    {
        std::lock_guard<std::mutex> lock{allocation_mutex_};
        if (reusable)
            intervals_.release(owner);
        else
            intervals_.quarantine(owner);
    }

    int device_{-1};
    char* base_{};
    usize capacity_{};
    mutable std::mutex allocation_mutex_{};
    cudaEvent_t transient_completion_{};
    bool transient_pending_{};
    ArenaIntervals intervals_{};
    ArenaIntervals::Owner next_owner_{1};
};

inline ArenaLease::~ArenaLease()
{
    release();
}

[[nodiscard]] inline auto ArenaLease::take(usize bytes) -> void*
{
    if (not arena_ or owner_ == 0)
    {
        set_err(QNPEPS_ERR_INTERNAL);
        return nullptr;
    }
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

inline auto ArenaLease::release() noexcept -> void
{
    if (not arena_ or owner_ == 0) return;
    auto reusable = not arena_->lease_has_reservations(owner_);
    auto caller_device = int{};
    auto switched = false;
    if (not reusable)
    {
        const auto device_status = cudaGetDevice(&caller_device);
        reusable = device_status == cudaSuccess;
        if (device_status != cudaSuccess and err_state() == QNPEPS_OK) set_cuda_err(device_status);
        if (reusable and caller_device != device_)
        {
            const auto switch_status = cudaSetDevice(device_);
            reusable = switch_status == cudaSuccess;
            switched = reusable;
            if (switch_status != cudaSuccess and err_state() == QNPEPS_OK)
                set_cuda_err(switch_status);
        }
        if (reusable and not stream_)
        {
            reusable = false;
            if (err_state() == QNPEPS_OK) set_err(QNPEPS_ERR_INTERNAL);
        }
        if (reusable)
        {
            const auto completion_status = cudaStreamSynchronize(stream_);
            reusable = completion_status == cudaSuccess;
            if (completion_status != cudaSuccess and err_state() == QNPEPS_OK)
                set_cuda_err(completion_status);
        }
    }
    arena_->finish_lease(owner_, reusable);
    if (switched)
    {
        const auto restore_status = cudaSetDevice(caller_device);
        if (restore_status != cudaSuccess and err_state() == QNPEPS_OK)
            set_cuda_err(restore_status);
    }
    owner_ = 0;
    stream_ = nullptr;
    device_ = -1;
    arena_.reset();
}

[[nodiscard]] inline auto shared_device_arena(int device) -> std::shared_ptr<DeviceArena>
{
    static std::mutex registry_mutex{};
    static std::map<int, std::shared_ptr<DeviceArena>> arenas{};
    const std::lock_guard<std::mutex> lock{registry_mutex};
    const auto existing = arenas.find(device);
    if (existing != arenas.end()) return existing->second;
    auto arena = DeviceArena::create(device);
    if (arena) arenas.emplace(device, arena);
    return arena;
}
}

#endif
