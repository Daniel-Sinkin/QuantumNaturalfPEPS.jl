#ifndef QNPEPS_ARENA_CURSOR_CUH
#define QNPEPS_ARENA_CURSOR_CUH

#include "cuda_utils.cuh"

#include <cassert>
#include <limits>

namespace qnpeps
{
class ArenaCursor
{
  public:
    ArenaCursor() = default;

    [[nodiscard]] static auto measure() noexcept -> ArenaCursor
    {
        return ArenaCursor{Mode::measure, nullptr, 0};
    }

    [[nodiscard]] static auto carve(void* base, usize capacity) noexcept -> ArenaCursor
    {
        if (not base)
        {
            assert(base);
            set_err(QNPEPS_ERR_INTERNAL);
            return {};
        }
        return ArenaCursor{Mode::carve, static_cast<char*>(base), capacity};
    }

    template <typename T>
    auto take(usize count) -> T*
    {
        if (mode_ == Mode::unbound)
        {
            assert(mode_ != Mode::unbound);
            set_err(QNPEPS_ERR_INTERNAL);
            return nullptr;
        }

        constexpr auto max = std::numeric_limits<usize>::max();
        constexpr auto alignment_padding = k_device_malloc_align - 1;
        if (offset_ > max - alignment_padding)
        {
            set_err(QNPEPS_ERR_OOM);
            return nullptr;
        }
        const auto begin = device_align(offset_);
        if (count > (max - begin) / sizeof(T))
        {
            set_err(QNPEPS_ERR_OOM);
            return nullptr;
        }
        const auto end = begin + count * sizeof(T);
        if (end > max - alignment_padding or (mode_ == Mode::carve and end > capacity_))
        {
            set_err(QNPEPS_ERR_OOM);
            return nullptr;
        }

        offset_ = end;
        if (mode_ == Mode::measure) return nullptr;
        return reinterpret_cast<T*>(base_ + begin);
    }

    [[nodiscard]] auto take_subarena(usize bytes) -> ArenaCursor
    {
        auto* subarena = take<char>(bytes);
        if (err_state() != QNPEPS_OK) return {};
        if (mode_ == Mode::measure) return measure();
        return carve(subarena, bytes);
    }

    [[nodiscard]] auto total() const noexcept -> usize
    {
        if (mode_ == Mode::unbound) return 0;
        return device_align(offset_);
    }

    auto rewind() -> void
    {
        if (mode_ == Mode::unbound)
        {
            assert(mode_ != Mode::unbound);
            set_err(QNPEPS_ERR_INTERNAL);
            return;
        }
        offset_ = 0;
    }

  private:
    enum class Mode : u8
    {
        unbound,
        measure,
        carve
    };

    ArenaCursor(Mode mode, char* base, usize capacity) noexcept
        : mode_(mode), base_(base), capacity_(capacity)
    {
    }

    Mode mode_{Mode::unbound};
    char* base_{};
    usize capacity_{};
    usize offset_{};
};
}

#endif
