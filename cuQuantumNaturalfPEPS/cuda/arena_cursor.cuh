#ifndef QNPEPS_ARENA_CURSOR_CUH
#define QNPEPS_ARENA_CURSOR_CUH

#include "cuda_utils.cuh"

#include <algorithm>
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
            assert(false);
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
            assert(false);
            set_err(QNPEPS_ERR_INTERNAL);
            return nullptr;
        }

        constexpr auto alignment_padding = k_device_malloc_align - 1;
        constexpr auto max_safe_end = std::numeric_limits<usize>::max() - alignment_padding;
        const auto begin = device_align(offset_);
        const auto end_limit = [&]
        {
            auto out = max_safe_end;
            if (mode_ == Mode::carve) return std::min(capacity_, out);
            return out;
        }();
        if (begin > end_limit or count > (end_limit - begin) / sizeof(T))
        {
            set_err(QNPEPS_ERR_OOM);
            return nullptr;
        }
        const auto end = begin + count * sizeof(T);

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
            assert(false);
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
