#ifndef QNPEPS_RANGEFINDER_RNG_CUH
#define QNPEPS_RANGEFINDER_RNG_CUH

#include "types.cuh"

#include <random>
#include <span>

namespace qnpeps
{
class RangefinderRng
{
  public:
    explicit RangefinderRng(u64 seed) : engine_(seed) {}

    [[nodiscard]] static auto from_seed_and_width(u64 seed, int width) -> RangefinderRng
    {
        return RangefinderRng{seed ^ (static_cast<u64>(width) << k_width_seed_shift)};
    }

    template <typename Complex>
    auto fill_complex_normal(std::span<Complex> values) -> void
    {
        for (auto& value : values)
            value = Complex{normal_(engine_), normal_(engine_)};
    }

  private:
    static constexpr int k_width_seed_shift{20};

    std::mt19937_64 engine_{};
    std::normal_distribution<f32> normal_{0.0f, 1.0f};
};
}

#endif
