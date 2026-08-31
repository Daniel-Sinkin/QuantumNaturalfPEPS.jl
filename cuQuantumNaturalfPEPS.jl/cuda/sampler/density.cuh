#ifndef QNPEPS_SAMPLER_DENSITY_CUH
#define QNPEPS_SAMPLER_DENSITY_CUH

#include "core/qnpeps_ctx.cuh"

#include <span>
#include <vector>

namespace qnpeps::sampler
{
class DensityLane
{
  public:
    DensityLane(Sampler& sampler, int lane);
    ~DensityLane();
    DensityLane(const DensityLane&) = delete;
    DensityLane(DensityLane&&) = delete;
    auto operator=(const DensityLane&) -> DensityLane& = delete;
    auto operator=(DensityLane&&) -> DensityLane& = delete;

  private:
    Sampler& sampler_;
    int lane_{};
};

auto build_density_state_row(
    qnpeps_ctx& ctx,
    const SamplerConfig& config,
    int row,
    int environment_index,
    std::span<const int> input_bonds,
    std::vector<int>& output_bonds
) -> bool;
auto build_density_projected_row(
    qnpeps_ctx& ctx,
    const SamplerConfig& config,
    int row,
    int input_environment_index,
    int output_environment_index,
    std::span<const int> input_bonds,
    std::vector<int>& output_bonds
) -> bool;
}

#endif
